# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
Tensor = torch.Tensor


# ============================ 1) OU 数据生成（保留） ============================
def simulate_ou_process_batch(a, m, sigma, x0, T, N, batch_size, device):
    dt = T / N
    t = torch.linspace(0, T, N + 1, device=device)
    W = torch.randn(batch_size, N, device=device) * np.sqrt(dt)
    W = torch.cat([torch.zeros(batch_size, 1, device=device), W.cumsum(1)], dim=1)

    X = torch.zeros(batch_size, N + 1, device=device)
    X[:, 0] = x0
    for i in range(1, N + 1):
        X[:, i] = X[:, i - 1] + a * (m - X[:, i - 1]) * dt + sigma * (W[:, i] - W[:, i - 1])
    return t, W, X


# ============================ 2) Haar 基函数（保留） ============================
def haar_basis(N, T):
    basis = []
    basis.append(lambda t: torch.ones_like(t) / torch.sqrt(torch.tensor(T)))
    level = 0
    while len(basis) < N:
        num_segments = 2 ** level
        for i in range(num_segments):
            if len(basis) >= N:
                break
            def haar_fn(t, level=level, i=i):
                left = i * T / (2 ** level)
                middle = (i + 0.5) * T / (2 ** level)
                right = (i + 1) * T / (2 ** level)
                return torch.where(
                    (t >= left) & (t < middle),
                    torch.ones_like(t) * (2 ** (level / 2)) / torch.sqrt(torch.tensor(T)),
                    torch.where(
                        (t >= middle) & (t < right),
                        -torch.ones_like(t) * (2 ** (level / 2)) / torch.sqrt(torch.tensor(T)),
                        torch.zeros_like(t)
                    )
                )
            basis.append(haar_fn)
        level += 1
    return basis


# ============================ 4) Hermite orthonormal ============================
@torch.no_grad()
def hermite_polynomials_orthonormal_1d(xi: Tensor, M: int) -> Tensor:
    """
    Hermite / sqrt(m!)； if xi~N(0,1) then E[H_i H_j]=δ_ij
    input xi: [J, B]
    output H: [J, B, M+1]
    """
    J, B = xi.shape
    H = torch.empty(J, B, M + 1, device=xi.device, dtype=xi.dtype)
    H[..., 0] = 1.0
    if M >= 1:
        H[..., 1] = xi
    for m in range(2, M + 1):
        H[..., m] = xi * H[..., m - 1] - (m - 1) * H[..., m - 2]
        H[..., m] = H[..., m] / math.sqrt(math.factorial(m))
    return H


# ============================ 5) SDEONet（WCE + Two-MLP + 时间PE 内置） ============================
class SDEONet(nn.Module):
    """
    W ——(internal dW, Haar/Itô left-end point projection)→ ξ
    ξ ——(Hermite + Wick second order)→ features ——(branch-MLP)→ Ψ ∈ R^R
    t ——(PE or raw τ=t/T, through trunk-MLP)→ g(t) ∈ R^R
    output: pred(t) = <Ψ, g(t)>
    """
    class MLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, width: int, depth: int,
                     use_layernorm: bool = False, residual: bool = False):
            super().__init__()
            self.depth = depth
            self.residual = residual
            self.use_layernorm = use_layernorm
            self.project_in = nn.Linear(in_dim, width) if depth > 1 else nn.Linear(in_dim, out_dim)
            self.hiddens = nn.ModuleList()
            self.lns = nn.ModuleList() if use_layernorm else None
            for _ in range(max(0, depth - 2)):
                self.hiddens.append(nn.Linear(width, width))
                if use_layernorm:
                    self.lns.append(nn.LayerNorm(width))
            self.out = nn.Linear(width, out_dim) if depth > 1 else None

        def forward(self, x: Tensor) -> Tensor:
            if self.depth == 1:
                return self.project_in(x)
            h = F.gelu(self.project_in(x))
            for i, layer in enumerate(self.hiddens):
                h2 = F.gelu(layer(h))
                if self.use_layernorm:
                    h2 = self.lns[i](h2)
                h = h + h2 if (self.residual and h2.shape == h.shape) else h2
            return self.out(h)

    def __init__(self,
                 basis_in_dim: int,
                 basis_out_dim: int,
                 device,
                 T: float = 2.0,
                 hermite_M: int = 3,
                 wick_order: int = 2,
                 # trunk 时间位置编码
                 use_posenc: bool = True,
                 pe_dim: int = 32,                 # 必须为偶数（sin+cos）
                 include_raw_time: bool = True,    # 是否把 τ=t/T 原样拼入
                 # 两个 MLP 结构
                 branch_width: int = 256,
                 branch_depth: int = 3,
                 trunk_width: int = 256,
                 trunk_depth: int = 3,
                 use_layernorm: bool = False,
                 residual: bool = False):
        super().__init__()
        assert wick_order in (1, 2)
        if use_posenc:
            assert pe_dim % 2 == 0, "pe_dim must be even (sin and cos pairs)"
        self.device = device
        self.T = float(T)
        self.J = int(basis_in_dim)       # Haar basis number
        self.R = int(basis_out_dim)      # low-rank / trunk dimension
        self.M = int(hermite_M)
        self.wick_order = int(wick_order)
        self.use_posenc = bool(use_posenc)
        self.pe_dim = int(pe_dim)
        self.include_raw_time = bool(include_raw_time)

        # time Haar basis (function list)
        self.encoder_basis = haar_basis(self.J, self.T)

        # Wick second order off-diagonal indices
        if self.wick_order >= 2:
            pair_idx = torch.triu_indices(self.J, self.J, offset=1)
            self.register_buffer("pair_i", pair_idx[0])
            self.register_buffer("pair_j", pair_idx[1])

        # Branch input dimension: 1 + J + [J if M>=2] + J(J-1)/2
        in_dim = 1 + self.J
        if self.wick_order >= 2:
            diag2 = self.J if self.M >= 2 else 0
            in_dim += diag2 + (self.J * (self.J - 1)) // 2

        # Trunk input dimension: PE or raw τ or both concatenated
        trunk_in_dim = 1  # raw τ
        if self.use_posenc:
            trunk_in_dim = self.pe_dim + (1 if self.include_raw_time else 0)

        # two MLPs 
        self.branch = SDEONet.MLP(in_dim, self.R, branch_width, branch_depth,
                                  use_layernorm=use_layernorm, residual=residual)
        self.trunk  = SDEONet.MLP(trunk_in_dim, self.R, trunk_width, trunk_depth,
                                  use_layernorm=use_layernorm, residual=residual)

    # —— time features: PE or raw τ
    def _time_features(self, t_query: Tensor) -> Tensor:
        """
        input:
          t_query: [Tsteps]
        output:
          feats: [Tsteps, trunk_in_dim]
        """
        tau = (t_query / self.T).clamp(0.0, 1.0)              # normalized time
        if not self.use_posenc:
            return tau.unsqueeze(-1)

        L = self.pe_dim // 2
        # same as the implementation in the paper: freqs = (1..L) * π / T; equivalent to tau * (1..L) * π
        freqs = torch.arange(1, L + 1, device=t_query.device, dtype=t_query.dtype) * math.pi
        args = tau.unsqueeze(1) * freqs.unsqueeze(0)          # [Tsteps, L]
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [Tsteps, 2L] = [Tsteps, pe_dim]
        if self.include_raw_time:
            pe = torch.cat([tau.unsqueeze(1), pe], dim=1)     # [Tsteps, 1+pe_dim]
        return pe

    # —— Itô left-end point projection ξ_j = Σ_i e_j(t_{i-1}) ΔW_i
    def _ito_left_projection(self, W: Tensor, t_grid: Tensor) -> Tensor:
        B, Tsteps = W.shape
        dW = W[:, 1:] - W[:, :-1]  # [B, N]
        xis = []
        for phi in self.encoder_basis:
            phi_t = phi(t_grid)              # [Tsteps]
            coef = phi_t[:-1].unsqueeze(0)   # 左端点 [1, N]
            xis.append(torch.sum(coef * dW, dim=1))  # [B]
        return torch.stack(xis, dim=1)       # [B, J]

    # —— Wick second order features
    def _wick_features(self, xi: Tensor) -> Tensor:
        """
        xi: [B, J] → features: [B, D]
        D = 1 + J + [J if M>=2] + J(J-1)/2
        """
        B, J = xi.shape
        H = hermite_polynomials_orthonormal_1d(xi.transpose(0, 1), self.M)  # [J,B,M+1]
        H = H.permute(1, 0, 2).contiguous()                                  # [B,J,M+1]
        parts = [torch.ones(B, 1, device=xi.device, dtype=xi.dtype)]
        H1 = H[:, :, 1]
        parts.append(H1)
        if self.wick_order >= 2:
            if self.M >= 2:
                H2 = H[:, :, 2]
                parts.append(H2)
            pi, pj = self.pair_i, self.pair_j
            parts.append(H1[:, pi] * H1[:, pj])
        return torch.cat(parts, dim=1)

    def forward(self, W: Tensor, t_query: Tensor) -> Tensor:
        """
        W:       [B, Tsteps] —— Brownian path
        t_query: [Tsteps]    —— time grid
        output: pred: [B, Tsteps]
        """
        # 1) ξ（Itô left-end point）
        xi = self._ito_left_projection(W, t_query)   # [B, J]
        # 2) Wick   
        feats = self._wick_features(xi)              # [B, D]
        # 3) Branch Ψ
        Psi = self.branch(feats)                     # [B, R]
        # 4) Trunk g(t): PE or raw τ
        t_feats = self._time_features(t_query)       # [Tsteps, trunk_in_dim]
        g_t = self.trunk(t_feats)                    # [Tsteps, R]
        # 5) reconstruct
        pred = torch.bmm(g_t.unsqueeze(0).expand(Psi.shape[0], -1, -1),
                         Psi.unsqueeze(-1)).squeeze(-1)  # [B, Tsteps]
        return pred


# ============================ 6) training and visualization  ============================
def train_sdeonet(basis_in_dim=32, basis_out_dim=64, T=2.0, batch_size=32,
                  N=200, epochs=3000, a=1.0, m=0.0, sigma=1.0, x0=0.0,
                  hermite_M=3, wick_order=2,
                  # trunk time PE
                  use_posenc=True, pe_dim=32, include_raw_time=True,
                  # two MLPs
                  branch_width=256, branch_depth=3,
                  trunk_width=256, trunk_depth=3,
                  use_layernorm=False, residual=False,
                  lr=1e-3, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)

    model = SDEONet(
        basis_in_dim=basis_in_dim, basis_out_dim=basis_out_dim, device=device, T=T,
        hermite_M=hermite_M, wick_order=wick_order,
        use_posenc=use_posenc, pe_dim=pe_dim, include_raw_time=include_raw_time,
        branch_width=branch_width, branch_depth=branch_depth,
        trunk_width=trunk_width, trunk_depth=trunk_depth,
        use_layernorm=use_layernorm, residual=residual
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        t, W_batch, X_true_batch = simulate_ou_process_batch(
            a=a, m=m, sigma=sigma, x0=x0, T=T, N=N, batch_size=batch_size, device=device
        )
        pred = model(W_batch, t)
        loss = loss_fn(pred, X_true_batch)

        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model, device, dict(N=N, T=T, a=a, m=m, sigma=sigma, x0=x0)


def plot_sdeonet_prediction(model, device, meta, title_suffix=""):
    t, W, X_true = simulate_ou_process_batch(
        a=meta["a"], m=meta["m"], sigma=meta["sigma"], x0=meta["x0"],
        T=meta["T"], N=meta["N"], batch_size=1, device=device
    )
    with torch.no_grad():
        pred = model(W, t).cpu().squeeze(0).numpy()
    t_np = t.cpu().numpy()
    X_true_np = X_true.cpu().squeeze(0).numpy()

    plt.figure(figsize=(8, 6))
    plt.plot(t_np, X_true_np, label='Ground Truth', linewidth=2)
    plt.plot(t_np, pred, '--', label='Prediction', linewidth=2)
    plt.xlabel('Time'); plt.ylabel('X(t)')
    title = 'OU via WCE + Two-MLP'
    if title_suffix:
        title += f' ({title_suffix})'
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()


# ============================ 7) parameters ============================
if __name__ == "__main__":
    # —— data and time —— #
    T = 1.0
    N = 128                 # divisable to J.
    a, m, sigma, x0 = 1.0, 0.0, 1.0, 0.0
    batch_size = 64
    epochs = 5000
    seed = 42

    # —— basis and rank —— #
    basis_in_dim = 64       # Haar J
    basis_out_dim = 96      # trunk output dimension.

    # —— Wick / Hermite —— #
    hermite_M = 12          # second order >=2
    wick_order = 2          # 1: only H1; 2: add H2 and H1×H1

    # —— time position encoding (PE) —— #
    use_posenc = True       # use positional encoding or not, default is yes
    pe_dim = 128            # sin+cos
    include_raw_time = False #

    # —— two MLPs structure —— #
    branch_width, branch_depth = 512, 4
    trunk_width, trunk_depth = 512, 4
    use_layernorm, residual = False, False

    # —— optimization —— #
    lr = 1e-3

    # training + visualization
    model, device, meta = train_sdeonet(
        basis_in_dim=basis_in_dim, basis_out_dim=basis_out_dim,
        T=T, batch_size=batch_size, N=N, epochs=epochs,
        a=a, m=m, sigma=sigma, x0=x0,
        hermite_M=hermite_M, wick_order=wick_order,
        use_posenc=use_posenc, pe_dim=pe_dim, include_raw_time=include_raw_time,
        branch_width=branch_width, branch_depth=branch_depth,
        trunk_width=trunk_width, trunk_depth=trunk_depth,
        use_layernorm=use_layernorm, residual=residual,
        lr=lr, seed=seed,
    )
    plot_sdeonet_prediction(model, device, meta,
                            title_suffix=f"PE={use_posenc}, pe_dim={pe_dim}, J={basis_in_dim}, R={basis_out_dim}")

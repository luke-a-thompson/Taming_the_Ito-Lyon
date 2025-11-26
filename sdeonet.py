"""
SDEONet: A faithful JAX rewrite of the PyTorch implementation from OU_dw.py.

Key components
- simulate_ou_process_batch: batched OU process and Brownian path generator
- haar_basis_matrix: evaluates first J Haar basis functions on a grid
- hermite_polynomials_orthonormal_1d: probabilists' Hermite polynomials, orthonormal
- SDEONet: branch/trunk MLPs, Wick features, and reconstruction

The implementation focuses on clarity and fidelity over heavy JIT/vectorization.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr


def simulate_ou_process_batch(
    a: float,
    m: float,
    sigma: float,
    x0: float,
    T: float,
    N: int,
    batch_size: int,
    *,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Simulate batched Ornstein–Uhlenbeck paths X driven by Brownian motion W.

    Returns (t, W, X):
    - t: shape (N+1,)
    - W: shape (B, N+1)
    - X: shape (B, N+1)
    """
    dt = T / float(N)
    t = jnp.linspace(0.0, float(T), N + 1)
    k_incr, k_noise = jr.split(key)
    dW = jr.normal(k_incr, shape=(batch_size, N)) * math.sqrt(dt)
    W = jnp.concatenate([jnp.zeros((batch_size, 1)), jnp.cumsum(dW, axis=1)], axis=1)

    def step(carry: jax.Array, i: int) -> tuple[jax.Array, jax.Array]:
        x_prev = carry
        incr = W[:, i] - W[:, i - 1]
        x_next = x_prev + a * (m - x_prev) * dt + sigma * incr
        return x_next, x_next

    x_init = jnp.full((batch_size,), float(x0))
    _, xs = jax.lax.scan(step, x_init, jnp.arange(1, N + 1))
    X = jnp.concatenate([x_init[None, :], xs], axis=0).T
    return t, W, X


def haar_basis_matrix(J: int, T: float, t: jax.Array) -> jax.Array:
    """
    Evaluate the first J Haar basis functions on the time grid t.

    Returns Phi with shape (J, Tsteps), where Phi[j, :] is the j-th basis.
    The first basis is the constant scaling function 1/sqrt(T).
    Subsequent entries follow the standard Haar wavelets with normalization.
    """
    Phi_list: list[jax.Array] = []
    Phi_list.append(jnp.ones_like(t) / math.sqrt(float(T)))
    level = 0
    while len(Phi_list) < J:
        num_segments = 2**level
        for i in range(num_segments):
            if len(Phi_list) >= J:
                break
            left = i * float(T) / (2**level)
            middle = (i + 0.5) * float(T) / (2**level)
            right = (i + 1) * float(T) / (2**level)
            scale = (2.0 ** (level / 2.0)) / math.sqrt(float(T))
            pos = jnp.where((t >= left) & (t < middle), scale, 0.0)
            neg = jnp.where((t >= middle) & (t < right), -scale, 0.0)
            Phi_list.append(pos + neg)
        level += 1
    Phi = jnp.stack(Phi_list[:J], axis=0)  # (J, Tsteps)
    return Phi


def hermite_polynomials_orthonormal_1d(xi: jax.Array, M: int) -> jax.Array:
    """
    Probabilists' Hermite polynomials, orthonormal: H_m / sqrt(m!).

    xi: shape (B, J)
    Returns H: shape (B, J, M+1)
    """
    B, J = xi.shape
    H = jnp.zeros((B, J, M + 1), dtype=xi.dtype)
    H = H.at[..., 0].set(1.0)
    if M >= 1:
        H = H.at[..., 1].set(xi)

    def body(m: int, carry: jax.Array) -> jax.Array:
        prev = carry
        hm1 = prev[..., m - 1]
        hm2 = prev[..., m - 2]
        # Normalized probabilists' Hermite recurrence:
        # h_m = (xi * h_{m-1} - sqrt(m-1) * h_{m-2}) / sqrt(m)
        sqrt_m = jnp.sqrt(jnp.asarray(m, dtype=xi.dtype))
        sqrt_m_minus_1 = jnp.sqrt(jnp.asarray(m - 1, dtype=xi.dtype))
        val = (xi * hm1 - sqrt_m_minus_1 * hm2) / sqrt_m
        return prev.at[..., m].set(val)

    H = jax.lax.fori_loop(2, M + 1, body, H) if M >= 2 else H
    return H


class MLP(eqx.Module):
    """
    Simple MLP with optional layer norm and residual updates matching the PyTorch shape logic.
    """

    layers: list[eqx.nn.Linear]
    norms: list[eqx.nn.LayerNorm] | None
    depth: int
    use_layernorm: bool
    residual: bool

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        *,
        use_layernorm: bool = False,
        residual: bool = False,
        key: jax.Array,
    ) -> None:
        self.depth = int(depth)
        self.use_layernorm = bool(use_layernorm)
        self.residual = bool(residual)
        k_main, *k_rest = jr.split(key, max(2, depth))
        if depth <= 1:
            self.layers = [eqx.nn.Linear(in_dim, out_dim, key=k_main)]
            self.norms = None
            return
        layers: list[eqx.nn.Linear] = []
        norms: list[eqx.nn.LayerNorm] = []
        k_iter = iter(jr.split(key, depth))
        layers.append(eqx.nn.Linear(in_dim, width, key=next(k_iter)))
        for _ in range(max(0, depth - 2)):
            layers.append(eqx.nn.Linear(width, width, key=next(k_iter)))
            if use_layernorm:
                norms.append(eqx.nn.LayerNorm(width))
        layers.append(eqx.nn.Linear(width, out_dim, key=next(k_iter)))
        self.layers = layers
        self.norms = norms if use_layernorm else None

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.depth <= 1:
            return self.layers[0](x)
        h = jnn.gelu(self.layers[0](x))
        norm_idx = 0
        for i in range(1, self.depth - 1):
            h2 = jnn.gelu(self.layers[i](h))
            if self.use_layernorm:
                h2 = self.norms[norm_idx](h2)
                norm_idx += 1
            h = h + h2 if (self.residual and h2.shape == h.shape) else h2
        return self.layers[-1](h)


class SDEONet(eqx.Module):
    """
    W --(Itô left-end Haar projection)--> xi --(Hermite + Wick)--> features --(branch)--> Psi
    t --(PE or raw tau)--> features --(trunk)--> g(t)
    output: pred(t) = <Psi, g(t)>
    """

    # Static configuration
    T: float
    J: int
    R: int
    M: int
    wick_order: int
    use_posenc: bool
    pe_dim: int
    include_raw_time: bool

    # Precomputed pair indices for Wick second order (off-diagonal)
    pair_i: jax.Array | None
    pair_j: jax.Array | None

    # Networks
    branch: MLP
    trunk: MLP

    def __init__(
        self,
        *,
        basis_in_dim: int,
        basis_out_dim: int,
        T: float = 2.0,
        hermite_M: int = 3,
        wick_order: int = 2,
        use_posenc: bool = True,
        pe_dim: int = 32,
        include_raw_time: bool = True,
        branch_width: int = 256,
        branch_depth: int = 3,
        trunk_width: int = 256,
        trunk_depth: int = 3,
        use_layernorm: bool = False,
        residual: bool = False,
        key: jax.Array,
    ) -> None:
        assert wick_order in (1, 2)
        if use_posenc:
            assert pe_dim % 2 == 0
        self.T = float(T)
        self.J = int(basis_in_dim)
        self.R = int(basis_out_dim)
        self.M = int(hermite_M)
        self.wick_order = int(wick_order)
        self.use_posenc = bool(use_posenc)
        self.pe_dim = int(pe_dim)
        self.include_raw_time = bool(include_raw_time)

        # Wick second order off-diagonal indices
        if self.wick_order >= 2:
            pi, pj = jnp.triu_indices(self.J, k=1)
            self.pair_i = pi
            self.pair_j = pj
        else:
            self.pair_i = None
            self.pair_j = None

        # Branch input: 1 + J + [J if M>=2] + J(J-1)/2
        in_dim = 1 + self.J
        if self.wick_order >= 2:
            diag2 = self.J if self.M >= 2 else 0
            in_dim += diag2 + (self.J * (self.J - 1)) // 2

        # Trunk input: PE or raw tau or both concatenated
        trunk_in_dim = 1
        if self.use_posenc:
            trunk_in_dim = self.pe_dim + (1 if self.include_raw_time else 0)

        kb, kt = jr.split(key)
        self.branch = MLP(
            in_dim=in_dim,
            out_dim=self.R,
            width=branch_width,
            depth=branch_depth,
            use_layernorm=use_layernorm,
            residual=residual,
            key=kb,
        )
        self.trunk = MLP(
            in_dim=trunk_in_dim,
            out_dim=self.R,
            width=trunk_width,
            depth=trunk_depth,
            use_layernorm=use_layernorm,
            residual=residual,
            key=kt,
        )

    def _time_features(self, t_query: jax.Array) -> jax.Array:
        tau = jnp.clip(t_query / self.T, 0.0, 1.0)
        if not self.use_posenc:
            return tau[:, None]
        L = self.pe_dim // 2
        freqs = jnp.arange(1, L + 1) * math.pi
        args = tau[:, None] * freqs[None, :]
        pe = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=1)
        if self.include_raw_time:
            pe = jnp.concatenate([tau[:, None], pe], axis=1)
        return pe

    def _ito_left_projection(self, W: jax.Array, t_grid: jax.Array) -> jax.Array:
        dW = W[1:] - W[:-1]  # (B, N)
        Phi = haar_basis_matrix(self.J, self.T, t_grid)  # (J, Tsteps)
        coef = Phi[:, :-1]  # (J, N) left endpoints
        xi = dW @ coef.T  # (B, J)
        return xi

    def _wick_features(self, xi: jax.Array) -> jax.Array:
        B, J = xi.shape
        H = hermite_polynomials_orthonormal_1d(xi, self.M)  # (B, J, M+1)
        parts: list[jax.Array] = [jnp.ones((B, 1), dtype=xi.dtype)]
        H1 = H[:, :, 1]
        parts.append(H1)
        if self.wick_order >= 2:
            if self.M >= 2:
                H2 = H[:, :, 2]
                parts.append(H2)
            assert self.pair_i is not None and self.pair_j is not None
            parts.append(H1[:, self.pair_i] * H1[:, self.pair_j])
        return jnp.concatenate(parts, axis=1)

    def __call__(self, W: jax.Array, t_query: jax.Array) -> jax.Array:
        """
        W: shape (B, Tsteps)
        t_query: shape (Tsteps,)
        Returns pred: shape (B, Tsteps)
        """
        xi = self._ito_left_projection(W, t_query)  # (B, J)
        feats = self._wick_features(xi)  # (B, D)
        Psi = self.branch(feats)  # (B, R)
        t_feats = self._time_features(t_query)  # (Tsteps, trunk_in_dim)
        g_t = self.trunk(t_feats)  # (Tsteps, R)
        pred = (g_t @ Psi.T).T  # (B, Tsteps)
        return pred


__all__ = [
    "simulate_ou_process_batch",
    "haar_basis_matrix",
    "hermite_polynomials_orthonormal_1d",
    "SDEONet",
]

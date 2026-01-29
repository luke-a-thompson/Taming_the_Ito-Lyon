from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from stochastax.manifolds.spd import SPDManifold


def _as_btc_first_channel(x: np.ndarray) -> np.ndarray:
    """Coerce input to shape (B, T) by flattening trailing dims and taking channel 0."""
    if x.ndim == 1:
        # (T,) -> (1, T)
        return x[None, :]
    if x.ndim == 2:
        # (B, T)
        return x
    if x.ndim >= 3:
        # (B, T, ...) -> (B, T, Cflat) then take channel 0
        x_flat = x.reshape(int(x.shape[0]), int(x.shape[1]), -1)
        return x_flat[:, :, 0]
    raise ValueError(f"Expected array with ndim >= 1, got shape {x.shape}")


def save_rough_volatility_two_panel_plot(
    *,
    left: np.ndarray,
    right: np.ndarray,
    out_file: str,
    n_plot: int = 8,
    left_title: str = "Targets (one batch)",
    right_title: str = "Preds (one batch)",
    left_color: str = "black",
    right_color: str = "red",
    alpha: float = 0.5,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    left_bt = _as_btc_first_channel(left)
    right_bt = _as_btc_first_channel(right)

    n_plot0 = min(int(n_plot), int(left_bt.shape[0]), int(right_bt.shape[0]))
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True
    )
    for i in range(n_plot0):
        ax_left.plot(left_bt[i], color=left_color, alpha=float(alpha))
        ax_right.plot(right_bt[i], color=right_color, alpha=float(alpha))
    ax_left.set_title(left_title)
    ax_right.set_title(right_title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def _to_spd_matrix_paths(x: np.ndarray) -> np.ndarray:
    """Convert SPD trajectories to explicit matrix form.

    Accepts:
    - (B, T, 6) vech representation (as used by SPD datasets in this repo)
    - (B, T, 3, 3) explicit SPD matrices
    """
    x_np = np.asarray(x)
    if x_np.ndim == 4 and x_np.shape[-2:] == (3, 3):
        return x_np
    if x_np.ndim == 3 and int(x_np.shape[-1]) == 6:
        mats = SPDManifold.unvech(jnp.asarray(x_np, dtype=jnp.float32))
        return np.asarray(jax.device_get(mats))
    raise ValueError(
        f"Expected SPD paths shaped (B,T,6) or (B,T,3,3); got {x_np.shape}"
    )


def save_spd_covariance_eigenvalue_trajectory_plot(
    *,
    targets: np.ndarray,
    preds: np.ndarray,
    out_file: str,
    n_plot: int = 4,
    figsize: tuple[float, float] = (10.0, 4.0),
    targets_title: str = "Targets (eigenvalues)",
    preds_title: str = "Preds (eigenvalues)",
    alpha: float = 0.6,
) -> None:
    """Plot eigenvalue trajectories for SPD covariance paths (targets vs preds).

    We compute eigenvalues of the 3x3 SPD matrix at each time index and plot the
    three eigenvalue curves. Multiple trajectories are overlaid with transparency.
    """
    targets_mats = _to_spd_matrix_paths(targets)  # (B,T,3,3)
    preds_mats = _to_spd_matrix_paths(preds)  # (B,T,3,3)

    if targets_mats.shape != preds_mats.shape:
        raise ValueError(
            f"targets and preds must have same shape, got {targets_mats.shape} and {preds_mats.shape}"
        )
    if targets_mats.ndim != 4 or targets_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (B,T,3,3), got {targets_mats.shape}")

    b = int(targets_mats.shape[0])
    t = int(targets_mats.shape[1])
    n_plot0 = min(int(n_plot), b)
    if n_plot0 <= 0 or t <= 0:
        return

    # Ensure symmetry to stabilize eigendecomposition under FP noise.
    targets_mats = 0.5 * (targets_mats + np.swapaxes(targets_mats, -1, -2))
    preds_mats = 0.5 * (preds_mats + np.swapaxes(preds_mats, -1, -2))

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]

    for i in range(n_plot0):
        eig_t = np.linalg.eigvalsh(targets_mats[i])  # (T,3)
        eig_p = np.linalg.eigvalsh(preds_mats[i])  # (T,3)
        for k in range(3):
            ax_t.plot(
                eig_t[:, k],
                color=colors[k],
                alpha=float(alpha),
                linewidth=1.25,
                label=labels[k] if i == 0 else None,
            )
            ax_p.plot(
                eig_p[:, k],
                color=colors[k],
                alpha=float(alpha),
                linewidth=1.25,
                label=labels[k] if i == 0 else None,
            )

    ax_t.set_title(targets_title)
    ax_p.set_title(preds_title)
    ax_t.set_xlabel("time index")
    ax_p.set_xlabel("time index")
    ax_t.set_ylabel("eigenvalue")
    ax_t.legend(loc="best", frameon=False)
    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    out_file = f"{base}.pdf"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def save_spd_covariance_eigenvalue_fan_plot(
    *,
    targets: np.ndarray,
    preds: np.ndarray,
    out_file: str,
    max_paths: int | None = None,
    figsize: tuple[float, float] = (10.0, 4.0),
    targets_title: str = "Targets (eigenvalues)",
    preds_title: str = "Preds (eigenvalues)",
    quantiles: tuple[float, float, float, float] = (0.1, 0.25, 0.75, 0.9),
    alpha_outer: float = 0.18,
    alpha_inner: float = 0.35,
) -> None:
    """Fan plot of eigenvalue distributions over time (targets vs preds).

    The fan shows quantile bands (outer and inner) and a median line for each
    eigenvalue across the batch dimension.
    """
    targets_mats = _to_spd_matrix_paths(targets)  # (B,T,3,3)
    preds_mats = _to_spd_matrix_paths(preds)  # (B,T,3,3)

    if targets_mats.shape != preds_mats.shape:
        raise ValueError(
            f"targets and preds must have same shape, got {targets_mats.shape} and {preds_mats.shape}"
        )
    if targets_mats.ndim != 4 or targets_mats.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (B,T,3,3), got {targets_mats.shape}")

    b = int(targets_mats.shape[0])
    t = int(targets_mats.shape[1])
    if b <= 0 or t <= 0:
        return
    if max_paths is not None:
        b_use = min(int(max_paths), b)
        targets_mats = targets_mats[:b_use]
        preds_mats = preds_mats[:b_use]

    # Ensure symmetry to stabilize eigendecomposition under FP noise.
    targets_mats = 0.5 * (targets_mats + np.swapaxes(targets_mats, -1, -2))
    preds_mats = 0.5 * (preds_mats + np.swapaxes(preds_mats, -1, -2))

    eig_t = np.linalg.eigvalsh(targets_mats)  # (B,T,3)
    eig_p = np.linalg.eigvalsh(preds_mats)  # (B,T,3)

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    q_low, q_inner_low, q_inner_high, q_high = quantiles
    time_idx = np.arange(int(eig_t.shape[1]))

    for k in range(3):
        t_k = eig_t[:, :, k]
        p_k = eig_p[:, :, k]

        t_q_low = np.quantile(t_k, q_low, axis=0)
        t_q_inner_low = np.quantile(t_k, q_inner_low, axis=0)
        t_q_inner_high = np.quantile(t_k, q_inner_high, axis=0)
        t_q_high = np.quantile(t_k, q_high, axis=0)
        t_med = np.quantile(t_k, 0.5, axis=0)

        p_q_low = np.quantile(p_k, q_low, axis=0)
        p_q_inner_low = np.quantile(p_k, q_inner_low, axis=0)
        p_q_inner_high = np.quantile(p_k, q_inner_high, axis=0)
        p_q_high = np.quantile(p_k, q_high, axis=0)
        p_med = np.quantile(p_k, 0.5, axis=0)

        ax_t.fill_between(
            time_idx, t_q_low, t_q_high, color=colors[k], alpha=float(alpha_outer)
        )
        ax_t.fill_between(
            time_idx,
            t_q_inner_low,
            t_q_inner_high,
            color=colors[k],
            alpha=float(alpha_inner),
        )
        ax_t.plot(
            time_idx,
            t_med,
            color=colors[k],
            linewidth=1.5,
            label=labels[k],
        )

        ax_p.fill_between(
            time_idx, p_q_low, p_q_high, color=colors[k], alpha=float(alpha_outer)
        )
        ax_p.fill_between(
            time_idx,
            p_q_inner_low,
            p_q_inner_high,
            color=colors[k],
            alpha=float(alpha_inner),
        )
        ax_p.plot(
            time_idx,
            p_med,
            color=colors[k],
            linewidth=1.5,
            label=labels[k],
        )

    ax_t.set_title(targets_title)
    ax_p.set_title(preds_title)
    ax_t.set_xlabel("time index")
    ax_p.set_xlabel("time index")
    ax_t.set_ylabel("eigenvalue")
    ax_t.legend(loc="best", frameon=False)
    fig.tight_layout()

    base, _ = os.path.splitext(out_file)
    out_file = f"{base}.pdf"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    fig.savefig(out_file)
    plt.close(fig)


def save_sg_so3_sphere_plot(
    *,
    preds: np.ndarray,
    targets: np.ndarray,
    out_file: str,
    n_plot: int = 1,
    figsize: tuple[float, float] = (10.0, 5.0),
    labels: list[str] | None = None,
) -> None:
    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)

    # Ensure 4D shape: (B, T, 3, 3)
    if preds_np.ndim == 3:
        preds_np = preds_np[None, ...]
    if targets_np.ndim == 3:
        targets_np = targets_np[None, ...]

    if preds_np.ndim != 4 or preds_np.shape[-2:] != (3, 3):
        raise ValueError(f"Expected preds (B, T, 3, 3), got {preds_np.shape}")
    if targets_np.ndim != 4 or targets_np.shape[-2:] != (3, 3):
        raise ValueError(f"Expected targets (B, T, 3, 3), got {targets_np.shape}")

    batch_size = int(preds_np.shape[0])
    n_plot0 = min(int(n_plot), batch_size)

    # Project z-axis onto sphere: R @ [0, 0, 1] = R[:, :, 2] (third column)
    preds_pts = preds_np[..., :, 2]  # (B, T, 3)
    targets_pts = targets_np[..., :, 2]  # (B, T, 3)

    fixed_azim_deg = float(os.environ.get("SG_SO3_FIXED_AZIM", "35"))
    fixed_elev_deg = float(os.environ.get("SG_SO3_FIXED_ELEV", "20"))
    azim = np.deg2rad(fixed_azim_deg)
    elev = np.deg2rad(fixed_elev_deg)
    rot_z = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0.0],
            [np.sin(azim), np.cos(azim), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_y = np.array(
        [
            [np.cos(elev), 0.0, np.sin(elev)],
            [0.0, 1.0, 0.0],
            [-np.sin(elev), 0.0, np.cos(elev)],
        ]
    )
    rot = rot_z @ rot_y
    preds_pts = np.einsum("ij,btj->bti", rot, preds_pts)
    targets_pts = np.einsum("ij,btj->bti", rot, targets_pts)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        # Draw sphere surface for reference
        u = np.linspace(0.0, 2.0 * np.pi, 40)
        v = np.linspace(0.0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("SO(3) trajectories")
        ax.view_init(elev=fixed_elev_deg, azim=fixed_azim_deg)

        colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, n_plot0))
        if labels is None:
            labels = [f"path {i + 1}" for i in range(n_plot0)]
        for i in range(n_plot0):
            ax.plot(
                preds_pts[i, :, 0],
                preds_pts[i, :, 1],
                preds_pts[i, :, 2],
                color=colors[i],
                alpha=0.7,
                linewidth=1.5,
                label=labels[i] if i < len(labels) else None,
            )

        if int(targets_pts.shape[0]) > 0:
            ax.plot(
                targets_pts[0, :, 0],
                targets_pts[0, :, 1],
                targets_pts[0, :, 2],
                linestyle="None",
                marker="o",
                markerfacecolor="black",
                markeredgecolor="0.6",
                markeredgewidth=0.6,
                markersize=4.0,
                alpha=1.0,
            )

        ax.legend(loc="best", frameon=False)
        plt.tight_layout()
        base, _ = os.path.splitext(out_file)
        out_file = f"{base}.pdf"
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        plt.savefig(out_file, dpi=300)
        plt.close(fig)

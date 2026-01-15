from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


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


def save_sg_so3_sphere_plot(
    *,
    preds: np.ndarray,
    targets: np.ndarray,
    out_file: str,
    n_plot: int = 1,
    figsize: tuple[float, float] = (10.0, 5.0),
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

    fig = plt.figure(figsize=figsize)
    ax_t = fig.add_subplot(1, 2, 1, projection="3d")
    ax_p = fig.add_subplot(1, 2, 2, projection="3d")

    # Draw sphere surface for reference
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    for ax in (ax_t, ax_p):
        ax.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))

    ax_t.set_title("Target Trajectories")
    ax_p.set_title("Predicted Trajectories")

    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, n_plot0))
    for i in range(n_plot0):
        ax_t.plot(
            targets_pts[i, :, 0],
            targets_pts[i, :, 1],
            targets_pts[i, :, 2],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
        )
        ax_p.plot(
            preds_pts[i, :, 0],
            preds_pts[i, :, 1],
            preds_pts[i, :, 2],
            color=colors[i],
            alpha=0.7,
            linewidth=1.5,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    plt.savefig(out_file, dpi=100)
    plt.close(fig)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ResultsDict:
    eval_metric: float | None
    results_times: list[float]
    results: list[float]


class ResultsGatheringFn(Protocol):
    def __call__(
        self,
        preds: list[jax.Array] | jax.Array,
        targets: list[jax.Array] | jax.Array,
        epoch_idx: int,
        times_to_save: list[int] = [0, 21, 43, 64],
        n_plot: int | None = None,
    ) -> ResultsDict: ...


def get_rough_volatility_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    epoch_idx: int,
    times_to_save: list[int] = [0, 21, 43, 64],
    n_plot: int | None = None,
) -> ResultsDict:
    preds_batches = preds if isinstance(preds, list) else [preds]
    targets_batches = targets if isinstance(targets, list) else [targets]

    preds0 = np.array(jax.device_get(preds_batches[0]))
    targets0 = np.array(jax.device_get(targets_batches[0]))
    preds0_flat = preds0.reshape(preds0.shape[0], preds0.shape[1], -1)
    targets0_flat = targets0.reshape(targets0.shape[0], targets0.shape[1], -1)
    n_plot0 = 8 if n_plot is None else int(n_plot)
    n_plot0 = min(n_plot0, int(targets0_flat.shape[0]))
    if epoch_idx % 10 == 0:
        fig, (ax_targets, ax_preds) = plt.subplots(
            1, 2, figsize=(10, 4), sharex=True, sharey=True
        )
        for i in range(n_plot0):
            ax_targets.plot(targets0_flat[i, :, 0], color="black", alpha=0.5)
            ax_preds.plot(preds0_flat[i, :, 0], color="red", alpha=0.5)
        ax_targets.set_title("Targets (one batch)")
        ax_preds.set_title("Preds (one batch)")
        fig.tight_layout()
        os.makedirs("z_paper_content/rough_volatility_samples_by_epoch", exist_ok=True)
        fig.savefig(
            f"z_paper_content/rough_volatility_samples_by_epoch/batch_plot_{epoch_idx}.png"
        )
        plt.close(fig)

    preds_all = np.concatenate(
        [np.array(jax.device_get(p)) for p in preds_batches], axis=0
    )
    targets_all = np.concatenate(
        [np.array(jax.device_get(t)) for t in targets_batches], axis=0
    )
    ks_stats: list[float] = []
    for t in times_to_save:
        # KS test at time step t, flattened across all remaining dimensions
        pred_samples = preds_all[:, t, ...].ravel()
        target_samples = targets_all[:, t, ...].ravel()
        ks_res, p_val = ks_2samp(pred_samples, target_samples)  # type: ignore
        ks_stats.append(float(ks_res))  # type: ignore

    results_dict = ResultsDict(
        eval_metric=float(np.median(ks_stats)),
        results_times=[float(t) for t in times_to_save],
        results=[float(s) for s in ks_stats],
    )

    return results_dict


def get_sg_so3_simulation_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    epoch_idx: int,
    times_to_save: list[int] = [0, 21, 43, 64],
    n_plot: int | None = None,
) -> ResultsDict:
    # We receive rotation matrices shaped (B, T, 3, 3) (or a single trajectory (T, 3, 3)).
    # preds0 = preds[0] if isinstance(preds, list) else preds
    # targets0 = targets[0] if isinstance(targets, list) else targets

    # preds_np = np.array(jax.device_get(preds0))
    # targets_np = np.array(jax.device_get(targets0))

    # jax.debug.print(f"targets_np.shape: {targets_np.shape}")

    # targets = targets0 if targets0.ndim == 3 else targets0[0]
    # dR = jnp.einsum(
    #     "tij,tkj->tik", targets[:-1].transpose(0, 2, 1), targets[1:]
    # )  # R_t^T R_{t+1}
    # cos = (jnp.trace(dR, axis1=-2, axis2=-1) - 1.0) / 2.0
    # ang = jnp.arccos(jnp.clip(cos, -1.0, 1.0))
    # jax.debug.print(f"ang: {ang}")  # (19,)
    # jax.debug.print(f"ang.mean(): {ang.mean()}")

    # if preds_np.ndim == 3 and preds_np.shape[-2:] == (3, 3):
    #     preds_np = preds_np[None, ...]
    # if targets_np.ndim == 3 and targets_np.shape[-2:] == (3, 3):
    #     targets_np = targets_np[None, ...]

    # if preds_np.ndim != 4 or preds_np.shape[-2:] != (3, 3):
    #     raise ValueError(f"Expected preds shaped (B, T, 3, 3), got {preds_np.shape}")
    # if targets_np.ndim != 4 or targets_np.shape[-2:] != (3, 3):
    #     raise ValueError(
    #         f"Expected targets shaped (B, T, 3, 3), got {targets_np.shape}"
    #     )

    # # Project onto the sphere via R_t ẑ.
    # ez = np.array([0.0, 0.0, 1.0], dtype=preds_np.dtype)
    # preds_pts = preds_np @ ez  # (B, T, 3)
    # targets_pts = targets_np @ ez  # (B, T, 3)

    # # Plot only a couple of matched trajectories (same indices in preds/targets) so
    # # the figure stays readable even for large batches.
    # n_plot0 = 1
    # out_dir = os.environ.get(
    #     "SG_SO3_SPHERE_PLOT_DIR", "z_paper_content/sg_so3_sphere_by_epoch"
    # )
    # os.makedirs(out_dir, exist_ok=True)

    # fig = plt.figure(figsize=(10, 5))
    # ax_t = fig.add_subplot(1, 2, 1, projection="3d")
    # ax_p = fig.add_subplot(1, 2, 2, projection="3d")

    # # Sphere surface for context.
    # u = np.linspace(0.0, 2.0 * np.pi, 40)
    # v = np.linspace(0.0, np.pi, 20)
    # xs = np.outer(np.cos(u), np.sin(v))
    # ys = np.outer(np.sin(u), np.sin(v))
    # zs = np.outer(np.ones_like(u), np.cos(v))
    # ax_t.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0)
    # ax_p.plot_surface(xs, ys, zs, color="lightgray", alpha=0.12, linewidth=0)

    # for ax in (ax_t, ax_p):
    #     ax.set_xlim(-1.05, 1.05)
    #     ax.set_ylim(-1.05, 1.05)
    #     ax.set_zlim(-1.05, 1.05)
    #     ax.set_box_aspect((1.0, 1.0, 1.0))

    # ax_t.set_title("Targets: $R_t\\hat{z}$ on $S^2$")
    # ax_p.set_title("Preds: $R_t\\hat{z}$ on $S^2$")

    # for i in range(n_plot0):
    #     ax_t.plot(
    #         targets_pts[i + 10, :, 0],
    #         targets_pts[i + 10, :, 1],
    #         targets_pts[i + 10, :, 2],
    #         color="black",
    #         alpha=0.55,
    #         linewidth=1.2,
    #     )
    #     ax_p.plot(
    #         preds_pts[i + 10, :, 0],
    #         preds_pts[i + 10, :, 1],
    #         preds_pts[i + 10, :, 2],
    #         color="red",
    #         alpha=0.55,
    #         linewidth=1.2,
    #     )

    # fig.tight_layout()
    # fig.savefig(os.path.join(str(out_dir), f"sphere_epoch_{int(epoch_idx):05d}.png"))
    # plt.close(fig)

    # Use validation loss for early stopping/selection in `experiment.py`.
    return ResultsDict(eval_metric=None, results_times=[], results=[])

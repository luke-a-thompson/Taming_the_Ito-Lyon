import matplotlib.pyplot as plt
import jax
import numpy as np
from scipy.stats import ks_2samp
import os


def get_rough_volatility_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    epoch_idx: int,
    ks_time_steps: list[int] = [0, 21, 43, 64],
    n_plot: int | None = None,
) -> float:
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
    for t in ks_time_steps:
        # KS test at time step t, flattened across all remaining dimensions
        pred_samples = preds_all[:, t, ...].ravel()
        target_samples = targets_all[:, t, ...].ravel()
        ks_res, p_val = ks_2samp(pred_samples, target_samples)  # type: ignore
        ks_stats.append(float(ks_res))  # type: ignore

    return float(np.median(ks_stats))

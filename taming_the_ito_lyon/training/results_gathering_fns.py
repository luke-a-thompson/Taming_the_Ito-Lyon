import matplotlib

matplotlib.use("Agg")
from taming_the_ito_lyon.utils.mpl_style import apply_mpl_style
from taming_the_ito_lyon.training.results_plotting import (
    save_rough_volatility_two_panel_plot,
    save_sg_so3_sphere_plot,
)
from taming_the_ito_lyon.training.losses import (
    frobenius_loss,
    rotational_geodesic_loss,
)
from taming_the_ito_lyon.config.config import Config
from taming_the_ito_lyon.config.config_options import LossType
import jax
import numpy as np
from scipy.stats import ks_2samp
import os
from dataclasses import dataclass
from typing import Protocol


apply_mpl_style()


@dataclass
class ResultsDict:
    eval_metric: float | None
    results_times: list[float]
    results: list[float]
    extra_metrics: dict[str, list[float]] | None = None
    extra_scalar_metrics: dict[str, float] | None = None


class ResultsGatheringFn(Protocol):
    def __call__(
        self,
        preds: list[jax.Array] | jax.Array,
        targets: list[jax.Array] | jax.Array,
        epoch_idx: int,
        model_name: str,
        times_to_save: list[int] | None = None,
        n_plot: int | None = None,
        save_plot_every: int = 1,
        config: Config | None = None,
    ) -> ResultsDict: ...


def get_rough_volatility_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = [128, 256, 384, 512],
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    preds_batches = preds if isinstance(preds, list) else [preds]
    targets_batches = targets if isinstance(targets, list) else [targets]

    preds0 = np.array(jax.device_get(preds_batches[0]))
    targets0 = np.array(jax.device_get(targets_batches[0]))
    preds0_flat = preds0.reshape(preds0.shape[0], preds0.shape[1], -1)
    targets0_flat = targets0.reshape(targets0.shape[0], targets0.shape[1], -1)
    if n_plot is not None and epoch_idx % save_plot_every == 0:
        n_plot0 = min(int(n_plot), int(targets0_flat.shape[0]))
        out_dir = os.path.join(
            "z_paper_content", "rough_volatility_samples_by_epoch", model_name
        )
        os.makedirs(out_dir, exist_ok=True)
        save_rough_volatility_two_panel_plot(
            left=targets0_flat,
            right=preds0_flat,
            out_file=os.path.join(out_dir, f"batch_plot_epoch_{epoch_idx}.png"),
            n_plot=n_plot0,
            left_title="Targets (one batch)",
            right_title="Preds (one batch)",
        )

    preds_all = np.concatenate(
        [np.array(jax.device_get(p)) for p in preds_batches], axis=0
    )
    targets_all = np.concatenate(
        [np.array(jax.device_get(t)) for t in targets_batches], axis=0
    )
    if times_to_save is None:
        return ResultsDict(eval_metric=None, results_times=[], results=[])

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
    model_name: str,
    times_to_save: list[int] | None = None,
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    """Visualize SO(3) predictions on a sphere.

    Args:
        preds: Predicted rotation matrices (B, T, 3, 3) (or list of batches)
        targets: Target rotation matrices (B, T, 3, 3) (or list of batches)
        epoch_idx: Current epoch number
        times_to_save: Unused (for compatibility)
        n_plot: Number of trajectories to plot (default: 1)

    Returns:
        Empty ResultsDict (plots saved to disk)
    """
    # We only ever plot one batch. If a list is passed, take the first batch.
    preds0 = preds[0] if isinstance(preds, list) else preds
    targets0 = targets[0] if isinstance(targets, list) else targets

    # Convert to numpy for plotting
    preds_np = np.array(jax.device_get(preds0))
    targets_np = np.array(jax.device_get(targets0))

    if n_plot is not None and epoch_idx % save_plot_every == 0:
        # Create output directory
        base_out_dir = os.environ.get(
            "SG_SO3_SPHERE_PLOT_DIR", "z_paper_content/sg_so3_sphere_by_epoch"
        )
        out_dir = os.path.join(base_out_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        save_sg_so3_sphere_plot(
            preds=preds_np,
            targets=targets_np,
            out_file=os.path.join(out_dir, f"sphere_epoch_{epoch_idx:05d}.png"),
            n_plot=int(n_plot),
        )

    if config is None:
        raise ValueError("config must be provided to compute RGE for SG_SO3 results.")

    rge_fn = rotational_geodesic_loss(config)
    rge_value = float(jax.device_get(rge_fn(preds0, targets0)))
    eval_metric: float | None = None
    if config.experiment_config.loss == LossType.RGE:
        eval_metric = rge_value
    elif config.experiment_config.loss == LossType.FROBENIUS:
        frob_fn = frobenius_loss(config)
        eval_metric = float(jax.device_get(frob_fn(preds0, targets0)))

    return ResultsDict(
        eval_metric=eval_metric,
        results_times=[],
        results=[],
        extra_scalar_metrics={"rge": rge_value},
    )

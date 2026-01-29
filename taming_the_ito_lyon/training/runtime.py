import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from cyreal.loader import DataLoader, _LoaderState

from taming_the_ito_lyon.config import Config
from taming_the_ito_lyon.config.config_options import Datasets, ModelType, TrainingMode
from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.factories import (
    create_dataloaders,
    create_grad_batch_loss_fns,
    create_results_gathering_fn,
    create_unconditional_control_sampler_batched,
)
from taming_the_ito_lyon.training.results_gathering_fns import ResultsGatheringFn


@dataclasses.dataclass
class ExperimentRuntime:
    config: Config
    loss_label: str
    mode: TrainingMode
    batch_size: int
    ts_full: jax.Array
    input_path_dim: int
    output_head_dim: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_loader_state: _LoaderState
    val_loader_state: _LoaderState
    test_loader_state: _LoaderState
    train_iterate: Callable[
        [_LoaderState], tuple[dict[str, jax.Array], _LoaderState, jax.Array]
    ]
    val_iterate: Callable[
        [_LoaderState], tuple[dict[str, jax.Array], _LoaderState, jax.Array]
    ]
    test_iterate: Callable[
        [_LoaderState], tuple[dict[str, jax.Array], _LoaderState, jax.Array]
    ]
    unconditional_control_sampler: (
        Callable[[jax.Array, jax.Array, int], jax.Array] | None
    )
    grad_fn: Callable[
        [Model, jax.Array, jax.Array, jax.Array], tuple[jax.Array, optax.Updates]
    ]
    batch_loss_fn: Callable[[Model, jax.Array, jax.Array, jax.Array], jax.Array]
    loss_on_preds_fn: Callable[
        [jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
    ]
    eval_step: Callable[[jax.Array, jax.Array, jax.Array, Model], jax.Array]
    predict_batch: Callable[[jax.Array, Model], jax.Array]
    results_gathering_fn: ResultsGatheringFn


def build_runtime(config: Config, loader_key: jax.Array) -> ExperimentRuntime:
    loss_label: str = str(config.experiment_config.loss.value)
    mode = config.experiment_config.training_mode

    train_loader, val_loader, test_loader = create_dataloaders(config=config)
    train_loader_state = train_loader.init_state(loader_key)
    val_loader_state = val_loader.init_state(loader_key)
    test_loader_state = test_loader.init_state(loader_key)

    train_iterate = jax.jit(train_loader.next)
    val_iterate = jax.jit(val_loader.next)
    test_iterate = jax.jit(test_loader.next)

    # Get shapes
    shape_batch, _, _ = test_iterate(test_loader_state)

    batch_size, timesteps, input_channels = shape_batch["driver"].shape
    # `solution` may be either (B, T, C) or (B, T, 3, 3) depending on dataset.
    # Do not infer model output head size from this blindly.
    ts_full = jnp.linspace(0.0, 1.0, timesteps, dtype=shape_batch["solution"].dtype)

    # For SO(3) rotation-matrix datasets with rotational-geodesic loss we train a
    # 6D head and Gram–Schmidt it into a (3,3) rotation matrix via SO3.from_6d.
    is_so3_rge = config.experiment_config.dataset_name in (
        Datasets.SG_SO3_SIMULATION,
        Datasets.OXFORD_MULTIMOTION_STATIC,
        Datasets.OXFORD_MULTIMOTION_TRANSLATIONAL,
        Datasets.OXFORD_MULTIMOTION_UNCONSTRAINED,
    )
    if is_so3_rge and config.experiment_config.model_type == ModelType.NRDE:
        output_head_dim = 9
    else:
        output_head_dim = 6 if is_so3_rge else int(shape_batch["solution"].shape[-1])

    # In unconditional mode, the model consumes a sampled Brownian control with time
    # prepended. For Wishart/SPD experiments, a 1D driver is often too restrictive
    # (it yields rank-1 quadratic variation in vech-space), so for MNRDE we use a
    # higher-dimensional latent Brownian by default.
    if mode == TrainingMode.UNCONDITIONAL:
        driver_dim = 1
        if config.experiment_config.model_type == ModelType.MNRDE:
            driver_dim = int(output_head_dim)
        input_path_dim = int(driver_dim) + 1  # (t, W^driver_dim)
    elif config.experiment_config.extrapolation_scheme is not None:
        input_path_dim = int(input_channels) + 1  # time concat
    else:
        input_path_dim = int(input_channels)

    unconditional_control_sampler = None
    if mode == TrainingMode.UNCONDITIONAL:
        unconditional_control_sampler = create_unconditional_control_sampler_batched(
            driver_dim=int(input_path_dim) - 1,
        )

    grad_fn, batch_loss_fn, loss_on_preds_fn = create_grad_batch_loss_fns(
        config=config,
        # Only used for SIGKER; for SO3+RGE we keep targets as (3,3) matrices.
        output_path_dim=int(output_head_dim),
    )

    @eqx.filter_jit
    def eval_step(
        control_values_b: jax.Array,
        target_b: jax.Array,
        gt_driver_b: jax.Array,
        model: Model,
    ) -> jax.Array:
        return batch_loss_fn(model, control_values_b, target_b, gt_driver_b)

    @eqx.filter_jit
    def predict_batch(control_values_b: jax.Array, model: Model) -> jax.Array:
        return jax.vmap(model)(control_values_b)

    results_gathering_fn = create_results_gathering_fn(config)

    return ExperimentRuntime(
        config=config,
        loss_label=loss_label,
        mode=mode,
        batch_size=int(batch_size),
        ts_full=ts_full,
        input_path_dim=int(input_path_dim),
        output_head_dim=int(output_head_dim),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_loader_state=train_loader_state,
        val_loader_state=val_loader_state,
        test_loader_state=test_loader_state,
        train_iterate=train_iterate,
        val_iterate=val_iterate,
        test_iterate=test_iterate,
        unconditional_control_sampler=unconditional_control_sampler,
        grad_fn=grad_fn,
        batch_loss_fn=batch_loss_fn,
        loss_on_preds_fn=loss_on_preds_fn,
        eval_step=eval_step,
        predict_batch=predict_batch,
        results_gathering_fn=results_gathering_fn,
    )

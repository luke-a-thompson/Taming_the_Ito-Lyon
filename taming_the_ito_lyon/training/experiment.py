import atexit
import json
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from datetime import datetime
from tqdm.auto import tqdm
import os
import shutil
import signal
import time
from collections.abc import Callable

import optax
from cyreal.loader import DataLoader, _LoaderState
from taming_the_ito_lyon.training.results_gathering_fns import (
    get_rough_volatility_results,
    ResultsDict,
)
from taming_the_ito_lyon.training.factories import (
    create_model,
    create_optimizer,
    create_dataloaders,
    create_unconditional_control_sampler_batched,
    create_grad_batch_loss_fns,
    configure_jax,
)
from taming_the_ito_lyon.config import (
    Config,
)
from taming_the_ito_lyon.config.config_options import TrainingMode
from taming_the_ito_lyon.models import Model

SAVED_MODELS_DIR = "saved_models"


def _get_run_dirname(model_name: str) -> str:
    """Generate a human-readable directory name like 'nrde_10_25pm_26_11_25'."""
    now = datetime.now()
    time_str = now.strftime("%I_%M%p").lower()
    date_str = now.strftime("%d_%m_%y")
    return f"{model_name}_{time_str}_{date_str}"


def experiment(config: Config, config_path: str | None = None) -> None:
    configure_jax()
    model_name = config.experiment_config.model_type.value
    loss_label: str = str(config.experiment_config.loss.value)
    model_key, loader_key = jr.split(jr.PRNGKey(config.experiment_config.seed), 2)
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
    target_channels = shape_batch["solution"].shape[-1]
    ts_full = jnp.linspace(0.0, 1.0, timesteps, dtype=shape_batch["solution"].dtype)

    if mode == TrainingMode.UNCONDITIONAL:
        # Model consumes (time + sampled driver channels)
        uncond_dim = config.experiment_config.unconditional_driver_dim
        assert uncond_dim is not None
        input_path_dim = int(uncond_dim) + 1
    else:
        input_path_dim = input_channels

    model = create_model(
        config=config,
        input_path_dim=input_path_dim,
        output_path_dim=target_channels,
        key=model_key,
    )

    trainable_leaves = jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array)
    )
    num_params = int(sum(int(x.size) for x in trainable_leaves))

    optim = create_optimizer(
        optimizer_name=config.experiment_config.optimizer,
        learning_rate=config.experiment_config.learning_rate,
        weight_decay=config.experiment_config.weight_decay,
        max_grad_norm=config.experiment_config.max_grad_norm,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    unconditional_control_sampler = None
    if mode == TrainingMode.UNCONDITIONAL:
        uncond_dim = config.experiment_config.unconditional_driver_dim
        assert uncond_dim is not None
        assert config.experiment_config.unconditional_driver_kind is not None
        assert config.experiment_config.unconditional_hurst is not None
        unconditional_control_sampler = create_unconditional_control_sampler_batched(
            driver_kind=config.experiment_config.unconditional_driver_kind,
            driver_dim=int(uncond_dim),
            hurst=float(config.experiment_config.unconditional_hurst),
        )

    grad_fn, batch_loss_fn = create_grad_batch_loss_fns(
        loss_type=config.experiment_config.loss,
        output_path_dim=int(target_channels),
    )

    @eqx.filter_jit(donate="all")
    def train_step(
        control_values_b: jax.Array,
        target_b: jax.Array,
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        loss_value, grads = grad_fn(model, control_values_b, target_b)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, new_opt_state = optim.update(grads, opt_state, params)
        updated_model: Model = eqx.apply_updates(model, updates)
        return loss_value, updated_model, new_opt_state

    @eqx.filter_jit
    def eval_step(
        control_values_b: jax.Array,
        target_b: jax.Array,
        model: Model,
    ) -> jax.Array:
        return batch_loss_fn(model, control_values_b, target_b)

    @eqx.filter_jit
    def predict_batch(control_values_b: jax.Array, model: Model) -> jax.Array:
        return jax.vmap(model)(control_values_b)

    train_key = None
    val_key = None
    test_key = None
    if mode == TrainingMode.UNCONDITIONAL:
        train_key = jr.PRNGKey(config.experiment_config.seed + 1)
        val_key = jr.PRNGKey(config.experiment_config.seed + 2)
        test_key = jr.PRNGKey(config.experiment_config.seed + 3)

    # Setup temporary checkpoint path
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    temp_best_path = os.path.join(SAVED_MODELS_DIR, "best.eqx")

    # Cleanup handler for temp file
    def cleanup_temp() -> None:
        if os.path.exists(temp_best_path):
            os.remove(temp_best_path)

    def sigint_handler(signum: int, frame: object) -> None:
        tqdm.write("\nInterrupted. Cleaning up temporary checkpoint...")
        cleanup_temp()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, sigint_handler)
    atexit.register(cleanup_temp)

    min_val_metric = float("inf")
    min_train_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    patience = int(config.experiment_config.early_stopping_patience)

    epochs = int(config.experiment_config.epochs)
    training_start = time.perf_counter()
    final_epoch = 0

    def train_epoch(
        model: Model,
        opt_state: optax.OptState,
        training_mode: TrainingMode,
        epoch_key: jax.Array | None,
        epoch_idx: int,
    ) -> tuple[float, Model, optax.OptState]:
        nonlocal train_loader_state
        total_loss = 0.0
        pbar = tqdm(
            range(train_loader.steps_per_epoch),
            desc="Training batches",
            position=1,
            leave=False,
        )
        for step_idx in pbar:
            data_time = time.perf_counter()
            batch, train_loader_state, _ = train_iterate(train_loader_state)

            # Get control values based on mode
            if training_mode == TrainingMode.UNCONDITIONAL:
                if epoch_key is None or unconditional_control_sampler is None:
                    raise ValueError(
                        "epoch_key and unconditional_control_sampler required for UNCONDITIONAL"
                    )
                step_key = jr.fold_in(jr.fold_in(epoch_key, epoch_idx), step_idx)
                control_values_b = unconditional_control_sampler(
                    ts_full, step_key, batch_size
                )
            else:
                control_values_b = batch["driver"]

            data_time_elapsed = time.perf_counter() - data_time
            train_time = time.perf_counter()
            loss_value, model, opt_state = train_step(
                control_values_b,
                batch["solution"],
                model,
                opt_state,
            )
            total_loss += float(loss_value)
            train_time_elapsed = time.perf_counter() - train_time
            pbar.set_postfix(
                {
                    f"train_{loss_label}": f"{float(loss_value) * 1e2:.3f}×10⁻²",
                    "data_time": f"{data_time_elapsed:.2f}s",
                    "train_time": f"{train_time_elapsed:.2f}s",
                }
            )

        avg_loss = total_loss / max(1, int(train_loader.steps_per_epoch))
        return avg_loss, model, opt_state

    def eval_epoch(
        model: Model,
        loader: DataLoader,
        iterate: Callable[
            [_LoaderState], tuple[dict[str, jax.Array], _LoaderState, jax.Array]
        ],
        loader_state: _LoaderState,
        epoch_key: jax.Array | None,
        epoch_idx: int,
    ) -> tuple[float, ResultsDict, _LoaderState]:
        total_loss = 0.0
        preds_batches: list[jax.Array] = []
        targets_batches: list[jax.Array] = []

        pbar = tqdm(
            range(loader.steps_per_epoch),
            desc="Evaluating batches",
            position=1,
            leave=False,
        )
        for step_idx in pbar:
            batch, loader_state, _ = iterate(loader_state)

            # Get control values based on mode
            if mode == TrainingMode.UNCONDITIONAL:
                if epoch_key is None or unconditional_control_sampler is None:
                    raise ValueError(
                        "epoch_key and unconditional_control_sampler required for UNCONDITIONAL"
                    )
                step_key = jr.fold_in(jr.fold_in(epoch_key, epoch_idx), step_idx)
                control_values_b = unconditional_control_sampler(
                    ts_full, step_key, batch_size
                )
            else:
                control_values_b = batch["driver"]

            total_loss += float(eval_step(control_values_b, batch["solution"], model))

            preds_batches.append(predict_batch(control_values_b, model))
            targets_batches.append(batch["solution"])

        avg_loss = total_loss / max(1, int(loader.steps_per_epoch))

        # Compute KS test if predictions were collected
        results_dict = get_rough_volatility_results(
            preds_batches,
            targets_batches,
            epoch_idx,
            ks_time_steps=[-1],
            n_plot=batch_size,
        )
        return avg_loss, results_dict, loader_state

    epoch_bar = tqdm(
        range(epochs),
        desc="Epochs",
        position=0,
        leave=True,
    )
    for epoch_idx in epoch_bar:
        final_epoch = epoch_idx
        train_loss, model, opt_state = train_epoch(
            model, opt_state, mode, train_key, epoch_idx
        )
        min_train_loss = min(min_train_loss, train_loss)
        val_loss, val_results_dict, val_loader_state = eval_epoch(
            model,
            val_loader,
            val_iterate,
            val_loader_state,
            val_key,
            epoch_idx,
        )

        if val_results_dict.eval_metric < min_val_metric:
            min_val_metric = val_results_dict.eval_metric
            best_epoch = epoch_idx
            eqx.tree_serialise_leaves(temp_best_path, model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        epoch_bar.set_postfix(
            {
                f"train_{loss_label}": f"{train_loss * 1e2:.3f}×10⁻²",
                f"best_val_{loss_label}": f"{min_val_metric * 1e2:.3f}×10⁻² at epoch {best_epoch}",
                "val_metric": f"{val_results_dict.eval_metric:.3f}",
            }
        )

        if epochs_since_improve >= patience:
            tqdm.write(
                f"Early stopping at epoch {epoch_idx} (no val improvement for {patience} epochs)."
            )
            break

    training_elapsed = time.perf_counter() - training_start

    # Evaluate best model on test set with KS test
    best_model: Model = eqx.tree_deserialise_leaves(temp_best_path, model)
    inference_start = time.perf_counter()
    test_loss, test_results_dict, test_loader_state = eval_epoch(
        best_model,
        test_loader,
        test_iterate,
        test_loader_state,
        test_key,
        0,
    )
    inference_elapsed = time.perf_counter() - inference_start

    # Create run directory and save final artifacts
    run_dirname = _get_run_dirname(model_name)
    run_dir = os.path.join(SAVED_MODELS_DIR, run_dirname)
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(run_dir, "best.eqx")
    last_path = os.path.join(run_dir, "last.eqx")
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_save_path = os.path.join(run_dir, "config.toml")

    # Move temp best to final location
    os.rename(temp_best_path, best_path)
    # Unregister cleanup since we moved the file
    atexit.unregister(cleanup_temp)

    # Save last model
    eqx.tree_serialise_leaves(last_path, model)

    # Copy config file if provided
    if config_path is not None and os.path.exists(config_path):
        shutil.copy2(config_path, config_save_path)

    # Save metrics
    metrics = {
        "run": {
            "name": run_dirname,
            "total_epochs": final_epoch + 1,
            "best_epoch": best_epoch,
        },
        "model": {
            "type": model_name,
            "num_params": num_params,
        },
        "timings": {
            "training_s": training_elapsed,
            "inference_s": inference_elapsed,
        },
        str(config.experiment_config.loss.value): {
            "test": test_results_dict.eval_metric,
            "min_val": min_val_metric,
            "min_train": min_train_loss,
        },
        "test_results_dict": test_results_dict,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    tqdm.write(
        f"\nResults: {num_params:,} params | "
        f"train {training_elapsed:.1f}s | "
        f"inference {inference_elapsed * 1000:.1f}ms | "
        f"test metric {test_results_dict.eval_metric:.4f}"
    )
    tqdm.write(f"Saved to: {run_dir}/")

import atexit
import json
import jax
import jax.random as jr
import equinox as eqx
from datetime import datetime
from tqdm.auto import tqdm
import os
import shutil
import signal
import time

from taming_the_ito_lyon.training.train import train_epoch, eval_epoch
from taming_the_ito_lyon.training.factories import (
    create_model,
    create_optimizer,
    create_dataset,
)
from taming_the_ito_lyon.data import (
    split_train_val_test,
    make_dataloader,
)
from taming_the_ito_lyon.config import Config, NCDEConfig, NRDEConfig, SDEONetConfig
from taming_the_ito_lyon.models import Model

SAVED_MODELS_DIR = "saved_models"


def _get_model_name(config: Config) -> str:
    """Get model name from config type."""
    match config.nn_config:
        case NCDEConfig():
            return "ncde"
        case NRDEConfig():
            return "nrde"
        case SDEONetConfig():
            return "sdeonet"
        case _:
            return "model"


def _get_run_dirname(model_name: str) -> str:
    """Generate a human-readable directory name like 'nrde_10_25pm_26_11_25'."""
    now = datetime.now()
    time_str = now.strftime("%I_%M%p").lower()
    date_str = now.strftime("%d_%m_%y")
    return f"{model_name}_{time_str}_{date_str}"


def experiment(config: Config, config_path: str | None = None) -> None:
    model_name = _get_model_name(config)
    key = jr.PRNGKey(config.experiment_config.seed)
    model_key, data_key, loader_key = jr.split(key, 3)

    ts_batched, solution, coeffs_batched = create_dataset(config=config, key=data_key)
    batch_count, length, target_channels = solution.shape
    data_channels = int(coeffs_batched[0].shape[-1])

    model = create_model(
        config=config,
        input_path_dim=data_channels,
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
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    (
        ts_train,
        target_train,
        coeffs_train,
        ts_val,
        target_val,
        coeffs_val,
        ts_test,
        target_test,
        coeffs_test,
    ) = split_train_val_test(
        ts_batched,
        solution,
        coeffs_batched,
        train_fraction=config.experiment_config.train_fraction,
        val_fraction=config.experiment_config.val_fraction,
        test_fraction=config.experiment_config.test_fraction,
    )

    batch_size = int(config.experiment_config.batch_size)
    num_batches = (ts_train.shape[0] + batch_size - 1) // batch_size
    loader = make_dataloader(
        timestep=ts_train,
        solution=target_train,
        drivers=coeffs_train,
        batch_size=batch_size,
        key=loader_key,
    )

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

    best_val_loss = float("inf")
    min_train_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    patience = int(config.experiment_config.early_stopping_patience)

    epochs = int(config.experiment_config.epochs)
    training_start = time.perf_counter()
    final_epoch = 0

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0, leave=True)
    for epoch_idx in epoch_bar:
        final_epoch = epoch_idx
        train_loss, model, opt_state = train_epoch(
            model=model,
            optim=optim,
            opt_state=opt_state,
            loader=loader,
            num_batches=num_batches,
        )

        min_train_loss = min(min_train_loss, train_loss)

        val_loss_value = eval_epoch(model, ts_val, target_val, coeffs_val)
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch_idx
            eqx.tree_serialise_leaves(temp_best_path, model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        epoch_bar.set_postfix(
            train=f"{train_loss * 1e2:.3f}×10⁻²",
            val=f"{val_loss_value * 1e2:.3f}×10⁻²",
            best_val=f"{best_val_loss * 1e2:.3f}×10⁻² at epoch {best_epoch}",
        )

        if epochs_since_improve >= patience:
            tqdm.write(
                f"Early stopping at epoch {epoch_idx} (no val improvement for {patience} epochs)."
            )
            break

    training_elapsed = time.perf_counter() - training_start

    # Evaluate best model on test set
    best_model: Model = eqx.tree_deserialise_leaves(temp_best_path, model)
    inference_start = time.perf_counter()
    test_loss_value = eval_epoch(best_model, ts_test, target_test, coeffs_test)
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
        "losses": {
            "test": test_loss_value,
            "min_val": best_val_loss,
            "min_train": min_train_loss,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    tqdm.write(
        f"\nResults: {num_params:,} params | "
        f"train {training_elapsed:.1f}s | "
        f"inference {inference_elapsed * 1000:.1f}ms | "
        f"test loss {test_loss_value:.4f}"
    )
    tqdm.write(f"Saved to: {run_dir}/")

import dataclasses
import json
import os
import shutil
from datetime import datetime

import equinox as eqx

from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.results_gathering_fns import ResultsDict


LOSS_METRICS: dict[str, dict[str, float | str]] = {
    "mse": {"scale": 1.0, "unit": ""},
    "rge": {"scale": 1.0, "unit": ""},
    "sigker": {"scale": 1.0, "unit": ""},
    "frobenius": {"scale": 1.0, "unit": ""},
}


def loss_meta(loss_label: str, value: float) -> tuple[float, str, float]:
    info = LOSS_METRICS.get(loss_label, {"scale": 1.0, "unit": ""})
    scale = float(info.get("scale", 1.0))
    unit = str(info.get("unit", ""))
    return value * scale, unit, scale


def format_loss(loss_label: str, value: float) -> str:
    scaled_value, unit, _ = loss_meta(loss_label, value)
    suffix = f" {unit}" if unit else ""
    return f"{scaled_value:.3f}{suffix}"


def get_run_dirname(model_name: str) -> str:
    """Generate a human-readable directory name like 'nrde_10_25pm_26_11_25'."""
    now = datetime.now()
    time_str = now.strftime("%I_%M%p").lower()
    date_str = now.strftime("%d_%m_%y")
    return f"{model_name}_{time_str}_{date_str}"


def finalize_training_run(
    *,
    run_dirname: str,
    model_name: str,
    model: Model,
    temp_best_path: str,
    config_path: str | None,
    num_params: int,
    final_epoch: int,
    best_epoch: int,
    training_elapsed: float,
    inference_elapsed: float,
    loss_label: str,
    test_eval_metric: float,
    min_val_metric: float,
    min_train_loss: float,
    test_results_dict: ResultsDict,
) -> str:
    run_dir = os.path.join("saved_models", run_dirname)
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(run_dir, "best.eqx")
    last_path = os.path.join(run_dir, "last.eqx")
    metrics_path = os.path.join(run_dir, "metrics.json")
    config_save_path = os.path.join(run_dir, "config.toml")

    # Move temp best to final location
    os.rename(temp_best_path, best_path)

    # Save last model
    eqx.tree_serialise_leaves(last_path, model)

    # Copy config file if provided
    if config_path is not None and os.path.exists(config_path):
        shutil.copy2(config_path, config_save_path)

    scaled_test, unit, scale = loss_meta(loss_label, test_eval_metric)
    scaled_min_val, _, _ = loss_meta(loss_label, min_val_metric)
    scaled_min_train, _, _ = loss_meta(loss_label, min_train_loss)
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
        loss_label: {
            "test": scaled_test,
            "min_val": scaled_min_val,
            "min_train": scaled_min_train,
            "unit": unit,
            "scale": scale,
        },
        "test_results_dict": dataclasses.asdict(test_results_dict),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return run_dir


def write_test_metrics(
    *,
    run_dir: str,
    model_name: str,
    num_params: int,
    inference_elapsed: float,
    loss_label: str,
    test_eval_metric: float,
    test_results_dict: ResultsDict,
    checkpoint_path: str,
) -> str:
    scaled_test, unit, scale = loss_meta(loss_label, test_eval_metric)
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    metrics = {
        "run": {
            "name": os.path.basename(run_dir),
            "checkpoint": checkpoint_path,
        },
        "model": {
            "type": model_name,
            "num_params": num_params,
        },
        "timings": {
            "inference_s": inference_elapsed,
        },
        loss_label: {
            "test": scaled_test,
            "unit": unit,
            "scale": scale,
        },
        "test_results_dict": dataclasses.asdict(test_results_dict),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics_path

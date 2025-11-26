import jax
import jax.random as jr
import equinox as eqx
from tqdm.auto import tqdm
import os
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
from taming_the_ito_lyon.config import Config
from taming_the_ito_lyon.models import Model


def experiment(config: Config) -> None:
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

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    patience = int(config.experiment_config.early_stopping_patience)
    ckpt_path = str(config.experiment_config.checkpoint_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    epochs = int(config.experiment_config.epochs)
    training_start = time.perf_counter()

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0, leave=True)
    for epoch_idx in epoch_bar:
        train_loss, model, opt_state = train_epoch(
            model=model,
            optim=optim,
            opt_state=opt_state,
            loader=loader,
            num_batches=num_batches,
        )

        val_loss_value = eval_epoch(model, ts_val, target_val, coeffs_val)
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch_idx
            eqx.tree_serialise_leaves(ckpt_path, model)
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

    best_model: Model = eqx.tree_deserialise_leaves(ckpt_path, model)
    inference_start = time.perf_counter()
    test_loss_value = eval_epoch(best_model, ts_test, target_test, coeffs_test)
    inference_elapsed = time.perf_counter() - inference_start
    tqdm.write(
        f"num_params={num_params} | train_s={training_elapsed:.3f} | infer_s={inference_elapsed:.3f} | test_loss={test_loss_value * 1e2:.3f}×10⁻²"
    )

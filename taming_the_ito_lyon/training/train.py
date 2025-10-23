import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tqdm.auto import tqdm
import os
import time

from taming_the_ito_lyon.models import Model
from taming_the_ito_lyon.training.factories import (
    create_model,
    create_optimizer,
    create_dataset,
)
from taming_the_ito_lyon.data import (
    split_train_val_test,
    make_dataloader,
)
from taming_the_ito_lyon.config.config import load_toml_config
from taming_the_ito_lyon.config import Config
import optax

os.environ["JAX_COMPILATION_CACHE_DIR"] = "jax_cache"
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


def eval_epoch(
    model: Model,
    ts_b: jax.Array,
    target_b: jax.Array,
    coeffs_b: tuple[jax.Array, ...],
) -> float:
    @eqx.filter_jit
    def loss_fn(
        model: Model,
        ts_b: jax.Array,
        target_b: jax.Array,
        coeffs_b: tuple[jax.Array, ...],
    ) -> jax.Array:
        def predict_path(t_i: jax.Array, c_i: tuple[jax.Array, ...]) -> jax.Array:
            return model(t_i, c_i)

        preds = jax.vmap(predict_path)(ts_b, coeffs_b)
        return jnp.mean((preds - target_b) ** 2)

    return float(loss_fn(model, ts_b, target_b, coeffs_b))


def train_epoch(
    model: Model,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loader,
    num_batches: int,
) -> tuple[float, Model, optax.OptState]:
    @eqx.filter_jit
    def loss_fn(
        model: Model,
        ts_b: jax.Array,
        target_b: jax.Array,
        coeffs_b: tuple[jax.Array, ...],
    ) -> jax.Array:
        def predict_path(t_i: jax.Array, c_i: tuple[jax.Array, ...]) -> jax.Array:
            return model(t_i, c_i)

        preds = jax.vmap(predict_path)(ts_b, coeffs_b)
        return jnp.mean((preds - target_b) ** 2)

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit(donate="all")
    def step(
        batch: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        ts_b, target_b, coeffs_b = batch
        loss_value, grads = grad_fn(model, ts_b, target_b, coeffs_b)
        updates, new_opt_state = optim.update(grads, opt_state)
        model: Model = eqx.apply_updates(model, updates)
        return loss_value, model, new_opt_state

    total_loss = 0.0
    for _ in range(num_batches):
        batch = next(loader)
        loss_value, model, opt_state = step(
            batch=batch, model=model, opt_state=opt_state
        )
        total_loss += float(loss_value)

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, model, opt_state


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

    # Count number of trainable parameters
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

    # Split dataset into train/val/test along the batch dimension (60/20/20)
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
    ) = split_train_val_test(ts_batched, solution, coeffs_batched)

    batch_size = config.experiment_config.batch_size
    num_batches = (ts_train.shape[0] + batch_size - 1) // batch_size
    loader = make_dataloader(
        arrays=(ts_train, target_train, coeffs_train),
        batch_size=batch_size,
        key=loader_key,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    patience = int(config.experiment_config.early_stopping_patience)
    ckpt_path = config.experiment_config.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    epochs = int(config.experiment_config.epochs)
    training_start = time.perf_counter()

    # Define JITted functions once for the training loop
    @eqx.filter_jit
    def loss_fn(
        model: Model,
        ts_b: jax.Array,
        target_b: jax.Array,
        coeffs_b: tuple[jax.Array, ...],
    ) -> jax.Array:
        def predict_path(t_i: jax.Array, c_i: tuple[jax.Array, ...]) -> jax.Array:
            return model(t_i, c_i)

        preds = jax.vmap(predict_path)(ts_b, coeffs_b)
        return jnp.mean((preds - target_b) ** 2)

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit(donate="all")
    def step(
        batch: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        ts_b, target_b, coeffs_b = batch
        loss_value, grads = grad_fn(model, ts_b, target_b, coeffs_b)
        updates, new_opt_state = optim.update(grads, opt_state)
        new_model: Model = eqx.apply_updates(model, updates)
        return loss_value, new_model, new_opt_state

    # Warmup compile to avoid freezing the first batch bar update
    warm_batch = next(loader)
    warm_loss, model, opt_state = step(warm_batch, model, opt_state)
    _ = jax.block_until_ready(warm_loss)

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0, leave=True)
    for epoch_idx in epoch_bar:
        epoch_loss_sum = 0.0
        for batch_idx in tqdm(
            range(num_batches), desc="Batches", position=1, leave=False
        ):
            batch = next(loader)
            loss_value, model, opt_state = step(batch, model, opt_state)
            loss_value = jax.block_until_ready(loss_value)
            epoch_loss_sum += float(loss_value)

        train_loss = epoch_loss_sum / max(1, num_batches)
        # Reflect the final train loss in the epoch bar postfix
        epoch_bar.set_postfix(train=f"{train_loss * 1e2:.4f}×10⁻²")

        # Evaluate on validation set at epoch end (unconditional)
        val_loss_value = eval_epoch(model, ts_val, target_val, coeffs_val)
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch_idx
            eqx.tree_serialise_leaves(ckpt_path, model)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        epoch_bar.set_postfix(
            val=f"{val_loss_value * 1e2:.4f}×10⁻²",
            best=f"{best_val_loss * 1e2:.4f}×10⁻² at epoch {best_epoch}",
        )

        if epochs_since_improve >= patience:
            tqdm.write(
                f"Early stopping at epoch {epoch_idx} (no val improvement for {patience} epochs)."
            )
            break

    training_elapsed = time.perf_counter() - training_start

    # Load best checkpoint for final test evaluation
    best_model: Model = eqx.tree_deserialise_leaves(ckpt_path, model)
    inference_start = time.perf_counter()
    test_loss_value = eval_epoch(best_model, ts_test, target_test, coeffs_test)
    inference_elapsed = time.perf_counter() - inference_start
    tqdm.write(
        f"num_params={num_params} | train_s={training_elapsed:.3f} | infer_s={inference_elapsed:.3f} | test_loss={test_loss_value * 1e2:.4f}×10⁻²"
    )


def main(config_path: str = "configs/nrde.toml") -> None:
    config = load_toml_config(config_path)
    experiment(config)


if __name__ == "__main__":
    main()

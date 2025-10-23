import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
from tqdm.auto import tqdm


def add_time_channel(values: jax.Array) -> jax.Array:
    """
    Append time as a leading channel to the provided path values.

    values: shape (B, T, C)
    returns: shape (B, T, C+1) where the leading channel is time in [0, 1].
    """
    batch_size, length, _ = values.shape
    ts = jnp.linspace(0.0, 1.0, length, dtype=values.dtype)
    time_channel = jnp.broadcast_to(ts[None, :, None], (batch_size, length, 1))
    return jnp.concatenate([time_channel, values], axis=-1)


def compute_cubic_coeffs_batch(ts: jax.Array, ys: jax.Array) -> tuple[jax.Array, ...]:
    """
    Compute batched Hermite cubic spline coefficients.

    ts: shape (T,)
    ys: shape (B, T, C)
    returns: tuple of arrays each of shape (B, T, C)
    """
    vmapped = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))
    coeffs = vmapped(ts, ys)
    return coeffs


def load_npz_dataset(npz_path: str) -> tuple[jax.Array, jax.Array]:
    """
    Load an NPZ dataset with keys 'solution' and 'driver' and return as JAX arrays.
    Returns (solution, driver).
    """
    data = np.load(npz_path)
    solution_np = data["solution"]  # (B, T, C_sol)
    driver_np = data["driver"]  # (B, T, C_drv)
    # Use float32 to avoid slow float64 kernels/compilation on many GPUs
    solution = jnp.asarray(solution_np, dtype=jnp.float32)
    driver = jnp.asarray(driver_np, dtype=jnp.float32)
    return solution, driver


def prepare_dataset(
    npz_path: str, *, key: jax.Array
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    """
    Load dataset and precompute cubic coefficients with time channel.

    Returns (ts_batched, solution, coeffs_batched) suitable for the dataloader.
    - ts_batched: shape (B, T)
    - solution: shape (B, T, C)
    - coeffs_batched: tuple of arrays each shape (B, T, data_channels)
    """
    del key  # key kept for future stochastic data transforms
    solution, driver = load_npz_dataset(npz_path)
    tqdm.write(
        f"Loaded dataset with shape dtype {solution.dtype} (batch: {solution.shape[0]}, length: {solution.shape[1]}, channels: {solution.shape[2]}) and {driver.dtype} (batch: {driver.shape[0]}, length: {driver.shape[1]}, channels: {driver.shape[2]})"
    )

    batch_size, length, _ = driver.shape
    ts = jnp.linspace(0.0, 1.0, length, dtype=driver.dtype)
    ts_batched = jnp.broadcast_to(ts[None, :], (batch_size, length))

    control_values = add_time_channel(driver)
    coeffs_batched = compute_cubic_coeffs_batch(ts, control_values)
    return ts_batched, solution, coeffs_batched


def split_train_val_test(
    ts_batched: jax.Array,
    solution: jax.Array,
    coeffs_batched: tuple[jax.Array, ...],
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array,
    tuple[jax.Array, ...],
]:
    """
    Split dataset into train/val/test along the batch dimension.
    Returns (ts_train, target_train, coeffs_train, ts_val, target_val, coeffs_val, ts_test, target_test, coeffs_test).
    """
    batch_count = solution.shape[0]

    # Base sizes
    train_size = max(1, int(train_fraction * batch_count))
    # Ensure validation size does not exceed remaining after train
    tentative_val = int(val_fraction * batch_count)
    val_size = max(0, min(tentative_val, batch_count - train_size))
    test_size = max(0, batch_count - train_size - val_size)

    ts_train = ts_batched[:train_size]
    target_train = solution[:train_size]
    coeffs_train = tuple(c[:train_size] for c in coeffs_batched)

    ts_val = ts_batched[train_size : train_size + val_size]
    target_val = solution[train_size : train_size + val_size]
    coeffs_val = tuple(c[train_size : train_size + val_size] for c in coeffs_batched)

    ts_test = ts_batched[train_size + val_size : train_size + val_size + test_size]
    target_test = solution[train_size + val_size : train_size + val_size + test_size]
    coeffs_test = tuple(
        c[train_size + val_size : train_size + val_size + test_size]
        for c in coeffs_batched
    )

    return (
        ts_train,
        target_train,
        coeffs_train,
        ts_val,
        target_val,
        coeffs_val,
        ts_test,
        target_test,
        coeffs_test,
    )


def make_dataloader(
    arrays: tuple[jax.Array, jax.Array, tuple[jax.Array, ...]],
    batch_size: int,
    *,
    key: jax.Array,
):
    """
    Simple shuffled minibatch generator yielding (ts_batch, target_batch, coeffs_batch).
    - ts_batch: shape (B, T)
    - target_batch: shape (B, T, C)
    - coeffs_batch: tuple of arrays, each shape (B, T, data_channels)
    """
    ts_all, target_all, coeffs_all = arrays
    dataset_size = target_all.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jr.split(key, 1)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = perm[start:end]
            ts_b = ts_all[batch_idx]
            target_b = target_all[batch_idx]
            coeffs_b = tuple(c[batch_idx] for c in coeffs_all)
            yield ts_b, target_b, coeffs_b

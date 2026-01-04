from typing import Generator, Never

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
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
    npz_path: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Load dataset with raw control values (including time channel).

    Returns:
        Tuple of (ts_batched, solution, control_values)
        - ts_batched: shape (B, T)
        - solution: shape (B, T, C)
        - control_values: shape (B, T, C+1) - includes time channel
    """
    solution, driver = load_npz_dataset(npz_path)
    tqdm.write(
        (
            f"Loaded dataset from {npz_path}:\n"
            f"  solution: dtype={solution.dtype}, shape=(batch={solution.shape[0]}, length={solution.shape[1]}, channels={solution.shape[2]})\n"
            f"  driver:   dtype={driver.dtype}, shape=(batch={driver.shape[0]}, length={driver.shape[1]}, channels={driver.shape[2]})"
        )
    )

    batch_size, length, _ = driver.shape
    ts = jnp.linspace(0.0, 1.0, length, dtype=driver.dtype)
    ts_batched = jnp.broadcast_to(ts[None, :], (batch_size, length))

    control_values = add_time_channel(driver)
    return ts_batched, solution, control_values


def split_train_val_test(
    ts_batched: jax.Array,
    solution: jax.Array,
    control_values: jax.Array,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """
    Split dataset into train/val/test along the batch dimension.

    Returns:
        (ts_train, target_train, control_train,
         ts_val, target_val, control_val,
         ts_test, target_test, control_test)
    """
    batch_count = solution.shape[0]
    # Normalize if they don't sum to 1.0 to honor intent
    total = train_fraction * 1.0 + val_fraction * 1.0 + test_fraction * 1.0
    if total <= 0:
        raise ValueError("At least one split fraction must be positive.")
    if abs(total - 1.0) > 1e-6:
        train_fraction = train_fraction / total
        val_fraction = val_fraction / total
        test_fraction = test_fraction / total

    train_size = int(round(train_fraction * batch_count))
    val_size = int(round(val_fraction * batch_count))
    # Ensure we don't exceed total; assign remainder to test
    train_size = max(0, min(train_size, batch_count))
    val_size = max(0, min(val_size, max(0, batch_count - train_size)))
    test_size = max(0, batch_count - train_size - val_size)

    ts_train = ts_batched[:train_size]
    target_train = solution[:train_size]
    control_train = control_values[:train_size]

    ts_val = ts_batched[train_size : train_size + val_size]
    target_val = solution[train_size : train_size + val_size]
    control_val = control_values[train_size : train_size + val_size]

    ts_test = ts_batched[train_size + val_size : train_size + val_size + test_size]
    target_test = solution[train_size + val_size : train_size + val_size + test_size]
    control_test = control_values[
        train_size + val_size : train_size + val_size + test_size
    ]

    return (
        ts_train,
        target_train,
        control_train,
        ts_val,
        target_val,
        control_val,
        ts_test,
        target_test,
        control_test,
    )


def make_dataloader(
    timestep: jax.Array,
    solution: jax.Array,
    control_values: jax.Array,
    batch_size: int,
    *,
    key: jax.Array,
) -> Generator[tuple[jax.Array, jax.Array, jax.Array], None, Never]:
    """
    Simple shuffled minibatch generator.

    Args:
        timestep: shape (N, T)
        solution: shape (N, T, C)
        control_values: shape (N, T, C+1) - raw control values with time channel
        batch_size: minibatch size
        key: random key for shuffling

    Yields:
        (timestep_b, solution_b, control_b) where control_b is raw control values
        of shape (B, T, C+1).
    """
    dataset_size = solution.shape[0]
    indices = jnp.arange(dataset_size)

    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jr.split(key, 1)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = perm[start:end]
            ts_b = timestep[batch_idx]
            sol_b = solution[batch_idx]
            control_b = control_values[batch_idx]
            yield ts_b, sol_b, control_b

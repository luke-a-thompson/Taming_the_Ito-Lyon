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


def compute_cubic_coeffs_batch(
    ts: jax.Array, ys: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Compute batched Hermite cubic spline coefficients.

    ts: shape (T,)
    ys: shape (B, T, C)
    returns: tuple of 4 arrays each of shape (B, T, C)
    """
    vmapped = jax.vmap(diffrax.backward_hermite_coefficients, in_axes=(None, 0))
    a, b, c, d = vmapped(ts, ys)
    return (a, b, c, d)


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
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
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
    coeffs_batched = compute_cubic_coeffs_batch(ts, control_values)
    return ts_batched, solution, coeffs_batched


def split_train_val_test(
    ts_batched: jax.Array,
    solution: jax.Array,
    coeffs_batched: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """
    Split dataset into train/val/test along the batch dimension.
    Returns (ts_train, target_train, coeffs_train, ts_val, target_val, coeffs_val, ts_test, target_test, coeffs_test).
    """
    batch_count = solution.shape[0]

    # Enforce all three fractions; adjust last slice to absorb rounding
    assert 0.0 <= train_fraction <= 1.0
    assert 0.0 <= val_fraction <= 1.0
    assert 0.0 <= test_fraction <= 1.0
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
    a, b, c, d = coeffs_batched
    coeffs_train = (a[:train_size], b[:train_size], c[:train_size], d[:train_size])

    ts_val = ts_batched[train_size : train_size + val_size]
    target_val = solution[train_size : train_size + val_size]
    coeffs_val = (
        a[train_size : train_size + val_size],
        b[train_size : train_size + val_size],
        c[train_size : train_size + val_size],
        d[train_size : train_size + val_size],
    )

    ts_test = ts_batched[train_size + val_size : train_size + val_size + test_size]
    target_test = solution[train_size + val_size : train_size + val_size + test_size]
    coeffs_test = (
        a[train_size + val_size : train_size + val_size + test_size],
        b[train_size + val_size : train_size + val_size + test_size],
        c[train_size + val_size : train_size + val_size + test_size],
        d[train_size + val_size : train_size + val_size + test_size],
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
    timestep: jax.Array,
    solution: jax.Array,
    drivers: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    batch_size: int,
    *,
    key: jax.Array,
):
    """
    Simple shuffled minibatch generator yielding (timestep_b, solution_b, drivers_b).
    - timestep_b: shape (B, T)
    - solution_b: shape (B, T, C)
    - drivers_b: tuple of arrays, each shape (B, T, data_channels)
    """
    dataset_size = solution.shape[0]
    indices = jnp.arange(dataset_size)
    a, b, c, d = drivers
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jr.split(key, 1)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = perm[start:end]
            ts_b = timestep[batch_idx]
            sol_b = solution[batch_idx]
            drv_b = (a[batch_idx], b[batch_idx], c[batch_idx], d[batch_idx])
            yield ts_b, sol_b, drv_b

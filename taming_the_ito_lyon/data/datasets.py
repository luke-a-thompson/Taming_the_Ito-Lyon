from typing import Generator, Never

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm.auto import tqdm
from taming_the_ito_lyon.config import Datasets


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


def make_sliding_windows(
    values: jax.Array,
    ts: jax.Array,
    *,
    n_recon: int,
    n_future: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Construct disjoint sliding windows:
      driver = seq[s : s+n_recon]
      solution = seq[s : s+n_recon+n_future]
    Returns:
      ts_windows: (B * W, n_recon + n_future)
      driver: (B * W, n_recon, C)
      solution: (B * W, n_recon + n_future, C)
    where W = T - (n_recon + n_future) + 1.
    """
    total_len = int(n_recon + n_future)
    if n_recon <= 0 or n_future <= 0:
        raise ValueError(
            f"n_recon and n_future must be positive, got n_recon={n_recon}, n_future={n_future}"
        )

    batch_size, length, channels = values.shape
    if length < total_len:
        raise ValueError(
            f"Sequence too short for windowing: length={length}, n_recon+n_future={total_len}"
        )

    start_idx = jnp.arange(length - total_len + 1)  # (W,)
    window_idx = start_idx[:, None] + jnp.arange(total_len)[None, :]  # (W, total_len)

    ts_windows_single = ts[window_idx]  # (W, total_len)
    ts_windows = jnp.repeat(ts_windows_single[None, :, :], batch_size, axis=0).reshape(
        batch_size * ts_windows_single.shape[0],
        total_len,
    )

    windows = jax.vmap(lambda seq: seq[window_idx], in_axes=0)(
        values
    )  # (B, W, total_len, C)
    windows = windows.reshape(
        batch_size * ts_windows_single.shape[0], total_len, channels
    )

    driver = windows[:, :n_recon, :]
    solution = windows
    return ts_windows, driver, solution


def prepare_dataset(
    dataset_name: Datasets,
    n_recon: int | None = None,
    n_future: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Load dataset with raw control values (including time channel).

    Returns:
        Tuple of (ts_batched, solution, control_values)
        - ts_batched: shape (B, T)
        - solution: shape (B, T, C)
        - control_values: shape (B, T, C+1) - includes time channel
    """
    dataset_path = Datasets(dataset_name).value
    data = np.load(dataset_path)

    solution_np: np.ndarray | jax.Array
    driver_np: np.ndarray | jax.Array

    match dataset_name := Datasets(dataset_name):
        case Datasets.OU_PROCESS | Datasets.ROUGH_OU_PROCESS:
            solution_np = data["solution"]  # (B, T, C_sol)
            driver_np = data["driver"]  # (B, T, C_drv)
        case Datasets.BLACK_SCHOLES | Datasets.BERGOMI | Datasets.ROUGH_BERGOMI:
            solution_np = data["solution"]  # (B, T, C_sol)
            driver_np = data["driver"]  # (B, T, C_drv)
            # Keep only the price (index 0) but preserve the channel axis.
            # Example: (3000, 1001, 2) -> (3000, 1001, 1)
            solution_np = solution_np[..., 0:1]
            driver_np = driver_np[..., 0:1]
        case Datasets.SG_SO3_SIMULATION:
            # Simulation data has quarternion scalar last, so is scipy compatible
            damping0_rotmat = data["R_sim_damped0"]
            damping1_rotmat = data["R_sim_damped1"]
            damping2_rotmat = data["R_sim_damped2"]
            damping3_rotmat = data["R_sim_damped3"]
            # Flatten 3x3 -> 9 so the model sees shape (B, T, 9)
            rotmats = np.stack(
                [damping0_rotmat, damping1_rotmat, damping2_rotmat, damping3_rotmat],
                axis=2,
            )
            b, t, boxes, _, _ = rotmats.shape
            rotmats = rotmats.reshape(b * boxes, t, 3, 3)
            solution_np = rotmats.reshape(b * boxes, t, 9)
            # For this dataset, the driver is the observed rotation sequence (per box)
            driver_np = solution_np
        case Datasets.OXFORD_MULTIMOTION_STATIC:
            box1_rotmat = data["R_box1"]
            box2_rotmat = data["R_box2"]
            box3_rotmat = data["R_box3"]
            box4_rotmat = data["R_box4"]
            # Keep boxes as separate sequences by folding the box axis into the batch axis:
            # (B, T, 4, 3, 3) -> (B*4, T, 3, 3) -> (B*4, T, 9)
            rotmats = np.stack(
                [box1_rotmat, box2_rotmat, box3_rotmat, box4_rotmat], axis=2
            )
            b, t, boxes, _, _ = rotmats.shape
            rotmats = rotmats.reshape(b * boxes, t, 3, 3)
            solution_np = rotmats.reshape(b * boxes, t, 9)
            # For this dataset, the driver is the observed rotation sequence (per box)
            driver_np = solution_np
        case _:
            raise ValueError(
                f"Dataset not supported by prepare_dataset: {dataset_name}"
            )

    # Use float32 to avoid slow float64 kernels/compilation on many GPUs
    solution = jnp.asarray(solution_np, dtype=jnp.float32)
    driver = jnp.asarray(driver_np, dtype=jnp.float32)

    # Time grid for the original (unwindowed) sequence.
    batch_size, length, _ = driver.shape
    ts_full = jnp.linspace(0.0, 1.0, length, dtype=driver.dtype)  # (T,)

    # Sliding-window construction (applied to any dataset when both values are provided).
    driver_recon: jax.Array | None = None
    if n_recon is not None and n_future is not None:
        ts_batched, driver_recon, solution = make_sliding_windows(
            driver, ts_full, n_recon=n_recon, n_future=n_future
        )
    else:
        ts_batched = jnp.broadcast_to(ts_full[None, :], (batch_size, length))

    tqdm.write(
        (
            f"Loaded dataset {dataset_name} from {dataset_path}:\n"
            f"  solution: dtype={solution.dtype}, shape=(batch={solution.shape[0]}, length={solution.shape[1]}, channels={solution.shape[2]})\n"
            f"  driver:   dtype={driver.dtype}, shape=(batch={driver.shape[0]}, length={driver.shape[1]}, channels={driver.shape[2]})"
        )
    )

    # Control values include a time channel.
    # In extrapolation/windowing mode we return ONLY the reconstruction control values:
    # (ts_recon, driver_recon). Future timestamps remain available in `ts_batched`,
    # but no dummy future driver values are provided.
    if n_recon is not None and n_future is not None:
        assert driver_recon is not None
        ts_recon = ts_batched[:, : int(n_recon)]
        time_channel = ts_recon[:, :, None]
        control_values = jnp.concatenate([time_channel, driver_recon], axis=-1)
    elif n_recon is not None:
        # Backwards-compatible behavior if a caller provides n_recon without n_future.
        # (This should usually be prevented by config validation.)
        ts_recon = ts_batched[:, : int(n_recon)]
        time_channel = ts_recon[:, :, None]
        control_values = jnp.concatenate(
            [time_channel, driver[:, : int(n_recon)]], axis=-1
        )
    else:
        control_values = add_time_channel(driver)
    return ts_batched, solution, control_values


def split_train_val_test(
    ts_batched: jax.Array,
    solution: jax.Array,
    control_values: jax.Array,
    *,
    key: jax.Array,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
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
    Split dataset into train/val/test along the batch dimension, using a random
    permutation of the batch indices.

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

    indices = jnp.arange(batch_count)
    perm = jr.permutation(key, indices)

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size : train_size + val_size + test_size]

    ts_train = ts_batched[train_idx]
    target_train = solution[train_idx]
    control_train = control_values[train_idx]

    ts_val = ts_batched[val_idx]
    target_val = solution[val_idx]
    control_val = control_values[val_idx]

    ts_test = ts_batched[test_idx]
    target_test = solution[test_idx]
    control_test = control_values[test_idx]
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


Dataloader = Generator[tuple[jax.Array, jax.Array, jax.Array], None, Never]


def make_dataloader(
    timestep: jax.Array,
    solution: jax.Array,
    control_values: jax.Array | None,
    batch_size: int,
    *,
    key: jax.Array,
) -> Generator[tuple[jax.Array, jax.Array, jax.Array], None, Never]:
    """
    Simple shuffled minibatch generator.

    Args:
        timestep: shape (N, T)
        solution: shape (N, T, C)
        control_values: optional. If provided, shape (N, T, C+1) - raw control values
            with time channel. If None, the dataloader yields an empty placeholder
            array for the third component.
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
            if control_values is None:
                control_b = jnp.empty((end - start, 0), dtype=sol_b.dtype)
            else:
                control_b = control_values[batch_idx]
            yield ts_b, sol_b, control_b

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.sources import DiskSource
from taming_the_ito_lyon.config import Config


@dataclass
class SPDWishartDiffusionDataset(DatasetProtocol):
    """Dataset wrapper for Wishart diffusion SPD paths."""

    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)
    _driver_np: np.ndarray = field(init=False, repr=False)
    _solution_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        raw = np.load(self.config.experiment_config.dataset_name.value, allow_pickle=False)
        if not isinstance(raw, np.lib.npyio.NpzFile):
            raise ValueError("Expected .npz file with keys {solution, ts, quadratic_variation}.")
        if "solution" not in raw:
            raise ValueError(f"Expected 'solution' in npz, got keys={list(raw.files)}")
        if "ts" not in raw:
            raise ValueError(f"Expected 'ts' in npz, got keys={list(raw.files)}")
        if "quadratic_variation" not in raw:
            raise ValueError(
                f"Expected 'quadratic_variation' in npz, got keys={list(raw.files)}"
            )
        ts = np.asarray(raw["ts"], dtype=np.float32)
        qv_density = np.asarray(raw["quadratic_variation"], dtype=np.float32)
        solution_raw = np.asarray(raw["solution"], dtype=np.float32)

        # Strict: solution is already in vech(X) coordinates: (B, T, 6).
        if solution_raw.ndim == 2:
            solution_raw = solution_raw[None, ...]
        if solution_raw.ndim != 3 or int(solution_raw.shape[-1]) != 6:
            raise ValueError(
                "Expected solution shaped (B, T, 6) (vech), "
                f"got {solution_raw.shape}"
            )
        solution = solution_raw

        if qv_density.ndim == 3:
            qv_density = qv_density[None, ...]
        if qv_density.ndim != 4 or qv_density.shape[-2:] != (6, 6):
            raise ValueError(
                f"Expected quadratic_variation shaped (B, T, 6, 6) (density), got {qv_density.shape}"
            )
        if int(qv_density.shape[0]) != int(solution.shape[0]):
            raise ValueError(
                f"quadratic_variation must align with solution batch size, got qv={qv_density.shape}, solution={solution.shape}"
            )
        if int(qv_density.shape[1]) == int(solution.shape[1]) - 1:
            # Quadratic variation density is per-interval; pad to align with solution.
            pad = np.zeros(
                (int(qv_density.shape[0]), 1, int(qv_density.shape[2]), int(qv_density.shape[3])),
                dtype=qv_density.dtype,
            )
            qv_density = np.concatenate([pad, qv_density], axis=1)
        elif int(qv_density.shape[1]) != int(solution.shape[1]):
            raise ValueError(
                "quadratic_variation must align with solution time length (T) or T-1 for per-interval density, "
                f"got qv={qv_density.shape}, solution={solution.shape}"
            )
        if ts.ndim != 1 or int(ts.shape[0]) != int(solution.shape[1]):
            raise ValueError(
                f"Expected ts shaped (T,) matching solution length, got ts={ts.shape}, solution={solution.shape}"
            )

        # Store as flattened (6x6) density in `driver`. In unconditional mode the model
        # does not consume dataset drivers, so this is a convenient side-channel for
        # losses that need quadratic variation information.
        driver = qv_density.reshape(qv_density.shape[0], qv_density.shape[1], 36)

        train_fraction = float(self.config.experiment_config.train_fraction)
        val_fraction = float(self.config.experiment_config.val_fraction)

        driver_split = _select_batch_split(
            driver,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        solution_split = _select_batch_split(
            solution,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

        # Keep NumPy arrays here: `DiskSource` expects NumPy, and the dataloader
        # pipeline will device-put as needed. `__getitem__` can still return JAX arrays
        # for interactive use.
        self._driver_np = np.asarray(driver_split, dtype=np.float32)
        self._solution_np = np.asarray(solution_split, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._driver_np.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "driver": jnp.asarray(self._driver_np[index], dtype=jnp.float32),
            "solution": jnp.asarray(self._solution_np[index], dtype=jnp.float32),
        }

    def make_disk_source(self) -> DiskSource:
        """Create a `DiskSource` streaming samples.

        Notes:
        - `driver` has shape (T, 36) and contains flattened 6x6 quadratic-variation density.
        - `solution` has shape (T, 6) and is already in vech(X) coordinates.
        """
        driver_np = self._driver_np
        solution_np = self._solution_np

        if driver_np.ndim != 3 or int(driver_np.shape[-1]) != 36:
            raise ValueError(f"Expected driver shaped (B, T, 36), got {driver_np.shape}")
        if solution_np.ndim != 3 or int(solution_np.shape[-1]) != 6:
            raise ValueError(
                f"Expected solution shaped (B, T, 6) (vech), got {solution_np.shape}"
            )
        if int(driver_np.shape[0]) != int(solution_np.shape[0]) or int(driver_np.shape[1]) != int(
            solution_np.shape[1]
        ):
            raise ValueError(
                f"driver/solution must align in (B,T), got driver={driver_np.shape}, solution={solution_np.shape}"
            )

        t_len = int(driver_np.shape[1])

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            return {
                "driver": np.asarray(driver_np[idx], dtype=np.float32),  # (T, 36)
                "solution": np.asarray(solution_np[idx], dtype=np.float32),  # (T, 6)
            }

        sample_spec = {
            "driver": jax.ShapeDtypeStruct(shape=(t_len, 36), dtype=jnp.float32),
            "solution": jax.ShapeDtypeStruct(shape=(t_len, 6), dtype=jnp.float32),
        }

        return DiskSource(
            length=int(driver_np.shape[0]),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=self.ordering,
            prefetch_size=self.config.experiment_config.batch_size,
        )


def _select_batch_split(
    array: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> np.ndarray:
    """Split along the batch axis (axis=0) without mixing trajectories."""
    if array.ndim < 1:
        raise ValueError(
            f"Expected array with at least 1 dim (B, ...), got shape {array.shape}"
        )
    n = int(array.shape[0])
    if n <= 0:
        raise ValueError("Array must be non-empty.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    train_end = min(max(int(n * train_fraction), 1), n)
    if val_fraction > 0.0:
        val_end = min(max(int(n * (train_fraction + val_fraction)), train_end + 1), n)
    else:
        val_end = train_end

    if split == "train":
        return array[:train_end, ...]
    if split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        return array[train_end:val_end, ...]
    return array[val_end:, ...]


if __name__ == "__main__":
    from taming_the_ito_lyon.config.config import load_toml_config

    config = load_toml_config("configs/spd_covariance/~m_nrde_wishart.toml")
    dataset = SPDWishartDiffusionDataset(config, "train")
    print(dataset._driver_np.shape)
    print(dataset._solution_np.shape)

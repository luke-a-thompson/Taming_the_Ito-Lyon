from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource, DiskSource
from taming_the_ito_lyon.config import Config


@dataclass
class SPDCovarianceDataset(DatasetProtocol):
    """Dataset wrapper for windowed SPD covariance sequences."""

    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)
    _driver_np: np.ndarray = field(init=False, repr=False)
    _solution_np: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        raw = np.load(
            self.config.experiment_config.dataset_name.value, allow_pickle=False
        )
        if isinstance(raw, np.lib.npyio.NpzFile):
            if "covariances" not in raw:
                raise ValueError(
                    f"Expected 'covariances' in npz, got keys={list(raw.files)}"
                )
            raw = raw["covariances"]

        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        # Strict: already vech'ed (B, T, 6).
        if arr.ndim != 3 or int(arr.shape[-1]) != 6:
            raise ValueError(
                f"Expected covariances shaped (B, T, 6) (vech), got {arr.shape}"
            )
        solution = arr
        driver = np.zeros((solution.shape[0], solution.shape[1], 1), dtype=np.float32)

        train_fraction = float(self.config.experiment_config.train_fraction)
        val_fraction = float(self.config.experiment_config.val_fraction)
        driver_split = _select_example_split(
            driver,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        solution_split = _select_example_split(
            solution,
            split=self.split,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

        self._driver = to_host_jax_array(driver_split)
        self._solution = to_host_jax_array(solution_split)
        self._driver_np = np.asarray(driver_split, dtype=np.float32)
        self._solution_np = np.asarray(solution_split, dtype=np.float32)

    def __len__(self) -> int:
        return int(self._driver.shape[0])

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        return {
            "driver": self._driver[index],
            "solution": self._solution[index],
        }

    def as_array_dict(self) -> dict[str, jax.Array]:
        """Expose the full dataset as a PyTree of JAX arrays."""
        return {"driver": self._driver, "solution": self._solution}

    def make_array_source(self) -> ArraySource:
        return ArraySource(self.as_array_dict(), ordering=self.ordering)

    def make_disk_source(self) -> DiskSource:
        driver_np = self._driver_np
        solution_np = self._solution_np
        if driver_np.ndim != 3 or int(driver_np.shape[-1]) != 1:
            raise ValueError(f"Expected driver shaped (B, T, 1), got {driver_np.shape}")
        if solution_np.ndim != 3 or int(solution_np.shape[-1]) != 6:
            raise ValueError(
                f"Expected solution shaped (B, T, 6), got {solution_np.shape}"
            )
        if int(driver_np.shape[0]) != int(solution_np.shape[0]) or int(
            driver_np.shape[1]
        ) != int(solution_np.shape[1]):
            raise ValueError(
                f"driver/solution must align in (B,T), got driver={driver_np.shape}, solution={solution_np.shape}"
            )
        t_len = int(driver_np.shape[1])

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            return {
                "driver": np.asarray(driver_np[idx], dtype=np.float32),  # (T, 1)
                "solution": np.asarray(solution_np[idx], dtype=np.float32),  # (T, 6)
            }

        sample_spec = {
            "driver": jax.ShapeDtypeStruct(shape=(t_len, 1), dtype=jnp.float32),
            "solution": jax.ShapeDtypeStruct(shape=(t_len, 6), dtype=jnp.float32),
        }

        return DiskSource(
            length=int(driver_np.shape[0]),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=self.ordering,
            prefetch_size=self.config.experiment_config.batch_size,
        )


def _select_example_split(
    array: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> np.ndarray:
    """Split along the time axis (axis=1) without mixing trajectories."""
    if array.ndim < 2:
        raise ValueError(
            f"Expected array with at least 2 dims (B, T, ...), got shape {array.shape}"
        )
    n = int(array.shape[1])
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
        return array[:, :train_end, ...]
    if split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        return array[:, train_end:val_end, ...]
    return array[:, val_end:, ...]


if __name__ == "__main__":
    pass

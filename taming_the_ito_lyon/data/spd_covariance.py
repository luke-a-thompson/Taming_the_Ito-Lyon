from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
from cyreal.sources import ArraySource
from stochastax.manifolds.spd import SPDManifold
from taming_the_ito_lyon.config import Config


@dataclass
class SPDCovarianceDataset(DatasetProtocol):
    """Dataset wrapper for windowed SPD covariance sequences."""

    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)

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

        matrices = np.asarray(raw, dtype=np.float32)
        if matrices.ndim == 3:
            matrices = matrices[None, ...]
        if matrices.ndim != 4:
            raise ValueError(
                f"Expected (B, T, N, N) or (T, N, N), got {matrices.shape}"
            )

        solution = np.asarray(SPDManifold.vech(jnp.asarray(matrices)), dtype=np.float32)
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
        dataset = SPDCovarianceDataset(config=self.config, split=self.split)
        return ArraySource(dataset.as_array_dict(), ordering=self.ordering)


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
    from taming_the_ito_lyon.config.config import load_toml_config

    config = load_toml_config("configs/spd_covariance/~m_nrde.toml")
    dataset = SPDCovarianceDataset(config, "train")
    print(dataset._driver.shape)
    print(dataset._solution.shape)
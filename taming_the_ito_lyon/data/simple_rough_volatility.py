from __future__ import annotations
from dataclasses import dataclass
from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.utils import to_host_jax_array
import jax
import numpy as np
from typing import Literal
from cyreal.datasets.time_utils import make_sequence_disk_source
from taming_the_ito_lyon.config import Config
from cyreal.sources import ArraySource, DiskSource
from dataclasses import field


@dataclass
class SimpleRoughVolatilityDataset(DatasetProtocol):
    """Dataset wrapper for `simple_rbergomi_data.npz`.

    The `.npz` file contains:
    - driver: (B, T, 1)
    - price: (B, T)
    - log_price: (B, T)
    - variance: (B, T)
    - dt: scalar

    This loader exposes a `(B, T, 1)` `solution` by using **log_price** as the
    target.
    """

    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        data = np.load(self.config.experiment_config.dataset_name.value)
        driver = np.asarray(data["driver"], dtype=np.float32)
        log_price = np.asarray(data["log_price"], dtype=np.float32)
        if log_price.ndim != 2:
            raise ValueError(
                f"Expected 'log_price' shaped (num_examples, T), got {log_price.shape}"
            )
        solution = log_price[..., None]  # (B, T, 1)

        if driver.shape != solution.shape:
            raise ValueError(
                f"driver and solution must have the same shape, got {driver.shape} and {solution.shape}"
            )
        if driver.ndim != 3:
            raise ValueError(
                f"Expected arrays shaped (num_examples, T, C), got driver shape {driver.shape}"
            )

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
        dataset = SimpleRoughVolatilityDataset(config=self.config, split=self.split)
        array_source = ArraySource(dataset.as_array_dict(), ordering=self.ordering)
        return array_source

    def make_disk_source(
        self,
    ) -> DiskSource:
        dataset = SimpleRoughVolatilityDataset(config=self.config, split=self.split)
        disk_source = make_sequence_disk_source(
            contexts=np.asarray(dataset._driver),
            targets=np.asarray(dataset._solution),
            ordering=self.ordering,
            prefetch_size=128,
        )
        return disk_source


def _select_example_split(
    array: np.ndarray,
    *,
    split: Literal["train", "val", "test"],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> np.ndarray:
    """Split independent examples along axis 0 (no overlap)."""
    n = int(len(array))
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
        return array[:train_end]
    if split == "val":
        if val_fraction == 0.0:
            raise ValueError("val_fraction must be > 0 when split='val'.")
        return array[train_end:val_end]
    return array[val_end:]


if __name__ == "__main__":
    from taming_the_ito_lyon.config.config import load_toml_config

    config = load_toml_config("configs/rough_volatility_simple/~m_nrde.toml")
    dataset = SimpleRoughVolatilityDataset(config, "train")
    print(dataset._driver.shape)
    print(dataset._solution.shape)

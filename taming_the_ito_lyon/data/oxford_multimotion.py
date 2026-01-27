from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from cyreal.datasets.dataset_protocol import DatasetProtocol
from cyreal.datasets.time_utils import prepare_seq_to_seq_windows
from taming_the_ito_lyon.config import Config
from cyreal.sources import DiskSource
from taming_the_ito_lyon.data.integrity_checks import (
    ensure_b_w_l_c,
    validate_window_alignment,
)


@dataclass
class OxfordMultimotionDataset(DatasetProtocol):
    config: Config
    split: Literal["train", "val", "test"]
    ordering: Literal["sequential", "shuffle"] = field(init=False)
    _driver_np: np.ndarray = field(init=False, repr=False)
    _solution_np: np.ndarray = field(init=False, repr=False)
    _num_windows: int = field(init=False, repr=False)
    _num_examples: int = field(init=False, repr=False)
    _dataset_len: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ordering = "shuffle" if self.split == "train" else "sequential"

        data = np.load(self.config.experiment_config.dataset_name.value)
        dt_sim = float(
            np.asarray(np.diff(data["t"])).mean()
        )  # This is a scalar dt for the whole simulation
        skip = int(1 / dt_sim)
        if skip < 1:
            skip = 1
        print(f"Downsample rate: {skip} (effective dt = {dt_sim * skip:.3f})")

        train_fraction = float(self.config.experiment_config.train_fraction)
        val_fraction = float(self.config.experiment_config.val_fraction)

        box1_rotmat = data["R_box1"]
        box2_rotmat = data["R_box2"]
        box3_rotmat = data["R_box3"]
        box4_rotmat = data["R_box4"]

        rotmats = np.stack(
            [box1_rotmat, box2_rotmat, box3_rotmat, box4_rotmat],
            axis=1,
        )
        timesteps, boxes, _, _ = rotmats.shape
        rotmats_flat = rotmats.reshape(timesteps, boxes, 9).transpose(1, 0, 2)
        driver_np, solution_np = prepare_seq_to_seq_windows(
            input_sequence=rotmats_flat,
            target_sequence=rotmats_flat,
            split=self.split,
            input_window_len=21,  # we will fit the polynomaial to 12 (n_recon) points only
            target_window_len=21,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            sliding_window_stride=1,
        )

        driver_np = ensure_b_w_l_c("driver", driver_np)
        solution_np = ensure_b_w_l_c("solution", solution_np)
        validate_window_alignment(driver_np, solution_np)

        self._driver_np = driver_np
        self._solution_np = solution_np
        self._num_examples = int(driver_np.shape[0])  # B*4
        self._num_windows = int(driver_np.shape[1])  # windows per example
        self._dataset_len = int(self._num_examples * self._num_windows)

    def __len__(self) -> int:
        # Flattened sample count: one sample per (example, window_start)
        return int(self._dataset_len)

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        # Keep this cheap: only materialize a single window pair.
        if index < 0 or index >= self._dataset_len:
            raise IndexError(f"Index out of range: {index} (len={self._dataset_len})")
        wi = index % self._num_windows
        bi = index // self._num_windows
        driver = jnp.asarray(self._driver_np[bi, wi], dtype=jnp.float32)  # (T, 9)
        solution_flat = jnp.asarray(
            self._solution_np[bi, wi], dtype=jnp.float32
        )  # (T, 9)
        solution = solution_flat.reshape(solution_flat.shape[0], 3, 3)  # (T, 3, 3)
        return {
            "driver": driver,
            "solution": solution,
        }

    def make_disk_source(
        self,
    ) -> DiskSource:
        driver_np = self._driver_np
        solution_np = self._solution_np

        _, _, ctx_len, channels = driver_np.shape
        _, _, tgt_len, channels2 = solution_np.shape
        if channels2 != channels:
            raise ValueError(
                f"Driver/solution channel mismatch: driver={channels}, solution={channels2}"
            )
        if int(channels) != 9:
            raise ValueError(
                f"Expected SO(3) rotations flattened as 9 channels, got channels={channels}. Total shape: {driver_np.shape}"
            )

        num_windows = int(self._num_windows)

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            wi = idx % num_windows
            bi = idx // num_windows
            driver = np.asarray(driver_np[bi, wi], dtype=np.float32)  # (T, 9)
            solution_flat = np.asarray(solution_np[bi, wi], dtype=np.float32)  # (T, 9)
            solution = solution_flat.reshape(int(tgt_len), 3, 3)  # (T, 3, 3)
            return {
                "driver": driver,
                "solution": solution,
            }

        sample_spec = {
            "driver": jax.ShapeDtypeStruct(
                shape=(ctx_len, channels), dtype=jnp.float32
            ),
            "solution": jax.ShapeDtypeStruct(shape=(tgt_len, 3, 3), dtype=jnp.float32),
        }

        return DiskSource(
            length=int(self._dataset_len),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=self.ordering,
            prefetch_size=self.config.experiment_config.batch_size,
        )


if __name__ == "__main__":
    from taming_the_ito_lyon.config.config import load_toml_config

    config = load_toml_config("configs/oxford_mm/m_nrde_mlp.toml")
    dataset = OxfordMultimotionDataset(config, "train")
    print(dataset._driver_np.shape)
    print(dataset._solution_np.shape)

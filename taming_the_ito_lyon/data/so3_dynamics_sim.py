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


@dataclass
class SO3DynamicsSim(DatasetProtocol):
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

        train_fraction = float(self.config.experiment_config.train_fraction)
        val_fraction = float(self.config.experiment_config.val_fraction)

        damping0_rotmat = data["R_sim_damped0"]
        damping1_rotmat = data["R_sim_damped1"]
        damping2_rotmat = data["R_sim_damped2"]
        damping3_rotmat = data["R_sim_damped3"]
        # Flatten 3x3 -> 9 so the model sees shape (B, T, 9)
        rotmats = np.stack(
            [damping0_rotmat, damping1_rotmat, damping2_rotmat, damping3_rotmat],
            axis=2,
        )
        batch_size, timesteps, boxes, _, _ = rotmats.shape
        # (Batch * 4, Timesteps, 9)
        rotmats_flat = rotmats.reshape(batch_size * boxes, timesteps, 9)
        # NOTE: `prepare_seq_to_seq_windows` returns NumPy arrays. With the updated
        # `cyreal.datasets.time_utils.sliding_window_many`, these are typically
        # *views* (stride-trick windows), not fully materialized copies.
        driver_np, solution_np = prepare_seq_to_seq_windows(
            input_sequence=rotmats_flat,
            target_sequence=rotmats_flat,
            split=self.split,
            input_window_len=20,  # we will fit the polynomaial to 12 (n_recon) points only
            target_window_len=20,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )

        # `np.lib.stride_tricks.sliding_window_view` appends the window axis at the end,
        # so for sequences shaped (B, T, C) the windowed result often comes out as
        # (B, W, C, L). We want (B, W, L, C) for downstream code and batching.
        if driver_np.ndim == 4 and int(driver_np.shape[-1]) == 20:
            driver_np = np.swapaxes(driver_np, -1, -2)
        if solution_np.ndim == 4 and int(solution_np.shape[-1]) == 20:
            solution_np = np.swapaxes(solution_np, -1, -2)

        if driver_np.ndim != 4:
            raise ValueError(
                f"Expected driver windows to have shape (B, W, L, C), got {driver_np.shape}"
            )
        if solution_np.ndim != 4:
            raise ValueError(
                f"Expected solution windows to have shape (B, W, L, C), got {solution_np.shape}"
            )
        if (
            driver_np.shape[0] != solution_np.shape[0]
            or driver_np.shape[1] != solution_np.shape[1]
        ):
            raise ValueError(
                "Driver/solution window counts are not aligned: "
                f"driver={driver_np.shape}, solution={solution_np.shape}"
            )

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
        return {
            "driver": jnp.asarray(self._driver_np[bi, wi], dtype=jnp.float32),
            "solution": jnp.asarray(self._solution_np[bi, wi], dtype=jnp.float32),
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

        num_windows = int(self._num_windows)

        def _read_sample(index: int | np.ndarray) -> dict[str, np.ndarray]:
            idx = int(np.asarray(index))
            wi = idx % num_windows
            bi = idx // num_windows
            return {
                "driver": np.asarray(driver_np[bi, wi], dtype=np.float32),
                "solution": np.asarray(solution_np[bi, wi], dtype=np.float32),
            }

        sample_spec = {
            "driver": jax.ShapeDtypeStruct(
                shape=(ctx_len, channels), dtype=jnp.float32
            ),
            "solution": jax.ShapeDtypeStruct(
                shape=(tgt_len, channels), dtype=jnp.float32
            ),
        }

        return DiskSource(
            length=int(self._dataset_len),
            sample_fn=_read_sample,
            sample_spec=sample_spec,
            ordering=self.ordering,
            prefetch_size=128,
        )

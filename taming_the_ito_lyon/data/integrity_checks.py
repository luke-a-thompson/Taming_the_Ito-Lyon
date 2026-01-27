from __future__ import annotations

import numpy as np


def ensure_b_w_l_c(
    name: str,
    windows: np.ndarray,
    channels: int = 9,
) -> np.ndarray:
    if windows.ndim != 4:
        raise ValueError(
            f"Expected {name} windows to have shape (B, W, L, C), got {windows.shape}"
        )
    # Expect one of the last two axes to be the SO(3) "channels" axis.
    if int(windows.shape[-1]) == channels:
        return windows  # already (B, W, L, C)
    if int(windows.shape[-2]) == channels:
        return np.swapaxes(windows, -1, -2)  # (B, W, C, L) -> (B, W, L, C)
    raise ValueError(
        f"Expected {name} windows to have channels={channels} on the last axis, "
        f"got shape={windows.shape}"
    )


def validate_window_alignment(
    driver_windows: np.ndarray,
    solution_windows: np.ndarray,
) -> None:
    if (
        driver_windows.shape[0] != solution_windows.shape[0]
        or driver_windows.shape[1] != solution_windows.shape[1]
        or driver_windows.shape[2] != solution_windows.shape[2]
    ):
        raise ValueError(
            "Driver/solution window counts are not aligned: "
            f"driver={driver_windows.shape}, solution={solution_windows.shape}"
        )

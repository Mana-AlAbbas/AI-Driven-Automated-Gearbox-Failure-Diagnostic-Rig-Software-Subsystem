"""Signal windowing helpers."""

from typing import Dict, List, Tuple

import numpy as np


def create_windows(
    x: np.ndarray,
    metadata: Dict,
    window_size: int = 12800,
    overlap: float = 0.5,
    fs: int = 25600,
) -> Tuple[List[np.ndarray], List[Dict]]:
    windows: List[np.ndarray] = []
    window_metadata: List[Dict] = []

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")

    if len(x) <= window_size:
        meta_copy = metadata.copy()
        meta_copy["window_id"] = 0
        meta_copy["window_start"] = 0.0
        meta_copy["window_end"] = len(x) / fs
        windows.append(x)
        window_metadata.append(meta_copy)
        return windows, window_metadata

    step_size = max(1, int(window_size * (1 - overlap)))
    num_windows = max(1, (len(x) - window_size) // step_size + 1)

    for index in range(num_windows):
        start = index * step_size
        end = start + window_size
        if end > len(x):
            break

        windows.append(x[start:end])

        meta_copy = metadata.copy()
        meta_copy["window_id"] = index
        meta_copy["window_start"] = start / fs
        meta_copy["window_end"] = end / fs
        window_metadata.append(meta_copy)

    return windows, window_metadata

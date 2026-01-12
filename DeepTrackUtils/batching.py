from __future__ import annotations
from typing import Iterable, List
import numpy as np


def window_video(x: np.ndarray, fps: float, window_sec: int, drop_last: bool = True) -> List[np.ndarray]:
    """
    Split frames into non-overlapping windows.

    x: [T, 480, 640, 1]
    return: list of [L, 480, 640, 1]
    """
    frames_per_window = int(round(window_sec * fps))
    if frames_per_window <= 0:
        return []
    T = x.shape[0]
    total = (T // frames_per_window) * frames_per_window
    if total == 0:
        return []
    if not drop_last and total < T:
        total = T
    windows = [x[i:i + frames_per_window] for i in range(0, total - frames_per_window + 1, frames_per_window)]
    return windows


def pad_tracks_to_capacity(tracks: np.ndarray, capacity: int, pad_value: float = np.nan) -> np.ndarray:
    """
    Ensure tracks tensor has shape [T, capacity, 2]. If smaller, pad with pad_value; if larger, truncate.

    tracks: [T, N, 2]
    """
    T, N, C = tracks.shape
    assert C == 2
    if N == capacity:
        return tracks
    if N > capacity:
        return tracks[:, :capacity, :]
    out = np.full((T, capacity, 2), pad_value, dtype=tracks.dtype)
    out[:, :N, :] = tracks
    return out

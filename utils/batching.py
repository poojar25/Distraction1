from __future__ import annotations
from typing import Iterable, List
import numpy as np


def window_video(x: np.ndarray, fps: float, window_sec: int, drop_last: bool = True) -> List[np.ndarray]:
    """
    Split into non-overlapping windows of length window_sec.

    x: [T, H, W, 1]
    returns: list of [L, H, W, 1]
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


def make_batches(windows: list[np.ndarray], batch_size: int) -> Iterable[np.ndarray]:
    """
    Yield batches of windows.

    Each batch: [B, L, H, W, 1]
    """
    if batch_size <= 0:
        batch_size = 1
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        yield np.stack(batch, axis=0)

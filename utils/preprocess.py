from __future__ import annotations
import numpy as np


def zscore_per_video(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-pixel z-score across time.

    x: [T, H, W, 1]
    returns: same shape
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def normalize_minus1_1(x: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Scale entire video globally to [-1, 1] using min/max across all elements.

    x: [T, H, W, 1]
    returns: same shape
    """
    x_min = x.min()
    x_max = x.max()
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x = np.nan_to_num(x)
        x_min = x.min()
        x_max = x.max()

    if x_max <= x_min + 1e-12:
        return np.zeros_like(x, dtype=x.dtype)

    y = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    if clip:
        y = np.clip(y, -1.0, 1.0)
    return y.astype(x.dtype, copy=False)

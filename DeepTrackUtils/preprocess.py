from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, detrend as sp_detrend


def detrend_per_pixel(x: np.ndarray) -> np.ndarray:
    """
    Linear detrend per pixel over time.
    x: [T, 480, 640, 1]
    """
    T, H, W, C = x.shape
    assert C == 1
    X = x.reshape(T, H * W)
    X = np.nan_to_num(X, copy=False)
    X_dt = sp_detrend(X, axis=0, type="linear")
    return X_dt.reshape(T, H, W, 1).astype(x.dtype, copy=False)


def bandpass_per_pixel(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """
    Butterworth band-pass + filtfilt along time per pixel.
    x: [T, 480, 640, 1]
    """
    if fs is None or fs <= 0:
        fs = 30.0
    T, H, W, C = x.shape
    assert C == 1
    X = x.reshape(T, H * W)
    X = np.nan_to_num(X, copy=False)

    nyq = 0.5 * fs
    lo = max(low, 0.0)
    hi = min(high, nyq - 1e-6)
    if lo >= hi:
        return x

    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    X_f = filtfilt(b, a, X, axis=0, method="gust")
    return X_f.reshape(T, H, W, 1).astype(x.dtype, copy=False)


def zscore_per_video(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std


def normalize_minus1_1(x: np.ndarray, clip: bool = True) -> np.ndarray:
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
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

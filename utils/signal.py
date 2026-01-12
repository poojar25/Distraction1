from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, detrend as sp_detrend


def detrend_per_pixel(x: np.ndarray) -> np.ndarray:
    """
    Detrend along time for each pixel.

    x: [T, H, W, 1]
    returns: same shape
    """
    T, H, W, C = x.shape
    assert C == 1, "Expected single-channel input"
    X = x.reshape(T, H * W)
    X = np.nan_to_num(X, copy=False)
    X_dt = sp_detrend(X, axis=0, type="linear")
    return X_dt.reshape(T, H, W, 1).astype(x.dtype, copy=False)


def bandpass_per_pixel(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """
    Band-pass filter along time using Butterworth + filtfilt.

    x: [T, H, W, 1]
    returns: same shape
    """
    if fs is None or fs <= 0:
        fs = 30.0
    T, H, W, C = x.shape
    assert C == 1, "Expected single-channel input"
    X = x.reshape(T, H * W)
    X = np.nan_to_num(X, copy=False)

    # Ensure frequency bounds are valid
    nyq = 0.5 * fs
    lo = max(low, 0.0)
    hi = min(high, nyq - 1e-6)
    if lo >= hi:
        # No-op if invalid
        return x

    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    # filtfilt along time axis (0)
    X_f = filtfilt(b, a, X, axis=0, method="gust")
    return X_f.reshape(T, H, W, 1).astype(x.dtype, copy=False)


def resample_per_pixel(x: np.ndarray, fs_in: float, fs_out: float | None) -> tuple[np.ndarray, float]:
    """
    Optional resampling along time. If fs_out is None, no-op.

    Returns (x_resampled, fs_effective)
    """
    if fs_out is None or fs_out <= 0 or abs(fs_out - fs_in) < 1e-6:
        return x, fs_in

    # For now, left as a no-op per user request. Keep signature for future.
    return x, fs_in

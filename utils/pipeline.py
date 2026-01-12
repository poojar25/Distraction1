from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np

from .config import Config
from .io import load_video
from .signal import detrend_per_pixel, bandpass_per_pixel, resample_per_pixel
from .preprocess import zscore_per_video, normalize_minus1_1
from .batching import window_video


def process_video_file(path: str, cfg: Config) -> List[np.ndarray]:
    """
    Process a single video into non-overlapping windows.

    Steps: load -> detrend -> band-pass -> (no-op resample) -> z-score -> scale [-1,1] -> windowing

    Returns list of windows, each [L, 36, 36, 1].
    """
    vid, fps = load_video(path, resize_hw=cfg.resize_hw, to_gray=cfg.to_gray)

    vid = detrend_per_pixel(vid)
    vid = bandpass_per_pixel(vid, fs=fps, low=cfg.bandpass[0], high=cfg.bandpass[1], order=cfg.filter_order)

    # Resampling left as a no-op for now
    vid, fps_eff = resample_per_pixel(vid, fs_in=fps, fs_out=cfg.resample_hz)

    vid = zscore_per_video(vid, eps=cfg.zscore_eps)
    vid = normalize_minus1_1(vid, clip=cfg.scale_clip)

    windows = window_video(vid, fps=fps_eff, window_sec=cfg.window_sec, drop_last=cfg.drop_last_window)
    return windows


def process_folder(folder: str, pattern: str = "*.avi", cfg: Config | None = None) -> Iterable[Tuple[str, List[np.ndarray]]]:
    """
    Iterate over a folder of videos and yield (path, windows) pairs.
    """
    if cfg is None:
        cfg = Config()
    for p in sorted(Path(folder).glob(pattern)):
        try:
            windows = process_video_file(str(p), cfg)
            yield str(p), windows
        except Exception as e:
            # Skip problematic files but continue
            print(f"[WARN] Skipping {p}: {e}")
            continue

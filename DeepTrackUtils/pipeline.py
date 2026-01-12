from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import numpy as np

from .config import DTConfig
from .io import load_video_640x480
from .preprocess import detrend_per_pixel, bandpass_per_pixel, zscore_per_video, normalize_minus1_1
from .tracker import DeepTrackTracker, track_sequence
from .batching import window_video


def process_video_file(path: str, cfg: DTConfig) -> List[np.ndarray]:
    """
    Process a single video into non-overlapping windows of tracked points.

    Steps: load (640x480) -> detrend -> band-pass -> z-score -> scale [-1,1]
           -> per-frame detection (LodeSTAR via deeplay) -> frame-to-frame tracking
           -> windowing -> pad to capacity

    Returns list of windows, each with shape [T, N, 2].
    """
    # Load
    vid, fps = load_video_640x480(path, to_gray=True)  # [T,480,640,1] in [0,1]
    # Preprocess
    vid = detrend_per_pixel(vid)
    vid = bandpass_per_pixel(vid, fs=fps, low=cfg.bandpass[0], high=cfg.bandpass[1], order=cfg.filter_order)
    vid = zscore_per_video(vid, eps=cfg.zscore_eps)
    vid = normalize_minus1_1(vid, clip=cfg.scale_clip)

    # Build tracker (deeplay LodeSTAR)
    tracker = DeepTrackTracker(
        n_transforms=4,
        lr=1e-4,
        capacity=cfg.capacity,
        max_distance=cfg.max_distance,
    )

    # Windowing by frames
    windows_frames = window_video(vid, fps=fps, window_sec=cfg.window_sec, drop_last=cfg.drop_last_window)

    outputs: List[np.ndarray] = []
    for wf in windows_frames:
        # Detect + track
        tracks = tracker.detect(wf)  # [T,N,2]
        outputs.append(tracks)

    return outputs


def process_folder(folder: str, pattern: str = "*.avi", cfg: Optional[DTConfig] = None) -> Iterable[Tuple[str, List[np.ndarray]]]:
    """
    Iterate over videos in a folder and yield (path, list_of_TxNx2_windows).
    """
    if cfg is None:
        cfg = DTConfig()
    for p in sorted(Path(folder).glob(pattern)):
        try:
            windows = process_video_file(str(p), cfg)
            yield str(p), windows
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

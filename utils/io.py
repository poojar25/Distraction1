from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def _ensure_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        # Default fallback if metadata missing
        fps = 30.0
    return float(fps)


def load_video(path: str, resize_hw: Tuple[int, int] = (36, 36), to_gray: bool = True) -> Tuple[np.ndarray, float]:
    """
    Load a .avi video using OpenCV.

    Returns
    -------
    frames : np.ndarray
        Array with shape [T, H, W, 1] in float32, values in [0, 1].
    fps : float
        Native frames-per-second of the input video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    fps = _ensure_fps(cap)
    frames = []
    H, W = resize_hw

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert RGB to single-channel by luminance if not gray
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=-1)  # [H, W, 1]
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from: {path}")

    vid = np.stack(frames, axis=0)  # [T, H, W, 1]
    return vid, fps

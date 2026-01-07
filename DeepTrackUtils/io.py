from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple


def _ensure_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    return float(fps)


def load_video_640x480(path: str, to_gray: bool = True) -> tuple[np.ndarray, float]:
    """
    Load a .avi video using OpenCV and keep original 640x480 spatial resolution.

    Returns
    -------
    frames : np.ndarray
        Array with shape [T, 480, 640, 1] in float32, values in [0, 1].
    fps : float
        Native frames-per-second of the input video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    fps = _ensure_fps(cap)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure 640x480 (OpenCV reads as WxH)
        h, w = frame.shape[:2]
        if (h, w) != (480, 640):
            # resize will remove the color channel if it exists
            if len(frame.shape) == 3:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            else:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        frame = frame.astype(np.float32) / 255.0
        if to_gray:
            frame = np.expand_dims(frame, axis=-1)  # [H, W, 1]
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from: {path}")

    vid = np.stack(frames, axis=0)  # [T, 480, 640, 1]
    return vid, fps

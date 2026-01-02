from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

# DeepTrack (deeplay) import — required via requirements.txt
import deeptrack.deeplay as dl  # type: ignore


###############################
# LodeSTAR tracker class (preferred API)
###############################

class DeepTrackTracker:
    """
    Tracker wrapper around deeptrack.deeplay LodeSTAR.

    Methods:
    - __init__: build LodeSTAR model with given hyperparameters
    - train: fit model using deeplay.DataLoader and deeplay.Trainer
    - detect: run per-frame detection and associate to tracks -> [T, N, 2]
    """

    def __init__(
        self,
        n_transforms: int = 4,
        lr: float = 1e-4,
        capacity: int = 500,
        max_distance: float = 10.0,
    ) -> None:
        self.capacity = capacity
        self.max_distance = max_distance
        self.model = dl.LodeSTAR(n_transforms=n_transforms, optimizer=dl.Adam(lr=lr)).build()
        self._trainer = None

    def _make_dataloader(self, training_dataset, batch_size: int = 8, shuffle: bool = True):
        return dl.DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle)

    def _ensure_trainer(self, max_epochs: int = 200):
        if self._trainer is None:
            self._trainer = dl.Trainer(max_epochs=max_epochs)
        else:
            # update epochs if larger requested
            if hasattr(self._trainer, "max_epochs") and max_epochs > getattr(self._trainer, "max_epochs"):
                self._trainer.max_epochs = max_epochs
        return self._trainer

    def train(self, training_dataset, batch_size: int = 8, shuffle: bool = True, max_epochs: int = 200):
        dataloader = self._make_dataloader(training_dataset, batch_size=batch_size, shuffle=shuffle)
        trainer = self._ensure_trainer(max_epochs=max_epochs)
        trainer.fit(self.model, dataloader)
        return self.model, trainer

    def evaluate(self, validation_dataset, batch_size: int = 8):
        """Minimal evaluation using deeplay DataLoader and model.evaluate if available."""
        dataloader = self._make_dataloader(validation_dataset, batch_size=batch_size, shuffle=False)
        if hasattr(self.model, "evaluate"):
            try:
                return self.model.evaluate(dataloader)
            except Exception:
                return {"status": "evaluate_failed"}
        return {"status": "no_evaluate_method"}

    def _detect_frame(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Detect points in a single frame using the LodeSTAR model. Fallback to blob detection if needed."""
        # Expect frame [H,W,1] in [-1,1]
        try:
            pred = self.model(frame) if callable(self.model) else (
                self.model.predict(frame) if hasattr(self.model, "predict") else None
            )
            if isinstance(pred, np.ndarray):
                if pred.ndim == 2 and pred.shape[1] >= 2:
                    return pred[:, :2].astype(np.float32)[: self.capacity]
                # If pred is heatmap, use fallback postprocess for peaks
        except Exception:
            pass
        # Fallback
        return _fallback_detect_points(frame, max_points=self.capacity)

    def detect(self, frames: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Detect and track across a sequence of frames.
        frames: [T,H,W,1] float32 in [-1,1]
        returns: [T, N, 2] with N=self.capacity, padded with NaN
        """
        T = frames.shape[0]
        dets_per_frame: List[np.ndarray] = []
        for t in range(T):
            dets = self._detect_frame(frames[t], threshold=threshold)
            # cap per-frame count
            if dets.shape[0] > self.capacity:
                dets = dets[: self.capacity]
            dets_per_frame.append(dets)
        tracks = track_sequence(
            dets_per_frame,
            T=T,
            capacity=self.capacity,
            max_distance=self.max_distance,
            pad_value=np.nan,
        )
        return tracks

    def infer_video(self, video_path: str, apply_preprocess: bool = True,
                    bandpass: Tuple[float, float] = (0.7, 2.5), filter_order: int = 3,
                    zscore_eps: float = 1e-6, scale_clip: bool = True) -> np.ndarray:
        """
        Load a video from path, optionally preprocess to [-1,1], then detect+track.
        Returns [T, N, 2] with N=self.capacity.
        """
        from .io import load_video_640x480
        from .preprocess import (
            detrend_per_pixel,
            bandpass_per_pixel,
            zscore_per_video,
            normalize_minus1_1,
        )

        frames, fps = load_video_640x480(video_path, to_gray=True)  # [T,480,640,1] in [0,1]
        if apply_preprocess:
            frames = detrend_per_pixel(frames)
            frames = bandpass_per_pixel(frames, fs=fps, low=bandpass[0], high=bandpass[1], order=filter_order)
            frames = zscore_per_video(frames, eps=zscore_eps)
            frames = normalize_minus1_1(frames, clip=scale_clip)
        else:
            # scale [0,1] -> [-1,1]
            frames = frames * 2.0 - 1.0
        return self.detect(frames)


###############################
# Detection and inference
###############################

def _fallback_detect_points(frame: np.ndarray, max_points: int = 500) -> np.ndarray:
    """
    Fallback detector using OpenCV SimpleBlobDetector on a normalized frame in [-1,1].
    Returns Kx2 array of (x,y) float coordinates.
    """
    img = ((frame.squeeze() + 1.0) * 0.5 * 255.0).astype(np.uint8)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 3.0
    params.maxArea = 1000.0
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    if pts.shape[0] > max_points:
        pts = pts[:max_points]
    return pts




def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance between a[K,2] and b[M,2]."""
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    return d.astype(np.float32)


def track_sequence(dets_per_frame: List[np.ndarray], T: int, capacity: int = 500, max_distance: float = 10.0,
                   pad_value: float = np.nan) -> np.ndarray:
    """
    Associate detections into tracks over T frames and return [T, N, 2] with padding to capacity.

    Parameters
    ----------
    dets_per_frame : list of np.ndarray
        Each element is Kx2 detections for that frame (x,y) in pixels.
    T : int
        Number of frames.
    capacity : int
        Max number of tracks (N). Extra detections start new tracks until capacity.
    max_distance : float
        Gating distance for association.
    pad_value : float
        Value to use when a track has no observation at a frame.

    Returns
    -------
    tracks_tensor : np.ndarray
        Shape [T, capacity, 2], float32.
    """
    # tracks is a dict: track_id -> last_position (2,) and column index in output
    next_track_col = 0
    active_track_cols: List[int] = []  # columns currently used (implicitly 0..next_track_col-1)
    last_positions: List[np.ndarray] = []  # last known positions for active tracks

    # Output initialized with pad_value
    out = np.full((T, capacity, 2), pad_value, dtype=np.float32)

    for t in range(T):
        dets = dets_per_frame[t]
        if dets is None or dets.size == 0:
            # No detections; just propagate pad_value
            continue
        dets = dets.astype(np.float32)

        if next_track_col == 0:
            # Initialize tracks with first frame's detections
            for k in range(min(dets.shape[0], capacity)):
                out[t, k, :] = dets[k]
                active_track_cols.append(k)
                last_positions.append(dets[k].copy())
                next_track_col += 1
            continue

        # Associate detections to existing tracks
        A = np.stack(last_positions, axis=0) if len(last_positions) else np.zeros((0, 2), dtype=np.float32)
        D = _pairwise_dist(A, dets)  # [num_tracks, num_dets]
        if D.size:
            row_ind, col_ind = linear_sum_assignment(D)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        assigned_tracks = set()
        assigned_dets = set()
        # Accept assignments under gating distance
        for r, c in zip(row_ind, col_ind):
            if D[r, c] <= max_distance:
                col = active_track_cols[r]
                out[t, col, :] = dets[c]
                last_positions[r] = dets[c].copy()
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # Unassigned detections -> start new tracks (until capacity)
        for det_idx in range(dets.shape[0]):
            if det_idx in assigned_dets:
                continue
            if next_track_col >= capacity:
                break
            col = next_track_col
            out[t, col, :] = dets[det_idx]
            active_track_cols.append(col)
            last_positions.append(dets[det_idx].copy())
            next_track_col += 1

        # Unassigned tracks get pad_value for this frame (already default in out)
        # Tracks remain active; we keep last_positions unchanged for next association

    return out


###############################
# Utility for padding
###############################

def pad_points_to_capacity(tracks: np.ndarray, capacity: int, pad_value: float = np.nan) -> np.ndarray:
    """Ensure [T,N,2] -> [T,capacity,2] by padding or truncation."""
    T, N, C = tracks.shape
    assert C == 2
    if N == capacity:
        return tracks
    if N > capacity:
        return tracks[:, :capacity, :]
    out = np.full((T, capacity, 2), pad_value, dtype=tracks.dtype)
    out[:, :N, :] = tracks
    return out


###############################
# Convenience: full inference from video path
###############################


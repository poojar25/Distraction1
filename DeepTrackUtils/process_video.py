#!/usr/bin/env python3
"""
How to use
----------

Install dependencies:
    pip install -r requirements.txt

Run on a folder of .avi videos and save BxTxNx2 arrays per video:
    python DeepTrackUtils/process_video.py /path/to/avi_folder --save-dir /path/to/output

Key options:
    --pattern "*.avi"    Glob pattern (default: *.avi)
    --window-sec 10      Window length in seconds (default: 10)
    --drop-last          Drop incomplete tail window
    --capacity 500       Max number of tracks N (default: 500)
    --threshold 0.5      Detection threshold for LodeSTAR (fallback uses blob detection)
    --max-distance 10    Association gate in pixels
    --model-path PATH    Optional path to a LodeSTAR model
    --per-window         Save each window as its own .npy (default saves one stacked [B,T,N,2] per video)

Notes:
    - Inputs are processed at 640x480 resolution (no spatial resizing).
    - Preprocessing: detrend -> band-pass [0.7,2.5] Hz -> z-score -> scale to [-1,1].
    - Output coordinates are (x,y) in pixel units with origin at top-left.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from .config import DTConfig
from .pipeline import process_folder


essential = (
    ("folder", str, "Folder containing video files"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepTrack LodeSTAR point tracking: export BxTxNx2 arrays from videos")
    p.add_argument("folder", type=str, help="Folder containing videos (e.g., .avi)")
    p.add_argument("--pattern", type=str, default="*.avi", help="Glob pattern for videos (default: *.avi)")
    p.add_argument("--save-dir", type=str, required=True, help="Directory to save outputs (.npy)")
    p.add_argument("--window-sec", type=int, default=10, help="Window length in seconds (default: 10)")
    p.add_argument("--drop-last", action="store_true", help="Drop the last incomplete window")
    p.add_argument("--capacity", type=int, default=500, help="Max number of tracks N (default: 500)")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    p.add_argument("--max-distance", type=float, default=10.0, help="Association gate distance in pixels (default: 10)")
    p.add_argument("--band-low", type=float, default=0.7, help="Band-pass low cutoff Hz (default: 0.7)")
    p.add_argument("--band-high", type=float, default=2.5, help="Band-pass high cutoff Hz (default: 2.5)")
    p.add_argument("--filter-order", type=int, default=3, help="Butterworth filter order (default: 3)")
    p.add_argument("--model-path", type=str, default=None, help="Optional path to LodeSTAR model")
    p.add_argument("--per-window", action="store_true", help="Save each window as its own .npy file")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = DTConfig(
        bandpass=(args.band_low, args.band_high),
        filter_order=args.filter_order,
        window_sec=args.window_sec,
        drop_last_window=args.drop_last,
        capacity=args.capacity,
        detection_threshold=args.threshold,
        max_distance=args.max_distance,
        model_path=args.model_path,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_windows = 0

    for path, windows in process_folder(args.folder, pattern=args.pattern, cfg=cfg, model=None):
        total_files += 1
        if not windows:
            print(f"File: {path} -> no windows")
            continue

        # windows: list of [T,N,2]; stack to [B,T,N,2]
        arr = np.stack(windows, axis=0)
        B, T, N, _ = arr.shape
        total_windows += B

        stem = Path(path).stem
        if args.per_window:
            for i, w in enumerate(windows):
                out = save_dir / f"{stem}_win{i:03d}.npy"
                np.save(out, w)
            print(f"File: {path} -> saved {B} windows as individual .npy files (shape each: {windows[0].shape})")
        else:
            out = save_dir / f"{stem}_tracks.npy"
            np.save(out, arr)
            print(f"File: {path} -> saved stacked array {arr.shape} to {out}")

    print("\nSummary")
    print(f"  Processed files: {total_files}")
    print(f"  Total windows:   {total_windows}")


if __name__ == "__main__":
    main()

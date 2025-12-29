#!/usr/bin/env python3
"""
How to use
----------

Install dependencies:
    pip install -r requirements.txt

Run on a folder of .avi videos and save windows as .npy:
    python scripts/process_videos.py /path/to/avi_folder --save-dir /path/to/output

Key options:
    --pattern "*.avi"       Glob pattern (default: *.avi)
    --resize "36,36"        Resize H,W (default: 36,36)
    --band-low 0.7          Band-pass low cutoff Hz (default: 0.7)
    --band-high 2.5         Band-pass high cutoff Hz (default: 2.5)
    --window-sec 10         Window length in seconds (default: 10)
    --drop-last             Drop incomplete tail window
    --no-gray               Skip forcing grayscale (still outputs single-channel)

Notes:
    - Resampling is currently a no-op; windows use the native FPS.
    - Each saved .npy is a window with shape [L, 36, 36, 1].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from utils.config import Config
from utils.pipeline import process_folder


def main():
    p = argparse.ArgumentParser(description="Process a folder of .avi videos into preprocessed 10s windows (36x36x1)")
    p.add_argument("folder", type=str, help="Folder containing .avi files")
    p.add_argument("--pattern", type=str, default="*.avi", help="Glob pattern for videos (default: *.avi)")
    p.add_argument("--save-dir", type=str, default=None, help="If set, saves windows as .npy files under this directory")
    p.add_argument("--no-gray", action="store_true", help="Do not force grayscale (still ends up single-channel)")
    p.add_argument("--window-sec", type=int, default=10, help="Window length in seconds (default: 10)")
    p.add_argument("--drop-last", action="store_true", help="Drop the last incomplete window")
    p.add_argument("--band-low", type=float, default=0.7, help="Band-pass low cutoff Hz (default: 0.7)")
    p.add_argument("--band-high", type=float, default=2.5, help="Band-pass high cutoff Hz (default: 2.5)")
    p.add_argument("--filter-order", type=int, default=3, help="Butterworth filter order (default: 3)")
    p.add_argument("--resize", type=str, default="36,36", help="Resize H,W (default: 36,36)")
    args = p.parse_args()

    H, W = [int(s) for s in args.resize.split(',')]

    cfg = Config(
        resize_hw=(H, W),
        to_gray=not args.no_gray,
        bandpass=(args.band_low, args.band_high),
        filter_order=args.filter_order,
        resample_hz=None,  # no-op
        window_sec=args.window_sec,
        drop_last_window=args.drop_last,
    )

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_windows = 0

    for path, windows in process_folder(args.folder, pattern=args.pattern, cfg=cfg):
        total_files += 1
        print(f"File: {path}")
        print(f"  Num windows: {len(windows)}")
        if len(windows):
            print(f"  Window shape: {windows[0].shape}")
        total_windows += len(windows)

        if save_dir and len(windows):
            base = Path(path).stem
            for i, w in enumerate(windows):
                out = save_dir / f"{base}_win{i:03d}.npy"
                np.save(out, w)

    print("\nSummary")
    print(f"  Processed files: {total_files}")
    print(f"  Total windows:   {total_windows}")


if __name__ == "__main__":
    main()

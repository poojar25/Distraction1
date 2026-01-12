from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    resize_hw: Tuple[int, int] = (36, 36)
    to_gray: bool = True
    bandpass: Tuple[float, float] = (0.7, 2.5)  # Hz
    filter_order: int = 3
    resample_hz: Optional[float] = None  # None -> no-op (skip resampling)
    window_sec: int = 10
    drop_last_window: bool = True
    zscore_eps: float = 1e-6
    scale_clip: bool = True

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DTConfig:
    # Preprocessing
    bandpass: Tuple[float, float] = (0.7, 2.5)  # Hz
    filter_order: int = 3
    zscore_eps: float = 1e-6
    scale_clip: bool = True

    # Windowing / batching
    window_sec: int = 10
    drop_last_window: bool = True

    # Detection / tracking
    capacity: int = 500  # N
    detection_threshold: float = 0.5
    max_distance: float = 10.0  # pixels for association gate

    # Model
    model_path: Optional[str] = None  # optional path to a LodeSTAR model
    use_opencv_fallback: bool = True  # in case DeepTrack is unavailable

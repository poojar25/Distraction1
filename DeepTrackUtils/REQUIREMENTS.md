# DeepTrack 2.0 Requirements

This project uses [DeepTrack 2.0](https://github.com/DeepTrackAI/DeepTrack-2.0) for point detection (LodeSTAR).

## Installation

1) Create/activate a Python environment (recommended)

2) Install project dependencies:
```bash
pip install -r requirements.txt
```

3) Ensure DeepTrack is importable:
```python
import deeptrack as dt
print(dt.__version__)
```

If you encounter backend errors (e.g., TensorFlow), install a compatible backend per DeepTrack's documentation (e.g., TensorFlow or PyTorch variant required by your installation).

## Notes
- LodeSTAR weights/model may be loaded via DeepTrack APIs. Provide a custom `--model-path` to the CLI if needed.
- GPU is optional but recommended for speed.

# Deep Learning Models for Physiological Signal Processing

This repository contains multiple deep learning approaches for physiological signal processing and analysis:

## 1. Linear Transformer for Blood Pressure Estimation

A PyTorch implementation of a Linear Transformer model for continuous blood pressure estimation from pulse waveforms.

### Key Files:
- `LinearTransformerModel.py` - Implements the Linear Transformer architecture
- `Train_LinearTransformer.ipynb` - Training and evaluation notebook
- `requirements.txt` - Python dependencies

### Features:
- Linear attention mechanism for efficient sequence processing
- Multi-head self-attention with ELU activation
- Dual-head output for BP estimation and pulse waveform prediction
- Support for attention masks from pretrained models
- Integration with Weights & Biases for experiment tracking

### Training:
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook Train_LinearTransformer.ipynb
```


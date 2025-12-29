# Encoder Training and Pretraining

This folder contains two pipelines:

- **Supervised encoding** for physiological prediction using `EncoderCAN`.
- **Unsupervised pretraining** via a weight-tying denoising autoencoder.

Key scripts and modules:
- `encoder.py` — PyTorch `EncoderCAN` model, `.npy` output writer, and attention mask exporters.
- `train_encoder.py` — Loads sequences, constructs triplet inputs, runs encoder, and writes outputs.
- `model_denoise_auto_encoder.py` — Denoising autoencoder sharing/tieing weights with the encoder’s raw branch.
- `pretrain.py` — Unsupervised pretraining by video reconstruction (MSE), saves weights to `.h5`.

## Data format

Place data under a directory like `get_Dyno/Data_Dyno/`.
- Each `.npy` file must be an array of shape `(L, 36, 36, 1)` (channels-last grayscale frames).
- The training scripts will scan the directory recursively.

From sequences `(N, L, 36, 36, 1)`, we construct triplets per time step t ≥ 2:
- `rawf_input = [frame(t-2), frame(t-1), frame(t)]` → shape `(M, 36, 36, 3)`
- `diff_input = [f(t-1)-f(t-2), f(t)-f(t-1), f(t)-f(t-2)]` → shape `(M, 36, 36, 3)`
- `M = N * (L - 2)`

## Supervised encoder run

Script: `train_encoder.py`

Purpose: run `EncoderCAN` to produce encoder outputs as `.npy` files (prefix `encoder_out_`) and optionally export attention masks equivalent to Keras layers `conv2d_5` and `conv2d_10`.

Basic usage:
```bash
cd Distraction1/Benefit_of_Distraction-master1/get_Dyno/training/
python train_encoder.py \
  --data_dir ../Data_Dyno \
  --save_dir encoder_outputs \
  --batch_size 128 \
  --save_masks
```

Outputs:
- Encoder predictions as multiple `.npy` files in `--save_dir` named `encoder_out_00000.npy`, `encoder_out_00128.npy`, ...
- If `--save_masks` is set:
  - `mask1.mat` — attention mask from the first gating stage (analogous to Keras `conv2d_5`), shape `(M, 1, 34, 34)`
  - `mask2.mat` — attention mask from the second gating stage (analogous to Keras `conv2d_10`), shape `(M, 1, 15, 15)`

Relevant APIs in `encoder.py`:
- `EncoderCAN` — model class.
- `encode_and_save(model, diff_input, rawf_input, out_dir, batch_size=128)` — writes `encoder_out_*.npy`.
- `save_attention_masks(model, diff_input, rawf_input, save_dir, batch_size=128)` — writes `mask1.mat`, `mask2.mat`.

## Unsupervised pretraining (reconstruction)

Script: `pretrain.py`

Purpose: pretrain the encoder as part of a weight-tying denoising autoencoder using reconstruction loss (MSE). This improves initialization prior to supervised training.

Basic usage:
```bash
cd Distraction1/Benefit_of_Distraction-master1/get_Dyno/training/
python pretrain.py \
  --data_dir ../Data_Dyno \
  --save_dir pretrain_outputs \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-3 \
  --noise_std 0.05
```

What it does:
- Converts video sequences to raw triplets `(M, 3, 36, 36)`.
- Adds Gaussian noise during training (`--noise_std`).
- Trains `DenoiseAutoEncoder` (`model_denoise_auto_encoder.py`) with MSE loss to reconstruct the clean triplets.
- Ties the decoder transposed-convolution weights to the encoder’s raw branch convs to encourage consistent representations.

Outputs:
- HDF5 weight files saved to `--save_dir`:
  - `dae_best.h5` — best validation loss
  - `dae_final.h5` — final epoch

## Using pretrained weights

The `.h5` files contain a dump of the PyTorch `state_dict` tensors for interoperability. To initialize the supervised encoder with pretrained weights, load the `.h5` tensors and map the corresponding keys to `EncoderCAN` parameters (or we can add a helper upon request).

## Environment

- Python 3.8+
- PyTorch
- NumPy, SciPy, h5py

Suggested install (conda):
```bash
conda create -n dyno python=3.10 -y
conda activate dyno
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheels
pip install numpy scipy h5py
```

## Notes

- Data values: if arrays are 0–255, scripts normalize triplets to [0,1] automatically for pretraining. Adjust as needed.
- GPU is used automatically if available; override with `--device cpu` or `--device cuda` in scripts that accept it.
- If you need an actual supervised training loop for physiological labels, we can wire one similarly to Keras using `MSELoss` and callbacks.

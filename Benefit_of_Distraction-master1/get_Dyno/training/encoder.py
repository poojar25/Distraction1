import os
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io


class MaskNorm(nn.Module):
    """
    Normalizes a single-channel spatial mask so its sum equals H*W*0.5, per-sample.
    Equivalent to Keras Lambda in the original model.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("MaskNorm expects input of shape (N, 1, H, W)")
        n, c, h, w = x.size()
        s = x.sum(dim=(2, 3), keepdim=True)  # (N,1,1,1)
        x = x / (s + self.eps) * (h * w * 0.5)
        return x


class EncoderCAN(nn.Module):
    """
    PyTorch reimplementation of the Keras dual-branch gated CNN from
    get_initial_BR_train_CAN_model.py.

    Expected input shapes:
      - diff_input: (N, 3, H, W)
      - rawf_input: (N, 3, H, W)

    Default H=W=36 to match the training script. If other sizes are used, the
    dense layers will be sized by probing a dummy forward pass at init.
    """

    def __init__(
        self,
        img_h: int = 36,
        img_w: int = 36,
        nb_filters1: int = 32,
        nb_filters2: int = 64,
        nb_dense: int = 32,
        dropout_rate1: float = 0.25,
        dropout_rate2: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2

        # Branch D (diff_input)
        self.d1 = nn.Conv2d(3, nb_filters1, kernel_size=3, padding=1)  # 'same'
        self.d2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=3, padding=0)  # 'valid'

        # Branch R (rawf_input)
        self.r1 = nn.Conv2d(3, nb_filters1, kernel_size=3, padding=1)  # 'same'
        self.r2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=3, padding=0)  # 'valid'

        # Gating 1 from R onto D
        self.g1_conv = nn.Conv2d(nb_filters1, 1, kernel_size=1, padding=0)
        self.g1_norm = MaskNorm()

        # After first pooling/dropout
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout_rate1)

        # Second conv block
        self.d5 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=3, padding=1)  # 'same'
        self.d6 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=3, padding=0)  # 'valid'

        self.r5 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=3, padding=1)  # 'same'
        self.r6 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=3, padding=0)  # 'valid'

        # Gating 2 from R onto D
        self.g2_conv = nn.Conv2d(nb_filters2, 1, kernel_size=1, padding=0)
        self.g2_norm = MaskNorm()

        self.drop2 = nn.Dropout(dropout_rate1)

        # We'll determine the flattened size dynamically via a dummy forward.
        flat_dim = self._infer_flat_dim(device=device)

        self.fc1 = nn.Linear(flat_dim, nb_dense)
        self.drop_fc = nn.Dropout(dropout_rate2)
        self.fc_out = nn.Linear(nb_dense, 1)

    def _infer_flat_dim(self, device: Optional[torch.device] = None) -> int:
        dev = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        with torch.no_grad():
            x_d = torch.zeros(1, 3, self.img_h, self.img_w, device=dev)
            x_r = torch.zeros(1, 3, self.img_h, self.img_w, device=dev)
            y = self._forward_features(x_d, x_r)
            flat_dim = y.view(1, -1).size(1)
        return flat_dim

    def _forward_features(self, diff_input: torch.Tensor, rawf_input: torch.Tensor) -> torch.Tensor:
        # First conv block with tanh activations
        d = torch.tanh(self.d1(diff_input))
        d = torch.tanh(self.d2(d))  # (N, nb_filters1, 34, 34)

        r = torch.tanh(self.r1(rawf_input))
        r = torch.tanh(self.r2(r))  # (N, nb_filters1, 34, 34)

        # Gating 1
        g1 = torch.sigmoid(self.g1_conv(r))  # (N,1,34,34)
        g1 = self.g1_norm(g1)
        d = d * g1  # gated

        # Pool + dropout
        d = self.drop1(self.pool(d))  # (N, nb_filters1, 17, 17)
        r = self.drop1(self.pool(r))  # (N, nb_filters1, 17, 17)

        # Second conv block
        d = torch.tanh(self.d5(d))
        d = torch.tanh(self.d6(d))  # (N, nb_filters2, 15, 15)

        r = torch.tanh(self.r5(r))
        r = torch.tanh(self.r6(r))  # (N, nb_filters2, 15, 15)

        # Gating 2
        g2 = torch.sigmoid(self.g2_conv(r))  # (N,1,15,15)
        g2 = self.g2_norm(g2)
        d = d * g2

        # Pool + dropout
        d = self.drop2(self.pool(d))  # (N, nb_filters2, 7, 7)
        return d

    def forward(self, diff_input: torch.Tensor, rawf_input: torch.Tensor) -> torch.Tensor:
        d = self._forward_features(diff_input, rawf_input)
        d = torch.flatten(d, start_dim=1)
        d = torch.tanh(self.fc1(d))
        d = self.drop_fc(d)
        out = self.fc_out(d)
        return out  # (N, 1)

    def masks(self, diff_input: torch.Tensor, rawf_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the two attention masks analogous to Keras layers 'conv2d_5' (g1 sigmoid output)
        and 'conv2d_10' (g2 sigmoid output), without MaskNorm applied, matching the original
        intermediate layer extraction behavior.

        Outputs shapes:
          - mask1: (N, 1, 34, 34)
          - mask2: (N, 1, 15, 15)
        """
        # First conv block
        r = torch.tanh(self.r1(rawf_input))
        r = torch.tanh(self.r2(r))  # (N, nb_filters1, 34, 34)
        mask1 = torch.sigmoid(self.g1_conv(r))  # before MaskNorm to match Keras intermediate

        # Continue raw branch to produce second-stage mask
        r_pooled = self.drop1(self.pool(r))  # (N, nb_filters1, 17, 17)
        r2_ = torch.tanh(self.r5(r_pooled))
        r2_ = torch.tanh(self.r6(r2_))  # (N, nb_filters2, 15, 15)
        mask2 = torch.sigmoid(self.g2_conv(r2_))

        return mask1, mask2


def _ensure_nchw(x: np.ndarray) -> np.ndarray:
    """Ensure array is NCHW (N, C, H, W). Accepts NHWC or NCHW."""
    if x.ndim != 4:
        raise ValueError("Input must be 4D: (N, H, W, C) or (N, C, H, W)")
    # If channels last
    if x.shape[-1] in (1, 3) and x.shape[1] != 1 and x.shape[1] != 3:
        x = np.transpose(x, (0, 3, 1, 2))
    return x


def encode_and_save(
    model: EncoderCAN,
    diff_input: np.ndarray,
    rawf_input: np.ndarray,
    out_dir: str,
    batch_size: int = 128,
    prefix: str = "encoder_out_",
    device: Optional[torch.device] = None,
) -> Tuple[int, str]:
    """
    Runs the encoder model on inputs and saves outputs as multiple .npy files
    named with the given prefix (e.g., encoder_out_00000.npy, ...).

    Returns (num_files_written, out_dir).
    """
    os.makedirs(out_dir, exist_ok=True)

    diff_np = _ensure_nchw(diff_input).astype(np.float32)
    rawf_np = _ensure_nchw(rawf_input).astype(np.float32)

    if diff_np.shape != rawf_np.shape:
        raise ValueError(f"diff_input and rawf_input must have same shape, got {diff_np.shape} vs {rawf_np.shape}")

    dev = device if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(dev)
    model.eval()

    n = diff_np.shape[0]
    num_files = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            d_batch = torch.from_numpy(diff_np[start:end]).to(dev)
            r_batch = torch.from_numpy(rawf_np[start:end]).to(dev)
            out = model(d_batch, r_batch).squeeze(-1).cpu().numpy()  # shape: (B,)

            fname = f"{prefix}{start:05d}.npy"
            fpath = os.path.join(out_dir, fname)
            np.save(fpath, out)
            num_files += 1

    return num_files, out_dir


def save_attention_masks(
    model: EncoderCAN,
    diff_input: np.ndarray,
    rawf_input: np.ndarray,
    save_dir: str,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[str, str]:
    """
    Computes attention masks corresponding to Keras 'conv2d_5' and 'conv2d_10' layers
    and saves them as MATLAB .mat files: mask1.mat and mask2.mat in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    diff_np = _ensure_nchw(diff_input).astype(np.float32)
    rawf_np = _ensure_nchw(rawf_input).astype(np.float32)

    if diff_np.shape != rawf_np.shape:
        raise ValueError(f"diff_input and rawf_input must have same shape, got {diff_np.shape} vs {rawf_np.shape}")

    dev = device if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(dev)
    model.eval()

    masks1_list = []
    masks2_list = []
    n = diff_np.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            d_batch = torch.from_numpy(diff_np[start:end]).to(dev)
            r_batch = torch.from_numpy(rawf_np[start:end]).to(dev)
            m1, m2 = model.masks(d_batch, r_batch)
            masks1_list.append(m1.cpu().numpy())
            masks2_list.append(m2.cpu().numpy())

    mask1 = np.concatenate(masks1_list, axis=0)
    mask2 = np.concatenate(masks2_list, axis=0)

    path1 = os.path.join(save_dir, 'mask1.mat')
    path2 = os.path.join(save_dir, 'mask2.mat')
    scipy.io.savemat(path1, mdict={'mask1': mask1})
    scipy.io.savemat(path2, mdict={'mask2': mask2})

    return path1, path2


__all__ = ["EncoderCAN", "encode_and_save", "save_attention_masks"]

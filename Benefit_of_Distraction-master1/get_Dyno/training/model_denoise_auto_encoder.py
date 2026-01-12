from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import EncoderCAN


class DenoiseAutoEncoder(nn.Module):
    """
    Weight-tying denoising auto-encoder that reuses the raw branch of EncoderCAN
    as the encoder, and mirrors it with a decoder whose transposed-convolution
    weights are tied to the encoder's raw-branch conv weights (r1, r2, r5, r6).

    Input:  raw triplet frames (N, 3, 36, 36)
    Output: reconstructed raw triplet frames (N, 3, 36, 36)

    Note: Due to 'valid' convolutions in the encoder, exact spatial inversion
    requires careful sizing. This implementation uses ConvTranspose2d with
    kernel/padding/output_padding configured to reconstruct 36x36.
    """

    def __init__(self, base_encoder: Optional[EncoderCAN] = None, dropout_rate1: float = 0.25):
        super().__init__()
        self.enc = base_encoder if base_encoder is not None else EncoderCAN()

        nf1 = self.enc.nb_filters1
        nf2 = self.enc.nb_filters2

        # Decoder mirrors raw branch: r6 (valid), r5 (same), pool, r2 (valid), r1 (same)
        # We'll upsample (nearest) for pool inverses, sandwiching between deconvs.
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Deconvs configured to mirror encoder convs
        self.de_r6 = nn.ConvTranspose2d(nf2, nf2, kernel_size=3, padding=0)  # invert r6 (valid)
        self.de_r5 = nn.ConvTranspose2d(nf2, nf1, kernel_size=3, padding=1)  # invert r5 (same)
        self.de_r2 = nn.ConvTranspose2d(nf1, nf1, kernel_size=3, padding=0)  # invert r2 (valid)
        self.de_r1 = nn.ConvTranspose2d(nf1, 3, kernel_size=3, padding=1)    # invert r1 (same) -> 3 ch

        # Tie weights: ensure decoder weights reference encoder weights where compatible
        self._tie_weights()

        self.drop1 = nn.Dropout(dropout_rate1)

    def _tie_weights(self):
        # Tie shapes match between Conv2d and ConvTranspose2d as per PyTorch weight layout
        # Conv2d: (out_ch, in_ch, kH, kW). ConvTranspose2d: (in_ch, out_ch, kH, kW)
        # For same in/out pairs they have identical shape; assign directly for tying.
        with torch.no_grad():
            # r6: (nf2 -> nf2)
            self.de_r6.weight = self.enc.r6.weight  # type: ignore
            self.de_r6.bias = None
            # r5: (nf2 -> nf1)
            self.de_r5.weight = self.enc.r5.weight  # type: ignore
            self.de_r5.bias = None
            # r2: (nf1 -> nf1)
            self.de_r2.weight = self.enc.r2.weight  # type: ignore
            self.de_r2.bias = None
            # r1: (nf1 -> 3)
            # Shapes match ConvTranspose2d(in_ch=nf1, out_ch=3) -> (nf1,3,3,3)
            self.de_r1.weight = self.enc.r1.weight  # type: ignore
            self.de_r1.bias = None

            # Freeze decoder weights to maintain tying during training
            for p in [self.de_r6.weight, self.de_r5.weight, self.de_r2.weight, self.de_r1.weight]:
                p.requires_grad = False

    def encode_features(self, rawf_input: torch.Tensor) -> torch.Tensor:
        # Raw branch through r1, r2, pool, r5, r6, pool (matching EncoderCAN)
        r = torch.tanh(self.enc.r1(rawf_input))
        r = torch.tanh(self.enc.r2(r))        # (N, nf1, 34, 34)
        r = self.drop1(self.enc.pool(r))      # (N, nf1, 17, 17)
        r = torch.tanh(self.enc.r5(r))
        r = torch.tanh(self.enc.r6(r))        # (N, nf2, 15, 15)
        r = self.drop1(self.enc.pool(r))      # (N, nf2, 7, 7)
        return r

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Inverse of encode: upsample, deconv r6, deconv r5, upsample, deconv r2, deconv r1
        x = self.up(z)                         # 7 -> 14
        x = torch.tanh(self.de_r6(x))          # 14 -> 16
        x = torch.tanh(self.de_r5(x))          # 16 -> 16
        x = self.up(x)                         # 16 -> 32
        x = torch.tanh(self.de_r2(x))          # 32 -> 34
        x = torch.sigmoid(self.de_r1(x))       # 34 -> 34, bound to [0,1]
        # Pad to 36x36 to match original input spatial size
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")  # (left,right,top,bottom)
        return x                                # (N,3,36,36)

    def forward(self, rawf_input: torch.Tensor, noise_std: float = 0.05, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add Gaussian noise for denoising objective
        if train and noise_std > 0:
            noisy = rawf_input + noise_std * torch.randn_like(rawf_input)
            noisy = torch.clamp(noisy, 0.0, 1.0)
        else:
            noisy = rawf_input
        z = self.encode_features(noisy)
        recon = self.decode(z)
        return recon, rawf_input


__all__ = ["DenoiseAutoEncoder"]

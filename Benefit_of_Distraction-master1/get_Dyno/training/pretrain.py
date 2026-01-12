import argparse
import glob
import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

from model_denoise_auto_encoder import DenoiseAutoEncoder
from encoder import EncoderCAN


def load_sequences(data_dir: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .npy files found under {data_dir}")
    arrays: List[np.ndarray] = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 4 or arr.shape[1:3] != (36, 36) or arr.shape[3] != 1:
            raise ValueError(f"File {f} has invalid shape {arr.shape}, expected (L,36,36,1)")
        arrays.append(arr.astype(np.float32))
    stacked = np.stack(arrays, axis=0)  # (N, L, 36, 36, 1)
    return stacked


def sequences_to_raw_triplets(seqs: np.ndarray) -> np.ndarray:
    """
    From (N,L,36,36,1) build raw triplets (M,3,36,36) in NCHW, values in [0,1].
    Triplet frames are [t-2, t-1, t].
    """
    if seqs.ndim != 5:
        raise ValueError("Expected (N,L,36,36,1)")
    N, L, H, W, C = seqs.shape
    if L < 3:
        raise ValueError("Need L >= 3 to form triplets")

    out: List[np.ndarray] = []
    for n in range(N):
        s = seqs[n]
        for t in range(2, L):
            f0 = s[t - 2]
            f1 = s[t - 1]
            f2 = s[t]
            trip = np.concatenate([f0, f1, f2], axis=-1)  # (36,36,3)
            # to NCHW
            trip = np.transpose(trip, (2, 0, 1))  # (3,36,36)
            out.append(trip)
    raw = np.stack(out, axis=0)
    # Normalize to [0,1] if values look like 0..255
    if raw.max() > 1.0:
        raw = raw / 255.0
    return raw.astype(np.float32)


def save_weights_h5(model: nn.Module, path: str):
    """
    Save a PyTorch state_dict into an HDF5 .h5 file for interoperability.
    """
    sd = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        for k, v in sd.items():
            f.create_dataset(k.replace('.', '/'), data=v.cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description="Pretrain EncoderCAN with a weight-tying denoising autoencoder (reconstruction loss)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../Data_Dyno"))
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "pretrain_outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    print(f"Loading sequences from {args.data_dir} ...")
    seqs = load_sequences(args.data_dir)
    print(f"Loaded {seqs.shape}")

    print("Building raw triplets ...")
    raw = sequences_to_raw_triplets(seqs)  # (M,3,36,36)

    # Dataset
    tensor = torch.from_numpy(raw)
    ds = TensorDataset(tensor)
    if args.val_split > 0:
        n_val = int(len(ds) * args.val_split)
        n_train = len(ds) - n_val
        ds_train, ds_val = random_split(ds, [n_train, n_val])
    else:
        ds_train, ds_val = ds, None

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False) if ds_val is not None else None

    # Model
    base_enc = EncoderCAN()
    dae = DenoiseAutoEncoder(base_encoder=base_enc).to(device)

    # Only train encoder raw-branch (and possibly batchnorm/dropout stats, though none here)
    # Decoder weights are tied and frozen; ensure encoder weights require grad
    for name, p in dae.named_parameters():
        p.requires_grad = True
    for p in [dae.de_r6.weight, dae.de_r5.weight, dae.de_r2.weight, dae.de_r1.weight]:
        p.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, dae.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = None
    for epoch in range(1, args.epochs + 1):
        dae.train()
        total_loss = 0.0
        for (xb,) in dl_train:
            xb = xb.to(device)
            recon, target = dae(xb, noise_std=args.noise_std, train=True)
            # Crop recon to 36x36 already handled in model; ensure shapes match
            loss = criterion(recon, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(ds_train)

        if dl_val is not None:
            dae.eval()
            vloss = 0.0
            with torch.no_grad():
                for (xb,) in dl_val:
                    xb = xb.to(device)
                    recon, target = dae(xb, noise_std=0.0, train=False)
                    vloss += criterion(recon, target).item() * xb.size(0)
            val_loss = vloss / len(ds_val)
            print(f"Epoch {epoch}: train {train_loss:.6f}  val {val_loss:.6f}")
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                save_path = os.path.join(args.save_dir, "dae_best.h5")
                print(f"  Saving best weights to {save_path}")
                save_weights_h5(dae, save_path)
        else:
            print(f"Epoch {epoch}: train {train_loss:.6f}")

    # Save final weights
    final_path = os.path.join(args.save_dir, "dae_final.h5")
    print(f"Saving final weights to {final_path}")
    save_weights_h5(dae, final_path)


if __name__ == "__main__":
    main()

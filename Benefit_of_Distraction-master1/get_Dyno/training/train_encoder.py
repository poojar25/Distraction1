import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch

from encoder import EncoderCAN, encode_and_save, save_attention_masks


def load_sequences(data_dir: str) -> np.ndarray:
    """
    Load all .npy arrays from data_dir and stack to (N, L, 36, 36, 1).
    Each file must be shape (L, 36, 36, 1).
    """
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


def sequences_to_triplets(seq_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sequences (N,L,36,36,1) into model inputs for triplet frames.

    For each sequence and time t in [2..L-1], we form:
      - rawf_input: [frame(t-2), frame(t-1), frame(t)] along channel -> (36,36,3)
      - diff_input: [f(t-1)-f(t-2), f(t)-f(t-1), f(t)-f(t-2)] along channel -> (36,36,3)

    Returns two arrays of shape (M, 36, 36, 3), where M = N*(L-2).
    """
    if seq_batch.ndim != 5:
        raise ValueError("Expected (N,L,36,36,1)")
    N, L, H, W, C = seq_batch.shape
    assert H == 36 and W == 36 and C == 1

    M = N * max(L - 2, 0)
    if M <= 0:
        raise ValueError("Sequences must have L >= 3 to build triplets")

    raw_list: List[np.ndarray] = []
    diff_list: List[np.ndarray] = []

    for n in range(N):
        seq = seq_batch[n]  # (L,36,36,1)
        for t in range(2, L):
            f0 = seq[t - 2]
            f1 = seq[t - 1]
            f2 = seq[t]
            raw = np.concatenate([f0, f1, f2], axis=-1)  # (36,36,3)
            d01 = f1 - f0
            d12 = f2 - f1
            d02 = f2 - f0
            diff = np.concatenate([d01, d12, d02], axis=-1)
            raw_list.append(raw)
            diff_list.append(diff)

    rawf_input = np.stack(raw_list, axis=0)  # (M,36,36,3)
    diff_input = np.stack(diff_list, axis=0)  # (M,36,36,3)

    return diff_input, rawf_input


def main():
    parser = argparse.ArgumentParser(description="Train/Run EncoderCAN on Data_Dyno sequences and save outputs/masks.")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../Data_Dyno"),
                        help="Directory containing .npy batched sequences (each (L,36,36,1))")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "encoder_outputs"),
                        help="Directory to save encoder outputs and masks")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_masks", action="store_true", help="Also compute and save attention masks")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading sequences from {args.data_dir} ...")
    seqs = load_sequences(args.data_dir)
    print(f"Loaded sequences: {seqs.shape} (N,L,36,36,1)")

    print("Building triplet inputs ...")
    diff_input, rawf_input = sequences_to_triplets(seqs)
    print(f"Triplet inputs: diff {diff_input.shape}, raw {rawf_input.shape}")

    print("Initializing model ...")
    model = EncoderCAN()

    print("Running encoder and saving outputs (.npy with prefix 'encoder_out_') ...")
    nfiles, outdir = encode_and_save(
        model,
        diff_input=diff_input,
        rawf_input=rawf_input,
        out_dir=args.save_dir,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Saved {nfiles} npy files to {outdir}")

    if args.save_masks:
        print("Computing and saving attention masks (mask1.mat, mask2.mat) ...")
        p1, p2 = save_attention_masks(
            model,
            diff_input=diff_input,
            rawf_input=rawf_input,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"Saved masks to {p1} and {p2}")


if __name__ == "__main__":
    main()

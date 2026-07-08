"""M7: prepare ESA Anomaly Dataset (Mission1) into windowed, MinMax-normalized tensors.

Loads the 16 co-sampled channels selected in data/loaders/esa.py, resamples them onto a
common regular hourly grid, and applies the same time-based train/test split + window-level
validation holdout used by prepare_data_pm25.py (ESA has no natural "unit" grouping either,
and no official train/test split file -- see data/loaders/esa.py docstring for the protocol
this project defines as the "standard protocol" referenced in reproduction_plan.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.base import MinMaxNormalizer, make_windows_from_series
from data.loaders.esa import CHANNELS, build_hourly_frame, fill_missing

DEFAULT_ZIP = "datasets/ESA Anomaly Dataset/ESA-Mission1.zip"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--out", default="data/processed/esa")
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"loading and resampling {len(CHANNELS)} ESA channels to hourly grid...")
    df = build_hourly_frame(args.zip, CHANNELS)
    df = fill_missing(df)
    feature_cols = list(df.columns)
    values = df.to_numpy(dtype=np.float32)
    print(f"hourly frame: {values.shape} spanning {df.index.min()} .. {df.index.max()}")

    split_idx = int(round(len(values) * (1 - args.test_ratio)))
    train_series, test_series = values[:split_idx], values[split_idx:]

    train_windows_all = make_windows_from_series(train_series, T=args.T, stride=args.stride)
    test_windows = make_windows_from_series(test_series, T=args.T, stride=args.stride)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(train_windows_all))
    n_val = int(round(len(perm) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_windows = train_windows_all[train_idx]
    val_windows = train_windows_all[val_idx] if n_val > 0 else None

    normalizer = MinMaxNormalizer().fit(train_windows)
    train_norm = normalizer.transform(train_windows)
    test_norm = normalizer.transform(test_windows)
    val_norm = normalizer.transform(val_windows) if val_windows is not None else None

    assert not np.isnan(train_norm).any(), "NaNs found in normalized train windows"
    assert not np.isnan(test_norm).any(), "NaNs found in normalized test windows"
    assert train_norm.min() >= -1e-3 and train_norm.max() <= 1 + 1e-3, "train windows not in [0,1]"
    assert train_norm.shape[1] == args.T

    np.save(os.path.join(args.out, "train.npy"), train_norm.astype(np.float32))
    np.save(os.path.join(args.out, "test.npy"), test_norm.astype(np.float32))
    if val_norm is not None:
        assert not np.isnan(val_norm).any(), "NaNs found in normalized val windows"
        np.save(os.path.join(args.out, "val.npy"), val_norm.astype(np.float32))
    normalizer.save(os.path.join(args.out, "normalizer.npz"))

    meta = {
        "dataset": "esa",
        "mission": "ESA-Mission1",
        "channels": feature_cols,
        "D": len(feature_cols),
        "T": args.T,
        "n_train": int(train_norm.shape[0]),
        "n_val": int(val_norm.shape[0]) if val_norm is not None else 0,
        "n_test": int(test_norm.shape[0]),
        "n_train_hours": int(len(train_series)),
        "n_test_hours": int(len(test_series)),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    val_shape = val_norm.shape if val_norm is not None else None
    print(f"train shape: {train_norm.shape}, val shape: {val_shape}, test shape: {test_norm.shape}")


if __name__ == "__main__":
    main()

"""M1: prepare C-MAPSS data into windowed, MinMax-normalized train/test tensors."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.base import MinMaxNormalizer, split_units
from data.loaders.cmapss import load_cmapss_raw, make_windows, select_feature_columns

DEFAULT_ZIP = "datasets/C-MAPSS/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=DEFAULT_ZIP)
    p.add_argument("--subset", default="FD001")
    p.add_argument("--out", default="data/processed/cmapss")
    p.add_argument("--T", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_cmapss_raw(args.zip, subset=args.subset, split="train")
    feature_cols = select_feature_columns(df)

    train_units, test_units = split_units(df["unit"].to_numpy(), test_ratio=args.test_ratio, seed=args.seed)
    df_train = df[df["unit"].isin(train_units)]
    df_test = df[df["unit"].isin(test_units)]

    train_windows = make_windows(df_train, feature_cols, T=args.T, stride=args.stride)
    test_windows = make_windows(df_test, feature_cols, T=args.T, stride=args.stride)

    normalizer = MinMaxNormalizer().fit(train_windows)
    train_norm = normalizer.transform(train_windows)
    test_norm = normalizer.transform(test_windows)

    assert not np.isnan(train_norm).any(), "NaNs found in normalized train windows"
    assert not np.isnan(test_norm).any(), "NaNs found in normalized test windows"
    assert train_norm.min() >= -1e-3 and train_norm.max() <= 1 + 1e-3, "train windows not in [0,1]"
    assert train_norm.shape[1] == args.T

    np.save(os.path.join(args.out, "train.npy"), train_norm.astype(np.float32))
    np.save(os.path.join(args.out, "test.npy"), test_norm.astype(np.float32))
    normalizer.save(os.path.join(args.out, "normalizer.npz"))

    meta = {
        "subset": args.subset,
        "feature_cols": feature_cols,
        "D": len(feature_cols),
        "T": args.T,
        "n_train": int(train_norm.shape[0]),
        "n_test": int(test_norm.shape[0]),
        "n_train_units": int(len(train_units)),
        "n_test_units": int(len(test_units)),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"train shape: {train_norm.shape}, test shape: {test_norm.shape}")


if __name__ == "__main__":
    main()

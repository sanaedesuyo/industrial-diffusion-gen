"""C-MAPSS turbofan degradation dataset loader.

Raw layout: datasets/C-MAPSS/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
contains a nested CMAPSSData.zip with train_FD00{n}.txt / test_FD00{n}.txt, 26
whitespace-separated columns with no header: unit, cycle, op1-3, s1-s21.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load_cmapss_raw(zip_path: str, subset: str = "FD001", split: str = "train") -> pd.DataFrame:
    """Load a C-MAPSS split into a DataFrame with named columns, reading zips in-memory."""
    with zipfile.ZipFile(zip_path) as outer_zip:
        inner_name = next(n for n in outer_zip.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer_zip.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zip:
        member = f"{split}_{subset}.txt"
        raw_bytes = inner_zip.read(member)
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=r"\s+", header=None, names=COLUMNS)
    return df


def select_feature_columns(df_train: pd.DataFrame, std_thresh: float = 1e-5) -> list[str]:
    """Pick non-constant feature columns using train-set statistics only."""
    candidate_cols = [c for c in df_train.columns if c not in ("unit", "cycle")]
    stds = df_train[candidate_cols].std()
    return [c for c in candidate_cols if stds[c] > std_thresh]


def make_windows(df: pd.DataFrame, feature_cols: list[str], T: int = 24, stride: int = 1) -> np.ndarray:
    """Slide fixed-length windows of length T per unit; skip units shorter than T."""
    windows = []
    for unit_id, unit_df in df.groupby("unit"):
        unit_df = unit_df.sort_values("cycle")
        values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        n = len(values)
        if n < T:
            print(f"[warn] unit {unit_id} has only {n} cycles (< T={T}), skipping")
            continue
        for start in range(0, n - T + 1, stride):
            windows.append(values[start : start + T])
    if not windows:
        raise ValueError("no windows produced - check T/stride against data length")
    return np.stack(windows, axis=0)

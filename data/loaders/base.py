"""Generic MinMax normalization and unit-grouped train/test split utilities."""
from __future__ import annotations

import numpy as np


class MinMaxNormalizer:
    """Per-feature MinMax scaler fit on training windows only."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "MinMaxNormalizer":
        # windows: [N, T, D] -> flatten to [N*T, D] before computing per-feature min/max
        flat = windows.reshape(-1, windows.shape[-1])
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        return self

    def transform(self, windows: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        assert self.min_ is not None, "call fit() first"
        return (windows - self.min_) / (self.max_ - self.min_ + eps)

    def inverse_transform(self, windows: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        assert self.min_ is not None, "call fit() first"
        return windows * (self.max_ - self.min_ + eps) + self.min_

    def save(self, path: str) -> None:
        np.savez(path, min_=self.min_, max_=self.max_)

    @classmethod
    def load(cls, path: str) -> "MinMaxNormalizer":
        data = np.load(path)
        obj = cls()
        obj.min_ = data["min_"]
        obj.max_ = data["max_"]
        return obj


def split_units(unit_ids: np.ndarray, test_ratio: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split unique unit ids into train/test unit id arrays (shuffled, unit-scoped)."""
    unique_units = np.unique(unit_ids)
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_units)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_units = shuffled[:n_test]
    train_units = shuffled[n_test:]
    return train_units, test_units

"""Predictive score (TSTR): train a GRU next-step predictor on synthetic data, test MAE on real data."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 24, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out)


def predictive_score(
    synthetic: np.ndarray,
    real_test: np.ndarray,
    seed: int = 0,
    d_hidden: int = 24,
    n_iters: int = 2000,
    batch_size: int = 128,
    device: str = "cpu",
) -> float:
    """Train on synthetic[:, :-1] -> synthetic[:, 1:] (next-step, all features), test MAE on real_test."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    syn_t = torch.from_numpy(synthetic.astype(np.float32)).to(device)
    real_t = torch.from_numpy(real_test.astype(np.float32)).to(device)

    model = Predictor(d_in=synthetic.shape[-1], d_hidden=d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_train = len(syn_t)
    for it in range(n_iters):
        idx = torch.randint(0, n_train, (min(batch_size, n_train),))
        batch = syn_t[idx]
        x_in, y_true = batch[:, :-1, :], batch[:, 1:, :]
        y_pred = model(x_in)
        loss = torch.mean(torch.abs(y_pred - y_true))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_in, y_true = real_t[:, :-1, :], real_t[:, 1:, :]
        y_pred = model(x_in)
        mae = torch.mean(torch.abs(y_pred - y_true)).item()

    return mae

"""Discriminative score: 2-layer LSTM real/fake classifier, report |acc - 0.5| (TimeGAN protocol)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


class Discriminator(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 24, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


def discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    seed: int = 0,
    d_hidden: int = 24,
    n_iters: int = 2000,
    batch_size: int = 128,
    device: str = "cpu",
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    x = np.concatenate([real, fake], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = Discriminator(d_in=x.shape[-1], d_hidden=d_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    n_train = len(x_train_t)
    for it in range(n_iters):
        idx = torch.randint(0, n_train, (min(batch_size, n_train),))
        logits = model(x_train_t[idx])
        loss = loss_fn(logits, y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(x_test_t)) > 0.5).float()
        acc = (preds == y_test_t).float().mean().item()

    return abs(acc - 0.5)

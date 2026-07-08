"""GRU encoder/decoder autoencoder mapping x_{1:T} <-> h_{1:T} (paper Sec. 2, Eq. 3)."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, num_layers=n_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_in] -> h: [B, T, d_hidden] (all timestep hidden states, h_t = e(h_{t-1}, x_t))
        h, _ = self.gru(x)
        return h


class GRUDecoder(nn.Module):
    def __init__(self, d_hidden: int, d_in: int, n_layers: int = 1, output_sigmoid: bool = True):
        super().__init__()
        self.gru = nn.GRU(d_hidden, d_hidden, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(d_hidden, d_in)
        self.output_sigmoid = output_sigmoid

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_hidden] -> x_hat: [B, T, d_in]
        out, _ = self.gru(h)
        x_hat = self.fc(out)
        # Data is MinMax-normalized to [0,1]; a sigmoid ties the decoder output to that
        # exact support so out-of-range spill can't be an easy real/fake tell for the
        # discriminator (and so sampler-produced latents decode into valid range).
        if self.output_sigmoid:
            x_hat = torch.sigmoid(x_hat)
        return x_hat


class Autoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 1, output_sigmoid: bool = True):
        super().__init__()
        self.d_hidden = d_hidden
        self.encoder = GRUEncoder(d_in, d_hidden, n_layers)
        self.decoder = GRUDecoder(d_hidden, d_in, n_layers, output_sigmoid=output_sigmoid)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return h, x_hat

    @staticmethod
    def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return torch.mean((x - x_hat) ** 2)

"""Per-dimension standardization of AE latents before diffusion.

Diagnosis: the GRU encoder learns very tight latents (measured std ~0.01-0.13 per
dim on C-MAPSS, overall ~0.28) because reconstruction is trivial for this data. The
VP SDE defaults (beta_min=0.1, beta_max=20) assume roughly unit-variance input, as
in Song et al. (2021). With input std two orders of magnitude below 1, most of the
diffusion time range is spent on a signal already swamped by noise relative to its
own scale, and the concatenated [h_s_t, h_cond] score-network input mixes very
different magnitude regimes (h_cond ~0.1, h_s_t sweeping up to ~1 near s=1) -- this
starves training signal and degrades sample fidelity (observed as a high
discriminative score). Standardizing latents to zero-mean/unit-std before the score
network (and inverse-standardizing before decoding) fixes the scale mismatch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentStandardizer(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d_hidden))
        self.register_buffer("std", torch.ones(d_hidden))

    @torch.no_grad()
    def fit(self, h: torch.Tensor, eps: float = 1e-6) -> "LatentStandardizer":
        # h: [..., d_hidden]
        flat = h.reshape(-1, h.shape[-1])
        self.mean.copy_(flat.mean(dim=0))
        self.std.copy_(flat.std(dim=0).clamp_min(eps))
        return self

    def transform(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean) / self.std

    def inverse_transform(self, h: torch.Tensor) -> torch.Tensor:
        return h * self.std + self.mean

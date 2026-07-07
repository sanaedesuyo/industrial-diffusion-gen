"""VP / subVP SDEs (Song et al., 2021 "Score-Based Generative Modeling through SDEs").

Continuous diffusion time s in [0, 1]. The paper explicitly excludes VE SDEs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class SDE(ABC):
    @abstractmethod
    def sde(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward SDE drift f(x,s) and diffusion g(s)."""

    @abstractmethod
    def marginal_prob(self, x0: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of p(x_s | x_0)."""

    def prior_sampling(self, shape, device=None) -> torch.Tensor:
        return torch.randn(*shape, device=device)


def _log_mean_coeff(s: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return -0.25 * s**2 * (beta_max - beta_min) - 0.5 * s * beta_min


def _beta(s: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    return beta_min + s * (beta_max - beta_min)


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sde(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_s = _beta(s, self.beta_min, self.beta_max).view(-1, *([1] * (x.dim() - 1)))
        drift = -0.5 * beta_s * x
        diffusion = torch.sqrt(_beta(s, self.beta_min, self.beta_max).clamp_min(0))
        return drift, diffusion

    def marginal_prob(self, x0: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = _log_mean_coeff(s, self.beta_min, self.beta_max)
        mean = torch.exp(log_mean_coeff).view(-1, *([1] * (x0.dim() - 1))) * x0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std


class SubVPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sde(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta_s = _beta(s, self.beta_min, self.beta_max).view(-1, *([1] * (x.dim() - 1)))
        drift = -0.5 * beta_s * x
        discount = 1.0 - torch.exp(2.0 * _log_mean_coeff(s, self.beta_min, self.beta_max))
        diffusion = torch.sqrt((_beta(s, self.beta_min, self.beta_max) * discount).clamp_min(0))
        return drift, diffusion

    def marginal_prob(self, x0: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = _log_mean_coeff(s, self.beta_min, self.beta_max)
        mean = torch.exp(log_mean_coeff).view(-1, *([1] * (x0.dim() - 1))) * x0
        std = 1.0 - torch.exp(2.0 * log_mean_coeff)
        return mean, std


def build_sde(sde_type: str, beta_min: float = 0.1, beta_max: float = 20.0) -> SDE:
    if sde_type == "vp":
        return VPSDE(beta_min, beta_max)
    if sde_type == "subvp":
        return SubVPSDE(beta_min, beta_max)
    raise ValueError(f"unknown sde_type: {sde_type} (expected 'vp' or 'subvp')")

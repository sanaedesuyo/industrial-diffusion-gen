"""Exponential moving average of model parameters, applied to the score network only."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def apply_to(self, model: nn.Module) -> nn.Module:
        """Return a copy of model with EMA weights loaded (does not mutate model)."""
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(self.shadow)
        return ema_model

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

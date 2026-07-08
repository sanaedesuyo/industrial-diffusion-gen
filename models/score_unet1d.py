"""Conditional score network M_theta(s, h_t^s, h_{t-1}).

Design (confirmed with user): the paper defines the score network as a function of
single latent vectors (h_t^s, h_{t-1} in R^d_hidden), not the whole [T, d_hidden]
sequence -- this is required so the recursive sampler can call it one timestep at a
time. We implement the "1D U-Net" by treating the concatenated vector
[h_t^s ++ h_{t-1}] (length 2*d_hidden) as a length-(2*d_hidden), single-channel 1D
signal and running a conv1d U-Net along that feature axis, with the diffusion time s
injected via FiLM (scale+shift) at every block. During training the T dimension is
flattened into the batch dimension (see models/tsgm.py), which is mathematically
equivalent since the network has no cross-timestep parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B] (continuous diffusion time in [0,1]) -> [B, dim]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = s.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class FiLMConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, d_t: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.SiLU()
        self.film = nn.Linear(d_t, 2 * c_out)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.act(x)


class ConditionalScoreUNet1D(nn.Module):
    def __init__(
        self,
        d_hidden: int = 64,
        d_t: int = 64,
        channels: list[int] = (32, 64, 64),
        depth: int = 3,
    ):
        super().__init__()
        assert depth == len(channels), "depth must match len(channels)"
        self.d_hidden = d_hidden
        in_len = 2 * d_hidden  # concatenated [h_s_t ++ h_cond]

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_t),
            nn.Linear(d_t, d_t),
            nn.SiLU(),
            nn.Linear(d_t, d_t),
        )

        self.in_conv = nn.Conv1d(1, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        c_prev = channels[0]
        for c in channels:
            self.down_blocks.append(FiLMConvBlock(c_prev, c, d_t, stride=2))
            c_prev = c

        self.mid_block = FiLMConvBlock(c_prev, c_prev, d_t, stride=1)

        self.up_blocks = nn.ModuleList()
        for c in reversed(channels[:-1]):
            self.up_blocks.append(FiLMConvBlock(c_prev + c, c, d_t, stride=1))
            c_prev = c
        # final up-block back to channels[0] resolution
        self.up_blocks.append(FiLMConvBlock(c_prev + channels[0], channels[0], d_t, stride=1))

        # flatten length after 'depth' stride-2 downsamples
        down_len = in_len
        for _ in range(depth):
            down_len = math.ceil(down_len / 2)
        self._down_len = down_len

        self.out_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[0] * in_len, 4 * d_hidden),
            nn.SiLU(),
            nn.Linear(4 * d_hidden, d_hidden),
        )

    def forward(self, h_s_t: torch.Tensor, h_cond: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # h_s_t, h_cond: [B, d_hidden]; s: [B] -> predicted score [B, d_hidden]
        x = torch.cat([h_s_t, h_cond], dim=-1).unsqueeze(1)  # [B, 1, 2*d_hidden]
        t_emb = self.time_embed(s)

        x = self.in_conv(x)
        skips = [x]
        for block in self.down_blocks:
            x = block(x, t_emb)
            skips.append(x)
        skips.pop()  # drop the deepest (bottleneck) skip, keep it as mid input

        x = self.mid_block(x, t_emb)

        for block in self.up_blocks:
            skip = skips.pop()
            x = torch.nn.functional.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        x = torch.nn.functional.interpolate(x, size=2 * self.d_hidden, mode="nearest")
        return self.out_proj(x)


class FiLMResBlock(nn.Module):
    """Pre-norm residual MLP block with per-block FiLM (scale+shift) time conditioning."""

    def __init__(self, width: int, d_t: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.film = nn.Linear(d_t, 2 * width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.lin1(h))
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = self.lin2(h)
        return x + h


class ConditionalScoreMLP(nn.Module):
    """Conditional score network M_theta(s, h_t^s, h_{t-1}) as a residual MLP + FiLM.

    The score net operates on low-dimensional latent *vectors* with no spatial/temporal
    structure along the feature axis, so a plain residual MLP with FiLM time conditioning
    is the principled model (vs. the conv-over-features U-Net, whose locality prior is
    meaningless here). Same interface and "direct score output" convention as
    ConditionalScoreUNet1D, so the training loss and sampler are unchanged.
    """

    def __init__(self, d_hidden: int = 64, d_t: int = 64, width: int = 256, n_blocks: int = 4):
        super().__init__()
        self.d_hidden = d_hidden

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(d_t),
            nn.Linear(d_t, d_t),
            nn.SiLU(),
            nn.Linear(d_t, d_t),
        )
        self.in_proj = nn.Linear(2 * d_hidden, width)
        self.blocks = nn.ModuleList([FiLMResBlock(width, d_t) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, d_hidden)

    def forward(self, h_s_t: torch.Tensor, h_cond: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # h_s_t, h_cond: [B, d_hidden]; s: [B] -> predicted score [B, d_hidden]
        t_emb = self.time_embed(s)
        x = self.in_proj(torch.cat([h_s_t, h_cond], dim=-1))
        for block in self.blocks:
            x = block(x, t_emb)
        return self.out_proj(self.out_norm(x))


def build_score_net(model_cfg: dict, d_hidden: int) -> nn.Module:
    """Build the score network selected by model_cfg['score_net_type'] ('mlp' | 'unet').

    Defaults to 'unet', matching the paper's proposed 1D U-Net architecture (Sec.
    "Conditional Score Network"). 'mlp' (ResMLP+FiLM) is kept as an off-paper variant
    that was found to lower the discriminative score on C-MAPSS; both construction paths
    must stay reachable from train/sample/eval so checkpoints of either type still load.
    """
    net_type = model_cfg.get("score_net_type", "unet")
    d_t = model_cfg["d_t"]
    if net_type == "mlp":
        return ConditionalScoreMLP(
            d_hidden=d_hidden,
            d_t=d_t,
            width=model_cfg.get("mlp_width", 256),
            n_blocks=model_cfg.get("mlp_blocks", 4),
        )
    if net_type == "unet":
        return ConditionalScoreUNet1D(
            d_hidden=d_hidden,
            d_t=d_t,
            channels=model_cfg["unet_channels"],
            depth=model_cfg["unet_depth"],
        )
    raise ValueError(f"unknown score_net_type: {net_type} (expected 'mlp' or 'unet')")

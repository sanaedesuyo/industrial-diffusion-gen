"""M4: recursive PC sampling from a trained checkpoint, using EMA weights for the score net."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.base import MinMaxNormalizer
from models import tsgm
from models.autoencoder import Autoencoder
from models.ema import EMA
from models.latent_norm import LatentStandardizer
from models.score_unet1d import ConditionalScoreUNet1D
from models.sde import build_sde
from scripts.config_utils import get_default_device, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cmapss.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--sde-type", default=None, help="override sde type (vp/subvp)")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--out", default="outputs/samples/cmapss.npy")
    p.add_argument("--device", default=get_default_device())
    return p.parse_args()


def load_model(checkpoint_path: str, cfg: dict, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    D, T, d_hidden = ckpt["D"], ckpt["T"], ckpt["d_hidden"]

    ae = Autoencoder(d_in=D, d_hidden=d_hidden, n_layers=cfg["model"]["ae_layers"]).to(device)
    ae.load_state_dict(ckpt["ae"])

    score_net = ConditionalScoreUNet1D(
        d_hidden=d_hidden,
        d_t=cfg["model"]["d_t"],
        channels=cfg["model"]["unet_channels"],
        depth=cfg["model"]["unet_depth"],
    ).to(device)
    score_net.load_state_dict(ckpt["score_net"])

    ema = EMA(score_net, decay=cfg["train"]["ema_decay"])
    ema.load_state_dict(ckpt["ema"])
    ema_score_net = ema.apply_to(score_net).to(device)

    latent_standardizer = LatentStandardizer(d_hidden).to(device)
    if "latent_standardizer" in ckpt:
        latent_standardizer.load_state_dict(ckpt["latent_standardizer"])
    else:
        print("[warn] checkpoint predates latent standardization; using identity (mean=0, std=1)")

    return ae, ema_score_net, D, T, d_hidden, latent_standardizer


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device

    ae, score_net, D, T, d_hidden, latent_standardizer = load_model(args.checkpoint, cfg, device)
    sde_type = args.sde_type or cfg["sde"]["type"]
    sde = build_sde(sde_type, cfg["sde"]["beta_min"], cfg["sde"]["beta_max"])
    n_steps = args.n_steps or cfg["sde"]["n_steps_sample"]

    x_hat = tsgm.sample(
        ae,
        score_net,
        sde,
        n_samples=args.n_samples,
        T=T,
        d_hidden=d_hidden,
        n_steps=n_steps,
        device=device,
        latent_standardizer=latent_standardizer,
    )
    x_hat_np = x_hat.cpu().numpy()

    normalizer_path = os.path.join(cfg["data"]["processed_dir"], "normalizer.npz")
    if os.path.exists(normalizer_path):
        normalizer = MinMaxNormalizer.load(normalizer_path)
        x_hat_np = normalizer.inverse_transform(x_hat_np)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, x_hat_np.astype(np.float32))
    print(f"saved samples: shape={x_hat_np.shape} nan={np.isnan(x_hat_np).any()} -> {args.out}")


if __name__ == "__main__":
    main()

"""Algorithm 1: pretrain AE, then train score net (optionally alternating AE fine-tune)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import tsgm
from models.autoencoder import Autoencoder
from models.ema import EMA
from models.score_unet1d import ConditionalScoreUNet1D
from models.sde import build_sde
from scripts.config_utils import get_default_device, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cmapss.yaml")
    p.add_argument("--out", default="outputs/checkpoints/cmapss")
    p.add_argument("--smoke-test", action="store_true", help="use configs['smoke_test'] iter counts")
    p.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    p.add_argument("--device", default=get_default_device())
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = args.device
    os.makedirs(args.out, exist_ok=True)

    train_np = np.load(os.path.join(cfg["data"]["processed_dir"], "train.npy"))
    D = train_np.shape[-1]
    T = train_np.shape[1]

    loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=cfg["train"]["batch_size"], shuffle=True)

    d_hidden = cfg["model"]["d_hidden"]
    ae = Autoencoder(d_in=D, d_hidden=d_hidden, n_layers=cfg["model"]["ae_layers"]).to(device)
    score_net = ConditionalScoreUNet1D(
        d_hidden=d_hidden,
        d_t=cfg["model"]["d_t"],
        channels=cfg["model"]["unet_channels"],
        depth=cfg["model"]["unet_depth"],
    ).to(device)
    sde = build_sde(cfg["sde"]["type"], cfg["sde"]["beta_min"], cfg["sde"]["beta_max"])
    ema = EMA(score_net, decay=cfg["train"]["ema_decay"])

    ae_opt = torch.optim.Adam(ae.parameters(), lr=cfg["train"]["lr_ae"])
    score_opt = torch.optim.Adam(score_net.parameters(), lr=cfg["train"]["lr_score"])

    if args.smoke_test:
        iter_pre = cfg["smoke_test"]["iter_pre"]
        iter_main = cfg["smoke_test"]["iter_main"]
        save_every = cfg["smoke_test"]["save_every"]
    else:
        iter_pre = cfg["train"]["iter_pre"]
        iter_main = cfg["train"]["iter_main"]
        save_every = cfg["train"]["save_every"]

    print(f"== pretraining AE for {iter_pre} iters ==")
    tsgm.train_autoencoder(ae, loader, ae_opt, n_iters=iter_pre, device=device, log_every=max(1, iter_pre // 10))
    torch.save(ae.state_dict(), os.path.join(args.out, "ae_pretrain.pt"))

    print(f"== training score net for {iter_main} iters (use_alt={cfg['train']['use_alt']}) ==")

    def checkpoint(step: int):
        torch.save(
            {
                "ae": ae.state_dict(),
                "score_net": score_net.state_dict(),
                "ema": ema.state_dict(),
                "step": step,
                "d_hidden": d_hidden,
                "T": T,
                "D": D,
            },
            os.path.join(args.out, f"ckpt_{step}.pt"),
        )
        torch.save(
            {
                "ae": ae.state_dict(),
                "score_net": score_net.state_dict(),
                "ema": ema.state_dict(),
                "step": step,
                "d_hidden": d_hidden,
                "T": T,
                "D": D,
            },
            os.path.join(args.out, "ckpt_latest.pt"),
        )

    n_done = 0
    while n_done < iter_main:
        chunk = min(save_every, iter_main - n_done)
        tsgm.train_score(
            ae,
            score_net,
            sde,
            loader,
            score_opt,
            n_iters=chunk,
            use_alt=cfg["train"]["use_alt"],
            ae_optimizer=ae_opt,
            ema=ema,
            device=device,
            log_every=max(1, chunk // 5),
        )
        n_done += chunk
        checkpoint(n_done)
        print(f"[checkpoint] saved at step {n_done}/{iter_main}")

    print("training complete")


if __name__ == "__main__":
    main()

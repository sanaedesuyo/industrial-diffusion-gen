"""Algorithm 1: pretrain AE, then train score net (optionally alternating AE fine-tune)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.discriminative import discriminative_score
from models import tsgm
from models.autoencoder import Autoencoder
from models.ema import EMA
from models.latent_norm import LatentStandardizer
from models.score_unet1d import build_score_net
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

    # Same-distribution holdout for best-checkpoint selection (see prepare_data.py / evaluate.py).
    val_path = os.path.join(cfg["data"]["processed_dir"], "val.npy")
    val_np = np.load(val_path) if os.path.exists(val_path) else None

    d_hidden = cfg["model"]["d_hidden"]
    ae = Autoencoder(
        d_in=D,
        d_hidden=d_hidden,
        n_layers=cfg["model"]["ae_layers"],
        output_sigmoid=cfg["model"].get("decoder_sigmoid", True),
    ).to(device)
    score_net = build_score_net(cfg["model"], d_hidden).to(device)
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

    ae_noise_std = cfg["train"].get("ae_noise_std", 0.0)
    print(f"== pretraining AE for {iter_pre} iters (ae_noise_std={ae_noise_std}) ==")
    tsgm.train_autoencoder(
        ae, loader, ae_opt, n_iters=iter_pre, device=device, log_every=max(1, iter_pre // 10), ae_noise_std=ae_noise_std
    )
    torch.save(ae.state_dict(), os.path.join(args.out, "ae_pretrain.pt"))

    # Fit latent standardizer on the pretrained AE's encoder outputs so the score
    # network trains on roughly unit-variance latents, matching the scale the VP/subVP
    # SDE defaults were designed for (see models/latent_norm.py for the diagnosis).
    ae.eval()
    with torch.no_grad():
        h_all = ae.encoder(torch.from_numpy(train_np).to(device))
    latent_standardizer = LatentStandardizer(d_hidden).to(device).fit(h_all)
    print(
        f"[latent standardizer] fit on train latents: "
        f"mean_abs={latent_standardizer.mean.abs().mean().item():.4f} "
        f"std_range=[{latent_standardizer.std.min().item():.4f}, {latent_standardizer.std.max().item():.4f}]"
    )

    print(f"== training score net for {iter_main} iters (use_alt={cfg['train']['use_alt']}) ==")

    def make_state(step: int) -> dict:
        return {
            "ae": ae.state_dict(),
            "score_net": score_net.state_dict(),
            "ema": ema.state_dict(),
            "latent_standardizer": latent_standardizer.state_dict(),
            "step": step,
            "d_hidden": d_hidden,
            "T": T,
            "D": D,
        }

    def checkpoint(step: int):
        state = make_state(step)
        torch.save(state, os.path.join(args.out, f"ckpt_{step}.pt"))
        torch.save(state, os.path.join(args.out, "ckpt_latest.pt"))

    # Best-checkpoint selection: every save_every steps, sample a small batch with EMA
    # weights (reduced n_steps for speed) and score discriminative against the val holdout;
    # keep ckpt_best.pt. evaluate.py can then use ckpt_best.pt instead of ckpt_latest.pt.
    select_best = cfg["train"].get("select_best", True) and val_np is not None
    select_n_steps = cfg["train"].get("select_n_steps", 100)
    select_n_samples = cfg["train"].get("select_n_samples", min(len(val_np), 512) if val_np is not None else 0)
    select_n_seeds = cfg["train"].get("select_n_seeds", 1)
    select_n_fake_batches = cfg["train"].get("select_n_fake_batches", 1)
    best_disc = float("inf")

    def maybe_select_best(step: int):
        nonlocal best_disc
        if not select_best:
            return
        ema_net = ema.apply_to(score_net).to(device)
        # Resample the fake batch itself select_n_fake_batches times (each tsgm.sample call
        # draws fresh noise from the global RNG) and average select_n_seeds classifier seeds
        # per batch. A single static fake batch scored with only multiple classifier seeds
        # does NOT capture generation-draw variance -- observed: checkpoints picked that way
        # read ~0.002-0.03 internally but ~0.31-0.36 under the full 10-seed/3736-sample
        # protocol. Resampling the generation itself closes most of that gap.
        discs = []
        for _ in range(select_n_fake_batches):
            fake = tsgm.sample(
                ae,
                ema_net,
                sde,
                n_samples=select_n_samples,
                T=T,
                d_hidden=d_hidden,
                n_steps=select_n_steps,
                device=device,
                verbose=False,
                latent_standardizer=latent_standardizer,
            )
            fake_np = fake.cpu().numpy()
            discs.extend(discriminative_score(val_np, fake_np, seed=s, device=device) for s in range(select_n_seeds))
        disc = float(np.mean(discs))
        marker = ""
        if disc < best_disc:
            best_disc = disc
            torch.save(make_state(step), os.path.join(args.out, "ckpt_best.pt"))
            marker = " (new best -> ckpt_best.pt)"
        print(
            f"[select] step {step}/{iter_main} val_discriminative={disc:.4f} "
            f"({select_n_fake_batches} fake batches x {select_n_seeds} seeds={discs}) best={best_disc:.4f}{marker}"
        )
        ae.train() if cfg["train"]["use_alt"] else ae.eval()
        score_net.train()

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
            latent_standardizer=latent_standardizer,
            device=device,
            log_every=max(1, chunk // 5),
        )
        n_done += chunk
        checkpoint(n_done)
        maybe_select_best(n_done)
        print(f"[checkpoint] saved at step {n_done}/{iter_main}")

    print("training complete")


if __name__ == "__main__":
    main()

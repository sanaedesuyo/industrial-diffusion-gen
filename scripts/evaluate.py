"""M5: discriminative / predictive / t-SNE evaluation, TimeGAN protocol, in normalized [0,1] space."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.discriminative import discriminative_score
from metrics.predictive import predictive_score
from metrics.visualization import tsne_plot
from models import tsgm
from models.sde import build_sde
from scripts.config_utils import get_default_device, load_config
from scripts.sample import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cmapss.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=None, help="default: len(test set)")
    p.add_argument("--n-seeds", type=int, default=None, help="default: cfg['eval']['n_seeds']")
    p.add_argument("--n-steps-sample", type=int, default=None)
    p.add_argument("--out-dir", default="outputs/reports")
    p.add_argument("--device", default=get_default_device())
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    test_np = np.load(os.path.join(cfg["data"]["processed_dir"], "test.npy"))

    # Discriminative real reference: same-distribution holdout (val.npy) carved from the
    # train units, following the TimeGAN/TSGM protocol. The generator only ever saw the
    # train distribution, so comparing fake against a slice of it isolates generation
    # quality. Fall back to test.npy for pre-val datasets (this reintroduces the
    # train/test engine-unit shift, so a warning is printed).
    val_path = os.path.join(cfg["data"]["processed_dir"], "val.npy")
    if os.path.exists(val_path):
        disc_real_np = np.load(val_path)
    else:
        print("[warn] val.npy not found; discriminative falls back to test.npy (train/test shift inflates the score). Re-run prepare_data.py.")
        disc_real_np = test_np

    n_samples = args.n_samples or len(test_np)
    n_seeds = args.n_seeds or cfg["eval"]["n_seeds"]
    n_steps = args.n_steps_sample or cfg["sde"]["n_steps_sample"]

    ae, score_net, D, T, d_hidden, latent_standardizer = load_model(args.checkpoint, cfg, device)
    sde = build_sde(cfg["sde"]["type"], cfg["sde"]["beta_min"], cfg["sde"]["beta_max"])

    print(f"generating {n_samples} synthetic windows (n_steps={n_steps}) for evaluation...")
    fake = tsgm.sample(
        ae,
        score_net,
        sde,
        n_samples=n_samples,
        T=T,
        d_hidden=d_hidden,
        n_steps=n_steps,
        n_corrector_steps=cfg["sde"].get("n_corrector_steps", 1),
        snr=cfg["sde"].get("snr", 0.16),
        device=device,
        latent_standardizer=latent_standardizer,
    )
    fake_np = fake.cpu().numpy()

    disc_scores, pred_scores = [], []
    for seed in range(n_seeds):
        d = discriminative_score(disc_real_np, fake_np, seed=seed, device=device)
        p = predictive_score(fake_np, test_np, seed=seed, device=device)
        disc_scores.append(d)
        pred_scores.append(p)
        print(f"seed {seed}: discriminative={d:.4f} predictive={p:.4f}")

    disc_mean, disc_std = float(np.mean(disc_scores)), float(np.std(disc_scores))
    pred_mean, pred_std = float(np.mean(pred_scores)), float(np.std(pred_scores))

    report_path = os.path.join(args.out_dir, "cmapss_metrics.csv")
    report_body = "metric,mean,std\n" f"discriminative,{disc_mean},{disc_std}\n" f"predictive,{pred_mean},{pred_std}\n"
    try:
        with open(report_path, "w") as f:
            f.write(report_body)
    except PermissionError:
        # e.g. the CSV is open in Excel on Windows; don't lose the whole run.
        import time

        report_path = os.path.join(args.out_dir, f"cmapss_metrics_{int(time.time())}.csv")
        with open(report_path, "w") as f:
            f.write(report_body)
        print(f"[warn] default report path was locked; wrote to {report_path} instead")

    tsne_path = os.path.join(args.out_dir, "cmapss_tsne.png")
    tsne_plot(disc_real_np, fake_np, tsne_path)

    print(f"discriminative: {disc_mean:.4f} +/- {disc_std:.4f}")
    print(f"predictive:     {pred_mean:.4f} +/- {pred_std:.4f}")
    print(f"report -> {report_path}")
    print(f"t-SNE  -> {tsne_path}")


if __name__ == "__main__":
    main()

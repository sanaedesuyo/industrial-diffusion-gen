"""Run the trained TSGM model on the ESA Anomaly Dataset (Mission1): load a checkpoint,
generate synthetic telemetry windows via recursive PC sampling, and save comparison
plots + a metrics/summary report to disk.

This is a standalone "inference demo" script (distinct from scripts/evaluate.py's
10-seed protocol run) meant to be run ad hoc against any ESA checkpoint, with progress
logging at every stage so long-running generation can be monitored.

Usage:
    python scripts/run_esa_anomaly_inference.py \
        --checkpoint outputs/checkpoints/esa_full/ckpt_best.pt \
        --n-samples 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.base import MinMaxNormalizer
from metrics.discriminative import discriminative_score
from metrics.predictive import predictive_score
from metrics.visualization import tsne_plot
from models import tsgm
from models.sde import build_sde
from scripts.config_utils import get_default_device, load_config
from scripts.sample import load_model


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/esa.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/esa_full/ckpt_best.pt")
    p.add_argument("--n-samples", type=int, default=64, help="number of synthetic windows to generate")
    p.add_argument("--n-steps", type=int, default=None, help="PC sampler steps (default: cfg sde.n_steps_sample)")
    p.add_argument("--n-seeds", type=int, default=3, help="seeds for discriminative/predictive scoring")
    p.add_argument("--n-example-windows", type=int, default=4, help="how many real-vs-fake windows to plot per channel")
    p.add_argument("--out-dir", default="outputs/esa_anomaly_inference")
    p.add_argument("--device", default=get_default_device())
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def plot_channel_grid(real_np, fake_np, channel_names, out_path, n_examples=4, seed=0):
    """Grid of small multiples: one subplot per channel, overlaying a few real and fake
    example windows so generation fidelity can be eyeballed per-sensor."""
    rng = np.random.RandomState(seed)
    D = real_np.shape[-1]
    real_idx = rng.choice(len(real_np), min(n_examples, len(real_np)), replace=False)
    fake_idx = rng.choice(len(fake_np), min(n_examples, len(fake_np)), replace=False)

    ncols = 4
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False)
    for d in range(D):
        ax = axes[d // ncols][d % ncols]
        for i, ri in enumerate(real_idx):
            ax.plot(real_np[ri, :, d], color="tab:blue", alpha=0.6, label="real" if i == 0 else None)
        for i, fi in enumerate(fake_idx):
            ax.plot(fake_np[fi, :, d], color="tab:red", alpha=0.6, linestyle="--", label="synthetic" if i == 0 else None)
        ax.set_title(channel_names[d], fontsize=9)
        ax.tick_params(labelsize=7)
    for d in range(D, nrows * ncols):
        axes[d // ncols][d % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("ESA Mission1: real vs. synthetic example windows, per channel")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mean_std_band(real_np, fake_np, out_path):
    """Per-timestep mean +/- std band, averaged over channels, real vs synthetic."""
    real_mean = real_np.mean(axis=(0, 2))
    real_std = real_np.std(axis=(0, 2))
    fake_mean = fake_np.mean(axis=(0, 2))
    fake_std = fake_np.std(axis=(0, 2))
    t = np.arange(real_np.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(t, real_mean, color="tab:blue", label="real mean")
    plt.fill_between(t, real_mean - real_std, real_mean + real_std, color="tab:blue", alpha=0.2)
    plt.plot(t, fake_mean, color="tab:red", label="synthetic mean")
    plt.fill_between(t, fake_mean - fake_std, fake_mean + fake_std, color="tab:red", alpha=0.2)
    plt.xlabel("timestep within window")
    plt.ylabel("value (channel-averaged)")
    plt.title("ESA Mission1: per-timestep mean +/- std, real vs. synthetic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"loading config {args.config}")
    cfg = load_config(args.config)
    device = args.device
    log(f"device={device}")

    meta_path = os.path.join(cfg["data"]["processed_dir"], "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    channel_names = meta.get("channels", [f"ch_{i}" for i in range(meta["D"])])

    log(f"loading checkpoint {args.checkpoint}")
    ae, score_net, D, T, d_hidden, latent_standardizer = load_model(args.checkpoint, cfg, device)
    log(f"model loaded: D={D} T={T} d_hidden={d_hidden}")

    sde = build_sde(cfg["sde"]["type"], cfg["sde"]["beta_min"], cfg["sde"]["beta_max"])
    n_steps = args.n_steps or cfg["sde"]["n_steps_sample"]

    log("loading real ESA test windows")
    test_np = np.load(os.path.join(cfg["data"]["processed_dir"], "test.npy"))
    val_path = os.path.join(cfg["data"]["processed_dir"], "val.npy")
    real_for_compare = np.load(val_path) if os.path.exists(val_path) else test_np
    log(f"test windows: {test_np.shape}, comparison-real windows: {real_for_compare.shape}")

    log(f"generating {args.n_samples} synthetic windows (n_steps={n_steps}) ...")
    gen_t0 = time.time()
    fake = tsgm.sample(
        ae, score_net, sde,
        n_samples=args.n_samples, T=T, d_hidden=d_hidden,
        n_steps=n_steps,
        n_corrector_steps=cfg["sde"].get("n_corrector_steps", 1),
        snr=cfg["sde"].get("snr", 0.16),
        device=device, verbose=True,
        latent_standardizer=latent_standardizer,
    )
    fake_np = fake.cpu().numpy()
    log(f"generation done in {time.time() - gen_t0:.1f}s, shape={fake_np.shape}, nan={np.isnan(fake_np).any()}")

    normalizer_path = os.path.join(cfg["data"]["processed_dir"], "normalizer.npz")
    fake_np_orig, real_np_orig = fake_np, real_for_compare[: len(fake_np)]
    if os.path.exists(normalizer_path):
        normalizer = MinMaxNormalizer.load(normalizer_path)
        fake_np_orig = normalizer.inverse_transform(fake_np)
        real_np_orig = normalizer.inverse_transform(real_for_compare[: len(fake_np)])
        log("inverse-transformed samples back to original (pre-normalization) scale")

    samples_path = os.path.join(args.out_dir, "esa_synthetic_samples.npy")
    np.save(samples_path, fake_np_orig.astype(np.float32))
    log(f"saved synthetic samples -> {samples_path}")

    log("plotting per-channel real-vs-synthetic example windows...")
    grid_path = os.path.join(args.out_dir, "esa_channel_comparison.png")
    plot_channel_grid(real_np_orig, fake_np_orig, channel_names, grid_path, n_examples=args.n_example_windows, seed=args.seed)
    log(f"saved -> {grid_path}")

    log("plotting per-timestep mean/std band...")
    band_path = os.path.join(args.out_dir, "esa_mean_std_band.png")
    plot_mean_std_band(real_np_orig, fake_np_orig, band_path)
    log(f"saved -> {band_path}")

    log("computing t-SNE (real vs. synthetic, normalized space)...")
    tsne_path = os.path.join(args.out_dir, "esa_tsne.png")
    tsne_plot(real_for_compare, fake_np, tsne_path)
    log(f"saved -> {tsne_path}")

    log(f"scoring discriminative/predictive over {args.n_seeds} seeds...")
    disc_scores, pred_scores = [], []
    for seed in range(args.n_seeds):
        d = discriminative_score(real_for_compare, fake_np, seed=seed, device=device)
        p = predictive_score(fake_np, test_np, seed=seed, device=device)
        disc_scores.append(d)
        pred_scores.append(p)
        log(f"  seed {seed}: discriminative={d:.4f} predictive={p:.4f}")

    disc_mean, disc_std = float(np.mean(disc_scores)), float(np.std(disc_scores))
    pred_mean, pred_std = float(np.mean(pred_scores)), float(np.std(pred_scores))
    log(f"discriminative: {disc_mean:.4f} +/- {disc_std:.4f}")
    log(f"predictive:     {pred_mean:.4f} +/- {pred_std:.4f}")

    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "device": device,
        "n_samples": args.n_samples,
        "n_steps": n_steps,
        "n_seeds": args.n_seeds,
        "discriminative_mean": disc_mean,
        "discriminative_std": disc_std,
        "predictive_mean": pred_mean,
        "predictive_std": pred_std,
        "generation_seconds": time.time() - gen_t0,
        "total_seconds": time.time() - t0,
        "channels": channel_names,
        "outputs": {
            "samples_npy": samples_path,
            "channel_comparison_png": grid_path,
            "mean_std_band_png": band_path,
            "tsne_png": tsne_path,
        },
    }
    summary_path = os.path.join(args.out_dir, "esa_inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"saved summary -> {summary_path}")

    log(f"all done in {time.time() - t0:.1f}s. outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()

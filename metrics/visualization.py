"""t-SNE visualization of real vs. synthetic window overlap (diversity check)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def tsne_plot(real: np.ndarray, fake: np.ndarray, out_path: str, n_samples: int = 500, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(real), len(fake))
    real_idx = rng.choice(len(real), n, replace=False)
    fake_idx = rng.choice(len(fake), n, replace=False)

    real_flat = real[real_idx].reshape(n, -1)
    fake_flat = fake[fake_idx].reshape(n, -1)
    combined = np.concatenate([real_flat, fake_flat], axis=0)

    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, n - 1))
    proj = tsne.fit_transform(combined)

    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:n, 0], proj[:n, 1], c="tab:blue", alpha=0.5, label="real")
    plt.scatter(proj[n:, 0], proj[n:, 1], c="tab:red", alpha=0.5, label="synthetic")
    plt.legend()
    plt.title("t-SNE: real vs. synthetic")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

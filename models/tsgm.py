"""TSGM assembly: two-stage training (Algorithm 1) + recursive PC sampling (M4)."""
from __future__ import annotations

import itertools

import torch
import torch.nn as nn

from models.autoencoder import Autoencoder
from models.ema import EMA
from models.sde import SDE


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train_autoencoder(ae: Autoencoder, train_loader, optimizer, n_iters: int, device="cpu", log_every: int = 100):
    ae.train()
    history = []
    data_iter = _cycle(train_loader)
    for it in range(n_iters):
        x = next(data_iter)
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        _, x_hat = ae(x)
        loss = ae.reconstruction_loss(x, x_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (it + 1) % log_every == 0 or it == 0:
            print(f"[ae pretrain] iter {it + 1}/{n_iters} recon_loss={loss.item():.6f}")
    return {"recon_loss": history}


def train_score(
    ae: Autoencoder,
    score_net: nn.Module,
    sde: SDE,
    train_loader,
    optimizer,
    n_iters: int,
    use_alt: bool = False,
    ae_optimizer=None,
    ema: EMA | None = None,
    device: str = "cpu",
    eps: float = 1e-5,
    log_every: int = 100,
):
    score_net.train()
    ae.train() if use_alt else ae.eval()
    history = {"score_loss": [], "recon_loss": []}
    data_iter = _cycle(train_loader)

    for it in range(n_iters):
        x = next(data_iter)
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(device)
        B, T, _ = x.shape

        if use_alt:
            h = ae.encoder(x)
        else:
            with torch.no_grad():
                h = ae.encoder(x)
        d_hidden = h.shape[-1]

        # h_0 = 0 by convention; condition h_{t-1} must be detached (no score-loss gradient into AE)
        h_detached = h.detach()
        h_prev = torch.cat([torch.zeros(B, 1, d_hidden, device=device), h_detached[:, :-1]], dim=1)

        h_flat = h_detached.reshape(B * T, d_hidden)
        h_prev_flat = h_prev.reshape(B * T, d_hidden)

        s = torch.rand(B * T, device=device) * (1 - eps) + eps
        mean, std = sde.marginal_prob(h_flat, s)
        z = torch.randn_like(h_flat)
        h_s = mean + std.unsqueeze(-1) * z

        pred = score_net(h_s, h_prev_flat, s)
        # canonical weighted denoising score matching loss with lambda(s)=std(s)^2,
        # written in the numerically stable "noise prediction" form
        score_loss = torch.mean((std.unsqueeze(-1) * pred + z) ** 2)

        optimizer.zero_grad()
        score_loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(score_net)

        recon_loss_val = float("nan")
        if use_alt:
            x_hat = ae.decoder(h)
            recon_loss = ae.reconstruction_loss(x, x_hat)
            ae_optimizer.zero_grad()
            recon_loss.backward()
            ae_optimizer.step()
            recon_loss_val = recon_loss.item()

        history["score_loss"].append(score_loss.item())
        history["recon_loss"].append(recon_loss_val)

        if (it + 1) % log_every == 0 or it == 0:
            msg = f"[score train] iter {it + 1}/{n_iters} score_loss={score_loss.item():.6f}"
            if use_alt:
                msg += f" recon_loss={recon_loss_val:.6f}"
            print(msg)

    return history


@torch.no_grad()
def pc_sample_step(
    score_fn,
    sde: SDE,
    h_prev: torch.Tensor,
    n_steps: int = 1000,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    eps: float = 1e-3,
    device: str = "cpu",
) -> torch.Tensor:
    """Predictor-Corrector sampler generating a single latent timestep h_t given h_{t-1}."""
    B, d_hidden = h_prev.shape
    h_s = sde.prior_sampling((B, d_hidden), device=device)
    time_steps = torch.linspace(1.0, eps, n_steps, device=device)
    ds = time_steps[0] - time_steps[1] if n_steps > 1 else torch.tensor(1.0 / max(n_steps, 1))

    for i in range(n_steps):
        s_val = time_steps[i]
        s_batch = torch.full((B,), s_val.item(), device=device)

        # Corrector: Langevin MCMC
        for _ in range(n_corrector_steps):
            grad = score_fn(h_s, h_prev, s_batch)
            noise = torch.randn_like(h_s)
            grad_norm = grad.flatten(1).norm(dim=-1).mean()
            noise_norm = noise.flatten(1).norm(dim=-1).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2
            h_s = h_s + step_size * grad + torch.sqrt(2 * step_size) * noise

        # Predictor: reverse-diffusion / Euler-Maruyama
        drift, diffusion = sde.sde(h_s, s_batch)
        score = score_fn(h_s, h_prev, s_batch)
        diffusion = diffusion.view(-1, *([1] * (h_s.dim() - 1)))
        reverse_drift = drift - diffusion**2 * score
        h_mean = h_s - reverse_drift * ds
        if i < n_steps - 1:
            h_s = h_mean + diffusion * torch.sqrt(ds) * torch.randn_like(h_s)
        else:
            h_s = h_mean

    return h_s


@torch.no_grad()
def recursive_generate(
    ae: Autoencoder,
    score_fn,
    sde: SDE,
    n_samples: int,
    T: int,
    d_hidden: int,
    n_steps: int = 1000,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate x_hat_{1:T} by recursively sampling h_1..h_T then batch-decoding."""
    h_prev = torch.zeros(n_samples, d_hidden, device=device)  # h_0 = 0
    h_seq = []
    for _ in range(T):
        h_t = pc_sample_step(
            score_fn, sde, h_prev, n_steps=n_steps, n_corrector_steps=n_corrector_steps, snr=snr, device=device
        )
        h_seq.append(h_t)
        h_prev = h_t
    h_all = torch.stack(h_seq, dim=1)  # [n_samples, T, d_hidden]
    x_hat = ae.decoder(h_all)
    return x_hat


def sample(
    ae: Autoencoder,
    score_net: nn.Module,
    sde: SDE,
    n_samples: int,
    T: int,
    d_hidden: int,
    n_steps: int = 1000,
    n_corrector_steps: int = 1,
    snr: float = 0.16,
    device: str = "cpu",
) -> torch.Tensor:
    ae.eval()
    score_net.eval()

    def score_fn(h_s_t, h_cond, s):
        return score_net(h_s_t, h_cond, s)

    return recursive_generate(
        ae, score_fn, sde, n_samples, T, d_hidden, n_steps=n_steps, n_corrector_steps=n_corrector_steps, snr=snr, device=device
    )

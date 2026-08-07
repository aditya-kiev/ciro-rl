"""Shared representation-learning trainer for CURL / marginal-HSIC / CIRO.

Method-agnostic: owns the data loop, optimizer, EMA key update and optional AMP
plumbing, and delegates only the per-step loss (and the penalty placement) to
the method modules. For SCM-MDP the image pairs are temporal neighbours
``(z_t, z_{t+window})`` rendered through the fixed decoder; DCS uses
distractor-augmented pairs (handled by the provider). Checkpoint/resume makes
long runs robust to Colab disconnects.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import torch
from torch import nn

from ..config import compute_total_steps
from ..methods.curl import CURLModel, curl_contrastive_loss


def _compute_loss(out: dict, loss_fn, model: CURLModel, method: str,
                  placement: str) -> dict:
    """Compute the per-step named losses for the requested method."""
    if method == "curl":
        c = curl_contrastive_loss(out["z_q"], out["z_k"], model.W)
        return {"total": c, "contrastive": c, "hsic": out["z_q"].sum() * 0.0}
    if method == "marginal_hsic":
        m = loss_fn(out["z_q"], out["z_k"], model.W)
        return {"total": m["total"], "contrastive": m["contrastive"],
                "hsic": m["hsic"]}
    m = loss_fn(out["h_q"], out["h_k"], out["z_q"], out["z_k"], model.W,
                placement=placement)
    return {"total": m["total"], "contrastive": m["contrastive"],
            "hsic": m["ciro"]}


def latest_checkpoint(output_dir) -> Optional[str]:
    """Most recent ``checkpoint_*.pt`` in ``output_dir``, or None."""
    pattern = os.path.join(str(output_dir), "checkpoint_*.pt")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    return paths[-1] if paths else None


def _save_checkpoint(output_dir, model, opt, step: int) -> str:
    os.makedirs(str(output_dir), exist_ok=True)
    path = os.path.join(str(output_dir), f"checkpoint_{step:06d}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": int(step),
    }, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def run_training(cfg, model: CURLModel, loss_fn: Optional[nn.Module],
                 provider, output_dir, device: Optional[torch.device] = None,
                 resume: bool = True):
    """Train ``model`` on ``provider`` for ``cfg.total_steps`` gradient steps.

    Mirrors causal-cir's train.py loop with a method-agnostic loss dispatch and
    a checkpoints/ directory per (method, dataset, seed) cell. Reuses existing
    checkpoints when ``resume`` is True (Colab-disconnect resilience).

    Returns:
        A dict summarising the run (started step, finished step, losses).
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)

    method = str(cfg.get("method", "ciro"))
    placement = str(cfg.get("placement", "encoder"))
    total = int(cfg.get("total_steps") or compute_total_steps(cfg))
    bs = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-6))
    momentum = float(cfg.get("momentum", 0.05))
    seed = int(cfg.get("seed", 0))
    save_every = int(cfg.get("save_every", 100))
    log_every = int(cfg.get("log_every", 25))

    os.makedirs(str(output_dir), exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    start = 0
    loaded = None
    if resume:
        cpath = latest_checkpoint(output_dir)
        if cpath:
            loaded = torch.load(cpath, map_location=device, weights_only=False)
    if loaded is not None:
        model.load_state_dict(loaded["model"])
        if loaded.get("optimizer"):
            opt.load_state_dict(loaded["optimizer"])
        start = int(loaded.get("step", 0))

    model.train()
    last = {"total": 0.0, "contrastive": 0.0, "hsic": 0.0}
    stream = provider.batches(bs, total, seed)
    for step in range(total):
        if step < start:
            try:
                next(stream)
            except StopIteration:
                break
            continue
        try:
            xq, xk, _lab = next(stream)
        except StopIteration:
            break
        out = model(xq, xk)
        met = _compute_loss(out, loss_fn, model, method, placement)
        opt.zero_grad(set_to_none=True)
        met["total"].backward()
        opt.step()
        model.update_momentum_key(momentum)
        last = {k: float(met[k].detach()) for k in ("total", "contrastive", "hsic")}
        if (step + 1) % log_every == 0:
            print(f"  [train/{method}/seed{seed} step {step + 1}/{total}] "
                  f"total={last['total']:.4f} c={last['contrastive']:.4f} "
                  f"h={last['hsic']:.4f}")
        if (step + 1) % save_every == 0:
            _save_checkpoint(output_dir, model, opt, step + 1)

    model.eval()
    return {"started": start, "finished": total, **last}
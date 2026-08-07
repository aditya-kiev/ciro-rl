"""Checkpoint / resume helpers for Colab-disconnect resilience.

A Colab session can drop mid-run. We write a checkpoint every `save_every`
steps that captures model + optimizer + RNG state so a run can be resumed
from the last checkpoint rather than restarted.

Note on reproducibility after resume: we store the Python / NumPy / torch RNG
states so that the post-resume random draws are exactly what they would have
been in an uninterrupted run (best-effort; DataLoader worker RNG is re-seeded).
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def checkpoint_path(output_dir: str, method: str, dataset: str, seed: int) -> str:
    """Per-(method, dataset, seed) checkpoint file path."""
    safe = dataset.replace("/", "_").replace("\\", "_")
    return os.path.join(output_dir, "checkpoints",
                        f"{method}__{safe}__seed{seed}.pt")


def save_checkpoint(path: str, model, optimizer, epoch: int, step: int,
                    config: Optional[dict] = None,
                    extra: Optional[dict] = None) -> None:
    """Persist a training snapshot that :func:`load_checkpoint` can restore."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "rng": {
            "python": None,
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": None,
        },
        "config": config,
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint dict (or None if missing/empty)."""
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng(state: dict) -> None:
    """Best-effort restore of Python/torch RNG onto the current process."""
    import random
    import numpy as np
    if "rng" not in state:
        return
    r = state["rng"]
    if r.get("numpy") is not None:
        np.random.set_state(r["numpy"])
    if r.get("python") is not None:
        random.setstate(r["python"])
    if r.get("torch") is not None:
        torch.set_rng_state(r["torch"])


def resume_or_init(config: dict, trainer, dataset_name: str) -> dict:
    """Resume a trainer from its checkpoint if one exists, else fresh.

    ``dataset_name`` is the value used to build the checkpoint filename
    (e.g. ``scm_independent``). Returns a small metadata dict.
    """
    out_dir = config.get("output_dir", "./outputs")
    method = config.get("method", "ciro")
    seed = int(config.get("seed", 0))
    path = checkpoint_path(out_dir, method, dataset_name, seed)
    ckpt = load_checkpoint(path, trainer.device)
    if ckpt is None:
        return {"resumed": False, "path": path, "epoch": 0, "step": 0}
    trainer.model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None:
        trainer.opt.load_state_dict(ckpt["optimizer"])
    restore_rng(ckpt)
    return {"resumed": True, "path": path,
            "epoch": ckpt.get("epoch", 0), "step": ckpt.get("step", 0)}


def save_final(config: dict, trainer, dataset_name: str, results: dict,
               metrics: dict) -> str:
    """Save a *results-only* bundle alongside the run's final state."""
    import json
    out_dir = config.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    seed = int(config.get("seed", 0))
    path = os.path.join(out_dir, f"final__{dataset_name}__seed{seed}.json")
    payload = {"config": config, "results": results, "metrics": metrics}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


__all__ = ["checkpoint_path", "save_checkpoint", "load_checkpoint",
           "restore_rng", "resume_or_init", "save_final"]

"""Configuration loading from the YAML files in `configs/`. 

Each cell is a plain dict but exposed with attribute access, plus helpers for
the two things the trainer and experiments need: ``total_steps`` and the seed
list. Defaults mirror the CIR paper's convention (batch 128-256, hsic_lambda
0.05, RFF D=128, seeds 0,1,2).
"""

import copy
import os
from pathlib import Path
from typing import Optional


class Config(dict):
    """Attribute-access uniform store for a run's parameters."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def with_overrides(self, overrides: dict) -> "Config":
        out = Config(self)
        out.update(overrides)
        return out


DEFAULTS = {
    "batch_size": 256,
    "d_rep": 64,
    "temperature": 0.1,
    "hsic_lambda": 0.05,
    "hsic_sigma": None,
    "hsic_mode": "rff",
    "rff_dim": 128,
    "placement": "encoder",
    "proj_enabled": False,
    "proj_hidden": 64,
    "proj_out": 64,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "momentum": 0.05,
    "mixed_precision": True,
    "device": "auto",
    "resume": True,
    "log_every": 10,
    "save_every": 50,
    "method": "ciro",
    "seed": 0,
    "output_dir": "./outputs",
    "n_episodes": 24,
    "episode_len": 120,
    "window": 4,
    "img_size": 32,
    "latent_dim": 8,
    "latent_seed": 7,
    "beta": 0.5,
    "confounded_pair": (0, 1),
    "dcs_resolution": 64,
    "dcs_n_frames": 16,
    "dcs_difficulty": "medium",
}


def load_config(path: str, overrides: Optional[dict] = None) -> Config:
    import yaml

    cfg = Config(DEFAULTS)
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg.update(data)
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_total_steps(cfg: Config) -> int:
    """Gradient steps from either epochs*steps_per_epoch (SCM) or DCS update
    budget."""
    if cfg.get("dcs", False) and cfg.get("dcs_steps"):
        return int(cfg["dcs_steps"])
    return int(cfg.get("epochs", 20) * cfg.get("steps_per_epoch", 200))


def ensure_output_dir(cfg: Config) -> Path:
    base = Path(cfg.get("output_dir", "./outputs"))
    base.mkdir(parents=True, exist_ok=True)
    return base
"""Small YAML-config loader (no OmegaConf dependency).

Reads a ``configs/*.yaml`` into a lightweight nested dict with attribute access
(``cfg.model.batch_size``) plus CLI overrides (``--quick-test``, ``--seed``,
``--out=...``). All experiment entrypoints share this loader.
"""

from __future__ import annotations

from typing import Any, Dict

import yaml


class AttrDict(dict):
    """Dot- and bracket-accessible dict."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:  # pragma: no cover - defensive
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _to_attrdict(node: Any) -> Any:
    if isinstance(node, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in node.items()})
    if isinstance(node, list):
        return [_to_attrdict(v) for v in node]
    return node


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> AttrDict:
    """Load a YAML config file and apply `overrides` (top-level keys win)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _to_attrdict(raw)
    # Fill defaults that every config must expose.
    cfg.setdefault("method", {})
    cfg.setdefault("model", {})
    cfg.setdefault("training", {})
    cfg.setdefault("data", {})
    cfg.setdefault("scm", {})
    cfg.setdefault("name", "config")
    if overrides:
        for k, v in overrides.items():
            if k == "seed":
                cfg["seed"] = v
            elif k in ("lambda", "proj_out", "penalty_on"):
                cfg["method"][k] = v
            else:
                cfg[k] = v
    return cfg
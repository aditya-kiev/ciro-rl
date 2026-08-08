"""Shared helpers for the experiment entry points.

Implements a single (method, dataset, seed) cell: build the provider, train the
representation encoder with checkpoint/resume, then evaluate the appropriate
diagnostic (ACS for SCM-MDPs, DTG for DCS tasks) and return the per-metric
results. DCS cells are skipped gracefully when the MuJoCo stack is unavailable.
"""

from pathlib import Path

import numpy as np

from ciro_rl.config import compute_total_steps, load_config
from ciro_rl.diagnostics.acs import average_causal_sensitivity
from ciro_rl.diagnostics.dtg import run_dtg
from ciro_rl.envs.dcs_wrapper import DCSError
from ciro_rl.methods import build_model, make_loss_fn
from ciro_rl.training.provider import DcsPairs, SMTemporalPairs

SCM_NAMES = {"scm_independent", "scm_chain", "scm_confounded"}
DCS_NAMES = {"dcs_cartpole_balance", "dcs_walker_walk"}
DATASET_NAMES = SCM_NAMES | DCS_NAMES
NAME_TO_CONFIG = {
    "scm_independent": "configs/scm_mdp_independent.yaml",
    "scm_chain": "configs/scm_mdp_chain.yaml",
    "scm_confounded": "configs/scm_mdp_confounded.yaml",
    "dcs_cartpole_balance": "configs/dcs_cartpole_balance.yaml",
    "dcs_walker_walk": "configs/dcs_walker_walk.yaml",
}
METHODS = ("curl", "marginal_hsic", "ciro")
SEEDS = (0, 1, 2)


def build_provider(cfg, device, is_dcs):
    """Build the training-stream provider.

    Returns (provider, error). `error` is an explanatory string when the
    dataset must be skipped (DCS unavailable), else None.
    """
    if is_dcs:
        try:
            prow = DcsPairs(
                cfg.domain,
                cfg.task,
                resolution=cfg.dcs_resolution,
                n_frames=cfg.dcs_n_frames,
                seed=cfg.seed,
                difficulty=cfg.dcs_difficulty,
                device=device,
            )
        except DCSError as e:
            return None, str(e)
        return prow, None
    prow = SMTemporalPairs(
        cfg.kind,
        window=cfg.window,
        n_episodes=cfg.n_episodes,
        episode_len=cfg.episode_len,
        seed=cfg.latent_seed,
        latent_dim=cfg.latent_dim,
        causal_parents=cfg.causal_parents,
        confounded_pair=tuple(cfg.confounded_pair),
        beta=cfg.beta,
        img_size=cfg.img_size,
        device=device,
    )
    return prow, None


def evaluate_diagnostics(model, provider, cfg, is_dcs, device):
    """Evaluate a trained model with its dataset's diagnostic (ACS or DTG)."""
    if is_dcs:
        fr_tr, lab_tr = provider.load_frames(
            cfg.dcs_train_pool, cfg.dcs_n_frames, cfg.dcs_pool_action_seed
        )
        fr_ev, lab_ev = provider.load_frames(
            cfg.dcs_eval_pool, cfg.dcs_n_frames, cfg.dcs_pool_action_seed + 7
        )
        out = run_dtg(
            model.query, fr_tr, lab_tr, fr_ev, lab_ev, device=device, seed=cfg.seed
        )
        return [
            ("score_id", out["score_id"]),
            ("score_ood", out["score_ood"]),
            ("dtg", out["dtg"]),
        ]
    acs = average_causal_sensitivity(
        provider.scm,
        model.query,
        n_samples=cfg.acs_n_samples,
        delta=cfg.acs_delta,
        h=cfg.acs_h,
        device=device,
    )
    return [("acs", acs["acs"])]


def run_cell(config_root: Path, dataset: str, method: str, seed: int,
             quick_overrides: dict, device, force: bool = False) -> dict:
    """Run one (dataset, method, seed) cell.

    Returns a dict with keys method/dataset/seed/skipped/metrics. `metrics` is
    a list of (metric, value) for the diagnostics defined on that dataset.
    """
    config_path = config_root / NAME_TO_CONFIG[dataset].rsplit("/", 1)[-1]
    overrides = {"method": method, "seed": seed}
    if quick_overrides:
        overrides.update(quick_overrides)
    overrides["method"] = method  # requested method always wins over defaults
    cfg = load_config(config_path, overrides)
    is_dcs = dataset in DCS_NAMES
    device = device

    provider, err = build_provider(cfg, device, is_dcs)
    if provider is None:
        return {"dataset": dataset, "method": method, "seed": seed,
                "skipped": True, "reason": err}

    model = build_model(cfg, device)
    loss_fn = make_loss_fn(method, cfg, placement=cfg.placement)
    cfg.total_steps = compute_total_steps(cfg)
    cfg.output_dir = Path(cfg.output_dir) / f"{dataset}_{method}_s{seed}"

    from ciro_rl.training.trainer import run_training
    run_training(cfg, model, loss_fn, provider, cfg.output_dir,
                 device=device, resume=cfg.resume)

    metrics = evaluate_diagnostics(model, provider, cfg, is_dcs, device)
    return {"dataset": dataset, "method": method, "seed": seed,
            "skipped": False, "metrics": metrics}


def cells_to_records(cells):
    """Flatten a list of cells into one record per (method, dataset, seed, metric)."""
    rows = []
    for c in cells:
        if c.get("skipped"):
            continue
        for metric, value in c["metrics"]:
            rows.append({
                "method": c["method"],
                "dataset": c["dataset"],
                "seed": c["seed"],
                "metric": metric,
                "value": value,
            })
    return rows


def summarize_by_dataset(records):
    """Aggregate per-(method, dataset, metric) mean/std over seeds.

    This is the per-dataset breakdown the CIRO paper Appendix D wants for
    reproducibility (mean +- std over the 3 seeds).
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r["method"], r["dataset"], r["metric"])].append(r["value"])
    summary = []
    for (method, dataset, metric), vals in sorted(groups.items()):
        vals = np.asarray(vals, dtype=float)
        summary.append({
            "method": method, "dataset": dataset, "metric": metric,
            "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
            "n_seeds": int(len(vals)),
        })
    return summary


def table1_from_records(records, metric_datasets):
    """Appendix D unweighted average: for each metric, average the per-dataset
    means over the datasets on which that metric is defined, and report the
    population std across those dataset means.

    Args:
        records: list of per-seed dicts (method, dataset, metric, value).
        metric_datasets: {'acs': SCM_NAMES, 'score_id': DCS_NAMES, ...}.
    """
    from collections import defaultdict
    dmean = defaultdict(list)  # (method, metric) -> list of per-dataset means
    per_dm = defaultdict(list)  # (method, dataset, metric) -> seed values
    for r in records:
        per_dm[(r["method"], r["dataset"], r["metric"])].append(r["value"])
    for (method, dataset, metric), vals in per_dm.items():
        dmean[(method, metric)].append(float(np.mean(vals)))

    rows = []
    for method in sorted({r["method"] for r in records}):
        row = {"method": method}
        for metric, dsets in metric_datasets.items():
            vals = dmean.get((method, metric), [])
            if not vals:
                row[metric] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_n"] = 0
            else:
                row[metric] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=0))
                row[f"{metric}_n"] = len(vals)
        rows.append(row)
    return rows


METRIC_DATASETS = {
    "acs": SCM_NAMES,
    "score_id": DCS_NAMES,
    "score_ood": DCS_NAMES,
    "dtg": DCS_NAMES,
}
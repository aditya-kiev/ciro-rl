"""run_matrix_reduced.py: full structural matrix at a DOCUMENTED reduced budget.

Runs the *complete* Table 1 structure (all SCM methods x datasets x seeds), the
Table 2 ablation, and the ACS-horizon sweep, at a reduced budget
(``batch_size=32``, ``epochs=1`` / ``steps_per_epoch=60`` => 60 gradient steps
per cell) instead of the config defaults (batch 256, 8000 steps).

WHY (read before trusting these numbers): the CIRO / marginal-HSIC penalty sums
per-pair-coordinate biased HSIC V-statistics, which is O(d^2 * n^3) per step
with d=64 representation dimensions; at batch 256 a single CIRO step costs
~100s of CPU, so the placeholder 8000-step scale is computationally infeasible
on this box (weeks of single-thread CPU). These CSVs are structurally-complete
REAL runs for pipeline/integration verification at a reduced budget -- NOT the
publication budget. The paper numbers must come from the GPU run described in
README ("run_table1.py --seeds 0 1 2" at default configs). DCS datasets are
excluded (no MuJoCo/DAVIS/MuJoCo-license on this box).

Robustness: each cell's trained model is checkpointed (resume=True), and each
cell's evaluated metrics are appended to ``records.jsonl`` immediately, so
killed invocations never lose completed cells and re-running the same command
skips already-done cells (checkpoint resume + record dedup).

Usage:
    python experiments/run_matrix_reduced.py --part table1
    python experiments/run_matrix_reduced.py --part table2
    python experiments/run_matrix_reduced.py --part horizon
    python experiments/run_matrix_reduced.py --part all
    add --method ciro / --seed 0 to run a slice (helps chunk long CPU runs)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(line_buffering=True)

from experiments.common import (  # noqa: E402
    METHODS, SEEDS, SCM_NAMES,
    run_cell, summarize_by_dataset,
    table1_from_records, METRIC_DATASETS,
)

# Documented reduced budget. Uniform across every cell so relative comparisons
# that the paper cares about are still meaningful.
OVERRIDES = {
    "batch_size": 32,
    "epochs": 1,
    "steps_per_epoch": 12,   # => 12 gradient steps per cell
    "acs_n_samples": 128,
    "acs_delta": 1.0,
    "acs_h": 0,
    "log_every": 6,
    "save_every": 12,
    "output_dir": "./outputs/matrix",
}

LAMBDAS = [0.01, 0.05, 0.10]
PLACEMENTS = ["encoder", "projection"]
HORIZONS = [0, 5, 20]

MATRIX_DIR = Path("./outputs/matrix")
RECORDS_PATH = MATRIX_DIR / "records.jsonl"


def _resolve_device():
    from ciro_rl.utils.seeding import resolve_device
    return resolve_device()


def _load_records():
    if not RECORDS_PATH.exists():
        return {}
    out = {}
    for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[(r["method"], r["dataset"], r["seed"], r["metric"])] = r["value"]
    return out


def _append_record(r):
    with open(RECORDS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")


def _run_cell_recorded(ds, m, s, extra_overrides, device):
    overrides = dict(OVERRIDES)
    overrides.update(extra_overrides)
    overrides.update({"seed": s})
    cell = run_cell(ROOT / "configs", ds, m, s, overrides, device)
    if cell.get("skipped"):
        print(f"    [cell] {ds} {m} s{s} SKIPPED ({cell.get('reason')})", flush=True)
        return 0
    n_added = 0
    for metric, value in cell["metrics"]:
        key = (m, ds, s, metric)
        if key in _load_records():
            continue
        _append_record({"method": m, "dataset": ds, "seed": s,
                        "metric": metric, "value": float(value)})
        n_added += 1
    print(f"    [cell] {ds} {m} s{s} done ({n_added} new records)", flush=True)
    return n_added


def run_table1(device, only_methods, only_seeds):
    datasets = sorted(SCM_NAMES)
    for ds in datasets:
        for m in METHODS:
            if only_methods and m not in only_methods:
                continue
            for s in SEEDS:
                if only_seeds and s not in only_seeds:
                    continue
                _run_cell_recorded(ds, m, s, {}, device)
    records = [
        {"method": m, "dataset": d, "seed": s, "metric": mt, "value": v}
        for (m, d, s, mt), v in _load_records().items()
        if d in SCM_NAMES
    ]
    if not records:
        print("[table1] no records yet; nothing to summarise.", flush=True)
        return
    summary = summarize_by_dataset(records)
    table = table1_from_records(records, METRIC_DATASETS)
    _write_rows(MATRIX_DIR / "results_by_dataset_summary.csv",
                ["method", "dataset", "metric", "mean", "std", "n_seeds"], summary)
    fields = ["method"] + [f"{mk}_{suf}"
                           for mk in ("acs", "score_id", "score_ood", "dtg")
                           for suf in ["mean", "std", "n"]]
    avg_rows = []
    for row in table:
        out = {"method": row["method"]}
        for mk in ("acs", "score_id", "score_ood", "dtg"):
            out[f"{mk}_mean"] = row.get(mk, "")
            out[f"{mk}_std"] = row.get(f"{mk}_std", "")
            out[f"{mk}_n"] = row.get(f"{mk}_n", "")
        avg_rows.append(out)
    _write_rows(MATRIX_DIR / "results_table1_avg.csv", fields, avg_rows)
    print("\n--- Table 1 (SCM, reduced budget) ---", flush=True)
    for r in summary:
        print(f"  {r['method']:<14}{r['dataset']:<20}{r['metric']:<8}"
              f"{r['mean']:.4f} +- {r['std']:.4f} (n={r['n_seeds']})", flush=True)


def run_table2(device):
    rows = []
    for lam in LAMBDAS:
        for placement in PLACEMENTS:
            overrides = dict(OVERRIDES)
            overrides.update({"method": "ciro", "hsic_lambda": lam,
                              "placement": placement, "seed": 0})
            cell = run_cell(ROOT / "configs", "scm_confounded", "ciro", 0,
                            overrides, device)
            if cell.get("skipped"):
                rows.append({"lambda": lam, "placement": placement,
                             "acs": "", "skipped": "1"})
                continue
            acs = dict(cell["metrics"]).get("acs", float("nan"))
            rows.append({"lambda": lam, "placement": placement,
                         "acs": f"{acs:.4f}", "skipped": ""})
    _write_rows(MATRIX_DIR / "results_table2.csv",
                ["lambda", "placement", "acs", "skipped"], rows)
    print("\n--- Table 2 ablation (confounded, seed 0, reduced budget) ---", flush=True)
    for r in rows:
        print(f"  lambda={r['lambda']} {r['placement']:<12} acs={r['acs']}", flush=True)


def run_horizon(device):
    rows = []
    for h in HORIZONS:
        overrides = dict(OVERRIDES)
        overrides.update({"method": "ciro", "acs_h": h, "seed": 0})
        cell = run_cell(ROOT / "configs", "scm_confounded", "ciro", 0,
                        overrides, device)
        if cell.get("skipped"):
            rows.append({"h": h, "acs": "", "skipped": "1"})
            continue
        acs = dict(cell["metrics"]).get("acs", float("nan"))
        rows.append({"h": h, "acs": f"{acs:.4f}", "skipped": ""})
    _write_rows(MATRIX_DIR / "results_acs_horizon.csv",
                ["h", "acs", "skipped"], rows)
    print("\n--- ACS rollout-horizon sweep (confounded, seed 0, reduced) ---", flush=True)
    for r in rows:
        print(f"  h={r['h']:<4} acs={r['acs']}", flush=True)


def _write_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[matrix] wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="CIRO reduced-budget matrix runner")
    ap.add_argument("--part", choices=["table1", "table2", "horizon", "all"],
                    default="all")
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--seed", type=int, action="append", default=None)
    args = ap.parse_args()

    device = _resolve_device()
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[matrix] device={device} budget override="
          f"batch={OVERRIDES['batch_size']} steps="
          f"{OVERRIDES['epochs'] * OVERRIDES['steps_per_epoch']}", flush=True)
    if args.part in ("all", "table1"):
        run_table1(device, args.method, args.seed)
    if args.part in ("all", "table2"):
        run_table2(device)
    if args.part in ("all", "horizon"):
        run_horizon(device)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
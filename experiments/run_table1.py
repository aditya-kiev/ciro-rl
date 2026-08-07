"""run_table1.py : 3-seed runner across methods x datasets.

For every (method in {curl, marginal_hsis, ciro}) x (dataset in {scm_independent,
scm_chain, scm_confounded, dcs_cartpole_balance, dcs_walker_walk}) x
(seed in {0,1,2}): train the encoder, evaluate ACS (SCM datasets) or DTG
(DCS datasets), and write:

  - results_by_dataset_summary.csv -- one row per (method, dataset, metric)
       mean over seeds (material Table-1 breakdown).
  - results_table1_avg.csv         -- per-method unweighted average of the
       per-dataset means, with SEPARATE denominators for ACS-defined (SCM) and
       DTG-defined (DCS) datasets, mirroring CIR Appendix D.

DCS datasets are skipped when the MuJoCo stack is unavailable.

Usage:
    python experiments/run_table1.py --out outputs --seeds 0 1 2
    python experiments/run_table1.py --quick-test   # tiny smoke run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from experiments.common import (  # noqa: E402
    DATASET_NAMES, METHODS, SEEDS,
    cells_to_records, run_cell, summarize_by_dataset,
    table1_from_records, METRIC_DATASETS,
)

AVG_METRICS = ["acs", "score_id", "score_ood", "dtg"]


def _resolve_device():
    from ciro_rl.utils.seeding import resolve_device
    return resolve_device()


def _load_quick() -> dict:
    cfg_path = ROOT / "configs" / "quick_test.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CIRO Table 1 runner")
    ap.add_argument("--seed", type=int, action="append", default=None,
                    help="seed to run (repeatable); default 0,1,2")
    ap.add_argument("--out", default="./outputs", help="output directory")
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--quick-test", action="store_true",
                    help="tiny smoke run (few steps, seeds=[0], ciro only)")
    args = ap.parse_args(argv)

    if args.quick_test:
        seeds, methods = [0], ["ciro"]
    else:
        seeds = args.seed or list(SEEDS)
        methods = args.method or list(("curl", "marginal_hsic", "ciro"))
    datasets = args.dataset or sorted(DATASET_NAMES)

    quick = _load_quick() if args.quick_test else {}
    device = _resolve_device()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = []
    for ds in datasets:
        for m in methods:
            for s in seeds:
                cells.append(run_cell(ROOT / "configs", ds, m, s, quick, device))

    records = cells_to_records(cells)
    summary = summarize_by_dataset(records)
    table = table1_from_records(records, METRIC_DATASETS)

    _write_by_dataset(summary, outdir / "results_by_dataset_summary.csv")
    _write_avg(table, outdir / "results_table1_avg.csv")
    _print_table(summary)

    print(f"\n[table1] wrote {outdir / 'results_by_dataset_summary.csv'} "
          f"and {outdir / 'results_table1_avg.csv'}")


def _write_by_dataset(summary, path) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "dataset", "metric",
                                           "mean", "std", "n_seeds"])
        w.writeheader()
        for r in summary:
            w.writerow(r)


def _write_avg(table, path) -> None:
    import csv
    fields = ["method"] + [f"{mk}_{suf}"
                           for mk in AVG_METRICS for suf in ("mean", "std", "n")]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in table:
            out = {"method": row["method"]}
            for mk in AVG_METRICS:
                out[f"{mk}_mean"] = row.get(mk, "")
                out[f"{mk}_std"] = row.get(f"{mk}_std", "")
                out[f"{mk}_n"] = row.get(f"{mk}_n", "")
            w.writerow(out)


def _print_table(summary) -> None:
    print("\n=== Table 1: per-(method,dataset,metric) means over seeds ===")
    print(f"{'method':<14}{'dataset':<20}{'metric':<10}{'mean':>10}{'std':>10}")
    for r in summary:
        print(f"{r['method']:<14}{r['dataset']:<20}{r['metric']:<10}"
              f"{r['mean']:>10.4f}{r['std']:>10.4f}")


if __name__ == "__main__":
    main()
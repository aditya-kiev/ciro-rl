"""run_table2_ablations.py: CIRO Table-2-style ablation on SCM-MDP (C).

Varies (a) the penalty weight hsic_lambda in {0.01, 0.05, 0.10} and (b) the
penalty placement in {encoder, projection} on the confounded benchmark, seed 0,
and writes a results_table2.csv summarising the resulting ACS per configuration.

Usage:
    python experiments/run_table2_ablations.py
    python experiments/run_table2_ablations.py --quick-test
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

from experiments.common import run_cell  # noqa: E402

DATASET = "scm_confounded"
LAMBDAS = [0.01, 0.05, 0.10]
PLACEMENTS = ["encoder", "projection"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CIRO Table 2 ablation runner")
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--quick-test", action="store_true")
    args = ap.parse_args(argv)

    quick = {}
    if args.quick_test:
        with open(ROOT / "configs" / "quick_test.yaml", encoding="utf-8") as fh:
            quick = yaml.safe_load(fh) or {}

    from ciro_rl.utils.seeding import resolve_device
    device = resolve_device()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lam in LAMBDAS:
        for placement in PLACEMENTS:
            overrides = dict(quick)
            overrides.update({"method": "ciro", "hsic_lambda": lam,
                              "placement": placement, "seed": 0})
            cell = run_cell(ROOT / "configs", DATASET, "ciro", 0, overrides, device)
            if cell.get("skipped"):
                rows.append({"lambda": lam, "placement": placement,
                             "acs": "", "skipped": "1"})
                continue
            acs = dict(cell["metrics"]).get("acs", float("nan"))
            rows.append({"lambda": lam, "placement": placement,
                         "acs": f"{acs:.4f}", "skipped": ""})

    path = outdir / "results_table2.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["lambda", "placement", "acs", "skipped"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n=== Table 2 ablation (confounded, seed 0) ===")
    print(f"{'lambda':>8}{'placement':<12}{'acs':>10}")
    for r in rows:
        print(f"{r['lambda']:>8}{r['placement']:<12}{r['acs']:>10}")
    print(f"[table2] wrote {path}")


if __name__ == "__main__":
    main()
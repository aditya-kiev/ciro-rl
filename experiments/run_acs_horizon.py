"""run_acs_horizon.py: ACS rollout-horizon sweep (CIRO App. C.3).

Evaluates ACS at h in {0, 5, 20} on the confounded benchmark (Dataset C), which
is where the "propagate the intervention through h transition steps before
encoding" aspect of the ACS protocol becomes meaningful. Training is identical
across rows (same config, same seed), so the sweep shows the *same* encoder's
concentration as the intervention-horizon grows.

Usage:
    python experiments/run_acs_horizon.py
    python experiments/run_acs_horizon.py --quick-test
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
HORIZONS = [0, 5, 20]


def main() -> None:
    ap = argparse.ArgumentParser(description="CIRO ACS rollout-horizon sweep")
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--quick-test", action="store_true")
    args = ap.parse_args()

    quick = {}
    if args.quick_test:
        with open(ROOT / "configs" / "quick_test.yaml", encoding="utf-8") as fh:
            quick = yaml.safe_load(fh) or {}

    from ciro_rl.utils.seeding import resolve_device
    device = resolve_device()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for h in HORIZONS:
        overrides = dict(quick)
        overrides.update({"method": "ciro", "acs_h": h, "seed": 0})
        cell = run_cell(ROOT / "configs", DATASET, "ciro", 0, overrides, device)
        if cell.get("skipped"):
            rows.append({"h": h, "acs": "", "skipped": "1"})
            continue
        acs = dict(cell["metrics"]).get("acs", float("nan"))
        rows.append({"h": h, "acs": f"{acs:.4f}", "skipped": ""})

    path = outdir / "results_acs_horizon.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["h", "acs", "skipped"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n=== ACS rollout-horizon sweep (confounded, seed 0) ===")
    print(f"{'h':>4}{'acs':>10}")
    for r in rows:
        print(f"{r['h']:>4}{r['acs']:>10}")
    print(f"[acs_horizon] wrote {path}")


if __name__ == "__main__":
    main()
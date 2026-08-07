"""quick_test.py: tiny end-to-end smoke run.

Runs a few gradient steps of every cell (SCM datasets; DCS attempted and
skipped when MuJoCo is absent) and prints the measured diagnostics, so you can
confirm the whole pipeline (provider -> train -> diagnostic) runs before a full
3-seed run. See also `python experiments/run_table1.py --quick-test`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from experiments.common import (  # noqa: E402
    SCM_NAMES, METHODS, cells_to_records, run_cell,
)


def main() -> None:
    with open(ROOT / "configs" / "quick_test.yaml", encoding="utf-8") as fh:
        quick = yaml.safe_load(fh) or {}
    from ciro_rl.utils.seeding import resolve_device
    device = resolve_device()

    print("quick_test: training a few steps per cell and evaluating...")
    cells = []
    for ds in sorted(SCM_NAMES):
        for m in METHODS:
            cells.append(run_cell(ROOT / "configs", ds, m, 0, quick, device))

    recs = cells_to_records(cells)
    print("\ncollected records:")
    for r in recs:
        print(f"  {r['method']:<14}{r['dataset']:<20}{r['metric']:<8}{r['value']:.4f}")

    n_skipped = sum(1 for c in cells if c.get("skipped"))
    print(f"completed {len(cells)} cells "
          f"({n_skipped} skipped, usually DCS/MuJoCo-unavailable).")
    print("SUCCESS: quick_test pipeline ran end-to-end.")


if __name__ == "__main__":
    main()
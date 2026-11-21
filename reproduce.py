"""One command from the shipped feature caches to every figure, table and number.

    python reproduce.py              # analyses -> figures -> tables -> claim check
    python reproduce.py --jobs 6     # same, with analyses run in parallel
    python reproduce.py --tests      # also run the physics/unit test suite first
    python reproduce.py --force      # recompute analyses even if outputs exist

Outputs:
    paper/results/*.json            raw analysis outputs (one per step)
    paper/results/logs/*.log        console output of each analysis
    paper/figures/fig*.pdf|png      the six figures of the manuscript
    paper/tables/*.tex, tables.md   the results tables of the manuscript
    paper/results/claims_report.md  every quantitative claim, paper vs. regenerated

This tier needs no raw data and no GPU. Rebuilding the feature caches
themselves from BioSR and W2S is documented in docs/reproduction.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*argv: str) -> int:
    print(f"\n$ python {' '.join(argv)}", flush=True)
    return subprocess.call([sys.executable, *argv], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tests", action="store_true")
    args = ap.parse_args()

    if args.tests and run("-m", "pytest", "-q") != 0:
        print("tests failed; stopping", file=sys.stderr)
        return 1
    analyses = ["paper/run_analyses.py", "--jobs", str(args.jobs)]
    if args.force:
        analyses.append("--force")
    if run(*analyses) != 0:
        return 1
    rc = run("paper/make_figures.py")
    rc |= run("paper/make_tables.py")
    rc |= run("paper/check_claims.py", "--markdown", "paper/results/claims_report.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())

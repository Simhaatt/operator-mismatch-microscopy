"""Rebuild every paper figure from cached results.

One entry point so a stale figure cannot survive a change to the data. Each
sub-script reads only cached `.npz` and `.json`, so the whole set regenerates in
seconds and every panel is stamped with the commit it came from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ["figure_bound", "figure_blind", "figure_haze",
           "figure_transfer", "figure_reconstructors"]


def main() -> None:
    failed = []
    for name in FIGURES:
        print(f"=== {name} ===")
        p = subprocess.run([sys.executable, f"scripts/{name}.py"], cwd=ROOT,
                           text=True, capture_output=True)
        print(p.stdout.rstrip() or p.stderr.rstrip()[-400:])
        if p.returncode != 0:
            failed.append(name)
    print()
    outs = sorted((ROOT / "figures").glob("*.pdf"))
    print(f"{len(outs)} figures in figures/:")
    for o in outs:
        print(f"  {o.name:28s} {o.stat().st_size / 1024:6.0f} KB")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

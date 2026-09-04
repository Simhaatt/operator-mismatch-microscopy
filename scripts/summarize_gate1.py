"""Summarise Gate 1 records across photon budgets.

Tests the mechanism's falsifiable prediction. If the measurement residual is
blind to operator mismatch *because* the model-error projection falls below the
photon noise floor, then raising the photon budget must lower the floor (as
1/sqrt(N)) and make the residual progressively more informative. If instead the
residual stays equally blind at every photon level, the noise-floor explanation
is wrong and something else is going on.

Reads the JSON records written by ``gate1_phenomenon.py --out``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="experiments/gate1/*.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no records matched {args.pattern!r}")
    recs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    recs.sort(key=lambda r: r["config"]["photons"])

    print("Mechanism check: does the residual become informative as the noise")
    print("floor drops? BioSR's real signal levels span 15-600 photons.")
    print()
    print("  photons  RL iters  noise floor  resid(nom)  resid(worst)  signal/floor")
    for r in recs:
        c, ref = r["config"], r["reference"]
        worst = max(cell["residual_mean"] for cell in r["cells"])
        sig = abs(worst - ref["nominal_residual"]) / ref["noise_floor_rel_residual"]
        print(f"  {c['photons']:7.0f}  {c['rl_n_iter']:8d}  {ref['noise_floor_rel_residual']:11.4f}"
              f"  {ref['nominal_residual']:10.4f}  {worst:12.4f}  {sig:12.4f}")

    print()
    print("  photons   d(error) severe    d(residual) severe    ratio    err ratio")
    for r in recs:
        sev = [c for c in r["cells"] if c["severity"] == "severe"]
        de = sum(c["d_error"] for c in sev) / len(sev)
        dr = sum(c["d_residual"] for c in sev) / len(sev)
        e0 = r["reference"]["nominal_error"]
        er = sum(c["error_mean"] for c in sev) / len(sev) / e0
        print(f"  {r['config']['photons']:7.0f}   {de:14.2f}    {dr:17.2f}"
              f"  {de / max(dr, 1e-9):7.1f}x    {er:8.2f}x")

    print()
    print("  photons   absolute error increase (severe - nominal)   spearman pooled  cell")
    for r in recs:
        sev = [c for c in r["cells"] if c["severity"] == "severe"]
        e0 = r["reference"]["nominal_error"]
        d_abs = sum(c["error_mean"] for c in sev) / len(sev) - e0
        g = r["gates"]
        print(f"  {r['config']['photons']:7.0f}   {d_abs:+43.4f}   {g['spearman_pooled']:+14.4f}"
              f"  {g['spearman_cell_means']:+.4f}")

    print()
    print("  photons   GATE 1   GATE 2")
    for r in recs:
        g = r["gates"]
        print(f"  {r['config']['photons']:7.0f}   "
              f"{'PASS' if g['gate1_phenomenon'] else 'FAIL':6s}   "
              f"{'PASS' if g['gate2_residual_insufficient'] else 'FAIL'}")

    print()
    print("Reading: a roughly constant ABSOLUTE error increase across photon levels")
    print("shows the aberration adds a fixed amount of structural error, while the")
    print("error RATIO is compressed at low photon counts purely because the noise")
    print("floor inflates the baseline -- which is why Gate 1 tests effect size.")
    print("A rising signal/floor with photon budget supports the noise-floor")
    print("mechanism; a flat one would refute it.")


if __name__ == "__main__":
    main()

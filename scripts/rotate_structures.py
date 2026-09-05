"""Rotate the held-out structure through all four, from cached features.

The content-shift result so far rests on a single held-out structure (CCPs),
which a reviewer would rightly call one lucky split. Rotating is free: features
are computed per sample, so only the *partition* changes. This re-partitions a
cached matrix rather than re-running any simulation.

Splits stay blocked by cell, and the model is refit for each rotation so no
information about the held-out structure reaches training or calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLITS = ("train", "cal", "test", "ood")


def load_flat(path: str) -> dict:
    """Concatenate the stored splits back into one labelled pool."""
    d = np.load(path, allow_pickle=True)
    out = {k: np.concatenate([d[f"{s}_{k}"] for s in SPLITS])
           for k in ("X", "e", "rms", "severity", "structure", "cell")}
    out["feature_names"] = [str(x) for x in d["feature_names"]]
    out["photons"] = float(d["photons"][0])
    return out


def partition(pool: dict, held_out: str, seed: int = 0) -> dict:
    """Cell-blocked train/cal/test partition with one structure held out."""
    is_ood = pool["structure"] == held_out
    cells = sorted(set(pool["cell"][~is_ood]))
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    n_tr, n_ca = int(0.6 * len(cells)), int(0.2 * len(cells))
    assign = {c: ("train" if i < n_tr else "cal" if i < n_tr + n_ca else "test")
              for i, c in enumerate(cells)}
    role = np.array([assign.get(c, "ood") for c in pool["cell"]])
    role[is_ood] = "ood"
    return {s: np.where(role == s)[0] for s in SPLITS}


def evaluate(pool: dict, idx: dict, alpha: float, seed: int = 0) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor

    X, e = pool["X"], pool["e"]
    q = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha, n_estimators=300,
                                  max_depth=3, learning_rate=0.05, random_state=seed)
    q.fit(X[idx["train"]], e[idx["train"]])
    scores = e[idx["cal"]] - q.predict(X[idx["cal"]])
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    corr = float(np.sort(scores)[min(max(k, 0), n - 1)])

    out = {}
    for split in ("test", "ood"):
        sel = idx[split]
        U = q.predict(X[sel]) + corr
        sev = pool["severity"][sel]
        out[split] = {
            "coverage": float((e[sel] <= U).mean()),
            "worst_group": min(float((e[sel][sev == g] <= U[sev == g]).mean())
                               for g in np.unique(sev)),
            "mean_width": float(U.mean()),
            "n": int(len(sel)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pool = load_flat(args.features)
    target = 1 - args.alpha
    structures = sorted(set(pool["structure"]))
    print(f"photons={pool['photons']:.0f}  {len(pool['e'])} samples  "
          f"nominal coverage {target:.0%}")
    print()
    print(f"  {'held out':14s} {'test cov':>9s} {'ood cov':>8s} {'ood worst':>10s} "
          f"{'ood width':>10s} {'n_ood':>6s}")
    results = {}
    for s in structures:
        idx = partition(pool, s)
        r = evaluate(pool, idx, args.alpha)
        results[s] = r
        print(f"  {s:14s} {r['test']['coverage']:9.3f} {r['ood']['coverage']:8.3f} "
              f"{r['ood']['worst_group']:10.3f} {r['ood']['mean_width']:10.4f} "
              f"{r['ood']['n']:6d}")

    cov = [results[s]["ood"]["coverage"] for s in structures]
    tst = [results[s]["test"]["coverage"] for s in structures]
    print()
    print(f"  in-distribution coverage : {np.mean(tst):.3f} +- {np.std(tst):.3f}")
    print(f"  held-out coverage        : {np.mean(cov):.3f} +- {np.std(cov):.3f}  "
          f"(range {min(cov):.3f} to {max(cov):.3f})")
    print()
    print("  A collapse present for every held-out structure, not just one, is what")
    print("  makes this a property of content shift rather than of one odd split.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

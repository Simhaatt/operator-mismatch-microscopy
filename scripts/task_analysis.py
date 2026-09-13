"""Conformal calibration of downstream task error.

Every coverage number in the paper calibrates a full-reference image metric.
Neither SSIM nor relative l2 is what a biologist reports: they report a number
extracted from the image, and the error that matters is the error in that
number. This script runs the same calibration-under-shift analysis against the
task readouts stored by ``extract_features.py --task-metrics``.

The motivation is empirical, not aesthetic. Phase 2 established that the two
full-reference metrics disagree about which specimen class is hard, by a factor
of 0.648 against 1.867 on the held-out structure. When two such metrics
disagree that strongly, neither can be assumed to proxy a task measurement.

Three readouts were extracted. Only two are used:

* ``puncta_count`` -- local maxima above threshold; what CCPs are imaged for.
* ``skeleton_length`` -- skeleton pixel count; what filaments are imaged for.
* ``branch_points`` -- **excluded by default.** Measured on the RL 100-photon
  cache, its median error is non-monotone across the severity ladder (0.223 at
  nominal, falling to 0.170 at moderate before rising to 0.940 at extreme) and
  its tail reaches a 10.7x relative error. A readout that does not increase with
  mismatch at low severity cannot be calibrated against mismatch, so reporting
  it would be reporting noise. ``--include-branch`` overrides this.

Rows whose reference measurement is zero carry a NaN task error -- relative
error is undefined there -- and are dropped explicitly rather than being
silently averaged in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import one_sided_correction  # noqa: E402
from scripts.phase2_analysis import shift_axes  # noqa: E402

DEFAULT_TASKS = ("puncta_count", "skeleton_length")
ALL_TASKS = ("puncta_count", "skeleton_length", "branch_points")
SCALE_FLOOR_FRAC = 0.10
MILD = ("nominal", "mild", "moderate")
ZERNIKE = ("defocus", "astigmatism", "coma", "spherical")


def load_task(path, tasks):
    d = np.load(path, allow_pickle=True)
    if "task_metrics" not in d.files or not bool(d["task_metrics"][0]):
        raise SystemExit(f"{path} was not extracted with --task-metrics")
    out = {}
    for split in ("train", "cal", "test", "ood"):
        rec = {
            "X": d[f"{split}_X"],
            "e_ssim": d[f"{split}_e"],
            "e_l2": d[f"{split}_e_l2"],
            "mag": d[f"{split}_mismatch_mag"],
            "severity": d[f"{split}_severity"],
            "mode": d[f"{split}_mode"],
            "structure": d[f"{split}_structure"],
        }
        for t in tasks:
            rec[f"task_{t}"] = d[f"{split}_task_{t}"]
        out[split] = rec
    meta = {
        "photons": float(d["photons"][0]),
        "reconstructor": str(d["reconstructor"][0]),
        "n_specs": int(d["n_specs"][0]),
    }
    return out, meta


def fit(train, key, seed=0):
    """Fit mean and scale for one error definition, dropping undefined rows."""
    from sklearn.ensemble import GradientBoostingRegressor

    y = train[key]
    ok = np.isfinite(y)
    X, y = train["X"][ok], y[ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    b = rng.permutation(n)[n // 2:]

    mu = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(X[b], y[b])
    resid = np.abs(y[b] - mu.predict(X[b]))
    sigma = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(X[b], resid)
    return mu, sigma, SCALE_FLOOR_FRAC * float(resid.mean())


def coverage(data, key, alpha, seeds):
    rows = {}
    for axis, (cal_mask, cal_split, test_split, test_mask) in shift_axes(data).items():
        per = []
        for seed in range(seeds):
            mu, sigma, floor = fit(data["train"], key, seed=seed)
            cal, test = data[cal_split], data[test_split]

            ce, cX = cal[key][cal_mask], cal["X"][cal_mask]
            ok = np.isfinite(ce)
            s = (ce[ok] - mu.predict(cX[ok])) / np.maximum(
                sigma.predict(cX[ok]), floor
            )
            q = one_sided_correction(s, alpha)

            te, tX = test[key][test_mask], test["X"][test_mask]
            ok = np.isfinite(te)
            te, tX = te[ok], tX[ok]
            U = mu.predict(tX) + np.maximum(sigma.predict(tX), floor) * q
            per.append((float((te <= U).mean()), float(U.mean()), int(len(te))))
        cov = np.array([p[0] for p in per])
        rows[axis] = {
            "coverage": (float(cov.mean()), float(cov.std())),
            "mean_bound": float(np.mean([p[1] for p in per])),
            "n": per[0][2],
        }
    return rows


def severity_profile(data, keys):
    """Median error per severity rung, pooled over the Zernike families."""
    tr = data["train"]
    zern = np.isin(tr["mode"], ZERNIKE) | (tr["mode"] == "none")
    out = {}
    for s in ("nominal", "mild", "moderate", "severe", "extreme"):
        sel = zern & (tr["severity"] == s)
        if not sel.any():
            continue
        row = {}
        for k in keys:
            v = tr[k][sel]
            v = v[np.isfinite(v)]
            row[k] = float(np.median(v)) if len(v) else float("nan")
        out[s] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--include-branch", action="store_true",
                    help="also report branch_points, which is non-monotone in "
                         "severity and is excluded by default")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = ALL_TASKS if args.include_branch else DEFAULT_TASKS
    data, meta = load_task(args.features, ALL_TASKS)
    keys = ["e_ssim", "e_l2"] + [f"task_{t}" for t in tasks]
    label = {"e_ssim": "1-SSIM", "e_l2": "rel l2",
             "task_puncta_count": "puncta count",
             "task_skeleton_length": "skeleton length",
             "task_branch_points": "branch points"}

    print(f"TASK-LEVEL CALIBRATION -- {meta['reconstructor']}, "
          f"photons={meta['photons']:.0f}, alpha={args.alpha}, {args.seeds} seeds")
    print()

    prof = severity_profile(data, keys)
    print("1. Does each error definition rise with mismatch severity?")
    print()
    print("   " + f"{'severity':10s}" + "".join(f"{label[k]:>16s}" for k in keys))
    base = prof.get("nominal", {})
    for s, row in prof.items():
        print("   " + f"{s:10s}" + "".join(f"{row[k]:16.4f}" for k in keys))
    if base:
        print("   " + f"{'fold rise':10s}" + "".join(
            f"{prof['extreme'][k]/base[k]:15.1f}x" if base.get(k) else f"{'-':>16s}"
            for k in keys))
    print()
    print("   The fold rise is the sensitivity of each readout to operator")
    print("   mismatch. They differ substantially, which is the point: how much")
    print("   mismatch matters depends on what is being measured.")
    print()

    print(f"2. Coverage under shift, per error definition (nominal "
          f"{1-args.alpha:.0%})")
    print()
    tabs = {k: coverage(data, k, args.alpha, args.seeds) for k in keys}
    axes = list(tabs[keys[0]])
    print("   " + f"{'shift axis':26s}" + "".join(f"{label[k]:>16s}" for k in keys))
    for ax in axes:
        cells = []
        for k in keys:
            c = tabs[k][ax]["coverage"]
            cells.append(f"{c[0]:9.3f}+/-{c[1]:<5.3f}")
        print("   " + f"{ax:26s}" + "".join(f"{c:>16s}" for c in cells))
    print()
    print("   Coverage is a property of the calibration, not of the metric, so")
    print("   in-distribution agreement is expected. Divergence under shift is")
    print("   the informative part.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "meta": meta,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "tasks_reported": list(tasks),
            "severity_profile": prof,
            "coverage": tabs,
        }, indent=1))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

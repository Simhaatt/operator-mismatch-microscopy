"""Feature-block ablation compared at matched coverage.

Comparing feature blocks by their correlation with mismatch, or by raw
coverage, is not a fair comparison: split conformal will hit its nominal
coverage with *any* score function, simply by widening the interval until it
does. A block that carries no information is not punished by loss of coverage;
it is punished by loss of **efficiency**. So the question a reliability paper
has to answer is not "which block covers?" but "which block covers at the
narrowest bound?"

This script fits the mean/scale predictor of :mod:`scripts.gate4_scaled` using
only the features in one block at a time, calibrates each at the same nominal
level, and reports the mean bound. Because every block is conformalised the
same way, all of them achieve nominal coverage in-distribution by construction,
and the mean bound is then directly comparable across blocks -- narrower is
strictly better at equal validity.

Two readings are reported:

* **in-distribution** -- coverage is matched by construction, so bound width is
  the whole story;
* **operator shift** -- coverage is *not* matched, because that is the failure
  under study; both coverage and width are reported so a block cannot look good
  merely by being vacuous.

The blocks follow ``BLOCK_SPECS`` in :mod:`scripts.gate5_bound`, so this
ablation and the bound analysis partition the features identically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import load, one_sided_correction  # noqa: E402
from scripts.gate5_bound import feature_blocks  # noqa: E402

MILD = ("nominal", "mild", "moderate")


def fit_on_columns(train, cols, seed=0):
    """Fit mean and scale using only ``cols``; alpha plays no part."""
    from sklearn.ensemble import GradientBoostingRegressor

    X = train["X"][:, cols]
    n = len(train["e"])
    rng = np.random.default_rng(seed)
    b = rng.permutation(n)[n // 2:]

    mu = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(X[b], train["e"][b])
    resid = np.abs(train["e"][b] - mu.predict(X[b]))
    sigma = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(X[b], resid)
    floor = 0.10 * float(resid.mean())
    return mu, sigma, floor


def bound_for(mu, sigma, floor, cols, X, q):
    s = np.maximum(sigma.predict(X[:, cols]), floor)
    return mu.predict(X[:, cols]) + s * q


def evaluate(data, names, alpha, seeds):
    blocks = feature_blocks(list(names))
    cal, test = data["cal"], data["test"]
    cm = np.isin(cal["severity"], MILD)
    sm = ~np.isin(test["severity"], MILD)

    out = {}
    for name, cols in blocks.items():
        cols = np.asarray(cols)
        per_seed = []
        for seed in range(seeds):
            mu, sigma, floor = fit_on_columns(data["train"], cols, seed=seed)

            # In-distribution: calibrate on everything, evaluate on everything.
            s_cal = (cal["e"] - mu.predict(cal["X"][:, cols])) / np.maximum(
                sigma.predict(cal["X"][:, cols]), floor
            )
            q = one_sided_correction(s_cal, alpha)
            U = bound_for(mu, sigma, floor, cols, test["X"], q)

            # Operator shift: calibrate on mild rungs, evaluate on severe ones.
            s_mild = (cal["e"][cm] - mu.predict(cal["X"][cm][:, cols])) / np.maximum(
                sigma.predict(cal["X"][cm][:, cols]), floor
            )
            q_mild = one_sided_correction(s_mild, alpha)
            U_op = bound_for(mu, sigma, floor, cols, test["X"][sm], q_mild)

            per_seed.append({
                "cov_in": float((test["e"] <= U).mean()),
                "width_in": float(U.mean()),
                "cov_op": float((test["e"][sm] <= U_op).mean()),
                "width_op": float(U_op.mean()),
            })

        def ms(key):
            v = np.array([r[key] for r in per_seed])
            return float(v.mean()), float(v.std())

        out[name] = {
            "n_features": int(len(cols)),
            "coverage_in": ms("cov_in"),
            "width_in": ms("width_in"),
            "coverage_op": ms("cov_op"),
            "width_op": ms("width_op"),
            "seeds": seeds,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, names, photons = load(args.features)
    res = evaluate(data, names, args.alpha, args.seeds)

    print(f"FEATURE ABLATION AT MATCHED COVERAGE -- photons={photons:.0f}, "
          f"alpha={args.alpha}, {args.seeds} seeds")
    print()
    print("  In-distribution coverage is matched by conformal construction, so")
    print("  the bound width is the comparison. Narrower is better.")
    print()
    print(f"  {'block':12s} {'k':>3s} {'cov_in':>14s} {'width_in':>15s} "
          f"{'cov_op':>14s} {'width_op':>15s}")
    order = sorted(res, key=lambda b: res[b]["width_in"][0])
    for b in order:
        r = res[b]
        print(f"  {b:12s} {r['n_features']:3d} "
              f"{r['coverage_in'][0]:8.3f}+/-{r['coverage_in'][1]:<5.3f} "
              f"{r['width_in'][0]:9.4f}+/-{r['width_in'][1]:<5.4f} "
              f"{r['coverage_op'][0]:8.3f}+/-{r['coverage_op'][1]:<5.3f} "
              f"{r['width_op'][0]:9.4f}+/-{r['width_op'][1]:<5.4f}")
    print()
    best = order[0]
    worst = order[-1]
    gain = 100 * (1 - res[best]["width_in"][0] / res[worst]["width_in"][0])
    print(f"  narrowest: '{best}' ({res[best]['n_features']} features)")
    print(f"  widest:    '{worst}' ({res[worst]['n_features']} features)")
    print(f"  '{best}' is {gain:.1f}% narrower than '{worst}' at matched coverage")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"photons": photons, "alpha": args.alpha, "seeds": args.seeds,
             "blocks": res}, indent=1))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

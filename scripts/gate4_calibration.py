"""Gate 4: does shift-aware calibration repair coverage?

Gate 3 established the failure this gate must fix. Conformal bounds built the
standard way collapse to 0.27-0.41 coverage against a nominal 0.90 once the test
distribution shifts. That is the gap the literature leaves open: every conformal
imaging method assumes calibration and test are exchangeable.

Three shift axes are evaluated, each a realistic way a microscope departs from
the conditions a method was calibrated under:

* **operator shift** -- calibrate on nominal/mild/moderate aberration, test on
  severe/extreme. The optics drift beyond anything seen at calibration.
* **content shift** -- calibrate on three structures, test on a fourth. A new
  specimen type.
* **combined** -- both at once, the honest worst case.

Against three calibration schemes:

1. **Split conformal** -- one global quantile. The standard baseline.
2. **Mondrian (group-conditional)** -- a separate quantile per stratum of an
   estimated mismatch score. Guaranteed per stratum by construction, provided
   the strata are defined without peeking at the label.
3. **RLCP** -- randomised localised conformal prediction (Hore & Barber,
   arXiv:2310.07850), which weights calibration points by kernel proximity in
   the mismatch score. Deterministic localisation is *not* used: Guan's LCP has
   marginal coverage that "may be arbitrarily high or arbitrarily low", and the
   randomisation is what restores a guarantee.

The mismatch score is the physics contribution. It is fit on a **disjoint half**
of the training split, so it is not circular with the error predictor, and it is
computed from blind features only -- the true wavefront error is used solely to
check how well the score tracks reality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def one_sided_correction(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal correction: the ceil((n+1)(1-alpha)) order stat."""
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(np.sort(scores)[min(max(k, 0), n - 1)])


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Weighted conformal quantile with the test point's mass at +infinity.

    Including the test point's own weight is what makes the bound valid rather
    than merely plausible: without it the quantile is taken over a sample that
    excludes the very point being predicted.
    """
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    total = w.sum()
    if total <= 0:
        return float(np.max(scores)) if len(scores) else float("inf")
    # The test point carries the residual mass; give it the largest score slot.
    w_test = 1.0 / (len(scores) + 1)
    w = w / total * (1.0 - w_test)
    cum = np.cumsum(w)
    idx = np.searchsorted(cum, 1 - alpha)
    return float(s[min(idx, len(s) - 1)]) if idx < len(s) else float("inf")


def split_conformal(sc, st, m_cal, m_test, alpha, **kw):
    return np.full(len(st), one_sided_correction(sc, alpha))


def mondrian(sc, st, m_cal, m_test, alpha, n_bins=4, **kw):
    """Group-conditional: one quantile per mismatch stratum.

    Bin edges come from the calibration distribution of the mismatch score, so
    the partition never sees a test label.
    """
    edges = np.quantile(m_cal, np.linspace(0, 1, n_bins + 1)[1:-1])
    cal_bin = np.digitize(m_cal, edges)
    test_bin = np.digitize(m_test, edges)
    global_corr = one_sided_correction(sc, alpha)
    corr = np.empty(len(st))
    for b in range(n_bins):
        sel = cal_bin == b
        # Fall back to the global quantile when a stratum is too thin to be valid.
        c = one_sided_correction(sc[sel], alpha) if sel.sum() >= 20 else global_corr
        corr[test_bin == b] = c
    return corr


def rlcp(sc, st, m_cal, m_test, alpha, bandwidth=None, seed=0, **kw):
    """Randomised localised conformal prediction, localised on the mismatch score."""
    rng = np.random.default_rng(seed)
    h = bandwidth if bandwidth is not None else 0.5 * np.std(m_cal)
    h = max(float(h), 1e-6)
    corr = np.empty(len(m_test))
    for i, m in enumerate(m_test):
        # Randomised localisation: sample the anchor near the test point.
        anchor = m + rng.normal(0.0, h)
        w = np.exp(-0.5 * ((m_cal - anchor) / h) ** 2)
        corr[i] = weighted_quantile(sc, w, alpha)
    return corr


SCHEMES = {
    "split_conformal": split_conformal,
    "mondrian": mondrian,
    "rlcp": rlcp,
}


def load(path):
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    out = {}
    for split in ("train", "cal", "test", "ood"):
        out[split] = {
            "X": d[f"{split}_X"], "e": d[f"{split}_e"], "rms": d[f"{split}_rms"],
            "severity": d[f"{split}_severity"], "structure": d[f"{split}_structure"],
        }
    return out, names, float(d["photons"][0])


def fit_models(train, alpha, seed=0):
    """Fit the mismatch score and the error quantile model on disjoint halves.

    Splitting the data rather than the feature set avoids circularity without
    crippling either model: the localiser and the predictor may use the same
    blind features, but never the same samples.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(train["e"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2 :]

    mismatch = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][a], train["rms"][a])

    quantile = GradientBoostingRegressor(
        loss="quantile", alpha=1 - alpha, n_estimators=300,
        max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][b], train["e"][b])
    return mismatch, quantile


def coverage_report(e, U, groups):
    cov = float((e <= U).mean())
    per = {str(g): float((e[groups == g] <= U[groups == g]).mean())
           for g in np.unique(groups)}
    return {
        "coverage": cov,
        "coverage_error": abs(cov - COVER_TARGET),
        "worst_group": min(per.values()),
        "per_group": per,
        "mean_width": float(U.mean()),
        "n": int(len(e)),
    }


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


MILD = ("nominal", "mild", "moderate")
SEVERE = ("severe", "extreme")
COVER_TARGET = 0.9


def build_axes(data):
    """Calibration / evaluation pairs for each shift axis."""
    cal, test, ood = data["cal"], data["test"], data["ood"]

    def sub(d, mask):
        return {k: v[mask] for k, v in d.items()}

    cal_mild = sub(cal, np.isin(cal["severity"], MILD))
    test_sev = sub(test, np.isin(test["severity"], SEVERE))
    ood_sev = sub(ood, np.isin(ood["severity"], SEVERE))
    return {
        "none (in-distribution)": (cal, test),
        "operator shift": (cal_mild, test_sev),
        "content shift": (cal, ood),
        "combined": (cal_mild, ood_sev),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global COVER_TARGET
    COVER_TARGET = 1 - args.alpha

    data, names, photons = load(args.features)
    mismatch, quantile = fit_models(data["train"], args.alpha)

    # How well does the blind mismatch score track the true wavefront error?
    for split in ("test", "ood"):
        m = mismatch.predict(data[split]["X"])
        print(f"  mismatch score vs true RMS wavefront error, {split}: "
              f"rho = {spearman(m, data[split]['rms']):+.3f}")
    print()

    axes = build_axes(data)
    results: dict[str, dict] = {}
    print(f"GATE 4 -- coverage under shift, photons={photons:.0f}, "
          f"nominal {COVER_TARGET:.0%}")
    print()
    print(f"  {'shift axis':22s} {'scheme':16s} {'coverage':>9s} {'|err|':>7s} "
          f"{'worst-grp':>10s} {'width':>8s}   n")
    for axis, (cal, test) in axes.items():
        sc = cal["e"] - quantile.predict(cal["X"])
        st = test["e"] - quantile.predict(test["X"])
        m_cal = mismatch.predict(cal["X"])
        m_test = mismatch.predict(test["X"])
        base = quantile.predict(test["X"])
        results[axis] = {}
        for name, fn in SCHEMES.items():
            corr = fn(sc, st, m_cal, m_test, args.alpha)
            rep = coverage_report(test["e"], base + corr, test["severity"])
            results[axis][name] = rep
            print(f"  {axis:22s} {name:16s} {rep['coverage']:9.3f} "
                  f"{rep['coverage_error']:7.3f} {rep['worst_group']:10.3f} "
                  f"{rep['mean_width']:8.4f}   {rep['n']}")
        print()

    print("  Reading: split conformal is the literature baseline. A scheme earns its")
    print("  place by pulling coverage back toward nominal under shift WITHOUT")
    print("  inflating the bound so much that it becomes vacuous.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"photons": photons, "alpha": args.alpha, "results": results},
            indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

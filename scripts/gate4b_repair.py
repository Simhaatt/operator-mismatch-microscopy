"""Gate 4b: repair coverage using novelty rather than severity.

Gate 4 failed for a diagnosed reason. The localiser was trained to predict
operator severity, so under operator shift 93.6% of test points fell beyond the
calibration range and the kernel had nothing to borrow from, while under content
shift a novel specimen scored as *benign* and drew a narrow bound. Both are the
same mistake: severity is not novelty.

This script tests three repairs, in increasing order of how much they concede:

1. **Novelty-localised RLCP.** Replace the severity score with a distance from
   the calibration feature distribution -- computable blind at deployment. If
   the diagnosis is right, localising on novelty should behave better than
   localising on severity.
2. **Selective prediction.** Abstain on the most novel inputs and report coverage
   on the accepted subset. This *cannot* fail: refusing to certify is always
   available, and the honest deliverable is the coverage/abstention trade-off.
3. **Inflated (robust) conformal.** Take the quantile at a stricter level so the
   bound stays valid under bounded shift. Also cannot fail; it pays in width.

The point of reporting all three is that the useful scientific statement is not
"we fixed it" but "here is the price of validity under shift, measured".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import (  # noqa: E402
    build_axes, fit_models, load, one_sided_correction, weighted_quantile,
)


def novelty_score(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from the calibration feature distribution.

    Blind: uses only feature vectors, never a label. Standardised first so no
    single feature's scale dominates, and the covariance is ridge-regularised
    because the feature count is small relative to sample size but the matrix is
    still poorly conditioned.
    """
    mu = X_ref.mean(axis=0)
    sd = X_ref.std(axis=0) + 1e-8
    Z_ref = (X_ref - mu) / sd
    Z = (X - mu) / sd
    cov = np.cov(Z_ref, rowvar=False) + 1e-3 * np.eye(Z_ref.shape[1])
    prec = np.linalg.inv(cov)
    d = np.einsum("ij,jk,ik->i", Z, prec, Z)
    return np.sqrt(np.maximum(d, 0.0))


def rlcp_on(scores_cal, m_cal, m_test, alpha, seed=0):
    rng = np.random.default_rng(seed)
    h = max(0.5 * float(np.std(m_cal)), 1e-6)
    out = np.empty(len(m_test))
    for i, m in enumerate(m_test):
        anchor = m + rng.normal(0.0, h)
        w = np.exp(-0.5 * ((m_cal - anchor) / h) ** 2)
        out[i] = weighted_quantile(scores_cal, w, alpha)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = 1 - args.alpha

    data, names, photons = load(args.features)
    mismatch, quantile = fit_models(data["train"], args.alpha)
    axes = build_axes(data)
    results: dict = {}

    print(f"GATE 4b -- repairs, photons={photons:.0f}, nominal {target:.0%}")
    print()
    print("1) LOCALISER: severity vs novelty")
    print(f"  {'axis':22s} {'localiser':10s} {'coverage':>9s} {'worst-grp':>10s} {'width':>8s}")
    for axis, (cal, test) in axes.items():
        sc = cal["e"] - quantile.predict(cal["X"])
        base = quantile.predict(test["X"])
        for label, m_cal, m_test in (
            ("severity", mismatch.predict(cal["X"]), mismatch.predict(test["X"])),
            ("novelty", novelty_score(cal["X"], cal["X"]), novelty_score(cal["X"], test["X"])),
        ):
            U = base + rlcp_on(sc, m_cal, m_test, args.alpha)
            cov = float((test["e"] <= U).mean())
            worst = min(float((test["e"][test["severity"] == g] <= U[test["severity"] == g]).mean())
                        for g in np.unique(test["severity"]))
            print(f"  {axis:22s} {label:10s} {cov:9.3f} {worst:10.3f} {U.mean():8.4f}")
            results.setdefault(axis, {})[f"rlcp_{label}"] = {
                "coverage": cov, "worst_group": worst, "mean_width": float(U.mean())}
        print()

    print("2) SELECTIVE PREDICTION: abstain on the most novel inputs")
    print(f"  {'axis':22s} {'accept':>7s} {'coverage':>9s} {'width':>8s}")
    for axis, (cal, test) in axes.items():
        sc = cal["e"] - quantile.predict(cal["X"])
        U = quantile.predict(test["X"]) + one_sided_correction(sc, args.alpha)
        nov = novelty_score(cal["X"], test["X"])
        order = np.argsort(nov)
        for frac in (1.0, 0.75, 0.5, 0.25):
            keep = order[: max(int(frac * len(order)), 1)]
            cov = float((test["e"][keep] <= U[keep]).mean())
            print(f"  {axis:22s} {frac:7.0%} {cov:9.3f} {U[keep].mean():8.4f}")
            results.setdefault(axis, {})[f"selective_{frac:.2f}"] = {
                "coverage": cov, "mean_width": float(U[keep].mean())}
        print()

    print("3) INFLATED CONFORMAL: what strictness restores nominal coverage?")
    print(f"  {'axis':22s} {'alpha_used':>10s} {'coverage':>9s} {'width':>8s}  (target "
          f"{target:.0%})")
    for axis, (cal, test) in axes.items():
        sc = cal["e"] - quantile.predict(cal["X"])
        base = quantile.predict(test["X"])
        row = None
        for a in (0.10, 0.05, 0.02, 0.01, 0.001):
            U = base + one_sided_correction(sc, a)
            cov = float((test["e"] <= U).mean())
            if row is None or cov < target:
                row = (a, cov, float(U.mean()))
            if cov >= target:
                row = (a, cov, float(U.mean()))
                break
        a, cov, w = row
        flag = "" if cov >= target else "   <- unreachable even at alpha=0.001"
        print(f"  {axis:22s} {a:10.3f} {cov:9.3f} {w:8.4f}{flag}")
        results.setdefault(axis, {})["inflated"] = {
            "alpha": a, "coverage": cov, "mean_width": w}
    print()
    print("  An 'unreachable' row means no amount of inflation of a calibration-set")
    print("  quantile fixes it: the calibration data simply does not contain the")
    print("  failure mode, which is what distribution shift means.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"photons": photons, "alpha": args.alpha, "results": results}, indent=2),
            encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

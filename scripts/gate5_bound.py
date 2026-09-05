"""Gate 5: bound the coverage loss, and estimate it blind.

The rotation experiment produced a spread that demands an explanation: held-out
coverage ranges from 0.29 to 0.97 depending only on which structure is held out.
A usable theory has to say *which* shifts are dangerous, before seeing the labels.

**The bound.** Split conformal fixes a threshold at the (1-alpha) quantile of the
calibration conformity scores. If calibration scores are drawn from P and test
scores from Q, then for the acceptance event A = {s <= q},

    | Q(A) - P(A) |  <=  d_TV(P, Q)

by the definition of total variation distance, and since P(A) >= 1 - alpha by the
usual exchangeable argument,

    coverage_test  >=  1 - alpha - d_TV(P, Q).                        (*)

This is elementary but it is the right statement for this setting: the shift that
matters is in the *conformity score* distribution, not in the images or the
features. It predicts that a large content shift which happens not to move the
score distribution is harmless, which is exactly what holding out ER does.

**The blind estimator.** Evaluating (*) needs test scores, which need ground
truth. So we also test whether a label-free proxy -- the divergence between
calibration and test *novelty* distributions -- predicts the realised coverage
loss. If it does, a practitioner gets a warning signal at deployment rather than
a post-hoc explanation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rotate_structures import load_flat, partition  # noqa: E402


def tv_distance(a: np.ndarray, b: np.ndarray, n_bins: int = 40) -> float:
    """Empirical total-variation distance between two 1-D samples."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=False)
    pb, _ = np.histogram(b, bins=edges, density=False)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def directional_excess(cal: np.ndarray, test: np.ndarray) -> float:
    """One-sided score-distribution shift: excess test mass in the upper tail.

    Total variation is the wrong quantity here, and measurably so. It is
    symmetric, but coverage loss is not: a shift that moves test conformity
    scores *downward* makes the calibration threshold conservative and coverage
    goes up. Only mass crossing above the threshold costs coverage.

    So we take the one-sided supremum

        d+ = sup_t [ Q(s > t) - P(s > t) ]_+

    which is zero for any benign shift, however large, and equals the coverage
    loss at the threshold actually used. The symmetric version reports a large
    shift for a harmless one -- e.g. holding out ER gives d_TV = 0.63 while
    coverage is 0.97.
    """
    grid = np.unique(np.concatenate([cal, test]))
    sc, st = np.sort(cal), np.sort(test)
    surv_cal = 1.0 - np.searchsorted(sc, grid, side="right") / len(sc)
    surv_test = 1.0 - np.searchsorted(st, grid, side="right") / len(st)
    return float(np.maximum(surv_test - surv_cal, 0.0).max())


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic, a lower bound on total variation."""
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def novelty(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    mu, sd = X_ref.mean(0), X_ref.std(0) + 1e-8
    Zr, Z = (X_ref - mu) / sd, (X - mu) / sd
    prec = np.linalg.inv(np.cov(Zr, rowvar=False) + 1e-3 * np.eye(Zr.shape[1]))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, prec, Z), 0.0))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def run_cell(pool, held_out, alpha, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor

    idx = partition(pool, held_out, seed=seed)
    X, e = pool["X"], pool["e"]
    q = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha, n_estimators=300,
                                  max_depth=3, learning_rate=0.05, random_state=seed)
    q.fit(X[idx["train"]], e[idx["train"]])

    s_cal = e[idx["cal"]] - q.predict(X[idx["cal"]])
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    corr = float(np.sort(s_cal)[min(max(k, 0), n - 1)])

    out = {}
    for split in ("test", "ood"):
        sel = idx[split]
        s = e[sel] - q.predict(X[sel])
        cov = float((s <= corr).mean())
        # Bound (*) : coverage >= 1 - alpha - d_TV between score distributions.
        tv = tv_distance(s_cal, s)
        ks = ks_distance(s_cal, s)
        dplus = directional_excess(s_cal, s)
        # Label-free proxy: divergence of the novelty distributions.
        nv_cal = novelty(X[idx["cal"]], X[idx["cal"]])
        nv_tst = novelty(X[idx["cal"]], X[sel])
        out[split] = {
            "coverage": cov,
            "tv_score": tv,
            "ks_score": ks,
            "bound": 1 - alpha - tv,
            "slack": cov - (1 - alpha - tv),
            "d_plus": dplus,
            "bound_1s": 1 - alpha - dplus,
            "slack_1s": cov - (1 - alpha - dplus),
            "tv_novelty_blind": tv_distance(nv_cal, nv_tst),
            "dplus_novelty_blind": directional_excess(nv_cal, nv_tst),
            "ks_novelty_blind": ks_distance(nv_cal, nv_tst),
            "n": int(len(sel)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = 1 - args.alpha

    rows, results = [], {}
    print(f"GATE 5 -- coverage bound  cov >= 1 - alpha - d_TV(score),  nominal {target:.0%}")
    print()
    print(f"  {'config':22s} {'held out':14s} {'cov':>6s} {'d_TV':>6s} {'d+':>6s} "
          f"{'bound1s':>8s} {'slack1s':>8s} {'blind d+':>9s}")
    for path in args.features:
        pool = load_flat(path)
        tag = Path(path).stem
        for s in sorted(set(pool["structure"])):
            r = run_cell(pool, s, args.alpha)["ood"]
            results.setdefault(tag, {})[s] = r
            rows.append(r)
            ok = "" if r["slack_1s"] >= 0 else "   <- BOUND VIOLATED"
            print(f"  {tag:22s} {s:14s} {r['coverage']:6.3f} {r['tv_score']:6.3f} "
                  f"{r['d_plus']:6.3f} {r['bound_1s']:8.3f} {r['slack_1s']:+8.3f} "
                  f"{r['dplus_novelty_blind']:9.3f}{ok}")
        print()

    cov = np.array([r["coverage"] for r in rows])
    tv = np.array([r["tv_score"] for r in rows])
    dp = np.array([r["d_plus"] for r in rows])
    blind_tv = np.array([r["tv_novelty_blind"] for r in rows])
    blind_dp = np.array([r["dplus_novelty_blind"] for r in rows])
    slack = np.array([r["slack"] for r in rows])
    slack1 = np.array([r["slack_1s"] for r in rows])

    print(f"  cells: {len(rows)}")
    print(f"  symmetric bound holds {int((slack >= 0).sum())}/{len(rows)}, "
          f"mean slack {slack.mean():+.3f}")
    print(f"  one-sided bound holds {int((slack1 >= 0).sum())}/{len(rows)}, "
          f"mean slack {slack1.mean():+.3f}")
    print()
    print("  Which quantity explains the spread in coverage?")
    print(f"    rho(coverage, -d_TV)      = {spearman(cov, -tv):+.3f}   symmetric")
    print(f"    rho(coverage, -d+)        = {spearman(cov, -dp):+.3f}   one-sided")
    print()
    print("  Can it be anticipated WITHOUT labels?")
    print(f"    rho(coverage, -blind d_TV) = {spearman(cov, -blind_tv):+.3f}")
    print(f"    rho(coverage, -blind d+)   = {spearman(cov, -blind_dp):+.3f}")
    print(f"    rho(d+, blind d+)          = {spearman(dp, blind_dp):+.3f}")
    print()
    print("  The first says the bound is the right explanation; the second and third")
    print("  say whether a practitioner could see the danger coming at deployment,")
    print("  with no ground truth, which is the only version that is actually usable.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

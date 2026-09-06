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

from src.metrics.stats import spearman  # noqa: E402
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


def novelty(X_ref: np.ndarray, X: np.ndarray, cols=None) -> np.ndarray:
    """Mahalanobis novelty of ``X`` against the reference cloud ``X_ref``.

    ``cols`` restricts the computation to a block of feature columns. That turns
    one blind score into a family of them, which is the point: the OTF block is
    physics-derived and built to detect operator mismatch from the residual
    spectrum alone, so it should predict coverage loss better than a generic
    distance over all 19 features, several of which merely track content.
    """
    if cols is not None:
        X_ref, X = X_ref[:, cols], X[:, cols]
    mu, sd = X_ref.mean(0), X_ref.std(0) + 1e-8
    Zr, Z = (X_ref - mu) / sd, (X - mu) / sd
    # np.cov collapses to a scalar for one column; atleast_2d keeps inv valid.
    cov = np.atleast_2d(np.cov(Zr, rowvar=False))
    prec = np.linalg.inv(cov + 1e-3 * np.eye(cov.shape[0]))
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, prec, Z), 0.0))


# Which columns each blind estimator is allowed to see. "otf" is the block added
# in commit 23 and never evaluated until now; "residual" is the Gate 2 baseline.
BLOCK_SPECS = {
    "all": None,
    "otf": ("otf_white_chi2", "otf_excess_inband", "otf_excess_at_cut",
            "otf_slope", "otf_noise_ref"),
    "oscr": "complement-of-otf",
    "residual": ("r_nominal",),
    # A sub-block dropping the two OTF features that carry no mismatch signal.
    # Measured against the oracle RMS wavefront error on the RL 100-photon
    # cache, otf_slope reaches |rho| 0.77 and otf_excess_at_cut 0.68, while
    # otf_excess_inband and otf_noise_ref sit below 0.09 everywhere. Two dead
    # directions dilute a 5-feature Mahalanobis distance, so this asks whether
    # the block does better without them.
    #
    # CAVEAT, and it matters: those two were identified by their correlation
    # with the oracle, so "otf_core" is selected on the label and its score
    # here is optimistic. It is a hypothesis to be confirmed on the GPU caches,
    # not a result to quote. The a-priori reading that motivates it is that
    # otf_noise_ref is the out-of-band *normaliser* (it tracks the photon
    # budget, not the operator) and otf_excess_inband is driven by specimen
    # content -- neither is a mismatch discriminator by construction.
    "otf_core": ("otf_white_chi2", "otf_excess_at_cut", "otf_slope"),
}


def feature_blocks(names: list[str]) -> dict[str, np.ndarray]:
    """Resolve :data:`BLOCK_SPECS` against the cache's own feature names.

    Blocks whose features are absent are dropped rather than faked, so a
    14-feature cache silently reports only the blocks it can support.
    """
    pos = {n: i for i, n in enumerate(names)}
    otf = [pos[n] for n in BLOCK_SPECS["otf"] if n in pos]
    out: dict[str, np.ndarray] = {"all": np.arange(len(names))}
    if otf:
        out["otf"] = np.asarray(otf)
        out["oscr"] = np.asarray([i for i in range(len(names)) if i not in set(otf)])
        core = [pos[n] for n in BLOCK_SPECS["otf_core"] if n in pos]
        if len(core) == len(BLOCK_SPECS["otf_core"]):
            out["otf_core"] = np.asarray(core)
    res = [pos[n] for n in BLOCK_SPECS["residual"] if n in pos]
    if res:
        out["residual"] = np.asarray(res)
    return out


def run_cell(pool, held_out, alpha, seed=0, blocks=None):
    from sklearn.ensemble import GradientBoostingRegressor

    if blocks is None:
        blocks = feature_blocks(pool["feature_names"])
    feature_index = {n: i for i, n in enumerate(pool["feature_names"])}
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
        # Label-free proxy: divergence of the novelty distributions. Computed
        # once per feature block so the blocks can be compared on equal footing
        # -- same partition, same fitted quantile model, same realised coverage.
        blind = {}
        # The model's OWN predicted quantile, as a blind statistic. Step 2
        # showed that piling features into a Mahalanobis novelty distance makes
        # the blind proxy worse, so the useful direction is a *low-dimensional*
        # statistic rather than more features. q.predict(X) is the natural one:
        # it is one number, it needs no labels at deployment, and it is already
        # trained to track the error the bound is about -- where a novelty
        # distance only measures how unusual the input looks, which Gate 4b
        # showed is the wrong question ("the dangerous cases look familiar").
        # Direction matters and is fixed a priori by the mechanism, not fitted:
        # under-coverage comes from intervals that are too NARROW, so the danger
        # signal is the predicted quantile shifting DOWN relative to
        # calibration, not up. A shift up widens the intervals and coverage
        # rises. Taking the excess the other way round gives a statistic that is
        # anti-correlated with coverage by construction, which is what the
        # first run of this showed (-0.800).
        qp_cal, qp = q.predict(X[idx["cal"]]), q.predict(X[sel])
        blind["qpred"] = {"tv": tv_distance(qp_cal, qp),
                          "d_plus": directional_excess(qp, qp_cal),
                          "ks": ks_distance(qp_cal, qp)}
        # A blind stand-in for the conformity score itself. The real score is
        # s = e - q(x), and only e needs a label. r_nominal is the measurement
        # residual under the assumed operator -- the one quantity that is both
        # observable at deployment and monotone in error. Standardising both on
        # the CALIBRATION split (never on test, which would leak) puts them on a
        # common scale so the difference is meaningful:
        #
        #     s_blind = z(r_nominal) - z(q(x))
        #
        # This is the natural construction rather than a search: it mirrors the
        # labelled score term for term. Whether it beats r_nominal alone is the
        # open question -- subtracting q could either remove the part of the
        # residual the model already anticipates, or just add the model's noise.
        ri = feature_index.get("r_nominal")
        if ri is not None:
            r_cal, r_tst = X[idx["cal"]][:, ri], X[sel][:, ri]
            mu_r, sd_r = r_cal.mean(), r_cal.std() + 1e-12
            mu_q, sd_q = qp_cal.mean(), qp_cal.std() + 1e-12
            sb_cal = (r_cal - mu_r) / sd_r - (qp_cal - mu_q) / sd_q
            sb_tst = (r_tst - mu_r) / sd_r - (qp - mu_q) / sd_q
            blind["blind_score"] = {"tv": tv_distance(sb_cal, sb_tst),
                                    "d_plus": directional_excess(sb_cal, sb_tst),
                                    "ks": ks_distance(sb_cal, sb_tst)}
        for name, cols in blocks.items():
            nv_cal = novelty(X[idx["cal"]], X[idx["cal"]], cols)
            nv_tst = novelty(X[idx["cal"]], X[sel], cols)
            blind[name] = {
                "tv": tv_distance(nv_cal, nv_tst),
                "d_plus": directional_excess(nv_cal, nv_tst),
                "ks": ks_distance(nv_cal, nv_tst),
            }
        nv_cal = novelty(X[idx["cal"]], X[idx["cal"]])
        nv_tst = novelty(X[idx["cal"]], X[sel])
        out[split] = {
            "blind_blocks": blind,
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


def collect_rows(features, alpha, seed):
    """Every (cache, held-out structure) cell for one partition seed."""
    rows, block_names, nfeat = [], [], {}
    for path in features:
        pool = load_flat(path)
        blocks = feature_blocks(pool["feature_names"])
        for b, cols in blocks.items():
            if b not in block_names:
                block_names.append(b)
            nfeat[b] = len(cols)
        for extra in ("qpred", "blind_score"):
            if extra not in block_names:
                block_names.append(extra)
        tag = Path(path).stem
        for st in sorted(set(pool["structure"])):
            r = run_cell(pool, st, alpha, seed=seed, blocks=blocks)["ood"]
            r = dict(r, tag=tag, structure=st)
            rows.append(r)
    return rows, block_names, nfeat


def key_stats(rows, block_names):
    """The handful of numbers the paper quotes, for one seed."""
    g = lambda k: np.array([r[k] for r in rows])
    cov, dp, tv = g("coverage"), g("d_plus"), g("tv_score")
    out = {
        "rho_cov_dplus": spearman(cov, -dp),
        "rho_cov_tv": spearman(cov, -tv),
        "holds_1s": int((g("slack_1s") >= 0).sum()),
        "n_cells": len(rows),
        "mean_slack_1s": float(g("slack_1s").mean()),
    }
    for name in block_names:
        got = [r for r in rows if name in r.get("blind_blocks", {})]
        if len(got) == len(rows):
            c = np.array([r["coverage"] for r in got])
            b = np.array([r["blind_blocks"][name]["d_plus"] for r in got])
            out[f"blind_{name}"] = spearman(c, -b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of partition seeds, 0..N-1. Seed 0 alone is "
                         "one split; conformal coverage is split-sensitive, so "
                         "more than one is what separates a result from an "
                         "anecdote.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = 1 - args.alpha

    rows, results = [], {}
    print(f"GATE 5 -- coverage bound  cov >= 1 - alpha - d_TV(score),  nominal {target:.0%}")
    print()
    print(f"  {'config':22s} {'held out':14s} {'cov':>6s} {'d_TV':>6s} {'d+':>6s} "
          f"{'bound1s':>8s} {'slack1s':>8s} {'blind d+':>9s}")
    block_names: list[str] = []
    nfeat: dict[str, int] = {}
    for path in args.features:
        pool = load_flat(path)
        blocks = feature_blocks(pool["feature_names"])
        for b, cols in blocks.items():
            if b not in block_names:
                block_names.append(b)
            nfeat[b] = len(cols)
        for extra in ("qpred", "blind_score"):
            if extra not in block_names:
                block_names.append(extra)
        tag = Path(path).stem
        for s in sorted(set(pool["structure"])):
            r = run_cell(pool, s, args.alpha, blocks=blocks)["ood"]
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
    print("  Which BLOCK of features makes the best blind estimator?")
    print("  The labelled bound is tight; the blind proxy is what limits the")
    print("  contribution, so this is the number that matters.")
    print()
    # n_cells matters: caches predating commit 23 carry no OTF block, so a block
    # can be supported by only part of the grid. Reporting rho over 4 cells as if
    # it were over 16 is exactly the error the standing rule forbids.
    print(f"    {'block':10s} {'n_feat':>6s} {'cells':>6s} {'rho(cov,-blind d+)':>20s} "
          f"{'rho(cov,-blind dTV)':>21s} {'rho(d+,blind d+)':>18s}")
    block_stats = {}
    for name in block_names:
        got = [r for r in rows if name in r.get("blind_blocks", {})]
        if len(got) < 3:
            continue
        c = np.array([r["coverage"] for r in got])
        d = np.array([r["d_plus"] for r in got])
        bdp = np.array([r["blind_blocks"][name]["d_plus"] for r in got])
        btv = np.array([r["blind_blocks"][name]["tv"] for r in got])
        st = {
            "n_cells": len(got),
            "n_features": int(nfeat.get(name, 0)),
            "rho_cov_dplus": spearman(c, -bdp),
            "rho_cov_tv": spearman(c, -btv),
            "rho_dplus_dplus": spearman(d, bdp),
        }
        block_stats[name] = st
        warn = "" if st["n_cells"] == len(rows) else "   <- PARTIAL GRID"
        print(f"    {name:10s} {st['n_features']:6d} {st['n_cells']:6d} "
              f"{st['rho_cov_dplus']:+20.3f} {st['rho_cov_tv']:+21.3f} "
              f"{st['rho_dplus_dplus']:+18.3f}{warn}")
    print()
    full = {k: v for k, v in block_stats.items() if v["n_cells"] == len(rows)}
    if full:
        best = max(full, key=lambda k: full[k]["rho_cov_dplus"])
        block_stats_full = full
        ref = full.get("all", {}).get("rho_cov_dplus")
        msg = f"    best blind block: {best!r} at rho {full[best]['rho_cov_dplus']:+.3f}"
        if ref is not None and best != "all":
            msg += f"  (vs {ref:+.3f} for the full feature set)"
        print(msg)
        partial = [k for k in block_stats if k not in full]
        if partial:
            print(f"    NOT ranked -- only part of the grid supports them: "
                  f"{', '.join(sorted(partial))}")
    print()
    print("  The first says the bound is the right explanation; the second and third")
    print("  say whether a practitioner could see the danger coming at deployment,")
    print("  with no ground truth, which is the only version that is actually usable.")

    seed_stats = None
    if args.seeds > 1:
        per_seed = [key_stats(rows, block_names)]
        for k in range(1, args.seeds):
            rk, bnk, _ = collect_rows(args.features, args.alpha, k)
            per_seed.append(key_stats(rk, bnk))
        keys = [k for k in per_seed[0] if k not in ("n_cells",)]
        seed_stats = {k: (float(np.mean([p[k] for p in per_seed if k in p])),
                          float(np.std([p[k] for p in per_seed if k in p])),
                          int(sum(1 for p in per_seed if k in p)))
                      for k in keys}
        print()
        print(f"  ACROSS {args.seeds} PARTITION SEEDS  (mean +- sd)")
        print("  Every number above is one split. These are the same quantities")
        print("  recomputed on independent cell-blocked partitions, which is the")
        print("  only way to tell a finding from a lucky draw.")
        print()
        for k in ("rho_cov_dplus", "rho_cov_tv"):
            m, sd, n = seed_stats[k]
            print(f"    {k:22s} {m:+.3f} +- {sd:.3f}   (n={n})")
        print(f"    {'holds_1s':22s} {seed_stats['holds_1s'][0]:.1f} +- "
              f"{seed_stats['holds_1s'][1]:.1f} of {len(rows)}")
        print(f"    {'mean_slack_1s':22s} {seed_stats['mean_slack_1s'][0]:+.3f} +- "
              f"{seed_stats['mean_slack_1s'][1]:.3f}")
        print()
        blinds = sorted((k for k in seed_stats if k.startswith("blind_")),
                        key=lambda k: -seed_stats[k][0])
        for k in blinds:
            m, sd, n = seed_stats[k]
            print(f"    {k:22s} {m:+.3f} +- {sd:.3f}")
        if len(blinds) >= 2:
            (m0, s0, _), (m1, s1, _) = seed_stats[blinds[0]], seed_stats[blinds[1]]
            gap = m0 - m1
            pooled = float(np.hypot(s0, s1))
            verdict = ("clear of the seed noise" if gap > 2 * pooled
                       else "NOT separated from seed noise")
            print()
            print(f"    best {blinds[0][6:]!r} leads {blinds[1][6:]!r} by {gap:+.3f}, "
                  f"pooled sd {pooled:.3f} -- {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"cells": results, "blind_blocks": block_stats,
                   "alpha": args.alpha, "seeds": args.seeds,
                   "across_seeds": seed_stats}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

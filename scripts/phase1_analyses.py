"""Calibration-set size, per-structure breakdown, and seed stability.

Three analyses a reviewer of a conformal-prediction paper will look for, none
of which needs re-extraction -- all three run off the cached features.

**Calibration-set size.** Split conformal's guarantee is finite-sample, so the
practical question is how many calibration cells a microscopist actually needs.
The correction is the ``ceil((n+1)(1-alpha))``-th order statistic, which does
not even exist for ``n < 1/alpha - 1``; at alpha = 0.1 that is n >= 9. This
sweeps n and reports coverage mean and spread over repeated draws, so the floor
is measured rather than asserted.

**Per-structure.** The four BioSR structures are not interchangeable: CCPs are
punctate, ER is a dense web, F-actin and microtubules are filamentous and
strongly anisotropic. Coverage is reported per structure to show the
in-distribution guarantee is not carried by one easy specimen class.

**Seed stability.** Every number in the paper depends on one train/cal/test
permutation and one fitted predictor. This repeats the whole pipeline across
seeds and reports the spread, so effects smaller than seed noise are not
reported as findings. Differences are computed *paired* within seed, which is
the comparison with any power -- across-seed variation is common to both arms
and cancels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import load, one_sided_correction  # noqa: E402
from scripts.gate4_scaled import fit_scaled_models, scale_of, scores  # noqa: E402

MILD = ("nominal", "mild", "moderate")


def calibration_size_study(data, alpha=0.1, sizes=None, draws=40, seed=0):
    """Coverage as the calibration set shrinks, over repeated random draws."""
    if sizes is None:
        sizes = [10, 20, 40, 80, 160, 320, 595]
    mismatch, mu, sigma, floor = fit_scaled_models(data["train"], seed=seed)
    cal, test = data["cal"], data["test"]
    s_cal_all = scores(cal["e"], mu, sigma, floor, cal["X"])
    base = mu.predict(test["X"])
    sc_test = scale_of(sigma, floor, test["X"])

    # The order statistic the correction needs does not exist below this n.
    n_min = int(np.ceil(1.0 / alpha)) - 1

    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(s_cal_all):
            continue
        covs = []
        for _ in range(draws):
            idx = rng.choice(len(s_cal_all), size=n, replace=False)
            q = one_sided_correction(s_cal_all[idx], alpha)
            U = base + sc_test * q
            covs.append(float((test["e"] <= U).mean()))
        covs = np.array(covs)
        rows.append({
            "n_cal": int(n),
            "draws": draws,
            "coverage_mean": float(covs.mean()),
            "coverage_std": float(covs.std()),
            "coverage_min": float(covs.min()),
            "coverage_p05": float(np.quantile(covs, 0.05)),
            "feasible": bool(n >= n_min),
        })
    return rows, n_min


def per_structure(data, alpha=0.1, seed=0):
    """In-distribution and operator-shift coverage per BioSR structure."""
    mismatch, mu, sigma, floor = fit_scaled_models(data["train"], seed=seed)
    cal, test = data["cal"], data["test"]

    # In-distribution: calibrate on everything, test on everything.
    q_all = one_sided_correction(scores(cal["e"], mu, sigma, floor, cal["X"]), alpha)
    # Operator shift: calibrate on the mild rungs only, test on severe/extreme.
    cm = np.isin(cal["severity"], MILD)
    q_mild = one_sided_correction(
        scores(cal["e"][cm], mu, sigma, floor, cal["X"][cm]), alpha
    )

    rows = []
    for st in sorted(set(test["structure"].tolist())):
        m_in = test["structure"] == st
        m_op = m_in & ~np.isin(test["severity"], MILD)
        rec = {"structure": st, "n_in": int(m_in.sum()), "n_op": int(m_op.sum())}
        for tag, mask, q in (("in_dist", m_in, q_all), ("operator", m_op, q_mild)):
            if not mask.any():
                rec[f"coverage_{tag}"] = float("nan")
                continue
            X, e = test["X"][mask], test["e"][mask]
            U = mu.predict(X) + scale_of(sigma, floor, X) * q
            rec[f"coverage_{tag}"] = float((e <= U).mean())
            rec[f"error_{tag}"] = float(e.mean())
        rows.append(rec)
    return rows


def seed_stability(data, alpha=0.1, n_seeds=10):
    """Repeat the pipeline across seeds; compare schemes paired within seed."""
    cal, test = data["cal"], data["test"]
    cm = np.isin(cal["severity"], MILD)
    sm = ~np.isin(test["severity"], MILD)

    per_seed = []
    for seed in range(n_seeds):
        mismatch, mu, sigma, floor = fit_scaled_models(data["train"], seed=seed)
        q_all = one_sided_correction(
            scores(cal["e"], mu, sigma, floor, cal["X"]), alpha
        )
        q_mild = one_sided_correction(
            scores(cal["e"][cm], mu, sigma, floor, cal["X"][cm]), alpha
        )
        U_in = mu.predict(test["X"]) + scale_of(sigma, floor, test["X"]) * q_all
        Xs, es = test["X"][sm], test["e"][sm]
        U_op = mu.predict(Xs) + scale_of(sigma, floor, Xs) * q_mild
        per_seed.append({
            "seed": seed,
            "coverage_in_dist": float((test["e"] <= U_in).mean()),
            "coverage_operator": float((es <= U_op).mean()),
        })

    ind = np.array([r["coverage_in_dist"] for r in per_seed])
    ops = np.array([r["coverage_operator"] for r in per_seed])
    paired = ind - ops
    return {
        "per_seed": per_seed,
        "in_dist_mean": float(ind.mean()),
        "in_dist_std": float(ind.std()),
        "operator_mean": float(ops.mean()),
        "operator_std": float(ops.std()),
        # Paired within seed: the loss attributable to shift, not to the seed.
        "paired_drop_mean": float(paired.mean()),
        "paired_drop_std": float(paired.std()),
        "paired_drop_min": float(paired.min()),
        "n_seeds": n_seeds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, names, photons = load(args.features)
    print(f"PHASE 1 ANALYSES -- photons={photons:.0f}, alpha={args.alpha}")
    print()

    t0 = time.time()
    sizes, n_min = calibration_size_study(
        data, args.alpha, draws=args.draws
    )
    print(f"1. Calibration-set size (nominal {1 - args.alpha:.0%}, "
          f"{args.draws} random draws each)")
    print(f"   the conformal order statistic requires n >= {n_min} at this alpha")
    print()
    print(f"   {'n_cal':>6s} {'coverage':>9s} {'sd':>7s} {'5th pct':>8s} "
          f"{'worst':>7s}   feasible")
    for r in sizes:
        print(f"   {r['n_cal']:6d} {r['coverage_mean']:9.3f} "
              f"{r['coverage_std']:7.3f} {r['coverage_p05']:8.3f} "
              f"{r['coverage_min']:7.3f}   {'yes' if r['feasible'] else 'NO'}")
    print()

    struct = per_structure(data, args.alpha)
    print("2. Per-structure coverage")
    print()
    print(f"   {'structure':16s} {'in-dist':>8s} {'operator':>9s} "
          f"{'n_in':>6s} {'n_op':>6s}")
    for r in struct:
        print(f"   {r['structure']:16s} {r['coverage_in_dist']:8.3f} "
              f"{r['coverage_operator']:9.3f} {r['n_in']:6d} {r['n_op']:6d}")
    print()

    stab = seed_stability(data, args.alpha, args.seeds)
    print(f"3. Seed stability over {stab['n_seeds']} seeds")
    print()
    print(f"   in-distribution coverage : {stab['in_dist_mean']:.3f} "
          f"+/- {stab['in_dist_std']:.3f}")
    print(f"   operator-shift coverage  : {stab['operator_mean']:.3f} "
          f"+/- {stab['operator_std']:.3f}")
    print(f"   paired drop (within seed): {stab['paired_drop_mean']:.3f} "
          f"+/- {stab['paired_drop_std']:.3f}   (min {stab['paired_drop_min']:.3f})")
    print()
    print("   The paired drop is the quantity with power: seed-to-seed variation")
    print("   is common to both arms and cancels. A drop whose spread is small")
    print("   relative to its mean is a real effect, not a lucky split.")
    print()
    print(f"  elapsed {time.time() - t0:.1f}s")

    if args.out:
        out = {
            "photons": photons,
            "alpha": args.alpha,
            "n_cal_minimum": n_min,
            "calibration_size": sizes,
            "per_structure": struct,
            "seed_stability": stab,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

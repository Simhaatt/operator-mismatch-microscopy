"""Phase 2 analysis: is the result an SSIM artefact, and do physical families behave?

Two questions, both answerable only from a 29-spec cache produced with
``--include-physical``.

**Is it an SSIM artefact?** Every coverage number in the paper calibrates
``e = 1 - SSIM``. The same cache now also stores ``e_l2``, the relative l2
error, computed from the identical scale-matched reconstruction -- so the only
thing that differs between the two analyses is the metric. If the coverage
conclusions hold under both, the finding is about calibration under operator
mismatch rather than about a perceptual metric. This runs the full
calibration-under-shift comparison twice and reports them side by side.

**Do the physically-motivated families behave like the Zernike ladder?** The
Zernike axis varies RMS wavefront error, which nobody measures. NA, emission
wavelength and focus drift are the quantities a microscopist is actually
uncertain about. Coverage is reported per family so the two axes can be
compared directly.

Because NA and wavelength perturbations have no RMS wavefront error, the
mismatch score is fitted against ``mismatch_mag`` -- the relative l2 difference
between the true and nominal kernels -- which is defined for every family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import one_sided_correction  # noqa: E402
from src.metrics.stats import spearman  # noqa: E402

MILD = ("nominal", "mild", "moderate")
SEVERE = ("severe", "extreme")
ZERNIKE = ("defocus", "astigmatism", "coma", "spherical")
PHYSICAL = ("na", "wavelength", "focus_drift")
SCALE_FLOOR_FRAC = 0.10


def load29(path):
    """Load a 29-spec cache, keeping the columns this analysis needs."""
    d = np.load(path, allow_pickle=True)
    if "e_l2" not in "".join(d.files):
        raise SystemExit(f"{path} has no e_l2 column; it predates Phase 2")
    out = {}
    for split in ("train", "cal", "test", "ood"):
        out[split] = {
            "X": d[f"{split}_X"],
            "e_ssim": d[f"{split}_e"],
            "e_l2": d[f"{split}_e_l2"],
            "mag": d[f"{split}_mismatch_mag"],
            "rms": d[f"{split}_rms"],
            "severity": d[f"{split}_severity"],
            "mode": d[f"{split}_mode"],
            "structure": d[f"{split}_structure"],
        }
    meta = {
        "photons": float(d["photons"][0]),
        "n_specs": int(d["n_specs"][0]) if "n_specs" in d.files else None,
        "reconstructor": str(d["reconstructor"][0]),
    }
    return out, meta


def fit(train, target_key, seed=0):
    """Fit mismatch score, mean and scale for one error definition."""
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(train[target_key])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2:]

    # Fitted against mismatch_mag, not rms: NA and wavelength have no RMS
    # wavefront error, so rms is NaN for a third of the specs.
    mismatch = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][a], train["mag"][a])

    mu = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][b], train[target_key][b])
    resid = np.abs(train[target_key][b] - mu.predict(train["X"][b]))
    sigma = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][b], resid)
    return mismatch, mu, sigma, SCALE_FLOOR_FRAC * float(resid.mean())


def bound(mu, sigma, floor, X, q):
    return mu.predict(X) + np.maximum(sigma.predict(X), floor) * q


def scores_of(e, mu, sigma, floor, X):
    return (e - mu.predict(X)) / np.maximum(sigma.predict(X), floor)


def shift_axes(data):
    """Calibration and evaluation masks for each shift axis.

    The 29-spec ladder makes the old definition wrong. Previously anything
    outside nominal/mild/moderate counted as "operator shift", which sweeps the
    physical families' rungs (labelled ``p1``..``p4``, not ``severe``) into the
    shifted set. Those rungs are small perturbations that retain coverage, so
    including them inflated the operator-shift figure and made it incomparable
    with the 17-spec results.

    So the axes are now defined by family as well as severity:

    * **operator shift (Zernike)** -- calibrate on the Zernike mild rungs, test
      on Zernike severe/extreme. Identical in composition to the 17-spec
      protocol, hence directly comparable to it.
    * **operator shift (physical)** -- the same calibration set, tested on the
      NA, wavelength and focus-drift families. This is the new axis.

    Calibration always excludes the physical families, so the calibration set is
    exactly the one the earlier experiments used.
    """
    cal, test, ood = data["cal"], data["test"], data["ood"]

    zern_cal = np.isin(cal["mode"], ZERNIKE + ("none",)) & np.isin(
        cal["severity"], MILD
    )
    zern_test_severe = np.isin(test["mode"], ZERNIKE) & np.isin(
        test["severity"], SEVERE
    )
    phys_test = np.isin(test["mode"], PHYSICAL)
    zern_ood_severe = np.isin(ood["mode"], ZERNIKE) & np.isin(
        ood["severity"], SEVERE
    )
    # In-distribution keeps the whole calibration and test splits, matching the
    # earlier protocol, so the 29-spec composition shows up here as a slightly
    # different mixture rather than as a redefinition.
    all_cal = np.ones(len(cal["severity"]), dtype=bool)
    all_test = np.ones(len(test["severity"]), dtype=bool)
    all_ood = np.ones(len(ood["severity"]), dtype=bool)

    return {
        "none (in-distribution)": (all_cal, "cal", "test", all_test),
        "operator shift (Zernike)": (zern_cal, "cal", "test", zern_test_severe),
        "operator shift (physical)": (zern_cal, "cal", "test", phys_test),
        "content shift": (all_cal, "cal", "ood", all_ood),
        "combined": (zern_cal, "cal", "ood", zern_ood_severe),
    }


def coverage_table(data, target_key, alpha, seeds):
    """Coverage under each shift axis, for one error definition."""
    rows = {}
    axes = shift_axes(data)
    for axis, (cal_mask, cal_split, test_split, test_mask) in axes.items():
        per_seed = []
        for seed in range(seeds):
            mismatch, mu, sigma, floor = fit(data["train"], target_key, seed=seed)
            cal, test = data[cal_split], data[test_split]
            s = scores_of(cal[target_key][cal_mask], mu, sigma, floor,
                          cal["X"][cal_mask])
            q = one_sided_correction(s, alpha)
            Xt = test["X"][test_mask]
            et = test[target_key][test_mask]
            U = bound(mu, sigma, floor, Xt, q)
            per_seed.append((float((et <= U).mean()), float(U.mean())))
        cov = np.array([p[0] for p in per_seed])
        wid = np.array([p[1] for p in per_seed])
        rows[axis] = {
            "coverage": (float(cov.mean()), float(cov.std())),
            "width": (float(wid.mean()), float(wid.std())),
            "n": int(len(et)),
        }
    return rows


def family_coverage(data, target_key, alpha, seed=0):
    """Coverage per mismatch family at its most severe rung."""
    mismatch, mu, sigma, floor = fit(data["train"], target_key, seed=seed)
    cal, test = data["cal"], data["test"]
    cm = np.isin(cal["severity"], MILD)
    s = scores_of(cal[target_key][cm], mu, sigma, floor, cal["X"][cm])
    q = one_sided_correction(s, alpha)

    rows = []
    for fam in list(ZERNIKE) + list(PHYSICAL):
        sel = test["mode"] == fam
        if not sel.any():
            continue
        # Worst rung available in this family: Zernike ladders end at
        # "extreme", the physical families at their largest magnitude.
        sevs = sorted(set(test["severity"][sel].tolist()))
        worst = "extreme" if "extreme" in sevs else sevs[-1]
        m = sel & (test["severity"] == worst)
        X, e = test["X"][m], test[target_key][m]
        U = bound(mu, sigma, floor, X, q)
        rows.append({
            "family": fam,
            "kind": "zernike" if fam in ZERNIKE else "physical",
            "worst_rung": worst,
            "n": int(m.sum()),
            "coverage": float((e <= U).mean()),
            "mean_error": float(e.mean()),
            "mean_mismatch_mag": float(test["mag"][m].mean()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, meta = load29(args.features)
    print(f"PHASE 2 -- {meta['reconstructor']}, photons={meta['photons']:.0f}, "
          f"{meta['n_specs']} specs, alpha={args.alpha}, {args.seeds} seeds")
    print()

    # --- 1. metric sensitivity ---------------------------------------------
    print("1. Does the conclusion depend on the error metric?")
    print()
    print(f"   {'shift axis':26s} {'coverage (1-SSIM)':>20s} "
          f"{'coverage (rel l2)':>20s}   n")
    tabs = {}
    for key, label in (("e_ssim", "1-SSIM"), ("e_l2", "rel l2")):
        tabs[key] = coverage_table(data, key, args.alpha, args.seeds)
    for axis in tabs["e_ssim"]:
        a = tabs["e_ssim"][axis]["coverage"]
        b = tabs["e_l2"][axis]["coverage"]
        print(f"   {axis:26s} {a[0]:11.3f}+/-{a[1]:<6.3f} "
              f"{b[0]:11.3f}+/-{b[1]:<6.3f}   {tabs['e_ssim'][axis]['n']}")
    print()
    diffs = [abs(tabs['e_ssim'][ax]['coverage'][0] - tabs['e_l2'][ax]['coverage'][0])
             for ax in tabs['e_ssim']]
    print(f"   largest coverage difference between metrics: {max(diffs):.3f}")
    print("   The two metrics are different functions of the reconstruction, so")
    print("   exact agreement is not expected. What matters is whether the")
    print("   ordering and the qualitative failure survive the swap.")
    print()

    # correlation between the two targets, for context
    rho_metrics = spearman(data["test"]["e_ssim"], data["test"]["e_l2"])
    print(f"   Spearman rho(1-SSIM, rel l2) on the test split: "
          f"{rho_metrics:+.3f}")
    print()

    # --- 2. physical vs Zernike families -----------------------------------
    print("2. Do the physically-motivated families behave like the Zernike ladder?")
    print()
    fam_ssim = family_coverage(data, "e_ssim", args.alpha)
    fam_l2 = family_coverage(data, "e_l2", args.alpha)
    by_l2 = {r["family"]: r for r in fam_l2}
    print(f"   {'family':13s} {'kind':9s} {'rung':9s} {'|mismatch|':>11s} "
          f"{'error':>8s} {'cov(SSIM)':>10s} {'cov(l2)':>8s}   n")
    for r in fam_ssim:
        print(f"   {r['family']:13s} {r['kind']:9s} {r['worst_rung']:9s} "
              f"{r['mean_mismatch_mag']:11.4f} {r['mean_error']:8.4f} "
              f"{r['coverage']:10.3f} {by_l2[r['family']]['coverage']:8.3f}   {r['n']}")
    print()
    zc = [r["coverage"] for r in fam_ssim if r["kind"] == "zernike"]
    pc = [r["coverage"] for r in fam_ssim if r["kind"] == "physical"]
    zm = [r["mean_mismatch_mag"] for r in fam_ssim if r["kind"] == "zernike"]
    pm = [r["mean_mismatch_mag"] for r in fam_ssim if r["kind"] == "physical"]
    if zc and pc:
        print(f"   Zernike  families: coverage {min(zc):.3f}-{max(zc):.3f}, "
              f"|mismatch| {min(zm):.4f}-{max(zm):.4f}")
        print(f"   physical families: coverage {min(pc):.3f}-{max(pc):.3f}, "
              f"|mismatch| {min(pm):.4f}-{max(pm):.4f}")
        print()
        print("   Read these together with the magnitude column. A physical family")
        print("   that retains coverage while producing a much smaller kernel")
        print("   perturbation has not disproved the mechanism -- it has placed")
        print("   realistic parameter uncertainty lower on the same axis.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "meta": meta,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "coverage_ssim": tabs["e_ssim"],
            "coverage_l2": tabs["e_l2"],
            "rho_ssim_l2": rho_metrics,
            "families_ssim": fam_ssim,
            "families_l2": fam_l2,
        }, indent=1))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

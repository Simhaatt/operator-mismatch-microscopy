"""Per-family, per-severity disaggregation of the mismatch phenomenon.

Every headline number in the paper pools the four aberration families together.
That pooling is the obvious thing a reviewer will object to: "defocus,
astigmatism, coma and spherical are different operators -- does your claim hold
for each, or is it an average over one family that misbehaves?"

The caches already answer it. ``mode`` and ``severity`` are stored per row, and
the ladder is balanced -- 35 cells for every one of the 17 (family, severity)
combinations in both the calibration and test splits, 54 in the held-out
structure split. Nothing needs re-extracting; the disaggregation is arithmetic.

Three questions are asked of each family separately:

1. **Does the phenomenon hold?** Error must rise with severity while the
   nominal residual stays comparatively flat. Reported as the ratio of each to
   its value at nominal, so the two are on one scale.
2. **Does coverage break the same way?** Calibrate on nominal/mild/moderate
   pooled across families, then test on the severe and extreme rungs of one
   family at a time. A family-specific collapse would mean the paper's
   "operator shift" axis is really "that one family".
3. **Is the blind signal family-specific?** Spearman between the fitted
   mismatch score and true wavefront error, within each family.

The residual column is ``r_nominal`` -- the data residual under the *assumed*
PSF, which is the only residual an operator could actually compute at run time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import load, one_sided_correction  # noqa: E402
from scripts.gate4_scaled import fit_scaled_models, scale_of, scores  # noqa: E402
from src.metrics.stats import spearman  # noqa: E402

MILD = ("nominal", "mild", "moderate")
SEVERE = ("severe", "extreme")
FAMILIES = ("defocus", "astigmatism", "coma", "spherical")
SEV_ORDER = ("nominal", "mild", "moderate", "severe", "extreme")


def phenomenon_table(data, feat_names):
    """Error and nominal-residual growth per family and severity."""
    r_idx = list(feat_names).index("r_nominal")
    # Pool every split so each cell is estimated from as many samples as
    # possible; this table is descriptive, not a held-out evaluation.
    e = np.concatenate([data[s]["e"] for s in ("train", "cal", "test")])
    r = np.concatenate([data[s]["X"][:, r_idx] for s in ("train", "cal", "test")])
    mode = np.concatenate([data[s]["mode"] for s in ("train", "cal", "test")])
    sev = np.concatenate([data[s]["severity"] for s in ("train", "cal", "test")])

    base_e = float(e[sev == "nominal"].mean())
    base_r = float(r[sev == "nominal"].mean())

    rows = []
    for fam in FAMILIES:
        for s in SEV_ORDER[1:]:
            m = (mode == fam) & (sev == s)
            if not m.any():
                continue
            rows.append({
                "family": fam,
                "severity": s,
                "n": int(m.sum()),
                "error": float(e[m].mean()),
                "residual": float(r[m].mean()),
                "error_ratio": float(e[m].mean() / base_e),
                "residual_ratio": float(r[m].mean() / base_r),
            })
    return rows, base_e, base_r


def coverage_by_family(data, alpha=0.1, seed=0):
    """Coverage when calibrated on mild rungs and tested on one family."""
    mismatch, mu, sigma, floor = fit_scaled_models(data["train"], seed=seed)

    cal, test = data["cal"], data["test"]
    cm = np.isin(cal["severity"], MILD)
    s_cal = scores(cal["e"][cm], mu, sigma, floor, cal["X"][cm])
    q = one_sided_correction(s_cal, alpha)

    rows = []
    for fam in list(FAMILIES) + ["ALL"]:
        for s in SEVERE:
            if fam == "ALL":
                m = np.isin(test["severity"], [s])
            else:
                m = (test["mode"] == fam) & (test["severity"] == s)
            if not m.any():
                continue
            X, e = test["X"][m], test["e"][m]
            U = mu.predict(X) + scale_of(sigma, floor, X) * q
            rows.append({
                "family": fam,
                "severity": s,
                "n": int(m.sum()),
                "coverage": float((e <= U).mean()),
                "mean_bound": float(U.mean()),
                "mean_error": float(e.mean()),
            })
    # In-distribution reference: the same bound on the mild rungs it saw.
    m = np.isin(test["severity"], MILD)
    X, e = test["X"][m], test["e"][m]
    U = mu.predict(X) + scale_of(sigma, floor, X) * q
    reference = {
        "family": "(calibrated rungs)",
        "severity": "nominal+mild+moderate",
        "n": int(m.sum()),
        "coverage": float((e <= U).mean()),
        "mean_bound": float(U.mean()),
        "mean_error": float(e.mean()),
    }
    return rows, reference, mismatch


def blind_by_family(data, mismatch):
    """Does the blind mismatch score track wavefront error within a family?"""
    test = data["test"]
    m_all = mismatch.predict(test["X"])
    rows = []
    for fam in FAMILIES:
        sel = np.isin(test["mode"], [fam, "none"])
        rows.append({
            "family": fam,
            "n": int(sel.sum()),
            "rho_score_vs_rms": spearman(m_all[sel], test["rms"][sel]),
        })
    rows.append({
        "family": "ALL (pooled)",
        "n": int(len(m_all)),
        "rho_score_vs_rms": spearman(m_all, test["rms"]),
    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data, names, photons = load(args.features)
    # ``load`` keeps only a fixed set of columns; mode is needed here.
    d = np.load(args.features, allow_pickle=True)
    for split in ("train", "cal", "test", "ood"):
        data[split]["mode"] = d[f"{split}_mode"]

    print(f"PER-FAMILY DISAGGREGATION -- photons={photons:.0f}, alpha={args.alpha}")
    if "background" in d.files:
        bg = float(d["background"][0])
        print(f"  haze fraction = {bg / (photons + bg):.3f}")
    print()

    rows, base_e, base_r = phenomenon_table(data, names)
    print("1. Does the phenomenon hold per family?")
    print(f"   nominal baselines: error={base_e:.4f}  residual(r_nominal)={base_r:.4f}")
    print()
    print(f"   {'family':14s} {'severity':10s} {'error':>8s} {'x nom':>7s} "
          f"{'resid':>8s} {'x nom':>7s} {'ratio':>7s}   n")
    for r in rows:
        ratio = r["error_ratio"] / r["residual_ratio"]
        print(f"   {r['family']:14s} {r['severity']:10s} {r['error']:8.4f} "
              f"{r['error_ratio']:7.2f} {r['residual']:8.4f} "
              f"{r['residual_ratio']:7.2f} {ratio:7.2f}   {r['n']}")
    print()
    print("   'ratio' is error growth divided by residual growth. Above 1 means")
    print("   the error outruns the residual an operator could compute.")
    print()

    cov, ref, mismatch = coverage_by_family(data, args.alpha)
    print(f"2. Coverage per family (calibrated on {'+'.join(MILD)}, nominal "
          f"{1 - args.alpha:.0%})")
    print()
    print(f"   {'family':14s} {'severity':10s} {'coverage':>9s} {'bound':>8s} "
          f"{'error':>8s}   n")
    print(f"   {ref['family']:14s} {ref['severity']:10s} {ref['coverage']:9.3f} "
          f"{ref['mean_bound']:8.4f} {ref['mean_error']:8.4f}   {ref['n']}")
    print("   " + "-" * 60)
    for r in cov:
        print(f"   {r['family']:14s} {r['severity']:10s} {r['coverage']:9.3f} "
              f"{r['mean_bound']:8.4f} {r['mean_error']:8.4f}   {r['n']}")
    print()
    spread = [r["coverage"] for r in cov if r["family"] != "ALL"]
    print(f"   coverage spread across families: {min(spread):.3f} to "
          f"{max(spread):.3f}  (range {max(spread) - min(spread):.3f})")
    print()

    blind = blind_by_family(data, mismatch)
    print("3. Blind mismatch score vs true wavefront error, per family")
    print()
    print(f"   {'family':16s} {'rho':>7s}   n")
    for r in blind:
        print(f"   {r['family']:16s} {r['rho_score_vs_rms']:+7.3f}   {r['n']}")
    print()

    if args.out:
        out = {
            "photons": photons,
            "alpha": args.alpha,
            "nominal_error": base_e,
            "nominal_residual": base_r,
            "phenomenon": rows,
            "coverage_reference": ref,
            "coverage_by_family": cov,
            "blind_by_family": blind,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

"""Check every quantitative claim of the paper against the regenerated results.

    python paper/check_claims.py               # prints a table, exits 1 on any mismatch
    python paper/check_claims.py --markdown paper/results/claims_report.md

Each claim names where it appears in the manuscript, the value printed there,
and the JSON field of ``paper/results/`` that reproduces it. A claim passes when
the regenerated value, rounded half-up to the precision the paper prints,
equals the printed value. Ranges ("0.754--0.760") are checked as their two endpoints
against the minimum and maximum of the underlying values.

Results depend on scikit-learn's gradient-boosting implementation. With the
pinned versions in ``requirements-lock.txt`` every claim matches exactly; other
versions can move individual values in the last printed digit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


@dataclass
class Claim:
    where: str
    what: str
    paper: float
    dp: int
    get: Callable[[], float]
    source: str
    tol: float | None = None  # set only for claims the paper words as approximate

    def check(self):
        try:
            v = float(self.get())
        except C.MissingResult as exc:
            return None, "missing", str(exc)
        if self.tol is not None:
            return v, ("approx" if abs(v - self.paper) <= self.tol else "DIFFERS"), ""
        ok = abs(C.round_half_up(v, self.dp) - self.paper) <= 1e-9
        return v, ("ok" if ok else "DIFFERS"), ""


CLAIMS: list[Claim] = []


def claim(where, what, paper, get, source, dp=None, tol=None):
    if dp is None:
        s = repr(paper)
        dp = len(s.split(".")[1]) if "." in s else 0
    CLAIMS.append(Claim(where, what, paper, dp, get, source, tol))


def claim_range(where, what, lo, hi, values, source, dp=3, tol=None):
    claim(where, f"{what} (min)", lo, lambda: min(values()), source, dp, tol)
    claim(where, f"{what} (max)", hi, lambda: max(values()), source, dp, tol)


def pm(where, what, mean, sd, get_pair, source, dp=3):
    claim(where, what, mean, lambda: get_pair()[0], source, dp)
    claim(where, f"{what} [sd]", sd, lambda: get_pair()[1], source, dp)


# ------------------------------------------------------------ Section IV-A --
FAM = "family_rl100"


def _fam(field, sev):
    return lambda: [r[field] for r in C.load(FAM)["phenomenon"]
                    if r["family"] in C.FAMILIES and r["severity"] == sev]


def _famcov(fam, sev):
    return lambda: next(r["coverage"] for r in C.load(FAM)["coverage_by_family"]
                        if r["family"] == fam and r["severity"] == sev)


claim("IV-A", "RL-100 nominal reconstruction error", 0.1345,
      lambda: C.load(FAM)["nominal_error"], FAM)
claim("IV-A", "RL-100 nominal residual", 0.1713,
      lambda: C.load(FAM)["nominal_residual"], FAM)
claim_range("IV-A / Fig.1a", "severe error ratio", 2.49, 2.90, _fam("error_ratio", "severe"), FAM, 2)
claim_range("IV-A / Fig.1b", "severe residual ratio", 0.88, 0.96, _fam("residual_ratio", "severe"), FAM, 2)
claim_range("IV-A / Fig.1a", "extreme error ratio", 4.48, 5.00, _fam("error_ratio", "extreme"), FAM, 2)
claim_range("IV-A / Fig.1b", "extreme residual ratio", 0.78, 0.90, _fam("residual_ratio", "extreme"), FAM, 2)
claim("IV-A", "alpha-independent in-domain coverage", 0.901,
      lambda: C.scaled_split("rl100", "none (in-distribution)")["coverage"], "scaled_rl100")
claim("IV-A", "pooled severe-extreme Zernike coverage", 0.664,
      lambda: C.scaled_split("rl100", "operator shift")["coverage"], "scaled_rl100")

# Supplementary family table (S-V)
_paper_family = {
    "defocus": (2.77, 0.93, 0.829, 5.00, 0.86, 0.486),
    "astigmatism": (2.49, 0.96, 0.686, 4.48, 0.90, 0.429),
    "coma": (2.84, 0.92, 0.771, 4.89, 0.84, 0.314),
    "spherical": (2.90, 0.88, 0.971, 4.84, 0.78, 0.829),
}
for fam, vals in _paper_family.items():
    for sev, (er, rr, cv) in (("severe", vals[:3]), ("extreme", vals[3:])):
        row = (lambda f=fam, s=sev: next(r for r in C.load(FAM)["phenomenon"]
                                         if r["family"] == f and r["severity"] == s))
        claim("Supp. family table", f"{fam} {sev} error/nom", er,
              lambda row=row: row()["error_ratio"], FAM, 2)
        claim("Supp. family table", f"{fam} {sev} residual/nom", rr,
              lambda row=row: row()["residual_ratio"], FAM, 2)
        claim("Supp. family table / Fig.1c", f"{fam} {sev} coverage", cv,
              _famcov(fam, sev), FAM, 3)
claim("Supp. family table", "pooled severe coverage", 0.814, _famcov("ALL", "severe"), FAM)
claim("Supp. family table", "pooled extreme coverage", 0.514, _famcov("ALL", "extreme"), FAM)

# Supplementary calibration audit (S-IV)
_alphas = ("0.1", "0.05", "0.02", "0.01", "0.005", "0.001")
for a, cov, wid in zip(_alphas, (0.901, 0.948, 0.983, 0.993, 0.997, 0.998),
                       (0.4055, 0.4402, 0.5145, 0.5533, 0.6660, 0.7309)):
    claim("Supp. calibration audit", f"RL-100 coverage at alpha={a}", cov,
          lambda a=a: C.scaled_split("rl100", "none (in-distribution)", a)["coverage"],
          "scaled_rl100", 3)
    claim("Supp. calibration audit", f"RL-100 mean width at alpha={a}", wid,
          lambda a=a: C.scaled_split("rl100", "none (in-distribution)", a)["mean_width"],
          "scaled_rl100", 4)
claim("Supp. calibration", "RL-100 content-shift coverage", 0.378,
      lambda: C.scaled_split("rl100", "content shift")["coverage"], "scaled_rl100")
claim("Supp. calibration", "RL-100 combined-shift coverage", 0.012,
      lambda: C.scaled_split("rl100", "combined")["coverage"], "scaled_rl100")
for (lab, tag, _, _), floor in zip(C.BIOSR_SETTINGS, (0.00310, 0.00235, 0.00223, 0.00219)):
    claim("Supp. calibration", f"{lab} scale floor", floor,
          lambda t=tag: C.load(f"scaled_{t}")["scale_floor"], f"scaled_{tag}", 5)
    claim("Supp. calibration", f"{lab} monotonicity violations", 0,
          lambda t=tag: C.load(f"scaled_{t}")["monotonicity_violations"], f"scaled_{tag}", 0)

# ------------------------------------------------------------ Section IV-B --
for (lab, tag, _, _), (m, s) in zip(C.BIOSR_SETTINGS, ((0.189, 0.029), (0.149, 0.021),
                                                       (0.175, 0.031), (0.189, 0.036))):
    ss = (lambda t=tag: C.load(f"phase1_{t}")["seed_stability"])
    pm("IV-B", f"{lab} paired coverage drop", m, s,
       lambda ss=ss: (ss()["paired_drop_mean"], ss()["paired_drop_std"]), f"phase1_{tag}")

_biosr2 = [p[2] for p in C.PHASE2 if p[0] == "BioSR"]


def _p2(axis, metric="ssim", tags=_biosr2):
    return lambda: [C.phase2_cov(t, axis, metric)[0] for t in tags]


claim_range("IV-B", "BioSR Zernike-shift coverage, 1-SSIM", 0.754, 0.760, _p2(C.AXIS_ZERN), "phase2_biosr_*")
claim_range("IV-B", "BioSR physical-shift coverage, 1-SSIM", 0.898, 0.922, _p2(C.AXIS_PHYS), "phase2_biosr_*")
claim_range("IV-B", "BioSR in-domain coverage, rel l2", 0.866, 0.903, _p2(C.AXIS_IN, "l2"), "phase2_biosr_*")
claim_range("IV-B", "BioSR Zernike-shift coverage, rel l2", 0.581, 0.704, _p2(C.AXIS_ZERN, "l2"), "phase2_biosr_*")
claim_range("IV-B", "BioSR physical-shift coverage, rel l2", 0.849, 0.925, _p2(C.AXIS_PHYS, "l2"), "phase2_biosr_*")

_paper_real = {
    ("BioSR", "RL"): (137, 0.368, 0.950, 0.017),
    ("BioSR", "TV"): (137, 0.374, 0.966, 0.010),
    ("BioSR", "residual U-Net"): (137, 0.358, 0.915, 0.028),
    ("BioSR", "unrolled"): (137, 0.354, 0.958, 0.013),
    ("W2S", "RL"): (288, 0.593, 0.891, 0.017),
}
for (ds, rec), (n, med, m, s) in _paper_real.items():
    row = (lambda ds=ds, rec=rec: next(r for r in C.load("real_coverage")["rows"]
                                       if r["dataset"] == ds and r["reconstructor"] == rec))
    where = "IV-B / IV-D / Supp. real table"
    claim(where, f"{ds}/{rec} real n_test", n, lambda row=row: row()["n_test"], "real_coverage", 0)
    claim(where, f"{ds}/{rec} real median 1-SSIM", med, lambda row=row: row()["median_error"],
          "real_coverage")
    pm(where, f"{ds}/{rec} real in-domain coverage", m, s,
       lambda row=row: (row()["coverage_mean"], row()["coverage_std"]), "real_coverage")
claim("IV-D", "W2S genuine-widefield measurements", 2160,
      lambda: sum(next(r for r in C.load("real_coverage")["rows"] if r["dataset"] == "W2S")[k]
                  for k in ("n_train", "n_cal", "n_test", "n_ood")), "real_coverage", 0)

# ------------------------------------------------------------ Section IV-C --
G5 = "gate5_biosr"
claim("IV-C / Table II", "BioSR one-sided bound holds (cells, mean over 10 seeds)", 16,
      lambda: C.seeds_stat(G5, "holds_1s")[0], G5, 0)
claim("IV-C / Table II", "BioSR one-sided bound holds [sd over seeds]", 0,
      lambda: C.seeds_stat(G5, "holds_1s")[1], G5, 0)
pm("IV-C / Table II", "BioSR rho(coverage, -d+)", 0.952, 0.025,
   lambda: C.seeds_stat(G5, "rho_cov_dplus"), G5)
pm("IV-C", "BioSR mean slack", 0.046, 0.006, lambda: C.seeds_stat(G5, "mean_slack_1s"), G5)
pm("IV-C", "BioSR rho(coverage, -d_TV^hist)", -0.064, 0.090,
   lambda: C.seeds_stat(G5, "rho_cov_tv"), G5)

# ------------------------------------------------------------ Section IV-D --
_w2s = [p[2] for p in C.PHASE2 if p[0] == "W2S"]
for axis, vals, lab in ((C.AXIS_IN, (0.891, 0.867), "in-domain"),
                        (C.AXIS_ZERN, (0.604, 0.593), "Zernike-shift"),
                        (C.AXIS_PHYS, (0.915, 0.897), "physical-shift")):
    for tag, v, ph in zip(_w2s, vals, (100, 15)):
        claim("IV-D / Table II", f"W2S RL-{ph} {lab} coverage", v,
              lambda t=tag, a=axis: C.phase2_cov(t, a)[0], f"phase2_{tag}")
G5W = "gate5_w2s"
claim("IV-D / Table II", "W2S one-sided bound holds (cells)", 6,
      lambda: C.seeds_stat(G5W, "holds_1s")[0], G5W, 0)
claim("IV-D / Table II", "W2S one-sided bound holds [sd]", 0,
      lambda: C.seeds_stat(G5W, "holds_1s")[1], G5W, 0)
claim("IV-D / Table II", "W2S mean slack", 0.052, lambda: C.seeds_stat(G5W, "mean_slack_1s")[0], G5W)
pm("IV-D / Table II", "W2S rho(coverage, -d+)", 0.638, 0.145,
   lambda: C.seeds_stat(G5W, "rho_cov_dplus"), G5W)
pm("IV-D", "W2S rho(coverage, -d_TV^hist)", 0.383, 0.273,
   lambda: C.seeds_stat(G5W, "rho_cov_tv"), G5W)

# ------------------------------------------------------------ Section IV-E --
_paper_width = {"rl100": (0.4015, 0.5658, 29.0), "rl15": (0.4738, 0.6230, 24.0),
                "tv100": (0.3856, 0.5354, 28.0), "tv15": (0.4381, 0.5537, 20.9)}
for lab, tag, _, _ in C.BIOSR_SETTINGS:
    a, r, red = _paper_width[tag]
    src = f"ablation_{tag}"
    claim("IV-E / Fig.4a / Supp. width table", f"{lab} width, all 19", a,
          lambda t=tag: C.ablation_width(t, "all")[0], src, 4)
    claim("IV-E / Fig.4a / Supp. width table", f"{lab} width, residual only", r,
          lambda t=tag: C.ablation_width(t, "residual")[0], src, 4)
    claim("IV-E / Fig.4a / Supp. width table", f"{lab} width reduction (%)", red,
          lambda t=tag: 100 * (1 - C.ablation_width(t, "all")[0]
                               / C.ablation_width(t, "residual")[0]), src, 1)
# Worded "approximately 0.89--0.90" in the paper; the regenerated range is
# 0.885--0.911, so this is checked to within 0.015 rather than to the digit.
claim_range("IV-E", "matched in-domain coverage of the ablation (approx.)", 0.89, 0.90,
            lambda: [C.ablation_coverage(t, b)[0] for _, t, _, _ in C.BIOSR_SETTINGS
                     for b in ("all", "residual")], "ablation_*", 3, tol=0.015)
pm("IV-E", "BioSR sim rho(coverage, conditional predictor)", 0.616, 0.044,
   lambda: C.seeds_stat(G5, "blind_qpred"), G5)
pm("IV-E", "BioSR sim rho(coverage, nominal residual)", 0.083, 0.061,
   lambda: C.seeds_stat(G5, "blind_residual"), G5)
_paper_monitor = {"blind_qpred": (0.222, 0.153), "blind_oscr": (0.197, 0.055),
                  "blind_otf": (0.182, 0.041), "blind_all": (0.166, 0.044),
                  "blind_otf_core": (-0.123, 0.067), "blind_residual": (-0.162, 0.084),
                  "blind_blind_score": (-0.310, 0.092)}
for key, label in C.BLIND_LABELS:
    m, s = _paper_monitor[key]
    pm("IV-E / Fig.4b / Supp. monitoring", f"real BioSR rho, {label}", m, s,
       lambda k=key: C.seeds_stat("gate5_real_grid", k), "gate5_real_grid")

# ------------------------------------------------------ Supplementary, rest --
GR = "gate5_real_grid"
claim("Supp. bound table", "BioSR real grid bound holds (cells)", 16,
      lambda: C.seeds_stat(GR, "holds_1s")[0], GR, 0)
pm("Supp. bound table", "BioSR real grid rho(coverage, -d+)", 0.933, 0.053,
   lambda: C.seeds_stat(GR, "rho_cov_dplus"), GR)
pm("Supp. bound table", "BioSR real grid mean slack", 0.088, 0.010,
   lambda: C.seeds_stat(GR, "mean_slack_1s"), GR)
GWR = "gate5_w2s_real"
claim("Supp. bound table", "W2S genuine-widefield bound holds (cells)", 3,
      lambda: C.seeds_stat(GWR, "holds_1s")[0], GWR, 0)
claim("Supp. bound table", "W2S genuine-widefield mean slack", 0.050,
      lambda: C.seeds_stat(GWR, "mean_slack_1s")[0], GWR, 3)

_paper_metric = {  # (ssim in, l2 in, ssim zern, l2 zern, ssim phys, l2 phys), each (mean, sd)
    "biosr_rl100": ((0.899, .014), (0.894, .006), (0.755, .024), (0.614, .023), (0.922, .014), (0.903, .009)),
    "biosr_rl15": ((0.895, .014), (0.866, .004), (0.757, .022), (0.581, .043), (0.898, .018), (0.849, .019)),
    "biosr_tv100": ((0.895, .007), (0.903, .012), (0.754, .034), (0.704, .018), (0.909, .017), (0.925, .012)),
    "biosr_tv15": ((0.887, .019), (0.889, .008), (0.760, .050), (0.667, .020), (0.906, .021), (0.923, .014)),
    "w2s_rl100": ((0.891, .011), (0.927, .006), (0.604, .017), (0.754, .024), (0.915, .015), (0.956, .009)),
    "w2s_rl15": ((0.867, .008), (0.923, .003), (0.593, .037), (0.794, .022), (0.897, .009), (0.966, .011)),
}
for inst, setting, tag in C.PHASE2:
    vals = _paper_metric[tag]
    cols = [(C.AXIS_IN, "ssim"), (C.AXIS_IN, "l2"), (C.AXIS_ZERN, "ssim"),
            (C.AXIS_ZERN, "l2"), (C.AXIS_PHYS, "ssim"), (C.AXIS_PHYS, "l2")]
    for (axis, metric), (m, s) in zip(cols, vals):
        short = {C.AXIS_IN: "in-domain", C.AXIS_ZERN: "Zernike", C.AXIS_PHYS: "physical"}[axis]
        pm("Table II / Fig.3 / Supp. metric table", f"{inst} {setting} {short} coverage, {metric}",
           m, s, lambda t=tag, a=axis, k=metric: C.phase2_cov(t, a, k), f"phase2_{tag}")

_paper_endpoint = {"biosr_rl100": (0.943, 0.914, 0.886), "biosr_rl15": (0.914, 0.914, 0.829),
                   "biosr_tv100": (0.943, 0.914, 0.914), "biosr_tv15": (0.971, 0.914, 0.857),
                   "w2s_rl100": (0.896, 0.896, 0.771), "w2s_rl15": (0.958, 0.917, 0.813)}
for inst, setting, tag in C.PHASE2:
    for fam, v in zip(("na", "wavelength", "focus_drift"), _paper_endpoint[tag]):
        claim("Supp. physical endpoint table", f"{inst} {setting} {fam} endpoint coverage", v,
              lambda t=tag, f=fam: C.family_extreme_physical(t)[f], f"phase2_{tag}")
for tag, v, inst in (("biosr_rl100", 0.681, "BioSR"), ("w2s_rl100", 0.394, "W2S")):
    claim("Supp. physical grid table", f"{inst} 200-nm focus delta_H", v,
          lambda t=tag: next(r["mean_mismatch_mag"] for r in C.load(f"phase2_{t}")["families_ssim"]
                             if r["family"] == "focus_drift"), f"phase2_{tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", default=None, help="also write a markdown report here")
    ap.add_argument("--verbose", action="store_true", help="print passing claims too")
    args = ap.parse_args()

    rows, counts = [], {"ok": 0, "approx": 0, "DIFFERS": 0, "missing": 0}
    for c in CLAIMS:
        v, status, note = c.check()
        counts[status] += 1
        rows.append((c, v, status, note))
        if status != "ok" or args.verbose:
            shown = "--" if v is None else C.fmt(v, c.dp)
            print(f"  {status:8s} {c.where:28s} {c.what:55s} paper {c.paper:<9g} "
                  f"got {shown:<9s} [{c.source}]")
    print()
    print(f"  {counts['ok']} of {len(CLAIMS)} claims reproduce exactly; "
          f"{counts['approx']} approximate claims within tolerance; "
          f"{counts['DIFFERS']} differ; {counts['missing']} missing results")

    if args.markdown:
        lines = ["# Paper claims vs. regenerated results", "",
                 "Generated by `python paper/check_claims.py --markdown ...`. "
                 "A claim passes when the regenerated value, rounded to the printed "
                 "precision, equals the value printed in the paper.", "",
                 f"**{counts['ok']} / {len(CLAIMS)} reproduce exactly; "
                 f"{counts['approx']} claims the paper words as approximate are within "
                 f"tolerance; {counts['DIFFERS']} differ.**", "",
                 "| Where | Quantity | Paper | Regenerated | Status | Source JSON |",
                 "|---|---|---|---|---|---|"]
        for c, v, status, _ in rows:
            shown = "--" if v is None else C.fmt(v, c.dp)
            lines.append(f"| {c.where} | {c.what} | {c.paper:.{c.dp}f} | {shown} | "
                         f"{status} | `{c.source}` |")
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {args.markdown}")
    return 0 if counts["DIFFERS"] == 0 and counts["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

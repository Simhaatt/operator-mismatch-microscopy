"""Rebuild the paper's results tables from ``paper/results/*.json``.

    python paper/make_tables.py        # writes paper/tables/*.tex and paper/tables/tables.md

Each ``.tex`` file holds the tabular rows of one table, formatted the way the
manuscript prints them, so it can be diffed against (or ``\\input`` into) the
LaTeX source. ``tables.md`` collects all of them for reading on GitHub.

    table2_headline          Table II   Principal coverage and reliability results
    supp_calibration_audit   Supp.      Coverage-level and nesting audit
    supp_family              Supp.      Family-resolved Zernike stress test
    supp_cross_setting       Supp.      Repeated 90% coverage across settings
    supp_physical_endpoint   Supp.      Coverage at the largest physical perturbation
    supp_metric_comparison   Supp.      1-SSIM vs relative l2
    supp_bound_summary       Supp.      Repeated one-sided bound validation
    supp_feature_width       Supp.      Matched-coverage bound-width ablation
    supp_monitoring          Supp.      Reference-free monitoring on real BioSR
    supp_real                Supp.      In-domain coverage on real measurements
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.ROOT / "paper" / "tables"


def f3(x):
    return C.fmt(x, 3)


def pmfmt(pair, tex=True):
    m, s = pair
    return f"${C.fmt(m)}\\pm{C.fmt(s)}$" if tex else f"{C.fmt(m)} ± {C.fmt(s)}"


class Table:
    def __init__(self, name, title, header):
        self.name, self.title, self.header, self.rows = name, title, header, []

    def add(self, *cells):
        self.rows.append([str(c) for c in cells])

    def tex(self):
        return "\n".join(" & ".join(r) + r" \\" for r in self.rows) + "\n"

    def md(self):
        import re

        def cells(c):
            """One markdown cell per column; a merged LaTeX cell is repeated."""
            m = re.fullmatch(r"\\multicolumn\{(\d+)\}\{c\}\{(.*)\}", c)
            n, c = (int(m.group(1)), m.group(2)) if m else (1, c)
            c = c.replace("\\rho(\\mathrm{coverage},-d^{+})", "ρ(coverage, −d⁺)")
            c = (c.replace("$", "").replace("\\pm", " ± ").replace("\\%", "%")
                 .replace("\\approx", "≈"))
            return [c] * n
        out = [f"### {self.title}", "", "| " + " | ".join(self.header) + " |",
               "|" + "---|" * len(self.header)]
        out += ["| " + " | ".join(x for c in r for x in cells(c)) + " |" for r in self.rows]
        return "\n".join(out) + "\n"


def table2():
    t = Table("table2_headline", "Table II -- Principal coverage and reliability results",
              ["Evaluation", "BioSR RL-100", "BioSR RL-15", "BioSR TV-100", "BioSR TV-15",
               "W2S RL-100", "W2S RL-15"])
    for label, axis in (("In-domain coverage", C.AXIS_IN),
                        ("Zernike-shift coverage", C.AXIS_ZERN),
                        ("Physical-shift coverage", C.AXIS_PHYS)):
        t.add(label, *[f3(C.phase2_cov(tag, axis)[0]) for _, _, tag in C.PHASE2])
    b = C.seeds_stat("gate5_biosr", "holds_1s")
    w = C.seeds_stat("gate5_w2s", "holds_1s")
    t.add("One-sided bound validity",
          f"\\multicolumn{{4}}{{c}}{{{b[0]:.0f}/16 cells (sd {b[1]:.1f}) on 10 partitions}}",
          f"\\multicolumn{{2}}{{c}}{{{w[0]:.0f}/6 cells (sd {w[1]:.1f}) on 10 partitions}}")
    t.add("$\\rho(\\mathrm{coverage},-d^{+})$",
          f"\\multicolumn{{4}}{{c}}{{{pmfmt(C.seeds_stat('gate5_biosr', 'rho_cov_dplus'))}}}",
          f"\\multicolumn{{2}}{{c}}{{{pmfmt(C.seeds_stat('gate5_w2s', 'rho_cov_dplus'))}}}")
    slack_b = C.fmt(C.seeds_stat("gate5_biosr", "mean_slack_1s")[0])
    slack_w = C.fmt(C.seeds_stat("gate5_w2s", "mean_slack_1s")[0])
    t.add("Mean bound slack",
          f"\\multicolumn{{4}}{{c}}{{{slack_b}}}",
          f"\\multicolumn{{2}}{{c}}{{{slack_w}}}")
    real = {(r["dataset"], r["reconstructor"]): r for r in C.load("real_coverage")["rows"]}
    rb, rw = real[("BioSR", "RL")], real[("W2S", "RL")]
    t.add("Real-data coverage (RL)",
          f"\\multicolumn{{4}}{{c}}{{{pmfmt((rb['coverage_mean'], rb['coverage_std']))}}}",
          f"\\multicolumn{{2}}{{c}}{{{pmfmt((rw['coverage_mean'], rw['coverage_std']))}}}")
    return t


def supp_calibration_audit():
    alphas = ("0.1", "0.05", "0.02", "0.01", "0.005", "0.001")
    t = Table("supp_calibration_audit", "Supp. -- Coverage-level audit, RL-100 (alpha-independent)",
              ["Quantity", *alphas])
    rows = [C.scaled_split("rl100", "none (in-distribution)", a) for a in alphas]
    t.add("Nominal coverage", *[f3(1 - float(a)) for a in alphas])
    t.add("Empirical coverage", *[f3(r["coverage"]) for r in rows])
    t.add("Mean bound width", *[C.fmt(r['mean_width'], 4) for r in rows])
    for lab, tag, _, _ in C.BIOSR_SETTINGS:
        d = C.load(f"scaled_{tag}")
        t.add(f"{lab}: scale floor / violations", C.fmt(d['scale_floor'], 5),
              str(d["monotonicity_violations"]), "", "", "", "")
    return t


def supp_family():
    d = C.load("family_rl100")
    t = Table("supp_family", "Supp. -- Family-resolved Zernike stress test, RL at 100 photons",
              ["Family", "Severe err/nom", "Severe resid/nom", "Severe cov",
               "Extreme err/nom", "Extreme resid/nom", "Extreme cov"])
    ph = {(r["family"], r["severity"]): r for r in d["phenomenon"]}
    cv = {(r["family"], r["severity"]): r["coverage"] for r in d["coverage_by_family"]}
    for fam in C.FAMILIES:
        cells = []
        for sev in ("severe", "extreme"):
            r = ph[(fam, sev)]
            cells += [C.fmt(r['error_ratio'], 2), C.fmt(r['residual_ratio'], 2), f3(cv[(fam, sev)])]
        t.add(fam.capitalize(), *cells)
    t.add("Pooled coverage", "", "", f3(cv[("ALL", "severe")]), "", "", f3(cv[("ALL", "extreme")]))
    return t


def supp_cross_setting():
    t = Table("supp_cross_setting", "Supp. -- Repeated 90% coverage across settings (5 seeds)",
              ["Configuration", "In domain", "Zernike shift", "Physical shift"])
    for inst, setting, tag in C.PHASE2:
        t.add(f"{inst} {setting}", *[f3(C.phase2_cov(tag, a)[0])
                                     for a in (C.AXIS_IN, C.AXIS_ZERN, C.AXIS_PHYS)])
    return t


def supp_physical_endpoint():
    t = Table("supp_physical_endpoint", "Supp. -- Coverage at the largest physical perturbation",
              ["Configuration", "NA endpoint", "Wavelength endpoint", "Focus, 200 nm"])
    for inst, setting, tag in C.PHASE2:
        fam = C.family_extreme_physical(tag)
        t.add(f"{inst} {setting}", f3(fam["na"]), f3(fam["wavelength"]), f3(fam["focus_drift"]))
    return t


def supp_metric_comparison():
    t = Table("supp_metric_comparison", "Supp. -- Coverage under 1-SSIM and relative l2",
              ["Configuration", "In 1-SSIM", "In rel l2", "Zernike 1-SSIM", "Zernike rel l2",
               "Physical 1-SSIM", "Physical rel l2"])
    for inst, setting, tag in C.PHASE2:
        cells = []
        for axis in (C.AXIS_IN, C.AXIS_ZERN, C.AXIS_PHYS):
            for metric in ("ssim", "l2"):
                cells.append(pmfmt(C.phase2_cov(tag, axis, metric)))
        t.add(f"{inst} {setting}", *cells)
    return t


def supp_bound_summary():
    t = Table("supp_bound_summary", "Supp. -- Repeated one-sided coverage-bound validation",
              ["Evaluation", "Cells", "Valid cells (mean over 10 splits)",
               "rho(coverage, -d+)", "Mean slack"])
    for label, name, n in (("BioSR simulation", "gate5_biosr", 16),
                           ("W2S simulation", "gate5_w2s", 6),
                           ("BioSR real grid", "gate5_real_grid", 16),
                           ("W2S genuine widefield", "gate5_w2s_real", 3)):
        h = C.seeds_stat(name, "holds_1s")
        rho = pmfmt(C.seeds_stat(name, "rho_cov_dplus")) if n > 3 else "--"
        t.add(label, n, f"{h[0]:.1f}/{n} (sd {h[1]:.1f})", rho,
              pmfmt(C.seeds_stat(name, "mean_slack_1s")))
    return t


def supp_feature_width():
    t = Table("supp_feature_width", "Supp. -- Matched-coverage bound-width ablation (5 seeds)",
              ["Setting", "All 19", "Residual", "Reduction"])
    for lab, tag, _, _ in C.BIOSR_SETTINGS:
        a = C.ablation_width(tag, "all")[0]
        r = C.ablation_width(tag, "residual")[0]
        t.add(lab, C.fmt(a, 4), C.fmt(r, 4), f"{C.fmt(100 * (1 - a / r), 1)}\\%")
    return t


def supp_monitoring():
    t = Table("supp_monitoring", "Supp. -- Reference-free monitoring on genuine BioSR measurements",
              ["Indicator", "Mean rho", "SD"])
    stats = [(lab, *C.seeds_stat("gate5_real_grid", k)) for k, lab in C.BLIND_LABELS]
    for lab, m, s in sorted(stats, key=lambda x: -x[1]):
        t.add(lab, f3(m), f3(s))
    return t


def supp_real():
    t = Table("supp_real", "Supp. -- In-domain 90% coverage on real measurements (10 seeds)",
              ["Dataset / reconstructor", "n_test", "Median 1-SSIM", "Coverage"])
    for r in C.load("real_coverage")["rows"]:
        t.add(f"{r['dataset']} / {r['reconstructor']}", r["n_test"], f3(r["median_error"]),
              pmfmt((r["coverage_mean"], r["coverage_std"])))
    return t


TABLES = [table2, supp_calibration_audit, supp_family, supp_cross_setting,
          supp_physical_endpoint, supp_metric_comparison, supp_bound_summary,
          supp_feature_width, supp_monitoring, supp_real]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Paper tables, regenerated", "",
          "Built by `python paper/make_tables.py` from `paper/results/*.json`. "
          "Every value is checked against the manuscript by `paper/check_claims.py`.", ""]
    failed = 0
    for fn in TABLES:
        try:
            t = fn()
        except C.MissingResult as exc:
            print(f"  {fn.__name__}: skipped -- {exc}")
            failed += 1
            continue
        (OUT / f"{t.name}.tex").write_text(t.tex(), encoding="utf-8")
        md.append(t.md())
        print(f"  wrote paper/tables/{t.name}.tex")
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("  wrote paper/tables/tables.md")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

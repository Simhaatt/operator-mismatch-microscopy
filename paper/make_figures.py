"""Rebuild the six figures of the paper from ``paper/results/*.json``.

    python paper/make_figures.py            # writes paper/figures/*.pdf and *.png

File names match the ``\\includegraphics`` calls in the manuscript, so the PDFs
can be dropped into the LaTeX source unchanged:

    fig1_operator_mismatch_v2.pdf   Fig. 1   family_rl100
    fig2a_dplus_bound.pdf           Fig. 2a  gate5_biosr (seed-0 cells)
    fig2b_tv_bound_corrected.pdf    Fig. 2b  gate5_biosr (seed-0 cells)
    fig3_cross_instrument.pdf       Fig. 3   phase2_* (5 seeds)
    fig4a_efficiency_gain.pdf       Fig. 4a  ablation_* (5 seeds)
    fig4b_real_monitoring.pdf       Fig. 4b  gate5_real_grid (10 seeds)

Run ``paper/run_analyses.py`` first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.ROOT / "paper" / "figures"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
MARKERS = ["o", "s", "^", "D"]
TARGET = 0.90

plt.rcParams.update({
    "font.size": 10,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "svg.hashsalt": "paper",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "paper/make_figures.py", "CreationDate": None}
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata=meta)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf")


def target_line(ax, label=True):
    ax.axhline(TARGET, ls="--", color=COLORS[0], lw=1.6)
    if label:
        ax.text(0.99, TARGET + 0.004, "Target 0.90", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9)


# ---------------------------------------------------------------- Fig. 1 ---
def fig1():
    d = C.load("family_rl100")
    sev = ["nominal", "mild", "moderate", "severe", "extreme"]
    labels = [s.capitalize() for s in sev]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"wspace": 0.35})
    for i, fam in enumerate(C.FAMILIES):
        rows = {r["severity"]: r for r in d["phenomenon"] if r["family"] == fam}
        err = [1.0] + [rows[s]["error_ratio"] for s in sev[1:]]
        res = [1.0] + [rows[s]["residual_ratio"] for s in sev[1:]]
        kw = dict(color=COLORS[i], marker=MARKERS[i], lw=2.5, ms=8, label=fam.capitalize())
        axs[0].plot(range(5), err, **kw)
        axs[1].plot(range(5), res, **kw)
        cov = {r["severity"]: r["coverage"] for r in d["coverage_by_family"]
               if r["family"] == fam}
        axs[2].plot([0, 1], [cov["severe"], cov["extreme"]], **kw)
    for ax, ylab in ((axs[0], "Error / nominal"), (axs[1], "Residual / nominal")):
        ax.axhline(1.0, ls="--", color=COLORS[0], lw=1.6)
        ax.set_xticks(range(5), labels, rotation=25, ha="right")
        ax.set_ylabel(ylab)
    axs[2].set_xticks([0, 1], ["Severe", "Extreme"])
    axs[2].set_ylabel("Coverage")
    target_line(axs[2])
    for ax, t in zip(axs, ("(a) Reconstruction error",
                           "(b) Nominal data-consistency residual",
                           "(c) 90% empirical coverage")):
        ax.set_title(t, y=-0.42, fontsize=11)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig1_operator_mismatch_v2")


# ---------------------------------------------------------------- Fig. 2 ---
def _scatter_cells(ax, cells, xkey):
    for i, (_, _, stem, label) in enumerate(C.BIOSR_SETTINGS):
        pts = [c for c in cells if c["tag"] == stem]
        ax.scatter([c[xkey] for c in pts], [c["coverage"] for c in pts], s=70,
                   color=COLORS[i], marker=MARKERS[i], label=label, zorder=3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Empirical coverage")


def fig2():
    cells = C.gate5_cells("gate5_biosr")
    xmax = max(max(c["d_plus"] for c in cells), max(c["tv_score"] for c in cells)) + 0.05

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    _scatter_cells(ax, cells, "d_plus")
    ax.plot([0, xmax], [TARGET, TARGET - xmax], ls="--", color=COLORS[0], lw=1.6,
            label=r"$1-\alpha-d^{+}$")
    ax.axhline(TARGET, ls=":", color=COLORS[0], lw=1.6, label=r"target $1-\alpha$")
    ax.set_xlim(-0.02, xmax)
    ax.set_xlabel(r"One-sided score shift $d^{+}$")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    save(fig, "fig2a_dplus_bound")

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    _scatter_cells(ax, cells, "tv_score")
    ax.axhline(TARGET, ls="--", color=COLORS[0], lw=1.6, label="Target 0.90")
    ax.set_xlim(-0.02, xmax)
    ax.set_xlabel(r"Histogram-TV score $d^{\mathrm{hist}}_{\mathrm{TV}}$")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    save(fig, "fig2b_tv_bound_corrected")


# ---------------------------------------------------------------- Fig. 3 ---
def fig3():
    axes = [C.AXIS_IN, C.AXIS_ZERN, C.AXIS_PHYS]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.25})
    panels = (("BioSR", [p for p in C.PHASE2 if p[0] == "BioSR"]),
              ("W2S", [p for p in C.PHASE2 if p[0] == "W2S"]))
    lows = []
    for ax, (inst, rows) in zip(axs, panels):
        for i, (_, setting, tag) in enumerate(rows):
            stats = [C.phase2_cov(tag, a) for a in axes]
            m = [s[0] for s in stats]
            sd = [s[1] for s in stats]
            lows.append(min(a - b for a, b in zip(m, sd)))
            recon, ph = setting.split("-")
            ax.errorbar(range(3), m, yerr=sd, color=COLORS[i], marker=MARKERS[i], lw=2.5,
                        ms=8, capsize=3, label=f"{recon}, {ph} ph")
        target_line(ax)
        ax.set_xticks(range(3), ["In-domain", "Zernike", "Physical"])
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"({'ab'[panels.index((inst, rows))]}) {inst}", y=-0.3, fontsize=11)
    lo = min(lows) - 0.02
    for ax in axs:
        ax.set_ylim(lo, 0.95)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig3_cross_instrument")


# ---------------------------------------------------------------- Fig. 4 ---
def fig4():
    import numpy as np

    labels = [s[0] for s in C.BIOSR_SETTINGS]
    res = [C.ablation_width(s[1], "residual") for s in C.BIOSR_SETTINGS]
    full = [C.ablation_width(s[1], "all") for s in C.BIOSR_SETTINGS]
    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(x - w / 2, [r[0] for r in res], w, yerr=[r[1] for r in res], capsize=3,
           color=COLORS[0], label="Residual only")
    ax.bar(x + w / 2, [f[0] for f in full], w, yerr=[f[1] for f in full], capsize=3,
           color=COLORS[1], label="All 19 features")
    for xi, r, f in zip(x, res, full):
        ax.text(xi - w / 2, r[0] + 0.02, f"{100 * (1 - f[0] / r[0]):.0f}%\nsmaller",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(r[0] for r in res) * 1.3)
    ax.set_ylabel("Mean in-domain bound width")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig4a_efficiency_gain")

    stats = [(lab, *C.seeds_stat("gate5_real_grid", key)) for key, lab in C.BLIND_LABELS]
    stats.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    y = np.arange(len(stats))
    ax.errorbar([s[1] for s in stats], y, xerr=[s[2] for s in stats], fmt="o",
                color=COLORS[0], ms=7, capsize=3, lw=1.6)
    ax.axvline(0, ls="--", color=COLORS[0], lw=1.6)
    ax.set_yticks(y, [s[0] for s in stats])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Spearman correlation $\rho$ with coverage")
    save(fig, "fig4b_real_monitoring")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGS), default=None)
    args = ap.parse_args()
    failed = 0
    for name in args.only or FIGS:
        try:
            FIGS[name]()
        except C.MissingResult as exc:
            print(f"  {name}: skipped -- {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

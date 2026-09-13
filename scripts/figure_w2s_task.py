"""Figures 6 and 7: the second instrument, and the choice of error definition.

Figure 6 puts the two instruments side by side. Left: the bound holds in every
evaluated cell on both, which is the paper's central claim and the thing that
transfers. Right: the separation between the one-sided shift and symmetric total
variation, which is large and consistent on BioSR and not established on W2S --
shown with a matched-size control so the reader can see how much of the gap is
sample size and how much is not.

Figure 7 is the task result, and it is drawn as an audit rather than as a
league table. Coverage alone would be misleading, because the held-out structure
is not equally hard for every error definition. Each point is therefore placed
at its *shift ratio* -- median held-out error over median calibration error --
so a bound that looks good only because the shift ran in its favour is visibly
to the left. The readable result is the vertical gap between SSIM at a
favourable ratio and the puncta count at an adverse one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import style  # noqa: E402

GATE = Path("experiments/gate4_haze31")
TASK = Path("experiments/features29_task")


def bound_panel(ax):
    """Cells evaluated per configuration; the bound holds in all of them.

    Drawn as a single bar with an explicit "n of n" annotation rather than as
    held-versus-total. A stacked or overlaid pair would put a grey bar exactly
    behind every blue one -- invisible, and a legend entry for something the
    reader cannot see is worse than no legend at all.
    """
    rows = [
        ("BioSR\n4 settings", "gate5_haze31.json"),
        ("BioSR\n2 settings", "gate5_biosr_2settings.json"),
        ("W2S\n2 settings", "gate5_w2s.json"),
        ("W2S\nreal", "gate5_w2s_real.json"),
    ]
    labels, held, total, colours = [], [], [], []
    for label, fn in rows:
        p = GATE / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        labels.append(label)
        held.append(d["across_seeds"]["holds_1s"][0])
        total.append(sum(len(v) for v in d["cells"].values()))
        colours.append(style.REAL if label.startswith("W2S") else style.SIM)

    x = np.arange(len(labels))
    ax.bar(x, total, color=colours, width=0.62)
    for xi, h, t in zip(x, held, total):
        ax.text(xi, t + 0.4, f"{h:.0f}/{t}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("cells evaluated")
    ax.set_ylim(0, max(total) * 1.28)
    ax.set_title("bound holds in every cell, every seed", fontsize=8, pad=4)


def contrast_panel(ax):
    """rho(coverage, -d+) against rho(coverage, TV), with the matched control."""
    rows = [
        ("BioSR, 16 cells", "gate5_haze31.json", style.SIM, "o"),
        ("BioSR, 8 cells", "gate5_biosr_2settings.json", style.PALETTE[4], "^"),
        ("W2S, 6 cells", "gate5_w2s.json", style.REAL, "s"),
    ]
    for i, (label, fn, colour, marker) in enumerate(rows):
        p = GATE / fn
        if not p.exists():
            continue
        a = json.load(open(p))["across_seeds"]
        for j, key in enumerate(("rho_cov_dplus", "rho_cov_tv")):
            m, s = a[key][0], a[key][1]
            ax.errorbar(i + (j - 0.5) * 0.24, m, yerr=s, fmt=marker, color=colour,
                        markersize=5, capsize=2.5,
                        markerfacecolor=colour if j == 0 else "white",
                        markeredgecolor=colour,
                        label=label if j == 0 else None)
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel(r"$\rho$ with coverage")
    ax.set_ylim(-0.7, 1.15)
    ax.text(0.02, 0.04, "filled: one-sided $d_+$    open: symmetric TV",
            transform=ax.transAxes, fontsize=6.5, color="0.35")


def task_panel(ax):
    """Coverage against shift ratio, separating RL from the confounded TV runs.

    Coverage alone would mislead: the held-out structure is not equally hard for
    every error definition, so a bound can look good merely because the shift ran
    in its favour. Plotting against the shift ratio makes that visible.

    Richardson-Lucy and Poisson-TV are drawn with filled and open markers because
    the two TV puncta points sit at ratios of 0.53 and 0.61 -- the held-out
    structure is about half as hard for a counting task there -- and their high
    coverage is an artefact of that, not evidence of transfer. Showing them with
    the same symbol as the RL points would assert exactly the claim the audit
    withdrew.
    """
    series = {
        "1-SSIM": ("e", "e_ssim", style.PALETTE[0], "o"),
        "puncta count": ("task_puncta_count", "task_puncta_count",
                         style.PALETTE[1], "s"),
        "skeleton length": ("task_skeleton_length", "task_skeleton_length",
                            style.PALETTE[2], "^"),
    }
    caches = ["rl_photons_100", "rl_photons_15", "tv_photons_100", "tv_photons_15"]
    for label, (col, covkey, colour, marker) in series.items():
        for tag in caches:
            f, c = TASK / f"{tag}.npz", GATE / f"task_{tag}.json"
            if not (f.exists() and c.exists()):
                continue
            d = np.load(f, allow_pickle=True)
            cal, ood = d[f"cal_{col}"], d[f"ood_{col}"]
            cal, ood = cal[np.isfinite(cal)], ood[np.isfinite(ood)]
            x = float(np.median(ood) / np.median(cal))
            y = json.load(open(c))["coverage"][covkey]["content shift"]["coverage"][0]
            is_rl = tag.startswith("rl")
            ax.scatter(x, y, s=38, marker=marker, zorder=3,
                       color=colour if is_rl else "white",
                       edgecolor=colour, linewidth=1.1,
                       label=label if (is_rl and tag == "rl_photons_100") else None)

    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    ax.axhline(0.9, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.3", "0.5", "1", "2", "4"])
    ax.set_xlabel("shift ratio   (held-out error / calibration error)")
    ax.set_ylabel("coverage under content shift")
    ax.set_ylim(-0.05, 1.12)
    ax.text(0.42, 1.06, "shift favours the bound", fontsize=6.5, color="0.35",
            ha="center")
    ax.text(2.4, 1.06, "shift works against it", fontsize=6.5, color="0.35",
            ha="center")
    ax.text(0.98, 0.02, "filled: Richardson-Lucy    open: Poisson-TV",
            transform=ax.transAxes, fontsize=6.5, color="0.35", ha="right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    style.use()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    bound_panel(axes[0])
    contrast_panel(axes[1])
    style.save(fig, "fig6_second_instrument")

    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    task_panel(ax)
    style.legend_above(fig, ax, ncol=3)
    style.save(fig, "fig7_task_audit")


if __name__ == "__main__":
    main()

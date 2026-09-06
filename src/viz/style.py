"""Shared figure style, so every panel in the paper looks like the same paper.

Two constraints shape this. Figures are read at print size in a single column,
so nothing may rely on a detail smaller than about a point; and the palette has
to survive greyscale printing and the common colour-vision deficiencies, so
series are distinguished by lightness order and marker as well as by hue.

The palette is Okabe-Ito, which is designed for exactly that and is ordered here
so that adjacent series also differ in lightness.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[2] / "figures"

#: Okabe-Ito, ordered so neighbouring series differ in lightness as well as hue.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

#: Consistent identity for the four reconstructors, used across every figure.
RECON = {
    "RL": PALETTE[0], "TV": PALETTE[1],
    "U-Net": PALETTE[2], "unrolled": PALETTE[3],
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: One colour for simulated, one for real, used the same way in every figure.
SIM, REAL = PALETTE[0], PALETTE[1]


def use() -> None:
    """Apply the paper style. Call once before building any figure."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "errorbar.capsize": 2,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save(fig, name: str, provenance: str | None = None) -> Path:
    """Write a figure to `figures/`, stamped with where its numbers came from.

    The stamp is not decoration. Eleven claims in this project were retracted,
    several because a number could not be traced to the run that produced it. A
    figure that names its cache and commit can be checked; one that does not has
    to be trusted.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    if provenance:
        # Reserve a strip first, or the stamp lands on top of the x-axis label.
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.text(0.005, 0.004, provenance, fontsize=4.5, color="0.55",
                 ha="left", va="bottom")
    else:
        fig.tight_layout()
    out = FIGDIR / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    return out


def stamp(*parts: str) -> str:
    """Provenance line: the git commit plus whatever sources are named."""
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=FIGDIR.parent, capture_output=True, text=True,
                           check=True).stdout.strip()
    except Exception:
        h = "unknown"
    return " | ".join((f"commit {h}", *parts))

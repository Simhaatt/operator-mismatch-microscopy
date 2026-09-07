"""Figure 1: the coverage bound, and why one-sidedness is the whole result.

Left panel puts every (cache, held-out structure) cell from every setting on the
same axes against its one-sided score shift `d+`, with the bound
`coverage >= 1 - alpha - d+` drawn as a line. A point above the line is a cell
where the bound holds. Right panel repeats it against the symmetric total
variation distance, which is the comparison that makes the point: the same cells,
a valid bound, and no predictive content at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import style  # noqa: E402

#: Each setting contributes its cells; the label says what the reader is seeing.
SETTINGS = [
    ("bound_bgprop_seeds.json", "simulated, RL and TV", style.SIM, "o"),
    ("bound_4recon.json", "simulated, four reconstructors", style.PALETTE[2], "^"),
    ("bound_real_grid.json", "REAL measurements", style.REAL, "s"),
]


def load(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    cov, dp, tv = [], [], []
    for cache in d["cells"].values():
        for cell in cache.values():
            cov.append(cell["coverage"])
            dp.append(cell["d_plus"])
            tv.append(cell["tv_score"])
    return np.array(cov), np.array(dp), np.array(tv), d["alpha"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="experiments/gate5")
    ap.add_argument("--name", default="fig1_bound")
    args = ap.parse_args()

    style.use()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    alpha, n_tot, n_held = 0.1, 0, 0
    for fname, label, colour, marker in SETTINGS:
        p = Path(args.dir) / fname
        if not p.exists():
            print(f"  skipping absent {fname}")
            continue
        cov, dp, tv, alpha = load(p)
        n_tot += cov.size
        n_held += int((cov >= 1 - alpha - dp).sum())
        for ax, x in zip(axes, (dp, tv)):
            ax.scatter(x, cov, s=14, c=colour, marker=marker, label=label,
                       alpha=0.85, edgecolors="none")

    grid = np.linspace(0, 1, 200)
    for ax, xlabel, title in zip(
            axes,
            (r"one-sided score shift  $d^+$", r"symmetric distance  $d_{TV}$"),
            (r"$\mathrm{coverage} \geq 1-\alpha-d^+$", "same cells, symmetric bound")):
        ax.plot(grid, 1 - alpha - grid, color="0.35", lw=1.0, ls="--",
                label=r"bound $1-\alpha-d$", zorder=1)
        ax.axhline(1 - alpha, color="0.65", lw=0.8, ls=":", zorder=0,
                   label=r"nominal $1-\alpha$")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("realised coverage")
    style.legend_above(fig, axes[0], ncol=5)

    out = style.save(fig, args.name, style.stamp(
        f"{n_held}/{n_tot} cells above the one-sided bound",
        "experiments/gate5/*.json"))
    print(f"wrote {out}  ({n_held}/{n_tot} cells satisfy the bound)")


if __name__ == "__main__":
    main()

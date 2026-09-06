"""Figure 3: the forward model's missing out-of-focus light, and what it costs.

Three panels, one claim each.

* Left: the measured out-of-focus fraction of real BioSR widefield measurements,
  per structure, against the 5% the project originally simulated.
* Middle: pairs of caches that share photon count *and* iteration count and
  differ only in haze. This is the panel that carries retraction 11 -- the
  original comparison moved dose, haze and iterations together, and once they
  are held fixed haze alone flips the correlation's sign.
* Right: dose at constant haze, which moves in the *opposite* direction to the
  retracted claim's prediction.

The middle and right panels are deliberately separate. Plotting both on a single
haze axis piles the constant-haze sweep into one vertical smear and hides the
very distinction the figure exists to draw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.stats import spearman  # noqa: E402
from src.viz import style  # noqa: E402

RHO_LABEL = r"$\rho(r_{\mathrm{nominal}},\ \mathrm{error})$"

#: Matched pairs: same dose, same iterations, only the haze differs.
PAIRS = [
    ("100 photons, 75 iters", "rl_photons_100", 5 / 105,
     "rl_photons_100_bg45", 45 / 145),
    ("15 photons", "rl_photons_15", 5 / 20, "rl_sweep_15", 6.74 / 21.74),
]

#: The dose sweep, all at haze 0.31 and 75 iterations.
SWEEP = [(15, "rl_sweep_15"), (30, "rl_sweep_30"), (60, "rl_sweep_60"),
         (100, "rl_photons_100_bg45"), (200, "rl_sweep_200")]


def residual_rho(cache: Path) -> float | None:
    """Spearman correlation of the plain residual with true error, all splits."""
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    i = names.index("r_nominal")
    X = np.concatenate([d[f"{s}_X"] for s in ("train", "cal", "test", "ood")])
    e = np.concatenate([d[f"{s}_e"] for s in ("train", "cal", "test", "ood")])
    return spearman(X[:, i], e)


def panel_measured(ax, hz) -> None:
    per = hz["per_structure"]
    order = sorted(per, key=lambda k: per[k]["median"])
    fracs = [[c["frac"] for c in hz["cells"] if c["structure"] == s] for s in order]
    parts = ax.violinplot(fracs, positions=np.arange(len(order)), widths=0.75,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(style.REAL)
        pc.set_alpha(0.45)
        pc.set_edgecolor("none")
    parts["cmedians"].set_color("0.2")
    parts["cmedians"].set_linewidth(1.2)
    ax.axhline(5 / 105, color=style.SIM, lw=1.3, ls="--",
               label="simulated (5%)")
    ax.axhline(hz["pooled_median"], color="0.35", lw=1.0, ls=":",
               label=f"measured ({hz['pooled_median']:.2f})")
    ax.set_xticks(np.arange(len(order)))
    short = {"Microtubules": "MT", "F-actin": "F-act"}
    ax.set_xticklabels([f"{short.get(s, s)}\nn={per[s]['n']}" for s in order],
                       fontsize=6.5)
    ax.set_ylabel("out-of-focus fraction")
    ax.set_title(f"Measured, {len(hz['cells'])} real cells")
    ax.set_ylim(0, 0.9)
    ax.legend(loc="upper left")


def panel_matched(ax, features: Path) -> None:
    for k, (lab, ca, ha, cb, hb) in enumerate(PAIRS):
        ra = residual_rho(features / f"{ca}.npz")
        rb = residual_rho(features / f"{cb}.npz")
        if ra is None or rb is None:
            print(f"  skipping pair {lab}: cache absent")
            continue
        colour = style.PALETTE[k]
        ax.plot([ha, hb], [ra, rb], "o", color=colour, label=lab, zorder=3)
        ax.annotate("", xy=(hb, rb), xytext=(ha, ra), zorder=2,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.1,
                                    shrinkA=4, shrinkB=4))
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("out-of-focus fraction")
    ax.set_ylabel(RHO_LABEL)
    ax.set_title("Haze alone flips the sign")
    ax.set_ylim(-0.45, 0.8)
    ax.set_xlim(0, 0.42)
    ax.legend(loc="upper right")


def panel_dose(ax, features: Path) -> None:
    xs, ys = [], []
    for ph, cache in SWEEP:
        r = residual_rho(features / f"{cache}.npz")
        if r is None:
            print(f"  skipping absent {cache}")
            continue
        xs.append(ph)
        ys.append(r)
    ax.plot(xs, ys, "-o", color=style.PALETTE[2], zorder=3)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlabel("photons (haze 0.31, 75 iters)")
    ax.set_title("Dose runs the other way")
    ax.set_ylim(-0.45, 0.8)
    ax.text(0.5, 0.96, "the retracted claim predicted\nan increase with dose",
            transform=ax.transAxes, fontsize=6, color="0.4",
            ha="center", va="top")
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--haze", default="experiments/realism/haze_fraction.json")
    ap.add_argument("--features", default="experiments/features19")
    ap.add_argument("--name", default="fig3_haze")
    args = ap.parse_args()

    style.use()
    import matplotlib.pyplot as plt

    hz = json.loads(Path(args.haze).read_text(encoding="utf-8"))
    features = Path(args.features)

    fig, (axl, axm, axr) = plt.subplots(1, 3, figsize=(7.0, 2.5))
    panel_measured(axl, hz)
    panel_matched(axm, features)
    xs, ys = panel_dose(axr, features)

    out = style.save(fig, args.name, style.stamp(
        Path(args.haze).name, "features19/*.npz", "n_iter fixed at 75"))
    print(f"wrote {out}")
    for ph, r in zip(xs, ys):
        print(f"  {ph:4d} ph at haze 0.31 -> rho {r:+.3f}")


if __name__ == "__main__":
    main()

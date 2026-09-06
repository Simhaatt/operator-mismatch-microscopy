"""Figure 2: no label-free warning signal survives contact with real data.

Seven blind statistics, each scored by how well its calibration-to-test
divergence predicts realised coverage loss, over ten partition seeds. The
simulated bars are the result the method would be sold on; the real bars are
what it does on real measurements. The gap between the two panels is the
project's sharpest limitation, so the figure is built to make that gap the thing
the eye lands on rather than the ranking within either panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import style  # noqa: E402

LABELS = {
    "qpred": "predicted\nquantile",
    "all": "all 19\nfeatures",
    "oscr": "14 OSCR\nfeatures",
    "otf": "5 OTF\nfeatures",
    "otf_core": "3 OTF\n(selected)",
    "residual": r"$r_{\mathrm{nominal}}$" + "\nalone",
    "blind_score": r"$z(r)-z(q)$",
}
ORDER = ["qpred", "all", "oscr", "otf", "otf_core", "residual", "blind_score"]


def read(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    a = d.get("across_seeds") or {}
    return {k[6:]: (v[0], v[1]) for k, v in a.items() if k.startswith("blind_")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", default="experiments/gate5/bound_bgprop_seeds.json")
    ap.add_argument("--real", default="experiments/gate5/bound_real_grid.json")
    ap.add_argument("--name", default="fig2_blind")
    args = ap.parse_args()

    sim, real = read(Path(args.sim)), read(Path(args.real))
    keys = [k for k in ORDER if k in sim or k in real]

    style.use()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(keys))
    w = 0.38
    for off, src, label, colour in ((-w / 2, sim, "simulated (16 cells)", style.SIM),
                                    (+w / 2, real, "REAL (16 cells)", style.REAL)):
        m = np.array([src.get(k, (np.nan, np.nan))[0] for k in keys])
        e = np.array([src.get(k, (np.nan, np.nan))[1] for k in keys])
        ax.bar(x + off, m, w, yerr=e, color=colour, label=label,
               error_kw={"lw": 0.8, "ecolor": "0.3"})

    ax.axhline(0, color="0.3", lw=0.8)
    # The threshold is not arbitrary: below it a statistic explains less than a
    # quarter of the coverage ordering, which is no use as a deployment warning.
    ax.axhspan(-1, 0.25, color="0.85", alpha=0.35, zorder=0)
    ax.text(-0.45, -0.53,
            "shaded: below 0.25 a statistic explains too little of the coverage"
            " ordering to serve as a deployment warning",
            fontsize=6, color="0.4", ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylabel(r"$\rho(\mathrm{coverage},\, -d^+_{\mathrm{blind}})$")
    ax.set_title("Can coverage loss be anticipated without ground truth?")
    ax.set_ylim(-0.55, 0.8)
    ax.legend(loc="upper right")

    out = style.save(fig, args.name, style.stamp(
        "10 partition seeds, mean +- sd",
        Path(args.sim).name, Path(args.real).name))
    print(f"wrote {out}")
    for k in keys:
        s = sim.get(k, (float('nan'),) * 2); r = real.get(k, (float('nan'),) * 2)
        print(f"  {k:12s} sim {s[0]:+.3f}+-{s[1]:.3f}   real {r[0]:+.3f}+-{r[1]:.3f}")


if __name__ == "__main__":
    main()

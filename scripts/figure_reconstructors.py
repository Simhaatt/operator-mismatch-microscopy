"""Figure 5: the mechanism does not depend on a data-fidelity term.

Left: the blind features' correlation with the oracle operator mismatch, on the
held-out structure, for all four reconstructors. The U-Net has no explicit
data-fidelity term and tracks Richardson-Lucy almost exactly -- which is the
decisive result, because it means the mechanism is a property of the inverse
problem rather than an artefact of fidelity-minimising algorithms.

The unrolled network is the informative outlier. Its forward pass takes explicit
gradient steps on the least-squares residual against the *nominal* operator, so
it minimises precisely what `r_nominal` measures under an operator that may be
wrong. `r_nominal` goes blind and `r_min` flips sign, while `r_gap` and
`otf_slope` reach their highest values of any reconstructor: the physics profile
earns most exactly where the plain residual earns least.

Right: `otf_excess_at_cut` per specimen type. All four vary somewhat, but the
U-Net varies three times as much as any other (range 0.909 against 0.106 for RL,
0.273 for TV and 0.311 for the unrolled net) and is the only one that changes
sign. It is weakest on the content it was trained on and strongest on the
structure it never saw, so a learned reconstructor can silently disable a
physics-derived feature -- and only for familiar specimens.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.stats import spearman  # noqa: E402
from src.viz import style  # noqa: E402

CACHES = [
    ("RL", "rl_photons_100_bg45"),
    ("TV", "tv_photons_100_bg45"),
    ("U-Net", "unet_photons_100_haze31"),
    ("unrolled", "unrolled_photons_100_haze31"),
]

#: The features that carry the argument, in the order it is made.
FEATURES = ["r_nominal", "r_min", "r_curvature", "r_gap", "otf_slope"]
PRETTY = {"r_nominal": r"$r_{\mathrm{nom}}$", "r_min": r"$r_{\min}$",
          "r_curvature": "curvature", "r_gap": r"$r_{\mathrm{gap}}$",
          "otf_slope": "OTF slope"}
STRUCTURES = ["F-actin", "Microtubules", "ER", "CCPs"]
SHORT = {"F-actin": "F-act", "Microtubules": "MT", "ER": "ER", "CCPs": "CCPs"}


def load(features: Path, cache: str):
    p = features / f"{cache}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    splits = ("train", "cal", "test", "ood")
    return {
        "names": names,
        "ood_X": d["ood_X"], "ood_rms": d["ood_rms"],
        "X": np.concatenate([d[f"{s}_X"] for s in splits]),
        "rms": np.concatenate([d[f"{s}_rms"] for s in splits]),
        "structure": np.concatenate([d[f"{s}_structure"] for s in splits]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="experiments/features19")
    ap.add_argument("--name", default="fig5_reconstructors")
    args = ap.parse_args()

    style.use()
    import matplotlib.pyplot as plt

    features = Path(args.features)
    data = {lab: load(features, c) for lab, c in CACHES}
    data = {k: v for k, v in data.items() if v is not None}

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.8, 2.9),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- left: key features, held-out structure --------------------------
    xs = np.arange(len(FEATURES))
    w = 0.8 / max(len(data), 1)
    for k, (lab, d) in enumerate(data.items()):
        vals = [spearman(d["ood_X"][:, d["names"].index(f)], d["ood_rms"])
                for f in FEATURES]
        axl.bar(xs + (k - (len(data) - 1) / 2) * w, vals, w,
                color=style.RECON.get(lab, style.PALETTE[k]), label=lab)
    axl.axhline(0, color="0.3", lw=0.8)
    axl.set_xticks(xs)
    axl.set_xticklabels([PRETTY[f] for f in FEATURES])
    axl.set_ylabel(r"$\rho(\mathrm{feature},\ \mathrm{oracle\ mismatch})$")
    axl.set_title("Held-out structure: U-Net tracks RL")
    axl.set_ylim(-1.05, 1.0)


    # ---- right: the one feature that is content-dependent ----------------
    i_ord = np.arange(len(STRUCTURES))
    for k, (lab, d) in enumerate(data.items()):
        j = d["names"].index("otf_excess_at_cut")
        vals = []
        for s in STRUCTURES:
            m = d["structure"] == s
            vals.append(spearman(d["X"][m, j], d["rms"][m]) if m.any() else np.nan)
        axr.plot(i_ord, vals, "-o", color=style.RECON.get(lab, style.PALETTE[k]),
                 label=lab, markersize=3.5)
    axr.axhline(0, color="0.3", lw=0.8)
    axr.set_xticks(i_ord)
    axr.set_xticklabels([SHORT[s] for s in STRUCTURES], fontsize=6.5)
    axr.set_ylabel(r"$\rho(\mathrm{otf\_excess\_at\_cut},\ \mathrm{mismatch})$",
                   fontsize=7)
    axr.set_title("Only the U-Net changes sign with content")
    axr.set_ylim(-1.0, 0.35)

    style.legend_above(fig, axl, ncol=4)

    out = style.save(fig, args.name, style.stamp(
        "100 photons, haze 0.31", "features19/*.npz"))
    print(f"wrote {out}")
    for lab, d in data.items():
        r = spearman(d["ood_X"][:, d["names"].index("r_nominal")], d["ood_rms"])
        g = spearman(d["ood_X"][:, d["names"].index("r_gap")], d["ood_rms"])
        print(f"  {lab:10s} r_nominal {r:+.3f}   r_gap {g:+.3f}")


if __name__ == "__main__":
    main()

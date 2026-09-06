"""Figure 4: calibrating on simulation fails; calibrating on real data works.

Left: the same held-out-structure experiment run on simulated and on real
measurements, both calibrated in-domain so the two are directly comparable. Two
claims at once -- CCPs reverses from the worst structure to the best, which the
haze diagnosis predicts since CCPs is the structure whose *simulated* features
were most distorted; and content shift does not disappear on real data, it
relocates to F-actin.

Right: in-distribution coverage on real measurements for all four
reconstructors, with the learned pair's much larger content-shift spread shown
as the error bar. The method does not depend on the reconstructor; its
robustness to content shift does.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz import style  # noqa: E402

#: Simulated per-structure held-out coverage, from the 10-seed rotation logs.
SIM_LOG = "experiments/logs/bg45_rotate_rl_photons_100_bg45.log"
REAL_LOG = "experiments/logs/real_grid_rotate_real_rl.log"

RECON_LOGS = [
    ("RL", "experiments/logs/real_grid_rotate_real_rl.log"),
    ("TV", "experiments/logs/real_grid_rotate_real_tv.log"),
    ("U-Net", "experiments/logs/real_grid_rotate_real_unet.log"),
    ("unrolled", "experiments/logs/real_grid_rotate_real_unrolled.log"),
]

STRUCTURES = ["CCPs", "ER", "F-actin", "Microtubules"]


def parse_rotation(path: Path) -> dict:
    """Pull per-structure held-out coverage and the summary out of a log.

    Parsing the log rather than a JSON because `rotate_structures.py` prints its
    10-seed table and only writes per-cell JSON; the printed table is the
    aggregated form the figure needs. Kept narrow and asserted so a format
    change fails loudly instead of yielding a plausible plot.
    """
    if not path.exists():
        return {}
    out = {"per": {}}
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"\s+(\S+)\s+([\d.]+) \+-([\d.]+)\s+([\d.]+) \+-([\d.]+)", line)
        if m and m.group(1) in STRUCTURES:
            out["per"][m.group(1)] = (float(m.group(4)), float(m.group(5)))
        m = re.search(r"in-distribution coverage\s*:\s*([\d.]+) \+- ([\d.]+)", line)
        if m:
            out["indist"] = (float(m.group(1)), float(m.group(2)))
        m = re.search(r"held-out coverage\s*:\s*([\d.]+) \+- ([\d.]+)", line)
        if m:
            out["heldout"] = (float(m.group(1)), float(m.group(2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="fig4_transfer")
    args = ap.parse_args()

    style.use()
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- left: the SAME experiment on simulated and on real measurements --
    # Both sides hold out a structure and calibrate in-domain, so they are
    # directly comparable. An earlier draft put the sim-to-real *transfer*
    # numbers on the left instead, which is a different experiment: it
    # calibrates on simulation and tests on real. Mixing the two would have
    # made the CCPs reversal look like it came from transfer rather than from
    # the change of measurement.
    sim = parse_rotation(Path(SIM_LOG))
    real = parse_rotation(Path(REAL_LOG))
    sim_cal = {s: sim["per"].get(s, (np.nan, np.nan))[0] for s in STRUCTURES}
    sim_err = [sim["per"].get(s, (np.nan, np.nan))[1] for s in STRUCTURES]
    x = np.arange(len(STRUCTURES))
    w = 0.38
    axl.bar(x - w / 2, [sim_cal.get(s, np.nan) for s in STRUCTURES], w,
            yerr=sim_err, color=style.SIM, label="simulated measurements",
            error_kw={"lw": 0.8, "ecolor": "0.3"})
    rm = [real["per"].get(s, (np.nan, np.nan))[0] for s in STRUCTURES]
    re_ = [real["per"].get(s, (np.nan, np.nan))[1] for s in STRUCTURES]
    axl.bar(x + w / 2, rm, w, yerr=re_, color=style.REAL,
            label="REAL measurements", error_kw={"lw": 0.8, "ecolor": "0.3"})
    axl.axhline(0.9, color="0.35", lw=1.0, ls="--")
    axl.text(len(STRUCTURES) - 0.55, 0.915, r"nominal $1-\alpha$", fontsize=6,
             color="0.4", ha="right", va="bottom")
    axl.set_xticks(x)
    axl.set_xticklabels(["CCPs", "ER", "F-act", "MT"])
    axl.set_ylabel("held-out coverage")
    axl.set_title("Held-out structure, in-domain calibration")
    axl.set_ylim(0, 1.09)
    axl.legend(loc="upper center", ncol=1)
    axl.annotate("CCPs reverses:\nworst to best", xy=(0.19, 1.0),
                 xytext=(0.55, 0.30), fontsize=6.5, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))

    # ---- right: real measurements, all four reconstructors ---------------
    labs, ind, indе, held, helde = [], [], [], [], []
    for lab, log in RECON_LOGS:
        r = parse_rotation(Path(log))
        if not r.get("indist"):
            print(f"  skipping {lab}: no summary in {log}")
            continue
        labs.append(lab)
        ind.append(r["indist"][0]); indе.append(r["indist"][1])
        held.append(r["heldout"][0]); helde.append(r["heldout"][1])
    xx = np.arange(len(labs))
    axr.errorbar(xx - 0.09, ind, yerr=indе, fmt="o", color=style.PALETTE[0],
                 label="in-distribution")
    axr.errorbar(xx + 0.09, held, yerr=helde, fmt="s", color=style.PALETTE[3],
                 label="held-out structure")
    axr.axhline(0.9, color="0.35", lw=1.0, ls="--")
    axr.set_xticks(xx)
    axr.set_xticklabels(labs, fontsize=6.5)
    axr.set_ylabel("coverage on real measurements")
    axr.set_title("In-domain calibration, four reconstructors")
    axr.set_ylim(0, 1.09)
    axr.set_xlim(-0.5, len(labs) - 0.5)
    axr.legend(loc="lower left")
    axr.text(len(labs) - 0.55, 0.915, r"nominal $1-\alpha$", fontsize=6,
             color="0.4", ha="right", va="bottom")

    out = style.save(fig, args.name, style.stamp(
        "10 partition seeds", "bg45_rotate_*.log", "real_grid_rotate_*.log"))
    print(f"wrote {out}")
    for s in STRUCTURES:
        print(f"  {s:14s} simulated {sim_cal.get(s, float('nan')):.3f}"
              f"   real {real['per'].get(s, (float('nan'),))[0]:.3f}")


if __name__ == "__main__":
    main()

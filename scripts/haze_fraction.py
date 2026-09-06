"""Measure the out-of-focus fraction of real BioSR widefield measurements.

The forward model's background level is now a *fraction* of the measurement mean
(`background_for` in `src.optics.forward_model`), and every learned
reconstructor has to be trained at the value this script reports. It was first
estimated from six cells, which is not enough to set a parameter that a GPU
session bakes in.

The estimate fits each real widefield measurement as

    y_real  ~  a * blur(GT) + b

and reports `b / mean(y_real)`. `a` absorbs the arbitrary scaling of a SIM
reconstruction and `b` is the flat pedestal the forward model has no way to
produce, because SIM reconstruction removes out-of-focus light from the ground
truth in the first place. See docs/sim2real.md.

Deliberately cheap: no deconvolution, one blur and one least-squares solve per
cell, so the whole dataset is minutes rather than hours.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realism_check import (  # noqa: E402
    WIDEFIELD_PIXEL_UM, decide_grouping, index_cell, real_widefield,
)
from scripts.sim2real import centre_crop, fourier_upsample  # noqa: E402
from src.data.mrc import read_mrc  # noqa: E402
from src.optics.forward_model import apply_psf  # noqa: E402
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402


def haze_of_cell(zf, raw_member, gt_member, groups, psf0, crop) -> dict:
    """Fit ``a * blur(GT) + b`` and report the pedestal as a fraction."""
    frames, _ = read_mrc(zf.read(raw_member))
    gt_arr, _ = read_mrc(zf.read(gt_member))
    frames = np.asarray(frames)
    gt = np.asarray(gt_arr[0], dtype=np.float64)
    if gt.shape[-1] != 2 * frames.shape[-1]:
        raise ValueError(f"GT {gt.shape} is not 2x the frames {frames.shape[-2:]}")

    y_cam = real_widefield(frames, WIDEFIELD_PIXEL_UM, groups)["y"]
    y = np.clip(centre_crop(fourier_upsample(y_cam, 2), crop), 0.0, None)

    gt_c = centre_crop(gt, crop)
    lo, hi = np.percentile(gt_c, [0.1, 99.9])
    gt_t = torch.from_numpy(np.clip((gt_c - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
                            ).to(torch.float32)
    blur = apply_psf(gt_t, psf0).numpy().ravel().astype(np.float64)

    A = np.stack([blur, np.ones_like(blur)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    pred = A @ np.array([a, b])
    ss = float(((y.ravel() - pred) ** 2).sum())
    tot = float(((y.ravel() - y.mean()) ** 2).sum())
    mean = float(y.mean())
    return {"a": float(a), "b": float(b), "mean": mean,
            "frac": float(max(b, 0.0) / max(mean, 1e-12)),
            "r2": float(1.0 - ss / tot) if tot > 0 else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", nargs="+",
                    default=["CCPs", "Microtubules", "F-actin", "ER"])
    ap.add_argument("--cells", type=int, default=0,
                    help="cells per structure; 0 means all of them")
    ap.add_argument("--level", type=int, default=0,
                    help="signal level; 0 means the middle one available")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--min-r2", type=float, default=0.5,
                    help="discard fits this poor; the two-term model has to "
                         "actually describe the measurement for b to mean "
                         "anything")
    ap.add_argument("--archives", default="data/biosr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(torch.float32)

    print("OUT-OF-FOCUS FRACTION of real BioSR widefield measurements")
    print("  fitting y_real ~ a * blur(GT) + b, reporting b / mean(y_real)")
    print(f"  crop {args.crop}, kernel {args.kernel}, discarding fits below "
          f"R2 {args.min_r2}")
    print()

    rows, skipped = [], []
    for structure in args.structures:
        path = Path(args.archives) / f"{structure}.zip"
        if not path.exists():
            skipped.append(f"{structure}: no archive"); continue
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            cells = sorted({n.split("/")[1] for n in names
                            if n.count("/") >= 2 and n.split("/")[1].startswith("Cell_")})
            if args.cells:
                cells = cells[: args.cells]
            done = 0
            for cell in cells:
                avail = index_cell(names, structure, cell)
                if not avail:
                    skipped.append(f"{structure}/{cell}: no raw/GT pair"); continue
                try:
                    fb, _ = read_mrc(zf.read(avail[max(avail)][0]))
                    groups, audit = decide_grouping(np.asarray(fb),
                                                    WIDEFIELD_PIXEL_UM)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell}: {exc}"); continue
                lv = args.level or sorted(avail)[len(avail) // 2]
                if lv not in avail:
                    skipped.append(f"{structure}/{cell}: no level {lv}"); continue
                try:
                    r = haze_of_cell(zf, avail[lv][0], avail[lv][1], groups,
                                     psf0, args.crop)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell}: {exc}"); continue
                if r["r2"] < args.min_r2:
                    skipped.append(f"{structure}/{cell}: R2 {r['r2']:.2f} too low")
                    continue
                r.update(structure=structure, cell=cell, level=lv,
                         ordering=audit["chosen"])
                rows.append(r)
                done += 1
                if done % 20 == 0:
                    print(f"  {structure}: {done} cells", flush=True)
            print(f"  {structure}: {done} usable of {len(cells)}", flush=True)

    if not rows:
        print("\n  nothing measured.")
        for s in skipped[:20]:
            print(f"    {s}")
        return
    summarise(rows, skipped, args)


def summarise(rows, skipped, args) -> None:
    allf = np.array([r["frac"] for r in rows])
    print()
    print(f"  {'structure':14s} {'n':>4s} {'median':>8s} {'mean':>8s} "
          f"{'p10':>7s} {'p90':>7s} {'median R2':>10s}")
    per = {}
    for st in sorted({r["structure"] for r in rows}):
        f = np.array([r["frac"] for r in rows if r["structure"] == st])
        r2 = np.array([r["r2"] for r in rows if r["structure"] == st])
        per[st] = {"n": int(f.size), "median": float(np.median(f)),
                   "mean": float(f.mean()), "p10": float(np.percentile(f, 10)),
                   "p90": float(np.percentile(f, 90)),
                   "median_r2": float(np.median(r2))}
        p = per[st]
        print(f"  {st:14s} {p['n']:4d} {p['median']:8.3f} {p['mean']:8.3f} "
              f"{p['p10']:7.3f} {p['p90']:7.3f} {p['median_r2']:10.3f}")
    print()
    print(f"  ALL {len(rows)} cells: median {np.median(allf):.3f}, "
          f"mean {allf.mean():.3f}, p10 {np.percentile(allf, 10):.3f}, "
          f"p90 {np.percentile(allf, 90):.3f}")
    print()
    print("  What this sets. --haze-frac takes ONE number, so the question is")
    print("  whether a single value is defensible. If the per-structure medians")
    print("  are close, the pooled median is the parameter. If they are far")
    print("  apart -- and the six-cell pilot suggested CCPs sits well above the")
    print("  rest -- then a single global fraction is itself a simplification,")
    print("  and the honest move is to say so and report sensitivity rather")
    print("  than to pick the median and be quiet about the spread.")
    spread = max(p["median"] for p in per.values()) - min(p["median"] for p in per.values())
    print()
    print(f"  spread of per-structure medians: {spread:.3f}")
    if spread > 0.15:
        print("  -> WIDE. A single --haze-frac cannot represent all four; train")
        print("     at the pooled median and report the per-structure values.")
    else:
        print("  -> narrow enough that the pooled median represents all four.")
    if skipped:
        print(f"\n  skipped {len(skipped)} cells; first few:")
        for s in skipped[:6]:
            print(f"    {s}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "per_structure": per, "cells": rows,
             "pooled_median": float(np.median(allf)), "skipped": skipped},
            indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

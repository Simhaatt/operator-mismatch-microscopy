"""Calibrate the reliability layer on REAL measurements instead of simulation.

`sim2real.py` showed that a predictor calibrated on simulated data does not
transfer: coverage 0.150-0.550 against a nominal 0.900, with the blind warning
signal inverting. That leaves the question this script answers -- is the failure
about the *forward model*, or about the reliability layer itself? Calibrating on
real measurements and testing on held-out real measurements separates the two.
If coverage comes back, the layer works and the forward model was the problem.
If it does not, the approach has a limit that no better simulator fixes.

Real measurements come from the raw SIM frames exactly as in `realism_check.py`:
summing the nine frames of one signal level gives a real widefield image paired
with the ground truth the error metric uses.

**What real data can and cannot offer as a shift axis.** There is no injected
aberration ladder -- the true PSF is whatever the microscope's was -- so the
operator-shift axis is unavailable and `rms` is unknown. Two real axes remain:

* **content shift**, by holding out a structure, exactly as in simulation;
* **signal shift**, by calibrating on bright levels and testing on dim ones.
  This one is arguably *more* honest than the simulated photon axis, because the
  dose really was lower rather than being scaled down by a factor.

Features are written in the same layout `extract_features.py` produces, so
`rotate_structures.py` and `gate5_bound.py` run on the output unchanged. The
signal level is stored as `severity`, which makes the existing worst-group
machinery report coverage per dose without modification.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realism_check import (  # noqa: E402
    WIDEFIELD_PIXEL_UM, decide_grouping, index_cell, real_widefield,
)
from scripts.sim2real import build_probes, centre_crop, fourier_upsample  # noqa: E402
from src.data.mrc import read_mrc  # noqa: E402
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.forward_model import apply_psf  # noqa: E402
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402
from src.reconstruction.tv import poisson_tv  # noqa: E402
from src.reconstruction.unet import UNet, normalise_measurement  # noqa: E402
from src.reconstruction.unrolled import UnrolledNet  # noqa: E402
from src.reliability.features import (  # noqa: E402
    FEATURE_NAMES, OTF_FEATURE_NAMES, oscr_features,
)


def prepare(zf, raw_member, gt_member, groups, psf0, crop, target_photons):
    """One real measurement on the pipeline's grid, with its oracle background.

    The background is fitted against the blurred ground truth. That is not
    deployable and is deliberately generous: estimating the pedestal blindly is
    its own hard problem (a low percentile misses by 2x in both directions), and
    the point here is to test the reliability layer, not a background estimator.
    Any failure that survives an oracle background is a failure of the layer.
    """
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
    (_, b), *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    b = float(max(b, 0.0))

    # Scale the SIGNAL part to the pipeline's budget, carrying the pedestal, so
    # real and simulated measurements share a decomposition rather than a total.
    scale = target_photons / max(y.mean() - b, 1e-9)
    return (torch.from_numpy(y * scale).to(torch.float32), gt_t,
            float(b * scale), float(y_cam.mean()))


def make_reconstructor(args, psf0):
    """The same four reconstructors `extract_features.py` offers.

    Four reconstructors on real data give a 16-cell grid, matching the
    simulated one. That matters for a specific reason: with a single cache the
    rotation yields only four cells, and every blind statistic came back between
    -0.72 and +0.14 with a seed standard deviation of 0.32 to 0.51. The
    comparison that separated `qpred` over 16 cells is simply not resolvable
    over four, and no number of seeds fixes a grid that small.
    """
    if args.reconstructor == "rl":
        return lambda y, bg: richardson_lucy(
            y, psf0, args.n_iter, background=bg,
            read_variance=args.read_noise_e**2)
    if args.reconstructor == "tv":
        return lambda y, bg: poisson_tv(
            y, psf0, args.tv_iter, weight=args.tv_weight, background=bg,
            prox_iter=args.tv_prox_iter)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck["model"] != args.reconstructor:
        raise ValueError(f"checkpoint holds {ck['model']!r}, not "
                         f"{args.reconstructor!r}")
    net = (UNet(base=32, depth=3) if args.reconstructor == "unet"
           else UnrolledNet(n_stages=5, width=32))
    net.load_state_dict(ck["state_dict"])
    net = net.to(torch.float32).eval()
    print(f"  loaded {args.reconstructor} (val SSIM {ck['val_ssim']:.4f}, "
          f"trained at haze {ck['args'].get('haze_frac')})")

    def run(y, bg):
        with torch.no_grad():
            if args.reconstructor == "unet":
                yn, sc = normalise_measurement(y)
                return net(yn) * sc
            return net(y, psf0)
    return run


def collect(args) -> dict:
    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(torch.float32)
    probes = build_probes(cfg, torch.device("cpu"), torch.float32)
    cutoff = 2 * cfg.na / cfg.wavelength_um * cfg.pixel_um
    reconstruct = make_reconstructor(args, psf0)

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
                    groups, _ = decide_grouping(np.asarray(fb), WIDEFIELD_PIXEL_UM)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell}: {exc}"); continue
                for level in args.levels:
                    if level not in avail:
                        continue
                    try:
                        y, gt, bg, native = prepare(
                            zf, avail[level][0], avail[level][1], groups, psf0,
                            args.crop, args.photons)
                    except ValueError as exc:
                        skipped.append(f"{structure}/{cell} L{level}: {exc}")
                        continue
                    x_hat = reconstruct(y[None], bg)
                    feats = oscr_features(y[None], x_hat, psf0, probes,
                                          cutoff_cyc_px=cutoff)[0].numpy()
                    e = float(1.0 - ssim(scale_match(x_hat[0], gt), gt,
                                         data_range=1.0))
                    rows.append({"X": feats, "e": e, "level": level,
                                 "cell": f"{structure}/{cell}",
                                 "structure": structure, "native": native,
                                 "background": bg})
                    done += 1
                if done and done % 30 == 0:
                    print(f"  {structure}: {done} samples", flush=True)
            print(f"  {structure}: {done} samples from "
                  f"{len({r['cell'] for r in rows if r['structure'] == structure})} "
                  f"cells", flush=True)
    return {"rows": rows, "skipped": skipped}


SPLITS = ("train", "cal", "test", "ood")


def write_cache(rows, args) -> None:
    """Store in `extract_features.py`'s layout so the gate scripts just work.

    ``severity`` holds the signal level rather than an aberration label, which
    is the meaningful notion of difficulty on real data and makes the existing
    worst-group machinery report coverage per dose for free. ``rms`` is filled
    with NaN and never read: there is no injected aberration to compare against,
    and a zero would read as "nominal operator" which is a claim we cannot make.
    """
    rng = np.random.default_rng(args.seed)
    is_ood = np.array([r["structure"] == args.held_out for r in rows])
    keys = sorted({r["cell"] for r, o in zip(rows, is_ood) if not o})
    rng.shuffle(keys)
    n_tr, n_ca = int(0.6 * len(keys)), int(0.2 * len(keys))
    role_of = {k: ("train" if i < n_tr else "cal" if i < n_tr + n_ca else "test")
               for i, k in enumerate(keys)}
    role = np.array([args.held_out and r["structure"] == args.held_out and "ood"
                     or role_of.get(r["cell"], "ood") for r in rows])

    store = {}
    n_feat = len(FEATURE_NAMES) + len(OTF_FEATURE_NAMES)
    for s in SPLITS:
        sel = [i for i, x in enumerate(role) if x == s]
        # Give the shape explicitly: np.asarray([]) is 1-D, so reshape(-1)
        # cannot infer a feature axis and an empty split raises. Empty splits
        # happen for real -- a structure with too few cells to divide, or a
        # held-out structure that is the only one present.
        store[f"{s}_X"] = (np.asarray([rows[i]["X"] for i in sel], dtype=np.float32)
                           if sel else np.zeros((0, n_feat), dtype=np.float32))
        store[f"{s}_e"] = np.asarray([rows[i]["e"] for i in sel], dtype=np.float32)
        store[f"{s}_rms"] = np.full(len(sel), np.nan, dtype=np.float32)
        store[f"{s}_severity"] = np.asarray([f"L{rows[i]['level']:02d}" for i in sel])
        store[f"{s}_mode"] = np.asarray(["real"] * len(sel))
        store[f"{s}_cell"] = np.asarray([rows[i]["cell"] for i in sel])
        store[f"{s}_structure"] = np.asarray([rows[i]["structure"] for i in sel])
        print(f"  {s}: {len(sel)} samples")
    store["feature_names"] = np.asarray(FEATURE_NAMES + OTF_FEATURE_NAMES)
    store["photons"] = np.asarray([args.photons])
    store["background"] = np.asarray([float(np.median([r["background"] for r in rows]))])
    store["haze_frac"] = np.asarray([float(np.median(
        [r["background"] / (args.photons + r["background"]) for r in rows]))])
    store["reconstructor"] = np.asarray([f"{args.reconstructor}_real"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"  wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", nargs="+",
                    default=["CCPs", "Microtubules", "F-actin", "ER"])
    ap.add_argument("--cells", type=int, default=0, help="per structure; 0 = all")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 3, 5, 7, 9])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--n-iter", type=int, default=75)
    ap.add_argument("--reconstructor", choices=("rl", "tv", "unet", "unrolled"),
                    default="rl")
    ap.add_argument("--checkpoint", default=None,
                    help="required for unet/unrolled")
    ap.add_argument("--tv-weight", type=float, default=0.02,
                    help="the value selected on the nominal operator for the "
                         "simulated TV caches, held fixed so real and simulated "
                         "TV differ only in the measurement")
    ap.add_argument("--tv-iter", type=int, default=100)
    ap.add_argument("--tv-prox-iter", type=int, default=10)
    ap.add_argument("--photons", type=float, default=100.0)
    ap.add_argument("--read-noise-e", type=float, default=1.5)
    ap.add_argument("--held-out", default="CCPs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--archives", default="data/biosr")
    ap.add_argument("--out", default="experiments/features19/real_rl.npz")
    args = ap.parse_args()
    if args.reconstructor in ("unet", "unrolled") and not args.checkpoint:
        ap.error(f"--reconstructor {args.reconstructor} needs --checkpoint")

    print(f"REAL-MEASUREMENT FEATURE EXTRACTION -- {args.reconstructor}")
    print(f"  levels {args.levels}, crop {args.crop}, held out {args.held_out}")
    print(f"  signal scaled to {args.photons:.0f} photons, oracle background")
    print()
    got = collect(args)
    rows = got["rows"]
    if not rows:
        print("  nothing collected.")
        for s in got["skipped"][:10]:
            print(f"    {s}")
        return
    print()
    print(f"  {len(rows)} samples from {len({r['cell'] for r in rows})} cells")
    e = np.array([r["e"] for r in rows])
    print(f"  true error: mean {e.mean():.4f}  median {np.median(e):.4f}  "
          f"range {e.min():.4f}-{e.max():.4f}")
    for lv in sorted({r["level"] for r in rows}):
        el = np.array([r["e"] for r in rows if r["level"] == lv])
        nat = np.median([r["native"] for r in rows if r["level"] == lv])
        print(f"    L{lv:02d}: n={el.size:4d}  native {nat:8.1f} e-  "
              f"median error {np.median(el):.4f}")
    print()
    write_cache(rows, args)
    if got["skipped"]:
        print(f"  skipped {len(got['skipped'])}")


if __name__ == "__main__":
    main()

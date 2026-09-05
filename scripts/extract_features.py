"""Extract and cache blind reliability features for the whole shift matrix.

Simulating measurements and deconvolving them costs ~35 minutes per photon
level. The calibration experiments in Gate 4 need to be re-run many times with
different schemes, so features are computed once here and cached to ``.npz``.

Everything stored is either blind (usable at deployment) or ground truth kept
strictly for *evaluation*:

* ``X``        -- blind features, safe to condition on
* ``e``        -- the target, 1 - SSIM, requires ground truth
* ``rms``      -- true RMS wavefront error of the operator that formed the data,
                  the oracle mismatch. Used only to evaluate how well an
                  estimated mismatch score tracks reality, never as an input.
* ``severity`` / ``mode`` / ``cell`` / ``structure`` -- grouping labels for
  stratified coverage and for split blocking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.biosr import index_biosr, index_npz, load_object, split_by_cell  # noqa: E402
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.aberrations import ABERRATION_MODES, ladder_specs  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    NoiseConfig, sample_seed, simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402
from src.reconstruction.tv import poisson_tv  # noqa: E402
from src.reliability.features import FEATURE_NAMES, oscr_features  # noqa: E402


def build_probes(cfg, device, dtype):
    probes = [pupil_psf(None, cfg, warn=False)[0]]
    for mode in ABERRATION_MODES.values():
        for amp in (0.05, 0.11, 0.18):
            probes.append(pupil_psf({mode: amp}, cfg, warn=False)[0])
    return [p.to(device=device, dtype=dtype) for p in probes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--photons", type=float, default=100.0)
    ap.add_argument("--n-iter", type=int, default=75)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--held-out", default="CCPs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", default=None,
                    help="compact .npz export; defaults to the zip archives")
    ap.add_argument("--reconstructor", choices=("rl", "tv"), default="rl")
    ap.add_argument("--tv-weight", type=float, default=None,
                    help="TV regularisation strength; selected on the nominal "
                         "operator if omitted, then held fixed across all cells")
    ap.add_argument("--tv-iter", type=int, default=100)
    ap.add_argument("--tv-prox-iter", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype, device = torch.float32, torch.device(args.device)
    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(device=device, dtype=dtype)
    probes = build_probes(cfg, device, dtype)
    noise = NoiseConfig(photons=args.photons, background=5.0, read_noise_e=1.5)
    specs = ladder_specs(
        ["defocus", "astigmatism", "coma", "spherical"],
        ["nominal", "mild", "moderate", "severe", "extreme"],
    )
    true_psfs = [
        (s, pupil_psf(s.coeffs, cfg, warn=False)[0].to(device=device, dtype=dtype))
        for s in specs
    ]

    idx = index_npz(args.source) if args.source else index_biosr()
    sp = split_by_cell(idx, held_out_structure=args.held_out)
    print(f"photons={args.photons} crop={args.crop} n_iter={args.n_iter} "
          f"probes={len(probes)} specs={len(specs)}")

    if args.reconstructor == "tv":
        weight = args.tv_weight
        if weight is None:
            # Select once on the NOMINAL operator, exactly as n_iter is chosen for
            # Richardson-Lucy. Retuning per cell would confound operator mismatch
            # with regularisation strength.
            print("  selecting TV weight on the nominal operator...")
            probe = [load_object(e, crop=args.crop, dtype=dtype).to(device)
                     for e in sp["train"][:8]]
            best, best_s = None, -1.0
            for w in (0.001, 0.005, 0.02, 0.05):
                tot = 0.0
                for x in probe:
                    y = simulate_measurement(x, psf0, noise, seed=sample_seed("wsel", w))
                    xh = poisson_tv(y[None], psf0, args.tv_iter, weight=w,
                                    background=noise.background,
                                    prox_iter=args.tv_prox_iter)
                    tot += float(ssim(scale_match(xh[0], x), x, data_range=1.0))
                sc = tot / len(probe)
                flag = ""
                if sc > best_s:
                    best, best_s, flag = w, sc, "   <- best"
                print(f"    weight={w:6.3f}  mean SSIM={sc:.4f}{flag}")
            weight = best
        print(f"  TV weight = {weight}, {args.tv_iter} iterations, held fixed")

        def reconstruct(y):
            return poisson_tv(y, psf0, args.tv_iter, weight=weight,
                              background=noise.background, prox_iter=args.tv_prox_iter)
    else:
        def reconstruct(y):
            return richardson_lucy(y, psf0, args.n_iter, background=noise.background,
                                   read_variance=noise.read_noise_e**2)

    store: dict[str, np.ndarray] = {}
    for split, entries in sp.items():
        X, e, sev, mode, cell, struct, rms = [], [], [], [], [], [], []
        for i, entry in enumerate(entries):
            x = load_object(entry, crop=args.crop, dtype=dtype).to(device)[None]
            for spec, psf_true in true_psfs:
                y = simulate_measurement(
                    x[0], psf_true, noise, seed=sample_seed(split, entry.key, spec.label)
                )[None]
                x_hat = reconstruct(y)
                X.append(oscr_features(y, x_hat, psf0, probes)[0].cpu().numpy())
                e.append(float(1.0 - ssim(scale_match(x_hat[0], x[0]), x[0], data_range=1.0)))
                sev.append(spec.severity)
                mode.append(spec.mode)
                cell.append(entry.key)
                struct.append(entry.structure)
                rms.append(spec.rms_waves)
            if (i + 1) % 20 == 0:
                print(f"  {split}: {i+1}/{len(entries)} cells", flush=True)
        store[f"{split}_X"] = np.asarray(X, dtype=np.float32)
        store[f"{split}_e"] = np.asarray(e, dtype=np.float32)
        store[f"{split}_rms"] = np.asarray(rms, dtype=np.float32)
        store[f"{split}_severity"] = np.asarray(sev)
        store[f"{split}_mode"] = np.asarray(mode)
        store[f"{split}_cell"] = np.asarray(cell)
        store[f"{split}_structure"] = np.asarray(struct)
        print(f"  {split}: {len(e)} samples")

    store["feature_names"] = np.asarray(FEATURE_NAMES)
    store["photons"] = np.asarray([args.photons])
    store["reconstructor"] = np.asarray([args.reconstructor])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

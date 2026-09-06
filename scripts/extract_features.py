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
    NoiseConfig, background_for, sample_seed, simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402
from src.reconstruction.unet import UNet, normalise_measurement  # noqa: E402
from src.reconstruction.unrolled import UnrolledNet  # noqa: E402
from src.reconstruction.tv import poisson_tv  # noqa: E402
from src.reliability.features import (  # noqa: E402
    FEATURE_NAMES, OTF_FEATURE_NAMES, oscr_features,
)


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
    ap.add_argument("--reconstructor", choices=("rl", "tv", "unet", "unrolled"),
                    default="rl")
    ap.add_argument("--checkpoint", default=None,
                    help="trained .pt from train_reconstructor.py; required for "
                         "--reconstructor unet/unrolled")
    ap.add_argument("--tv-weight", type=float, default=None,
                    help="TV regularisation strength; selected on the nominal "
                         "operator if omitted, then held fixed across all cells")
    ap.add_argument("--tv-iter", type=int, default=100)
    ap.add_argument("--tv-prox-iter", type=int, default=10)
    ap.add_argument("--background", type=float, default=5.0,
                    help="constant additive rate, in photons. The default of 5 "
                         "against a signal mean of 100 is 5%% background. Fitting "
                         "real BioSR widefield measurements as a*blur(GT)+b puts "
                         "the constant term at 19-68%% of the measurement mean "
                         "(median 31%%), i.e. 4-14x larger: SIM reconstruction "
                         "removes out-of-focus light, so a ground truth derived "
                         "from it carries none and blurring it cannot put it "
                         "back. Use ~45 to match the median real level.")
    ap.add_argument("--haze-frac", type=float, default=None,
                    help="out-of-focus fraction of the measurement mean; sets "
                         "the background as photons*f/(1-f), so the fraction is "
                         "held constant as the photon budget changes. PREFER "
                         "THIS over --background: a fixed rate is correct at "
                         "one photon level only (45 is 31%% at 100 photons but "
                         "75%% at 15). Real BioSR is 0.19-0.68, median 0.31.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype, device = torch.float32, torch.device(args.device)
    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(device=device, dtype=dtype)
    probes = build_probes(cfg, device, dtype)
    background = (background_for(args.photons, args.haze_frac)
                  if args.haze_frac is not None else args.background)
    noise = NoiseConfig(photons=args.photons, background=background,
                        read_noise_e=1.5)
    if args.haze_frac is not None:
        print(f"  haze fraction {args.haze_frac:.2f} -> background "
              f"{background:.2f} at {args.photons:.0f} photons")
    cutoff = 2 * cfg.na / cfg.wavelength_um * cfg.pixel_um  # OTF support, cyc/px
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
    elif args.reconstructor in ("unet", "unrolled"):
        if not args.checkpoint:
            ap.error(f"--reconstructor {args.reconstructor} needs --checkpoint")
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if ck["model"] != args.reconstructor:
            raise ValueError(f"checkpoint holds a {ck['model']!r} but "
                             f"--reconstructor is {args.reconstructor!r}")
        tr = ck["args"]
        trained_frac = tr.get("haze_frac")
        if trained_frac is None:
            trained_frac = float(tr.get("background", 5.0)) / (100.0 + float(
                tr.get("background", 5.0)))    # legacy: fixed rate, nominal 100 ph
        here_frac = float(background / (args.photons + background))
        trained_bg = background_for(args.photons, float(trained_frac))
        if abs(float(trained_frac) - here_frac) > 1e-3:
            raise ValueError(
                f"checkpoint was trained at haze fraction {float(trained_frac):.3f} "
                f"(background {trained_bg:.2f} here) but this extraction uses "
                f"{here_frac:.3f} ({background:.2f}). A learned reconstructor "
                f"evaluated under a different forward model than it was trained "
                f"on measures domain shift, not reliability. Re-train or match.")
        net = (UNet(base=32, depth=3) if args.reconstructor == "unet"
               else UnrolledNet(n_stages=5, width=32))
        net.load_state_dict(ck["state_dict"])
        net = net.to(device=device, dtype=dtype).eval()
        print(f"  loaded {args.reconstructor} from {args.checkpoint} "
              f"(val SSIM {ck.get('val_ssim', float('nan')):.4f}, "
              f"haze fraction {float(trained_frac):.3f})")

        def reconstruct(y):
            # Exactly the convention train_reconstructor.py used, or xhat_std
            # would land on a different scale from the classical reconstructors
            # and the comparison would measure units rather than reliability.
            with torch.no_grad():
                if args.reconstructor == "unet":
                    yn, sc = normalise_measurement(y)
                    return net(yn) * sc
                return net(y, psf0)
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
                X.append(oscr_features(y, x_hat, psf0, probes,
                                       cutoff_cyc_px=cutoff)[0].cpu().numpy())
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

    store["feature_names"] = np.asarray(FEATURE_NAMES + OTF_FEATURE_NAMES)
    store["otf_cutoff_cyc_px"] = np.asarray([cutoff])
    store["photons"] = np.asarray([args.photons])
    store["background"] = np.asarray([background])
    store["haze_frac"] = np.asarray([background / (args.photons + background)])
    store["reconstructor"] = np.asarray([args.reconstructor])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

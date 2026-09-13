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
from src.metrics.image import relative_l2, scale_match, ssim  # noqa: E402
from src.metrics.task import (  # noqa: E402
    TASK_NAMES,
    task_measurements,
)
from src.optics.aberrations import ABERRATION_MODES, ladder_specs  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    NoiseConfig, background_for, sample_seed, simulate_measurement,
)
from src.optics.physical_mismatch import (  # noqa: E402
    apply_overrides,
    physical_specs,
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
    ap.add_argument("--include-physical", action="store_true",
                    help="add the NA, wavelength and focus-drift mismatch "
                         "families (12 specs) to the 17 Zernike specs")
    ap.add_argument("--limit-cells", type=int, default=None,
                    help="use at most this many cells per split; for smoke "
                         "testing a pipeline change before a full extraction")
    ap.add_argument("--na", type=float, default=1.7,
                    help="objective numerical aperture (BioSR default 1.7; "
                         "W2S is 1.49)")
    ap.add_argument("--wavelength-um", type=float, default=0.525,
                    help="emission wavelength in um (BioSR default 0.525)")
    ap.add_argument("--pixel-um", type=float, default=0.0313,
                    help="object-grid pixel size in um (BioSR default 0.0313; "
                         "W2S SIM grid is 0.032)")
    ap.add_argument("--task-metrics", action="store_true",
                    help="also score each reconstruction by the relative "
                         "error in three downstream task readouts (puncta "
                         "count, skeleton length, branch points)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype, device = torch.float32, torch.device(args.device)
    # Optics default to the verified BioSR values; a second dataset on a
    # different instrument must override them or every PSF in the run is wrong.
    cfg = PupilConfig(
        n_out=args.kernel,
        na=args.na,
        wavelength_um=args.wavelength_um,
        pixel_um=args.pixel_um,
    )
    print(f"  optics: NA {cfg.na}, lambda {cfg.wavelength_um} um, "
          f"pixel {cfg.pixel_um} um")
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
    if args.include_physical:
        specs = specs + physical_specs(na=cfg.na, wavelength_um=cfg.wavelength_um)
    # The true kernel may come from a perturbed configuration (wrong NA or
    # wavelength); the nominal kernel psf0 above never is. That asymmetry is the
    # operator mismatch.
    true_psfs = [
        (
            s,
            pupil_psf(s.coeffs, apply_overrides(cfg, s), warn=False)[0]
            .to(device=device, dtype=dtype),
        )
        for s in specs
    ]
    # Family-agnostic mismatch magnitude in PSF space. RMS wavefront error is
    # undefined for the NA and wavelength families, so this is the only axis
    # comparable across all of them.
    psf0_norm = float(torch.linalg.vector_norm(psf0))
    mismatch_mag = {
        s.label: float(torch.linalg.vector_norm(p - psf0) / max(psf0_norm, 1e-20))
        for s, p in true_psfs
    }

    idx = index_npz(args.source) if args.source else index_biosr()
    sp = split_by_cell(idx, held_out_structure=args.held_out)
    if args.limit_cells:
        sp = {k: v[: args.limit_cells] for k, v in sp.items()}
        print(f"  LIMITED to {args.limit_cells} cells per split -- smoke test, "
              f"not a publishable cache")
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
        trained_bg_aware = bool(tr.get("bg_aware", False))
        # A background-aware checkpoint minimises a different objective.
        # Evaluating it with b = 0 would measure that mismatch, not
        # reliability, so the value is taken from the checkpoint rather
        # than from this script's flags.
        bg_fidelity = background if trained_bg_aware else 0.0
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
                return net(y, psf0, background=bg_fidelity)
                return net(y, psf0)
    else:
        def reconstruct(y):
            return richardson_lucy(y, psf0, args.n_iter, background=noise.background,
                                   read_variance=noise.read_noise_e**2)

    store: dict[str, np.ndarray] = {}
    for split, entries in sp.items():
        X, e, sev, mode, cell, struct, rms = [], [], [], [], [], [], []
        e_l2, mag = [], []
        tasks = {k: [] for k in TASK_NAMES} if args.task_metrics else {}
        for i, entry in enumerate(entries):
            x = load_object(entry, crop=args.crop, dtype=dtype).to(device)[None]
            # T(x) is identical for all specs of this cell; compute once.
            ref_task = task_measurements(x[0]) if args.task_metrics else None
            for spec, psf_true in true_psfs:
                y = simulate_measurement(
                    x[0], psf_true, noise, seed=sample_seed(split, entry.key, spec.label)
                )[None]
                x_hat = reconstruct(y)
                X.append(oscr_features(y, x_hat, psf0, probes,
                                       cutoff_cyc_px=cutoff)[0].cpu().numpy())
                # One scale-matched estimate, two metrics: the only difference
                # between e and e_l2 is the metric, not the reconstruction.
                matched = scale_match(x_hat[0], x[0])
                e.append(float(1.0 - ssim(matched, x[0], data_range=1.0)))
                e_l2.append(float(relative_l2(matched, x[0])))
                mag.append(mismatch_mag[spec.label])
                if args.task_metrics:
                    est_task = task_measurements(matched)
                    for k in TASK_NAMES:
                        denom = ref_task[k]
                        # Undefined when the reference measurement is
                        # zero; NaN so the analysis drops it explicitly
                        # rather than averaging in a meaningless number.
                        tasks[k].append(
                            abs(est_task[k] - denom) / denom
                            if denom > 0 else float('nan')
                        )
                sev.append(spec.severity)
                mode.append(spec.mode)
                cell.append(entry.key)
                struct.append(entry.structure)
                rms.append(spec.rms_waves)
            if (i + 1) % 20 == 0:
                print(f"  {split}: {i+1}/{len(entries)} cells", flush=True)
        store[f"{split}_X"] = np.asarray(X, dtype=np.float32)
        store[f"{split}_e"] = np.asarray(e, dtype=np.float32)
        store[f"{split}_e_l2"] = np.asarray(e_l2, dtype=np.float32)
        store[f"{split}_rms"] = np.asarray(rms, dtype=np.float32)
        store[f"{split}_mismatch_mag"] = np.asarray(mag, dtype=np.float32)
        for k, v in tasks.items():
            store[f"{split}_task_{k}"] = np.asarray(v, dtype=np.float32)
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
    store["include_physical"] = np.asarray([bool(args.include_physical)])
    store["task_metrics"] = np.asarray([bool(args.task_metrics)])
    if args.reconstructor in ("unet", "unrolled"):
        store["bg_aware"] = np.asarray([bool(trained_bg_aware)])
    store["n_specs"] = np.asarray([len(specs)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Does a reliability layer calibrated on SIMULATION hold on REAL measurements?

Every bound in this project is calibrated on measurements our own forward model
produced. `realism_check.py` established that the model's noise is Poisson in
shape and its spectra are off by a factor of ~1.9-5, but that is a property of
the *measurements*, not of the reliability layer built on top of them. The
question that actually matters for deployment is one step further on: take the
conformal predictor exactly as calibrated on simulated data, feed it a **real**
widefield measurement, and ask whether its 90% bound still covers the true error.

This is possible because BioSR ships the raw SIM frames. Summing the nine frames
of one level gives a real widefield measurement of a real specimen, paired with
the exact ground truth the error metric uses -- see `realism_check.py` for the
construction and for the two traps in it (per-structure frame ordering, and the
illumination-envelope leak).

**The two honest caveats**, both structural rather than incidental:

1. *No controlled operator mismatch.* On real data the true PSF is whatever the
   microscope's was; we cannot dial in a known aberration. So this tests the
   nominal-operator case and sim-to-real transfer, not the aberration ladder.
2. *Resampling.* The pipeline works at 31.3 nm with a 31.3 nm PSF; the camera
   samples at 62.6 nm. The real measurement is upsampled 2x by zero-padding in
   the Fourier domain, which is *exact* here -- the widefield band ends at
   2NA/lambda = 6.48 cyc/um, well inside the 62.6 nm Nyquist of 7.99 cyc/um, so
   nothing is aliased and nothing is invented. What it does change is the noise:
   white noise on the camera grid becomes band-limited after upsampling. That is
   a real difference from the simulated measurements and is reported, not hidden.
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
    PHOTONS_PER_COUNT, WIDEFIELD_PIXEL_UM, decide_grouping, index_cell,
    real_widefield,
)
from src.data.mrc import read_mrc  # noqa: E402
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.metrics.stats import spearman  # noqa: E402
from src.optics.aberrations import ABERRATION_MODES  # noqa: E402
from src.optics.forward_model import apply_psf  # noqa: E402
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402
from src.reliability.features import (  # noqa: E402
    FEATURE_NAMES, OTF_FEATURE_NAMES, oscr_features,
)


def fourier_upsample(img: np.ndarray, factor: int = 2) -> np.ndarray:
    """Exact band-limited upsampling by zero-padding the spectrum.

    Lossless for a field whose support fits inside the source Nyquist, which a
    widefield measurement's does by construction: the OTF ends at 2NA/lambda,
    below the camera's Nyquist. Interpolation in the image domain would smooth
    the noise and blur the signal; this does neither.
    """
    n = img.shape[-1]
    m = n * factor
    F = np.fft.fftshift(np.fft.fft2(img))
    out = np.zeros((m, m), dtype=complex)
    lo = (m - n) // 2
    out[lo : lo + n, lo : lo + n] = F
    up = np.fft.ifft2(np.fft.ifftshift(out)).real * (factor**2)
    return up


def centre_crop(a: np.ndarray, crop: int) -> np.ndarray:
    h, w = a.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    return a[..., top : top + crop, left : left + crop]


def build_probes(cfg, device, dtype):
    """The same probe family the cached features were computed with."""
    probes = [pupil_psf(None, cfg, warn=False)[0]]
    for mode in ABERRATION_MODES.values():
        for amp in (0.05, 0.11, 0.18):
            probes.append(pupil_psf({mode: amp}, cfg, warn=False)[0])
    return [p.to(device=device, dtype=dtype) for p in probes]


def real_sample(zf, raw_member, gt_member, groups, crop, target_mean, psf0):
    """One real measurement on the pipeline's own grid, plus its ground truth.

    Returns ``(y, x_gt)`` as 256x256 tensors at 31.3 nm, with ``y`` scaled to the
    photon budget the simulated calibration data was generated at. The scaling is
    a single global factor -- it cannot change the spectral content or the
    relative noise structure, only the absolute level, and matching it is
    necessary because two of the cached features (``y_mean``, ``y_std``) are
    scale-dependent by design.
    """
    frames, _ = read_mrc(zf.read(raw_member))
    gt_arr, _ = read_mrc(zf.read(gt_member))
    real = real_widefield(np.asarray(frames), WIDEFIELD_PIXEL_UM, groups)
    y_cam = real["y"]                       # photoelectrons, camera grid

    gt = np.asarray(gt_arr[0], dtype=np.float64)
    if gt.shape[-1] != 2 * y_cam.shape[-1]:
        raise ValueError(f"GT {gt.shape} is not 2x the frames {y_cam.shape}")

    y_fine = centre_crop(fourier_upsample(y_cam, 2), crop)
    # Upsampling can ring slightly negative in dark regions; the reconstructor
    # takes a photon count, so clamp exactly as load_object does for objects.
    y_fine = np.clip(y_fine, 0.0, None)

    gt_c = centre_crop(gt, crop)
    lo, hi = np.percentile(gt_c, [0.1, 99.9])
    gt_c = np.clip((gt_c - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    gt_t = torch.from_numpy(gt_c).to(torch.float32)

    # ORACLE background. A real widefield measurement is well described by
    # a*blur(GT) + b (R^2 0.72-0.96) where b is out-of-focus light that SIM
    # reconstruction removed from the ground truth, so the forward model has no
    # way to produce it. Estimating b blindly is its own hard problem -- a low
    # percentile misses by 2x in both directions, over-reading dense specimens
    # and under-reading sparse ones. So b is taken from a fit against the ground
    # truth. That is NOT deployable; it is deliberately the most favourable case,
    # so that a failure here cannot be blamed on background estimation.
    blur = apply_psf(gt_t, psf0).numpy().ravel()
    A = np.stack([blur, np.ones_like(blur)], axis=1)
    (a_fit, b_fit), *_ = np.linalg.lstsq(A, y_fine.ravel(), rcond=None)
    b_fit = float(max(b_fit, 0.0))

    # Scale so the SIGNAL part matches the calibration photon budget, carrying
    # the pedestal along with it, so the simulated and real measurements have
    # the same decomposition rather than the same total.
    signal_mean = max(y_fine.mean() - b_fit, 1e-9)
    scale = target_mean / signal_mean
    return (torch.from_numpy(y_fine * scale).to(torch.float32), gt_t,
            float(scale), float(y_cam.mean()), b_fit * scale)


def pick_level(zf, structure, cell, avail, groups, target_e):
    """The signal level whose native dose is closest to the simulated budget.

    Rescaling a Poisson measurement to a different mean does not rescale its
    noise correctly -- variance would need to scale with the mean, and a global
    factor scales it quadratically. Choosing the closest level first keeps that
    distortion as small as the data allows, and the residual factor is reported.
    """
    best, best_gap, best_mean = None, np.inf, None
    for lvl, (raw, _) in sorted(avail.items()):
        frames, _ = read_mrc(zf.read(raw))
        m = float(real_widefield(np.asarray(frames), WIDEFIELD_PIXEL_UM,
                                 groups)["y"].mean())
        gap = abs(np.log(max(m, 1e-9) / target_e))
        if gap < best_gap:
            best, best_gap, best_mean = lvl, gap, m
    return best, best_mean


def fit_on_simulation(cache, alpha, seed):
    """Train and calibrate the conformal predictor on SIMULATED data only.

    Deliberately uses the *nominal* operator rows alone. The real measurements
    carry whatever aberration the real microscope had, which is unknown but
    certainly not one of our injected ones, so the fair comparison is against
    the simulated nominal case rather than against a ladder average.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    d = np.load(cache, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    Xtr, etr = d["train_X"], d["train_e"]
    q = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha,
                                  n_estimators=300, max_depth=3,
                                  learning_rate=0.05, random_state=seed)
    q.fit(Xtr, etr)
    Xc, ec = d["cal_X"], d["cal_e"]
    s_cal = ec - q.predict(Xc)
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    corr = float(np.sort(s_cal)[min(max(k, 0), n - 1)])

    nominal = d["cal_severity"] == "nominal"
    return {"q": q, "corr": corr, "names": names, "s_cal": s_cal,
            "photons": float(d["photons"][0]),
            "cal_nominal_e": ec[nominal] if nominal.any() else ec,
            "cal_e": ec}


def evaluate_real(model, X, e, alpha):
    """Coverage of the simulation-calibrated bound on real measurements."""
    U = model["q"].predict(X) + model["corr"]
    covered = e <= U
    s = e - model["q"].predict(X)
    from scripts.gate5_bound import directional_excess
    dplus = directional_excess(model["s_cal"], s)
    return {
        "coverage": float(covered.mean()),
        "n": int(len(e)),
        "mean_width": float(U.mean()),
        "mean_error": float(e.mean()),
        "median_error": float(np.median(e)),
        "d_plus": float(dplus),
        "bound_1s": float(1 - alpha - dplus),
        "slack_1s": float(covered.mean() - (1 - alpha - dplus)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="experiments/features19/rl_photons_100.npz",
                    help="simulated cache to calibrate on")
    ap.add_argument("--structures", nargs="+",
                    default=["CCPs", "Microtubules", "F-actin", "ER"])
    ap.add_argument("--cells", type=int, default=6)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--n-iter", type=int, default=75)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--background", type=float, default=5.0)
    ap.add_argument("--read-noise-e", type=float, default=1.5)
    ap.add_argument("--archives", default="data/biosr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = torch.float32
    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(dtype)
    probes = build_probes(cfg, torch.device("cpu"), dtype)
    cutoff = 2 * cfg.na / cfg.wavelength_um * cfg.pixel_um

    model = fit_on_simulation(args.cache, args.alpha, seed=0)
    target_mean = model["photons"]
    print("SIM-TO-REAL TRANSFER")
    print(f"  calibrated on {args.cache} (photons={target_mean:.0f}, "
          f"nominal {1 - args.alpha:.0%})")
    print(f"  real measurement = sum of 9 raw SIM frames, upsampled 2x to "
          f"{cfg.pixel_um * 1000:.1f} nm, centre-cropped {args.crop}")
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
            for cell in cells[: args.cells]:
                avail = index_cell(names, structure, cell)
                if not avail:
                    skipped.append(f"{structure}/{cell}: no raw/GT pair"); continue
                try:
                    fb, _ = read_mrc(zf.read(avail[max(avail)][0]))
                    groups, audit = decide_grouping(np.asarray(fb), WIDEFIELD_PIXEL_UM)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell}: {exc}"); continue
                level, native = pick_level(zf, structure, cell, avail, groups,
                                           target_mean)
                raw_m, gt_m = avail[level]
                try:
                    y, gt, scale, cam_mean, bg_real = real_sample(
                        zf, raw_m, gt_m, groups, args.crop, target_mean, psf0)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell} L{level}: {exc}"); continue

                x_hat = richardson_lucy(y[None], psf0, args.n_iter,
                                        background=bg_real,
                                        read_variance=args.read_noise_e**2)
                feats = oscr_features(y[None], x_hat, psf0, probes,
                                      cutoff_cyc_px=cutoff)[0].numpy()
                err = float(1.0 - ssim(scale_match(x_hat[0], gt), gt, data_range=1.0))
                rows.append({"structure": structure, "cell": cell, "level": level,
                             "X": feats.tolist(), "e": err, "scale": scale,
                             "native_mean_e": native, "background": bg_real,
                             "ordering": audit["chosen"]})
                print(f"  {structure}/{cell} L{level:02d}  native={native:8.1f} e-  "
                      f"x{scale:5.2f}  oracle bg={bg_real:6.1f}  e={err:.4f}")

    if not rows:
        print("  nothing evaluated.")
        for s in skipped:
            print(f"    {s}")
        return
    summarise(rows, model, args, skipped)


def summarise(rows, model, args, skipped) -> None:
    X = np.asarray([r["X"] for r in rows], dtype=np.float32)
    e = np.asarray([r["e"] for r in rows], dtype=np.float32)
    res = evaluate_real(model, X, e, args.alpha)

    print()
    print(f"  REAL MEASUREMENTS: {res['n']} cells")
    print(f"    coverage of the simulation-calibrated {1 - args.alpha:.0%} bound"
          f"  = {res['coverage']:.3f}")
    print(f"    mean interval width                        = {res['mean_width']:.4f}")
    print(f"    mean / median true error (1 - SSIM)        = "
          f"{res['mean_error']:.4f} / {res['median_error']:.4f}")
    print(f"    d+ against the simulated calibration scores= {res['d_plus']:.3f}")
    print(f"    one-sided bound / slack                    = "
          f"{res['bound_1s']:.3f} / {res['slack_1s']:+.3f}")
    sim_e = model["cal_e"]
    print(f"    simulated calibration error, mean/median   = "
          f"{float(sim_e.mean()):.4f} / {float(np.median(sim_e)):.4f}")
    print()

    print(f"  {'structure':14s} {'n':>3s} {'coverage':>9s} {'mean e':>8s} "
          f"{'mean width':>11s}")
    per = {}
    for st in sorted({r["structure"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["structure"] == st]
        rs = evaluate_real(model, X[sel], e[sel], args.alpha)
        per[st] = rs
        print(f"  {st:14s} {rs['n']:3d} {rs['coverage']:9.3f} "
              f"{rs['mean_error']:8.4f} {rs['mean_width']:11.4f}")
    print()

    # Does the blind warning signal still work on real inputs? r_nominal is the
    # blind estimator that won on simulated data (rho +0.728 +- 0.049), so the
    # question is whether it still tracks true error when the measurement is real.
    ri = model["names"].index("r_nominal") if "r_nominal" in model["names"] else None
    if ri is not None and len(e) >= 5:
        print(f"  blind warning signal on real data:")
        print(f"    rho(r_nominal, true error) = {spearman(X[:, ri], e):+.3f}"
              f"   over {len(e)} cells")
    print()
    print("  How to read this. Coverage at or above nominal means the layer")
    print("  transfers: a bound calibrated entirely on simulated measurements")
    print("  still holds on real ones. Coverage far below nominal would mean the")
    print("  reliability layer is an artefact of the forward model, which is the")
    print("  single strongest objection to the whole approach. Note the real")
    print("  measurements carry the microscope's own unknown aberration, so this")
    print("  is not the nominal-operator case in the simulated sense -- it is a")
    print("  harder test, not an easier one.")

    if skipped:
        print("\n  skipped:")
        for s in skipped:
            print(f"    {s}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "overall": res, "per_structure": per,
             "rows": rows, "skipped": skipped}, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

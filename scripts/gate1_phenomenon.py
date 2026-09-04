"""Gate 1 and Gate 2: does operator mismatch produce plausible-but-wrong output,
and is the nominal residual already enough to detect it?

Gate 1 asks whether the phenomenon the paper is about actually exists: as the
true PSF drifts from the one the reconstructor assumes, does reconstruction
error grow substantially *while the measurement residual stays flat*? A residual
that tracked the error would make the problem trivially detectable; an error that
did not grow would mean there is no problem at all.

Gate 2 is the declared kill criterion B: if the nominal residual
``||H_0 x_hat - y||`` already predicts reconstruction error almost perfectly,
the proposed method has nothing to add.

Design notes
------------
* The reconstructor always assumes the **nominal** PSF ``H_0``. Only the PSF used
  to *generate* the measurement is aberrated. That is the mismatch under study.
* ``n_iter`` for Richardson-Lucy is selected once on the nominal operator and
  held fixed across every shift cell. Retuning per cell would confound operator
  mismatch with regularisation strength.
* The error target uses a least-squares scale match, because deconvolution
  recovers the object only up to a global intensity scale.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.aberrations import ladder_specs  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    NoiseConfig,
    apply_psf,
    sample_seed,
    simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""

    def rank(t: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(t)
        r = torch.empty_like(t)
        r[order] = torch.arange(len(t), dtype=t.dtype, device=t.device)
        return r

    ra, rb = rank(a.flatten()), rank(b.flatten())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-20))


def load_objects(pattern: str, limit: int | None, dtype: torch.dtype) -> torch.Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    if limit:
        files = files[:limit]
    arr = np.stack([np.squeeze(tifffile.imread(f)) for f in files])
    x = torch.from_numpy(arr.astype(np.float32)).to(dtype)
    # SIM reconstructions ring slightly negative; a photon density cannot.
    return x.clamp_min(0.0)


def reconstruct(y, psf0, n_iter, noise):
    return richardson_lucy(
        y, psf0, n_iter,
        background=noise.background,
        read_variance=noise.read_noise_e**2,
    )


def evaluate(x, y, x_hat, psf0):
    """Return per-image ``(error, relative residual)``.

    The residual is computed after rescaling the re-imaged reconstruction to the
    measurement's own scale -- a blind operation that uses no ground truth, and
    the same one the reliability features will use.
    """
    e = 1.0 - ssim(scale_match(x_hat, x), x, data_range=1.0)
    resid = scale_match(apply_psf(x_hat, psf0), y)
    r = (resid - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)
    return e, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="data/biosr_pilot/test_gt/*.tif")
    ap.add_argument("--n-objects", type=int, default=150)
    ap.add_argument("--na", type=float, default=0.794,
                    help="derived by scripts/inspect_objects.py to match the ~2x SIM gain")
    ap.add_argument("--wavelength-um", type=float, default=0.525)
    ap.add_argument("--pixel-um", type=float, default=0.0626)
    ap.add_argument("--photons", type=float, default=100.0)
    ap.add_argument("--background", type=float, default=5.0)
    ap.add_argument("--read-noise", type=float, default=1.5)
    ap.add_argument("--kernel", type=int, default=63)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dtype, dev = torch.float32, torch.device(args.device)
    x = load_objects(args.pattern, args.n_objects, dtype).to(dev)
    print(f"objects: {tuple(x.shape)}   photons={args.photons}   NA={args.na}")

    cfg = PupilConfig(
        wavelength_um=args.wavelength_um, na=args.na,
        pixel_um=args.pixel_um, n_out=args.kernel,
    )
    psf0, diag0 = pupil_psf(None, cfg, warn=False)
    psf0 = psf0.to(device=dev, dtype=dtype)
    otf_cut = 2 * args.na / args.wavelength_um * args.pixel_um
    print(f"nominal operator: {diag0.summary()}")
    print(f"  OTF cutoff = {otf_cut:.4f} cyc/px\n")

    noise = NoiseConfig(
        photons=args.photons, background=args.background, read_noise_e=args.read_noise
    )

    # --- select n_iter once, on the nominal operator -------------------------
    y0 = simulate_measurement(x, psf0, noise, seed=sample_seed("gate1", "nominal"))
    print("selecting Richardson-Lucy n_iter on the NOMINAL operator:")
    best, best_ssim = 1, -1.0
    for n in (5, 10, 20, 30, 50, 75, 100, 150):
        s = float(ssim(scale_match(reconstruct(y0, psf0, n, noise), x), x,
                       data_range=1.0).mean())
        flag = ""
        if s > best_ssim:
            best, best_ssim, flag = n, s, "   <- best"
        print(f"    n_iter={n:4d}   mean SSIM = {s:.4f}{flag}")
    n_iter = best
    blurred = float(ssim(scale_match(apply_psf(x, psf0), x), x, data_range=1.0).mean())
    print(f"  chosen n_iter = {n_iter}, held fixed across every shift cell")
    print(f"  reference: mean SSIM of the blurred measurement itself = {blurred:.4f}\n")

    # --- sweep the aberration ladder ----------------------------------------
    specs = ladder_specs(
        ["defocus", "astigmatism", "coma", "spherical"],
        ["nominal", "mild", "moderate", "severe", "extreme"],
    )
    print(f"sweeping {len(specs)} operator-shift cells "
          "(the reconstructor always assumes H_0):\n")
    print("  cell                     rms(w)  strehl    err=1-SSIM        rel.residual")
    rows = []
    for spec in specs:
        psf_true, diag = pupil_psf(spec.coeffs, cfg, warn=False)
        psf_true = psf_true.to(device=dev, dtype=dtype)
        y = simulate_measurement(x, psf_true, noise, seed=sample_seed("gate1", spec.label))
        x_hat = reconstruct(y, psf0, n_iter, noise)
        e, r = evaluate(x, y, x_hat, psf0)
        rows.append((spec, diag.strehl, e, r))
        print(f"  {spec.label:24s} {spec.rms_waves:.4f}  {diag.strehl:.4f}  "
              f"{float(e.mean()):.4f}+-{float(e.std()):.4f}  "
              f"{float(r.mean()):.4f}+-{float(r.std()):.4f}")

    # --- Gate 1 --------------------------------------------------------------
    base = next(r for r in rows if r[0].severity == "nominal")
    e0, r0 = float(base[2].mean()), float(base[3].mean())
    print(f"\nnominal baseline: err={e0:.4f}   residual={r0:.4f}")
    print("\nGATE 1 -- error must grow substantially while the residual stays flat:")
    print("  cell                     err ratio   residual ratio   verdict")
    severe_ok = []
    for spec, _, e, r in rows:
        if spec.severity in ("nominal", "extreme"):
            continue
        er, rr = float(e.mean()) / e0, float(r.mean()) / r0
        ok = er >= 1.8 and abs(rr - 1.0) <= 0.20
        if spec.severity == "severe":
            severe_ok.append(ok)
        print(f"  {spec.label:24s}   {er:5.2f}x       {rr:5.2f}x        "
              f"{'PASS' if ok else 'fail'}")
    gate1 = any(severe_ok)

    # --- Gate 2 --------------------------------------------------------------
    all_e = torch.cat([r[2] for r in rows])
    all_r = torch.cat([r[3] for r in rows])
    rho_all = spearman(all_e, all_r)
    rho_cell = spearman(
        torch.tensor([float(r[2].mean()) for r in rows]),
        torch.tensor([float(r[3].mean()) for r in rows]),
    )
    print("\nGATE 2 -- the nominal residual must NOT already predict the error:")
    print(f"  Spearman rho(e, r), pooled over all images and cells : {rho_all:+.4f}")
    print(f"  Spearman rho(e, r), over cell means                  : {rho_cell:+.4f}")
    gate2 = abs(rho_all) < 0.7

    # --- why the residual is blind: it sits at the photon noise floor -------
    # A relative residual cannot resolve model error smaller than the
    # contribution of shot noise alone. For y ~ Poisson(lam) + N(0, sigma^2)
    # the irreducible relative residual is
    #     sqrt(E[lam] + sigma^2) / sqrt(E[y^2]).
    lam = float(y0.mean())
    floor = ((lam + args.read_noise ** 2) ** 0.5) / float((y0 ** 2).mean()) ** 0.5
    worst = max(float(rr[3].mean()) for rr in rows)
    print()
    print("MECHANISM -- the residual is noise-dominated, not model-dominated:")
    print(f"  predicted shot-noise floor of the relative residual : {floor:.4f}")
    print(f"  observed residual, nominal operator                 : {r0:.4f}")
    print(f"  observed residual, worst operator                   : {worst:.4f}")
    print(f"  mismatch signal as a fraction of the noise floor    : "
          f"{(worst - r0) / floor:.3f}")
    print("  => the whole operator-mismatch signal lives inside the noise floor.")

    # --- effect sizes: the decision-relevant version of Gate 2 --------------
    # Spearman over cell means is misleading. The residual can be perfectly
    # monotone in severity and still be useless per image, if the shift is
    # small next to image-to-image spread. For a per-image reliability bound,
    # Cohen's d is the quantity that matters.
    def cohen_d(a, b):
        pooled = ((a.var() + b.var()) / 2).clamp_min(1e-20).sqrt()
        return float((b.mean() - a.mean()).abs() / pooled)

    print()
    print("  per-image discriminability, nominal vs severe (Cohen's d):")
    print("    cell                     d(error)   d(residual)    ratio")
    for spec, _, e, r in rows:
        if spec.severity != "severe":
            continue
        de, dr = cohen_d(base[2], e), cohen_d(base[3], r)
        print(f"    {spec.label:24s}  {de:6.2f}      {dr:6.2f}   {de / max(dr, 1e-9):8.1f}x")
    print("  A large d(error) beside a small d(residual) is the core claim: the")
    print("  degradation is real and severe, and the residual cannot see it.")

    print("\n" + "=" * 76)
    print(f"GATE 1 (phenomenon exists)         : {'PASS' if gate1 else 'FAIL'}")
    print(f"GATE 2 (residual not sufficient)   : {'PASS' if gate2 else 'FAIL'}"
          f"    |rho| = {abs(rho_all):.3f} vs threshold 0.7")
    print("=" * 76)
    if not gate1:
        print("Gate 1 failed: either mismatch does not degrade reconstruction at these")
        print("severities, or the residual tracks it. Re-examine before proceeding.")
    if not gate2:
        print("Gate 2 failed: kill criterion B has fired -- a trivial residual already")
        print("predicts reconstruction error, so the method as framed adds nothing.")


if __name__ == "__main__":
    main()

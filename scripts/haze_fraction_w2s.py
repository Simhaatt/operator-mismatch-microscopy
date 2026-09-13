"""Measure the out-of-focus fraction of the W2S real widefield measurements.

This is the W2S counterpart of ``haze_fraction.py`` and it must be run before
any W2S simulation. Haze was the single most consequential parameter in the
BioSR work: at 0.048 the nominal residual correlates +0.647 with reconstruction
error, and at 0.310 it correlates -0.251. Carrying BioSR's 0.31 over to a
different instrument without measuring it would repeat exactly the mistake that
took a full re-extraction to find.

The measurement is far simpler than BioSR's. BioSR had to reconstruct widefield
by summing raw SIM frames, which required working out the frame ordering from a
phase-cancellation test. W2S ships 400 genuine widefield captures per field, so
the noise-free widefield is just their mean.

The fit, as before, is

    y_real  ~  a * blur(GT) + b

with the pedestal ``b`` reported as a fraction of ``mean(y_real)``. ``a``
absorbs the arbitrary intensity scale of a SIM reconstruction; ``b`` is the flat
out-of-focus contribution the forward model cannot produce, because SIM
reconstruction removes out-of-focus light from the ground truth.

Sampling
--------
The SIM grid is 1024x1024 at an inferred 32 nm; the widefield is 512x512 at
64 nm. The model is applied in the physical order -- blur the object on the fine
grid, then bin 2x to the camera grid -- rather than upsampling the measurement,
so the camera's integration is represented rather than interpolated away.

Emission wavelength is not stated by W2S, only the excitation lines (488, 561,
640 nm). Stokes shifts of 20-30 nm are typical for the dyes used at these lines,
so emission is approximated per channel and the assumption is recorded in the
output. A 20 nm error moves the OTF cutoff by about 4%, which is small relative
to the haze fraction being measured.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optics.forward_model import apply_psf  # noqa: E402
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402

W2S_NA = 1.49
W2S_SIM_PIXEL_UM = 0.032
W2S_WF_PIXEL_UM = 0.064
# Excitation lines and the emission approximated from them.
W2S_EMISSION_UM = {0: 0.515, 1: 0.590, 2: 0.670}


def bin2(a: torch.Tensor) -> torch.Tensor:
    """2x2 sum-binning: the camera integrates, it does not interpolate."""
    h, w = a.shape[-2:]
    return a[..., : h // 2 * 2, : w // 2 * 2].reshape(
        *a.shape[:-2], h // 2, 2, w // 2, 2
    ).sum(dim=(-3, -1))


def haze_of_view(gt: np.ndarray, wf_mean: np.ndarray, psf: torch.Tensor) -> dict:
    """Fit ``a * bin(blur(GT)) + b`` against the real widefield mean."""
    g = torch.as_tensor(gt, dtype=torch.float64)
    g = g - g.min()
    blurred = apply_psf(g[None], psf, mode="reflect")[0]
    model = bin2(blurred)

    y = torch.as_tensor(wf_mean, dtype=torch.float64)
    if model.shape != y.shape:
        raise ValueError(f"shape mismatch: model {model.shape} vs widefield {y.shape}")

    # Least squares for (a, b) in y ~ a*model + b.
    X = torch.stack([model.flatten(), torch.ones(model.numel(), dtype=torch.float64)], 1)
    sol = torch.linalg.lstsq(X, y.flatten()[:, None]).solution[:, 0]
    a, b = float(sol[0]), float(sol[1])

    pred = a * model + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-20)
    ymean = float(y.mean())
    return {
        "a": a,
        "b": b,
        "r2": r2,
        "y_mean": ymean,
        "haze_fraction": b / ymean if ymean > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="W2S_raw.zip")
    ap.add_argument("--kernel", type=int, default=127)
    ap.add_argument("--limit-fov", type=int, default=None)
    ap.add_argument("--min-r2", type=float, default=0.5,
                    help="drop fits below this R^2; a bad fit means the model "
                         "did not describe the measurement, so its pedestal is "
                         "not a haze estimate")
    ap.add_argument("--out", default="experiments/realism/w2s_haze.json")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    sims = sorted(n for n in z.namelist() if re.search(r"/sim_channel\d\.npy$", n))
    if args.limit_fov:
        keep = sorted({re.search(r"(Image\d+)/", n).group(1) for n in sims})[: args.limit_fov]
        sims = [n for n in sims if re.search(r"(Image\d+)/", n).group(1) in keep]

    psfs = {
        ch: pupil_psf(
            None,
            PupilConfig(na=W2S_NA, wavelength_um=lam,
                        pixel_um=W2S_SIM_PIXEL_UM, n_out=args.kernel),
            warn=False,
        )[0].to(torch.float64)
        for ch, lam in W2S_EMISSION_UM.items()
    }

    print(f"W2S haze measurement over {len(sims)} views "
          f"(NA {W2S_NA}, SIM pixel {W2S_SIM_PIXEL_UM} um)")
    rows = []
    for i, sname in enumerate(sims):
        fov = re.search(r"(Image\d+)/", sname).group(1)
        ch = int(re.search(r"sim_channel(\d)\.npy$", sname).group(1))
        wname = f"raw/{fov}/wf_channel{ch}.npy"
        try:
            gt = np.load(io.BytesIO(z.read(sname)))
            wf = np.load(io.BytesIO(z.read(wname))).astype(np.float64).mean(axis=0)
        except KeyError:
            print(f"  {fov} ch{ch}: missing widefield, skipped")
            continue
        r = haze_of_view(gt, wf, psfs[ch])
        r.update({"fov": fov, "channel": ch, "emission_um": W2S_EMISSION_UM[ch]})
        rows.append(r)
        if (i + 1) % 30 == 0:
            good = [x["haze_fraction"] for x in rows if x["r2"] >= args.min_r2]
            print(f"  {i+1}/{len(sims)}  running median haze "
                  f"{np.median(good) if good else float('nan'):.3f}", flush=True)

    good = [r for r in rows if r["r2"] >= args.min_r2]
    h = np.array([r["haze_fraction"] for r in good])
    print()
    print(f"  fitted {len(rows)} views, {len(good)} with R^2 >= {args.min_r2}")
    if len(h):
        print(f"  haze fraction: median {np.median(h):.3f}, "
              f"mean {h.mean():.3f} +/- {h.std():.3f}, "
              f"range {h.min():.3f} to {h.max():.3f}")
        print(f"  R^2: median {np.median([r['r2'] for r in good]):.3f}")
        print()
        print(f"  BioSR comparison: median 0.33 (used as 0.31 in simulation)")
    per_ch = {}
    for ch in sorted(W2S_EMISSION_UM):
        v = [r["haze_fraction"] for r in good if r["channel"] == ch]
        if v:
            per_ch[ch] = float(np.median(v))
            print(f"    channel {ch} ({W2S_EMISSION_UM[ch]*1000:.0f} nm emission): "
                  f"median {np.median(v):.3f}  (n={len(v)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "na": W2S_NA,
        "sim_pixel_um": W2S_SIM_PIXEL_UM,
        "wf_pixel_um": W2S_WF_PIXEL_UM,
        "emission_um_assumed": W2S_EMISSION_UM,
        "emission_is_assumed": True,
        "min_r2": args.min_r2,
        "n_fitted": len(rows),
        "n_kept": len(good),
        "haze_median": float(np.median(h)) if len(h) else None,
        "haze_mean": float(h.mean()) if len(h) else None,
        "haze_std": float(h.std()) if len(h) else None,
        "haze_median_per_channel": per_ch,
        "rows": rows,
    }, indent=1))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()

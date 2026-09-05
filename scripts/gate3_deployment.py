"""Gate 3: can reliability be predicted from a SINGLE observation?

This is the deployment question, and it is not the one the earlier gates asked.

Gate 2 found that, holding a specimen fixed and varying the operator, the plain
residual tracks reconstruction error at rho ~ 0.9 for photon counts >= 100. But
that paired comparison is unavailable in practice: you never image the same
specimen under two operators. At deployment you hold one measurement and one
reconstruction and must say how wrong the reconstruction is, separating operator
mismatch from image content using that single observation.

So this script asks the question that actually decides the project:

  Given only (y, x_hat), how well can the error e = 1 - SSIM(x_hat, x) be
  predicted and *calibrated*, and does the PSF-family profile beat the plain
  nominal residual?

If the plain residual alone already gives tight, well-calibrated bounds, the
proposed method is unnecessary and we should know that before spending GPU time.

Protocol
--------
* Splits are blocked by **cell**, and one biological structure is held out
  entirely as a content-shift test.
* The predictor is fitted on train, conformalised on calibration, and evaluated
  on test (seen structures, unseen cells) and on the held-out structure.
* Bounds use one-sided conformalised quantile regression, so coverage is
  distribution-free under exchangeability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.stats import spearman  # noqa: E402

from src.data.biosr import index_biosr, load_object, split_by_cell  # noqa: E402
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.aberrations import ABERRATION_MODES, ladder_specs  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    NoiseConfig,
    sample_seed,
    simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402
from src.reliability.features import FEATURE_NAMES, oscr_features  # noqa: E402


def build_probes(cfg: PupilConfig, device, dtype) -> list[torch.Tensor]:
    """Operator family spanning plausible microscope states.

    Deliberately a coarse grid over aberration type and amplitude, and
    deliberately NOT containing the exact test operators -- if the true PSF were
    in the family the task would degenerate into a lookup.
    """
    probes = [pupil_psf(None, cfg, warn=False)[0]]
    for mode in ABERRATION_MODES.values():
        for amp in (0.05, 0.11, 0.18):
            probes.append(pupil_psf({mode: amp}, cfg, warn=False)[0])
    return [p.to(device=device, dtype=dtype) for p in probes]


def collect(entries, true_psfs, cfg, psf0, probes, noise, n_iter, crop, device, dtype, tag):
    """Feature matrix, targets and group labels for one split.

    ``true_psfs`` is a precomputed list of ``(spec, psf)``. Synthesising the PSF
    inside this loop instead would repeat a 1896-point FFT for every cell, which
    is 3876 redundant syntheses over a full run.
    """
    feats, errs, groups, cells = [], [], [], []
    for i, entry in enumerate(entries):
        x = load_object(entry, crop=crop, dtype=dtype).to(device)[None]
        for spec, psf_true in true_psfs:
            y = simulate_measurement(
                x[0], psf_true, noise, seed=sample_seed(tag, entry.key, spec.label)
            )[None]
            x_hat = richardson_lucy(
                y, psf0, n_iter, background=noise.background,
                read_variance=noise.read_noise_e**2,
            )
            feats.append(oscr_features(y, x_hat, psf0, probes)[0].cpu())
            errs.append(float(1.0 - ssim(scale_match(x_hat[0], x[0]), x[0], data_range=1.0)))
            groups.append(spec.severity)
            cells.append(entry.key)
        if (i + 1) % 10 == 0:
            print(f"    {tag}: {i+1}/{len(entries)} cells", flush=True)
    return torch.stack(feats).numpy(), np.array(errs), np.array(groups), np.array(cells)


def conformal_upper_bound(model, Xc, ec, Xt, alpha):
    """One-sided conformalised quantile regression.

    ``model`` predicts the (1-alpha) conditional quantile of the error. The
    conformity score ``e - q(z)`` is corrected by its empirical (1-alpha)
    quantile on the calibration split, which restores distribution-free marginal
    coverage even when the quantile model is badly misspecified.
    """
    n = len(ec)
    scores = ec - model.predict(Xc)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    correction = np.sort(scores)[min(max(k, 0), n - 1)]
    return model.predict(Xt) + correction


def evaluate(name, cols, Xtr, etr, Xc, ec, sets, alpha):
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        loss="quantile", alpha=1 - alpha, n_estimators=300,
        max_depth=3, learning_rate=0.05, random_state=0,
    )
    model.fit(Xtr[:, cols], etr)
    point = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0
    ).fit(Xtr[:, cols], etr)

    out = {}
    for split, (X, e, groups) in sets.items():
        U = conformal_upper_bound(model, Xc[:, cols], ec, X[:, cols], alpha)
        pred = point.predict(X[:, cols])
        cov = float((e <= U).mean())
        worst = min(
            float((e[groups == g] <= U[groups == g]).mean())
            for g in np.unique(groups)
        )
        out[split] = {
            "spearman": spearman(pred, e),
            "mae": float(np.abs(pred - e).mean()),
            "coverage": cov,
            "worst_group_coverage": worst,
            "mean_width": float(U.mean()),
            "n": int(len(e)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--photons", type=float, default=100.0)
    ap.add_argument("--n-iter", type=int, default=75)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--held-out", default="CCPs")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
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

    idx = index_biosr()
    sp = split_by_cell(idx, held_out_structure=args.held_out)
    if args.max_cells:
        sp = {k: v[: args.max_cells] for k, v in sp.items()}
    print(f"photons={args.photons}  crop={args.crop}  n_iter={args.n_iter}  "
          f"probes={len(probes)}  cells=" +
          ", ".join(f"{k}:{len(v)}" for k, v in sp.items()))

    print("  precomputing true operators once (a 1896-point FFT each)...")
    true_psfs = [
        (spec, pupil_psf(spec.coeffs, cfg, warn=False)[0].to(device=device, dtype=dtype))
        for spec in specs
    ]

    data = {}
    for split, entries in sp.items():
        print(f"  collecting {split} ({len(entries)} cells x {len(specs)} operators)...")
        data[split] = collect(entries, true_psfs, cfg, psf0, probes, noise,
                              args.n_iter, args.crop, device, dtype, split)

    Xtr, etr, _, _ = data["train"]
    Xc, ec, _, _ = data["cal"]
    sets = {s: (data[s][0], data[s][1], data[s][2]) for s in ("test", "ood")}

    plain = [FEATURE_NAMES.index("r_nominal")]
    full = list(range(len(FEATURE_NAMES)))
    results = {
        "plain_residual": evaluate("plain", plain, Xtr, etr, Xc, ec, sets, args.alpha),
        "oscr_full": evaluate("oscr", full, Xtr, etr, Xc, ec, sets, args.alpha),
    }

    print()
    print(f"GATE 3 -- single-observation reliability, nominal coverage {1-args.alpha:.0%}")
    print()
    hdr = f"  {'predictor':16s} {'split':6s} {'rho':>7s} {'MAE':>8s} {'cover':>7s} {'worst-grp':>10s} {'width':>8s}"
    print(hdr)
    for name, r in results.items():
        for split, m in r.items():
            print(f"  {name:16s} {split:6s} {m['spearman']:+7.3f} {m['mae']:8.4f} "
                  f"{m['coverage']:7.3f} {m['worst_group_coverage']:10.3f} {m['mean_width']:8.4f}")
    print()
    for split in ("test", "ood"):
        p, o = results["plain_residual"][split], results["oscr_full"][split]
        gain = (p["mean_width"] - o["mean_width"]) / max(p["mean_width"], 1e-9)
        print(f"  {split}: OSCR vs plain residual -> rho {p['spearman']:+.3f} to "
              f"{o['spearman']:+.3f}, bound width {gain:+.1%}")
    print()
    print("  A tighter bound at equal coverage is the only way OSCR earns its cost.")
    print("  If the plain residual matches it, the profile adds nothing usable.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "features": FEATURE_NAMES, "results": results},
            indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

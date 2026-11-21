"""In-domain coverage on real measurements, alpha-independent scheme.

Reproduces the real-measurement coverage table of the supplementary material
(Table "In-Domain 90% Coverage on Real Measurements") and the "Real-data
coverage" row of the paper's headline table.

The procedure is the alpha-independent normalised split-conformal construction
of ``gate4_scaled.py`` (Section II-B of the paper), restricted to the
in-domain axis: calibrate on the whole calibration split, evaluate on the
in-domain test split.

``gate4_scaled.py`` itself cannot be pointed at a real-measurement cache: it
also fits a mismatch localiser against the RMS wavefront error, and real
measurements have no known wavefront error (``rms`` is NaN). The localiser is
only used for the Mondrian/RLCP schemes, never for the split-conformal bound,
so this script fits the mean and scale models exactly as ``gate4_scaled`` does
and leaves the localiser out.

"Seed" here means: the random half of the training split used to fit the mean
and scale models, and the gradient-boosting random state. The cache's
train/calibration/test partition is fixed when the cache is built.

    python scripts/real_coverage.py --out paper/results/real_coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import one_sided_correction  # noqa: E402

SCALE_FLOOR_FRAC = 0.10

DEFAULT_CACHES = (
    ("BioSR", "RL", "experiments/features19/real_rl.npz"),
    ("BioSR", "TV", "experiments/features19/real_tv.npz"),
    ("BioSR", "residual U-Net", "experiments/features19/real_unet.npz"),
    ("BioSR", "unrolled", "experiments/features19/real_unrolled.npz"),
    ("W2S", "RL", "experiments/features19/real_w2s_rl.npz"),
)


def fit_mean_scale(X, e, seed):
    """Conditional mean and scale on a random half of the training split."""
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(e)
    perm = np.random.default_rng(seed).permutation(n)
    b = perm[n // 2:]
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
    mu = GradientBoostingRegressor(**kw).fit(X[b], e[b])
    resid = np.abs(e[b] - mu.predict(X[b]))
    sigma = GradientBoostingRegressor(**kw).fit(X[b], resid)
    return mu, sigma, SCALE_FLOOR_FRAC * float(resid.mean())


def in_domain_coverage(path, alpha, seeds):
    d = np.load(path, allow_pickle=True)
    covs, widths = [], []
    for seed in range(seeds):
        mu, sigma, floor = fit_mean_scale(d["train_X"], d["train_e"], seed)

        def scale(X):
            return np.maximum(sigma.predict(X), floor)

        s_cal = (d["cal_e"] - mu.predict(d["cal_X"])) / scale(d["cal_X"])
        q = one_sided_correction(s_cal, alpha)
        U = mu.predict(d["test_X"]) + scale(d["test_X"]) * q
        covs.append(float((d["test_e"] <= U).mean()))
        widths.append(float(U.mean()))
    return {
        "n_train": int(len(d["train_e"])),
        "n_cal": int(len(d["cal_e"])),
        "n_test": int(len(d["test_e"])),
        "n_ood": int(len(d["ood_e"])),
        "median_error": float(np.median(d["test_e"])),
        "coverage_mean": float(np.mean(covs)),
        "coverage_std": float(np.std(covs)),
        "width_mean": float(np.mean(widths)),
        "per_seed": covs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"REAL-MEASUREMENT IN-DOMAIN COVERAGE  nominal {1 - args.alpha:.3f}, "
          f"{args.seeds} seeds")
    print()
    print(f"  {'dataset':8s} {'reconstructor':16s} {'n_test':>6s} "
          f"{'median 1-SSIM':>14s} {'coverage':>16s}")
    rows = []
    for dataset, recon, path in DEFAULT_CACHES:
        if not Path(path).exists():
            print(f"  {dataset:8s} {recon:16s}  missing: {path}")
            continue
        r = in_domain_coverage(path, args.alpha, args.seeds)
        r.update(dataset=dataset, reconstructor=recon, cache=path)
        rows.append(r)
        print(f"  {dataset:8s} {recon:16s} {r['n_test']:6d} {r['median_error']:14.3f} "
              f"{r['coverage_mean']:9.3f} +- {r['coverage_std']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"alpha": args.alpha, "seeds": args.seeds, "rows": rows}, indent=1),
            encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

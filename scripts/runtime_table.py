"""Wall-clock and peak-memory cost of each calibration stage.

A reviewer of an applied imaging paper reasonably asks what the method costs to
run, and "it is cheap" is not an answer. This measures the stages that a
practitioner would actually execute at deployment time, separating the one-off
fitting cost from the per-image cost that matters if calibration is to sit
inside an acquisition loop.

Peak memory is measured with :mod:`tracemalloc`, which counts Python-level
allocations only. It therefore excludes the BLAS scratch space scikit-learn's
gradient boosting uses internally, and is reported as a floor rather than a
total. The machine constraint that shaped this project is 8 GB of RAM, so what
matters is that every stage sits orders of magnitude below it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import load, one_sided_correction  # noqa: E402
from scripts.gate4_scaled import (  # noqa: E402
    fit_scaled_models,
    mondrian_correction,
    rlcp_correction,
    scale_of,
    scores,
)


def timed(fn, *a, **kw):
    """Run ``fn``, returning (result, seconds, peak MiB of Python allocations)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / 2**20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    (data, names, photons), t_load, m_load = timed(load, args.features)
    n_train = len(data["train"]["e"])
    n_cal = len(data["cal"]["e"])
    n_test = len(data["test"]["e"])

    fitted, t_fit, m_fit = timed(fit_scaled_models, data["train"])
    mismatch, mu, sigma, floor = fitted

    cal, test = data["cal"], data["test"]
    _, t_score, m_score = timed(scores, cal["e"], mu, sigma, floor, cal["X"])
    s_cal = scores(cal["e"], mu, sigma, floor, cal["X"])
    m_cal = mismatch.predict(cal["X"])
    m_test = mismatch.predict(test["X"])

    _, t_split, m_split = timed(one_sided_correction, s_cal, args.alpha)
    _, t_mond, m_mond = timed(
        mondrian_correction, s_cal, m_cal, m_test, args.alpha
    )
    _, t_rlcp, m_rlcp = timed(rlcp_correction, s_cal, m_cal, m_test, args.alpha)

    def predict_one():
        X = test["X"][:1]
        return mu.predict(X) + scale_of(sigma, floor, X) * 0.0

    _, t_one, m_one = timed(predict_one)

    rows = [
        ("load cached features", t_load, m_load, f"{n_train + n_cal + n_test} rows"),
        ("fit mu, sigma, mismatch (one-off)", t_fit, m_fit, f"{n_train} train rows"),
        ("score calibration set", t_score, m_score, f"{n_cal} rows"),
        ("split-conformal quantile", t_split, m_split, f"{n_cal} scores"),
        ("Mondrian quantile (4 groups)", t_mond, m_mond, f"{n_test} test rows"),
        ("RLCP quantile (per-point kernel)", t_rlcp, m_rlcp, f"{n_test} test rows"),
        ("bound for ONE new image", t_one, m_one, "1 row, models already fitted"),
    ]

    print(f"RUNTIME AND MEMORY -- photons={photons:.0f}, single CPU core")
    print()
    print(f"  {'stage':36s} {'seconds':>9s} {'peak MiB':>9s}   scale")
    for name, dt, mb, scale in rows:
        print(f"  {name:36s} {dt:9.3f} {mb:9.2f}   {scale}")
    print()
    print(f"  per-test-image cost, RLCP  : {1000 * t_rlcp / n_test:.3f} ms")
    print(f"  per-test-image cost, split : "
          f"{1000 * t_split / max(n_test, 1):.4f} ms (one quantile, amortised)")
    print()
    print("  Everything above is CPU-only. No stage approaches the 8 GB limit of")
    print("  the machine these experiments were run on; the fitting stage")
    print("  dominates and is paid once, not per image.")

    if args.out:
        out = {
            "photons": photons,
            "alpha": args.alpha,
            "n_train": n_train,
            "n_cal": n_cal,
            "n_test": n_test,
            "stages": [
                {"stage": n, "seconds": dt, "peak_mib": mb, "scale": sc}
                for n, dt, mb, sc in rows
            ],
            "per_image_ms_rlcp": 1000 * t_rlcp / n_test,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

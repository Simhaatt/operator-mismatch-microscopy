"""Gate 4S: alpha-independent conformal calibration (normalised scores).

Why this script exists
----------------------
``gate4_calibration.py`` refits its base predictor at every miscoverage level,
as ``GradientBoostingRegressor(loss="quantile", alpha=1-alpha)``. That couples
the *predictor* to alpha, and below alpha ~= 0.02 the fit degenerates: asked for
the 0.999 quantile from ~884 training points, the quantile loss carries almost
no gradient signal and the model collapses toward a constant. Measured on
``rl_photons_100_bg45`` (haze 0.31), the spread of the base prediction over the
in-distribution test split falls

    alpha    0.100   0.050   0.020   0.010   0.005   0.001
    std      0.2420  0.2433  0.2422  0.0803  0.0218  0.0037

so by alpha = 0.001 the predictor is effectively unconditional. An
unconditional bound cannot adapt to a shifted operator, which is exactly what
the erratic small-alpha behaviour of Gate 4 reflects: operator-shift coverage
falls to 0.146 at alpha = 0.01 while the reported mean bound *narrows* by 40%.
Non-monotonicity in alpha is not a property of conformal prediction; it is an
artefact of refitting the predictor.

The fix
-------
Separate the predictor from the calibration level, which is what split
conformal assumes in the first place:

1. Fit a conditional mean ``mu(x)`` and a conditional scale ``sigma(x)`` once,
   with no reference to alpha.
2. Score each point by its scale-normalised residual ``s = (e - mu) / sigma``.
3. Take the usual finite-sample conformal quantile of ``s`` on calibration.
4. Report ``U(x) = mu(x) + sigma(x) * q_{1-alpha}``.

This is normalised (locally weighted) nonconformity -- Papadopoulos et al., and
Lei et al. (2018) -- applied to the mismatch setting. Two properties follow *by
construction* rather than by observation:

* **Monotonicity.** ``q_{1-alpha}`` is an order statistic of a score set that
  does not depend on alpha, so it is non-decreasing as alpha falls, and since
  ``sigma > 0``, every ``U(x)`` is non-decreasing too. The bound can never
  tighten when the requested confidence rises.
* **Preserved conditionality.** ``mu`` and ``sigma`` are fixed across alpha, so
  the bound stays as input-adaptive at alpha = 0.001 as at alpha = 0.1.

The marginal guarantee is unchanged: exchangeability of the calibration and
test scores still gives ``P(e <= U) >= 1 - alpha``, because normalising by a
function fitted on *disjoint* data leaves the scores exchangeable.

What this does and does not repair
----------------------------------
It repairs the *diagnostic*, not the phenomenon. Coverage under operator shift
is still lost -- that is the paper's finding, and it survives the fix. What the
fix removes is the possibility of reading the small-alpha numbers as evidence
about conformal prediction when they were evidence about a degenerate fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import (  # noqa: E402
    build_axes,
    coverage_report,
    load,
    one_sided_correction,
    weighted_quantile,
)
from src.metrics.stats import spearman  # noqa: E402

# Floor on the fitted scale. Without it a cell whose absolute residual is
# predicted near zero would divide its score by ~0 and dominate the quantile.
# Expressed as a fraction of the mean absolute residual so it carries no units.
SCALE_FLOOR_FRAC = 0.10


def fit_scaled_models(train, seed=0):
    """Fit mismatch score, conditional mean and conditional scale.

    All three are independent of alpha. The mismatch score and the
    mean/scale pair are fitted on disjoint halves of the training split, so the
    localiser used to define groups is never fitted on the samples that set the
    bound.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    n = len(train["e"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a, b = perm[: n // 2], perm[n // 2 :]

    mismatch = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][a], train["rms"][a])

    mu = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][b], train["e"][b])

    # Conditional scale from the absolute residual of the mean model, fitted on
    # the same half. Using the same half is deliberate: mu and sigma together
    # form one predictor, and it is the *calibration* split that must stay
    # disjoint from them for the conformal guarantee.
    resid = np.abs(train["e"][b] - mu.predict(train["X"][b]))
    sigma = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed
    ).fit(train["X"][b], resid)

    floor = SCALE_FLOOR_FRAC * float(resid.mean())
    return mismatch, mu, sigma, floor


def scale_of(sigma, floor, X):
    return np.maximum(sigma.predict(X), floor)


def scores(e, mu, sigma, floor, X):
    """Scale-normalised, one-sided conformity score."""
    return (e - mu.predict(X)) / scale_of(sigma, floor, X)


def mondrian_correction(s_cal, m_cal, m_test, alpha, n_groups=4):
    """Group-conditional quantile, groups from mismatch-score quantiles."""
    edges = np.quantile(m_cal, np.linspace(0, 1, n_groups + 1)[1:-1])
    g_cal = np.digitize(m_cal, edges)
    g_test = np.digitize(m_test, edges)
    out = np.empty(len(m_test))
    for g in range(n_groups):
        sel = g_cal == g
        q = one_sided_correction(s_cal[sel], alpha) if sel.sum() else float("inf")
        out[g_test == g] = q
    return out


def rlcp_correction(s_cal, m_cal, m_test, alpha, rng=None, bandwidth=None):
    """Randomised localised conformal prediction (Hore & Barber)."""
    rng = rng or np.random.default_rng(0)
    if bandwidth is None:
        bandwidth = float(np.std(m_cal)) * 0.5 or 1.0
    out = np.empty(len(m_test))
    for i, m in enumerate(m_test):
        anchor = m + rng.normal(0.0, bandwidth)
        w = np.exp(-0.5 * ((m_cal - anchor) / bandwidth) ** 2)
        out[i] = weighted_quantile(s_cal, w, alpha)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument(
        "--alphas",
        default="0.10,0.05,0.02,0.01,0.005,0.001",
        help="comma-separated miscoverage levels to sweep",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    data, names, photons = load(args.features)
    mismatch, mu, sigma, floor = fit_scaled_models(data["train"])

    for split in ("test", "ood"):
        m = mismatch.predict(data[split]["X"])
        print(f"  mismatch score vs true RMS wavefront error, {split}: "
              f"rho = {spearman(m, data[split]['rms']):+.3f}")

    # The claim the design is meant to establish: the base predictor's spread
    # no longer depends on alpha, because it is never refitted.
    test_scale = scale_of(sigma, floor, data["test"]["X"])
    print()
    print(f"  base mean prediction std (alpha-independent) : "
          f"{mu.predict(data['test']['X']).std():.4f}")
    print(f"  base scale prediction std (alpha-independent): {test_scale.std():.4f}")
    print(f"  scale floor                                  : {floor:.4f}")
    print()

    axes = build_axes(data)
    out: dict = {
        "photons": photons,
        "haze_frac": None,
        "scale_floor": floor,
        "alphas": alphas,
        "results": {},
    }
    d = np.load(args.features, allow_pickle=True)
    if "background" in d.files:
        bg = float(d["background"][0])
        out["haze_frac"] = bg / (photons + bg)

    print(f"GATE 4S -- alpha-independent normalised conformal, photons={photons:.0f}")
    print()
    header = (f"  {'shift axis':22s} {'scheme':10s} {'alpha':>6s} {'nominal':>8s} "
              f"{'coverage':>9s} {'worst-grp':>10s} {'bound':>8s}   n")
    print(header)
    for axis, (cal, test) in axes.items():
        m_cal = mismatch.predict(cal["X"])
        m_test = mismatch.predict(test["X"])
        s_cal = scores(cal["e"], mu, sigma, floor, cal["X"])
        base = mu.predict(test["X"])
        sc_test = scale_of(sigma, floor, test["X"])
        out["results"][axis] = {}
        for scheme in ("split", "mondrian", "rlcp"):
            out["results"][axis][scheme] = {}
            for alpha in alphas:
                if scheme == "split":
                    q = np.full(len(test["e"]), one_sided_correction(s_cal, alpha))
                elif scheme == "mondrian":
                    q = mondrian_correction(s_cal, m_cal, m_test, alpha)
                else:
                    q = rlcp_correction(s_cal, m_cal, m_test, alpha)
                U = base + sc_test * q
                globals_target = 1 - alpha
                rep = coverage_report_at(test["e"], U, test["severity"], globals_target)
                out["results"][axis][scheme][str(alpha)] = rep
                print(f"  {axis:22s} {scheme:10s} {alpha:6.3f} {1-alpha:8.3f} "
                      f"{rep['coverage']:9.3f} {rep['worst_group']:10.3f} "
                      f"{rep['mean_width']:8.4f}   {rep['n']}")
            print()

    # --- the constructive check: bounds must never tighten as alpha falls ----
    print("  MONOTONICITY CHECK (bound must not shrink as alpha falls):")
    violations = 0
    for axis, schemes in out["results"].items():
        for scheme, per_alpha in schemes.items():
            widths = [per_alpha[str(a)]["mean_width"] for a in alphas]
            finite = [w for w in widths if np.isfinite(w)]
            bad = sum(
                1
                for x, y in zip(finite, finite[1:])
                if y < x - 1e-12
            )
            violations += bad
            status = "ok" if bad == 0 else f"{bad} VIOLATION(S)"
            print(f"    {axis:22s} {scheme:10s} {status}")
    print()
    print(f"  total monotonicity violations: {violations}")
    out["monotonicity_violations"] = violations

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\n  wrote {args.out}")


def coverage_report_at(e, U, groups, target):
    """``coverage_report`` with an explicit target rather than the module global."""
    import scripts.gate4_calibration as g4

    saved = g4.COVER_TARGET
    g4.COVER_TARGET = target
    try:
        return coverage_report(e, U, groups)
    finally:
        g4.COVER_TARGET = saved


if __name__ == "__main__":
    main()

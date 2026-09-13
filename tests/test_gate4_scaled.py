"""Tests for the alpha-independent normalised conformal calibration.

The point of ``gate4_scaled`` is that two properties hold *by construction*
rather than by observation, so they are worth asserting rather than eyeballing
in a table:

* the bound never tightens as the requested confidence rises, and
* the base predictor keeps its conditionality at every alpha, because it is
  fitted once and never refitted per level.

The original ``gate4_calibration`` violates the first in 9 of 12 axis-by-scheme
combinations, which is what motivated the replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate4_calibration import one_sided_correction  # noqa: E402
from scripts.gate4_scaled import (  # noqa: E402
    SCALE_FLOOR_FRAC,
    fit_scaled_models,
    mondrian_correction,
    rlcp_correction,
    scale_of,
    scores,
)

ALPHAS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]


@pytest.fixture(scope="module")
def toy():
    """A small heteroscedastic regression problem with a usable train split.

    Deliberately synthetic: the properties under test are structural, so they
    must hold without reference to the microscopy caches (which are not
    redistributed and may be absent).
    """
    rng = np.random.default_rng(0)
    n, d = 600, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    rms = np.abs(X[:, 0]) * 0.1
    # Error scale grows with the first feature, so a conditional sigma has
    # something real to find.
    e = (0.2 + 0.5 * np.abs(X[:, 0])) * np.abs(rng.normal(size=n)) + 0.1
    return {
        "X": X.astype(np.float32),
        "e": e.astype(np.float32),
        "rms": rms.astype(np.float32),
    }


@pytest.fixture(scope="module")
def fitted(toy):
    return fit_scaled_models(toy, seed=0)


def test_scale_floor_is_positive_and_respected(toy, fitted):
    _, _, sigma, floor = fitted
    assert floor > 0.0
    s = scale_of(sigma, floor, toy["X"])
    assert np.all(s >= floor - 1e-12)
    assert np.all(np.isfinite(s))


def test_scale_floor_fraction_is_a_fraction():
    assert 0.0 < SCALE_FLOOR_FRAC < 1.0


def test_split_bound_is_monotone_in_alpha(toy, fitted):
    """The whole point: tightening alpha must never narrow the bound."""
    _, mu, sigma, floor = fitted
    s_cal = scores(toy["e"], mu, sigma, floor, toy["X"])
    base = mu.predict(toy["X"])
    sc = scale_of(sigma, floor, toy["X"])

    prev = None
    for alpha in ALPHAS:
        q = one_sided_correction(s_cal, alpha)
        U = base + sc * q
        if prev is not None:
            # Every element, not merely the mean.
            assert np.all(U >= prev - 1e-9), f"bound tightened at alpha={alpha}"
        prev = U


@pytest.mark.parametrize("scheme", ["mondrian", "rlcp"])
def test_group_schemes_are_monotone_in_alpha(toy, fitted, scheme):
    mismatch, mu, sigma, floor = fitted
    s_cal = scores(toy["e"], mu, sigma, floor, toy["X"])
    m_cal = mismatch.predict(toy["X"])
    m_test = mismatch.predict(toy["X"])

    prev = None
    for alpha in ALPHAS:
        if scheme == "mondrian":
            q = mondrian_correction(s_cal, m_cal, m_test, alpha)
        else:
            q = rlcp_correction(s_cal, m_cal, m_test, alpha,
                                rng=np.random.default_rng(0))
        if prev is not None:
            finite = np.isfinite(q) & np.isfinite(prev)
            assert np.all(q[finite] >= prev[finite] - 1e-9), (
                f"{scheme} correction tightened at alpha={alpha}"
            )
        prev = q


def test_base_predictor_conditionality_is_alpha_independent(toy):
    """mu and sigma must not depend on alpha -- they are never passed it.

    The original implementation refitted a quantile regressor per alpha, and
    its prediction spread collapsed from 0.242 to 0.0037 between alpha 0.1 and
    0.001, losing all input-dependence. Refitting here changes nothing, so the
    spread is identical across calls.
    """
    spreads = []
    for _ in ALPHAS:
        _, mu, sigma, floor = fit_scaled_models(toy, seed=0)
        spreads.append(float(mu.predict(toy["X"]).std()))
    assert np.allclose(spreads, spreads[0])
    # And it must actually be conditional, not a constant dressed up as one.
    assert spreads[0] > 1e-3


def test_marginal_coverage_holds_on_exchangeable_split(toy):
    """Split conformal's guarantee, checked on genuinely exchangeable halves."""
    rng = np.random.default_rng(1)
    n = len(toy["e"])
    perm = rng.permutation(n)
    tr, cal, te = perm[: n // 3], perm[n // 3 : 2 * n // 3], perm[2 * n // 3 :]
    train = {k: v[tr] for k, v in toy.items()}
    _, mu, sigma, floor = fit_scaled_models(train, seed=0)

    s_cal = scores(toy["e"][cal], mu, sigma, floor, toy["X"][cal])
    alpha = 0.1
    q = one_sided_correction(s_cal, alpha)
    U = mu.predict(toy["X"][te]) + scale_of(sigma, floor, toy["X"][te]) * q
    cov = float((toy["e"][te] <= U).mean())
    # Finite-sample slack on ~200 test points; this is a sanity band, not a
    # tight claim.
    assert cov >= 1 - alpha - 0.10, f"coverage {cov:.3f} far below nominal"


def test_scores_are_scale_normalised(toy, fitted):
    """A score is a residual divided by the predicted scale, nothing more."""
    _, mu, sigma, floor = fitted
    s = scores(toy["e"], mu, sigma, floor, toy["X"])
    expected = (toy["e"] - mu.predict(toy["X"])) / scale_of(sigma, floor, toy["X"])
    assert np.allclose(s, expected)


def test_mondrian_groups_partition_the_test_set(toy, fitted):
    """Every test point gets exactly one group quantile, none left unset."""
    mismatch, mu, sigma, floor = fitted
    s_cal = scores(toy["e"], mu, sigma, floor, toy["X"])
    m_cal = mismatch.predict(toy["X"])
    m_test = mismatch.predict(toy["X"])
    q = mondrian_correction(s_cal, m_cal, m_test, 0.1, n_groups=4)
    assert q.shape == m_test.shape
    assert not np.any(np.isnan(q))

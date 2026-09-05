"""Tests for the feature-block extension to the Gate 5 blind estimator.

The blind proxy is what limits the contribution -- the labelled bound is already
tight -- so which block of features it may look at is the substantive question.
These pin the plumbing that makes the comparison fair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gate5_bound import (  # noqa: E402
    directional_excess, feature_blocks, novelty, spearman, tv_distance,
)

FOURTEEN = [
    "r_nominal", "r_min", "r_mean", "r_std", "r_gap", "r_argmin_frac",
    "r_curvature", "band_0", "band_1", "band_2", "band_3", "y_mean", "y_std",
    "xhat_std",
]
OTF = ["otf_white_chi2", "otf_excess_inband", "otf_excess_at_cut", "otf_slope",
       "otf_noise_ref"]


def test_blocks_on_a_19_feature_cache_partition_the_columns() -> None:
    b = feature_blocks(FOURTEEN + OTF)
    assert sorted(b) == ["all", "oscr", "otf", "otf_core", "residual"]
    assert set(b["otf_core"]).issubset(set(b["otf"]))
    assert len(b["otf_core"]) == 3
    assert len(b["all"]) == 19 and len(b["otf"]) == 5 and len(b["oscr"]) == 14
    assert set(b["otf"]).isdisjoint(b["oscr"])
    assert sorted(np.concatenate([b["otf"], b["oscr"]])) == list(range(19))
    assert list(b["residual"]) == [0]


def test_blocks_on_a_14_feature_cache_omit_otf_rather_than_faking_it() -> None:
    """A cache predating the OTF block must report only what it can support.

    Silently substituting a different set of columns would make a 4-cell rho
    look like a 16-cell one, which is exactly the error the project's standing
    rule forbids.
    """
    b = feature_blocks(FOURTEEN)
    assert sorted(b) == ["all", "residual"]
    assert "otf_core" not in b
    assert len(b["all"]) == 14


def test_blocks_follow_the_cache_column_order_not_a_hardcoded_one() -> None:
    names = OTF + FOURTEEN          # OTF first
    b = feature_blocks(names)
    assert sorted(b["otf"]) == [0, 1, 2, 3, 4]
    assert b["residual"][0] == names.index("r_nominal")


def test_novelty_on_a_column_subset_matches_slicing_the_matrix_first() -> None:
    rng = np.random.default_rng(0)
    X_ref, X = rng.standard_normal((200, 19)), rng.standard_normal((50, 19))
    cols = np.array([14, 15, 16, 17, 18])
    assert np.allclose(novelty(X_ref, X, cols),
                       novelty(X_ref[:, cols], X[:, cols]))


def test_novelty_survives_a_single_column_block() -> None:
    """np.cov collapses to a scalar for one column; inv would fail unguarded."""
    rng = np.random.default_rng(1)
    X_ref, X = rng.standard_normal((100, 3)), rng.standard_normal((20, 3))
    d = novelty(X_ref, X, np.array([0]))
    assert d.shape == (20,) and np.all(np.isfinite(d)) and np.all(d >= 0)


def test_novelty_is_larger_for_a_shifted_cloud() -> None:
    rng = np.random.default_rng(2)
    X_ref = rng.standard_normal((300, 5))
    near = novelty(X_ref, rng.standard_normal((100, 5)))
    far = novelty(X_ref, rng.standard_normal((100, 5)) + 4.0)
    assert far.mean() > 3 * near.mean()


def test_directional_excess_is_one_sided_where_tv_is_not() -> None:
    """The Gate 5 contribution in one assertion.

    A shift that moves conformity scores *down* makes the calibration threshold
    conservative, so coverage rises and the bound must not penalise it. Total
    variation reports the shift regardless of direction; d+ must be zero.
    """
    rng = np.random.default_rng(3)
    cal = rng.standard_normal(500)
    benign = cal - 3.0        # scores move down: harmless
    harmful = cal + 3.0       # scores move up: costs coverage
    assert directional_excess(cal, benign) == pytest.approx(0.0, abs=1e-9)
    assert directional_excess(cal, harmful) > 0.8
    assert tv_distance(cal, benign) > 0.8    # symmetric: fooled by the benign one


def test_spearman_signs_and_bounds() -> None:
    a = np.arange(20.0)
    assert spearman(a, a) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def test_spearman_reports_no_correlation_against_a_constant() -> None:
    """The regression this fix exists for.

    The previous implementation ranked with ``argsort(argsort(x))``, which is
    only a rank when values are distinct. Against a constant variable it handed
    out ranks 0..n-1 and returned **+1.0** -- a perfect correlation where none
    is defined. Every correlation the project reports against ``rms``, which
    takes five distinct values, was computed on tied data.
    """
    a = np.arange(20.0)
    assert spearman(a, np.zeros(20)) == 0.0
    assert spearman(np.zeros(20), np.zeros(20)) == 0.0


def test_spearman_averages_tied_ranks() -> None:
    """Matches scipy.stats.spearmanr on tied input; the old version did not."""
    a = np.arange(20.0)
    b = np.repeat(np.arange(4.0), 5)          # heavy ties
    assert spearman(a, b) == pytest.approx(0.969458, abs=1e-6)
    # Ties in one variable only must still give a symmetric answer.
    assert spearman(b, a) == pytest.approx(spearman(a, b))

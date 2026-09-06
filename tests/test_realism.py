"""Tests for the forward-model realism check.

Two things are worth pinning here, both of which cost real time to get right:

* the noise estimator, which must recover a *known* Poisson-Gaussian gain from
  synthetic data before any number it reports about real data means anything;
* the frame-ordering decision, which a first implementation got wrong on half
  the dataset by trying to read illumination direction out of single frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.realism_check import (  # noqa: E402
    CANONICAL, STRIDE, WIDEFIELD_PIXEL_UM, _oob, block_reduce, decide_grouping,
    fit_residual, out_of_band, real_widefield, spectral_flatness, radial_power,
    variance_vs_mean,
)


def test_block_reduce_mean_and_sum_agree_by_the_area_factor() -> None:
    a = np.arange(36, dtype=np.float64).reshape(6, 6)
    assert np.allclose(block_reduce(a, 2, "sum"), 4.0 * block_reduce(a, 2, "mean"))
    assert block_reduce(a, 2).shape == (3, 3)


def test_block_reduce_sum_conserves_total() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((8, 8))
    assert block_reduce(a, 2, "sum").sum() == pytest.approx(a.sum())


def test_block_reduce_crops_odd_sizes_rather_than_padding() -> None:
    a = np.ones((7, 7))
    assert block_reduce(a, 2, "sum").shape == (3, 3)


def test_out_of_band_keeps_a_sane_fraction_and_kills_low_frequencies() -> None:
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    smooth = np.sin(2 * np.pi * xx / n) + np.cos(2 * np.pi * yy / n)
    filt, frac = out_of_band(smooth, WIDEFIELD_PIXEL_UM)
    assert 0.2 < frac < 0.8
    # A field made only of the lowest frequencies must be annihilated.
    assert filt.std() < 1e-9 * max(smooth.std(), 1e-30) + 1e-9


def test_out_of_band_variance_is_the_kept_fraction_of_white_noise() -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal((256, 256))
    filt, frac = out_of_band(z, WIDEFIELD_PIXEL_UM)
    assert filt.var() / frac == pytest.approx(1.0, rel=0.05)


def test_variance_vs_mean_recovers_a_known_poisson_gaussian_gain() -> None:
    """The estimator must return gain 1.0 where the truth is Poisson.

    This is the test that gives the real-data number its meaning: without it,
    a gain of 1.3 on BioSR could just as well be an artefact of the estimator.
    """
    rng = np.random.default_rng(2)
    n, read = 256, 1.5
    yy, xx = np.mgrid[0:n, 0:n]
    rate = 200.0 + 150.0 * np.exp(-((yy - 128) ** 2 + (xx - 100) ** 2) / (2 * 40.0**2))
    a = rng.poisson(rate) + read * rng.standard_normal((n, n))
    b = rng.poisson(rate) + read * rng.standard_normal((n, n))
    r = variance_vs_mean(0.5 * (a + b), *_oob(a - b, 2.0), tile=16,
                         read_var=read**2)
    assert r["gain_fixed"] == pytest.approx(1.0, abs=0.03)
    assert r["n_tiles"] > 100


def test_variance_vs_mean_sees_doubled_noise_as_doubled_gain() -> None:
    rng = np.random.default_rng(3)
    n = 256
    rate = np.full((n, n), 150.0)
    # 2 * Poisson(r) has mean 2r and variance 4r, i.e. twice its mean, so a
    # correct estimator must report gain 2 -- the signature of excess noise.
    a = 2.0 * rng.poisson(rate)
    b = 2.0 * rng.poisson(rate)
    r = variance_vs_mean(0.5 * (a + b), *_oob(a - b, 2.0), tile=16)
    assert r["gain_fixed"] == pytest.approx(2.0, rel=0.1)


def test_spectral_flatness_calls_white_noise_flat_and_smooth_fields_not() -> None:
    rng = np.random.default_rng(4)
    cutoff = 2 * 1.7 / 0.525
    f, p = radial_power(rng.standard_normal((256, 256)), WIDEFIELD_PIXEL_UM)
    assert spectral_flatness(f, p, cutoff)["hf_lf"] == pytest.approx(1.0, abs=0.2)
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    f, p = radial_power(np.sin(2 * np.pi * xx / n) + yy / n, WIDEFIELD_PIXEL_UM)
    assert spectral_flatness(f, p, cutoff)["hf_lf"] < 0.1


def test_fit_residual_removes_an_arbitrary_scale_and_offset_exactly() -> None:
    rng = np.random.default_rng(5)
    clean = rng.random((32, 32))
    assert np.abs(fit_residual(7.5 * clean + 3.0, clean)).max() < 1e-9


def synthetic_sim_frames(order: str, seed: int = 0, n: int = 256,
                         modulation: float = 0.6) -> np.ndarray:
    """Nine synthetic raw SIM frames, in a chosen acquisition order.

    Three illumination orientations, three phases at 0, 2pi/3, 4pi/3, on a
    structured specimen, with Poisson noise and an sCMOS-style fixed-pattern
    ridge on one axis -- the artefact that defeated the first implementation.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    obj = 300.0 + 200.0 * np.exp(-((yy - n / 2) ** 2 + (xx - n / 3) ** 2)
                                 / (2 * (n / 8) ** 2))
    obj += 120.0 * (np.sin(6 * np.pi * yy / n) > 0.4)      # some structure
    k = 0.26                                               # cycles per pixel
    dirs = [(k, 0.02), (-0.12, 0.23), (0.13, 0.22)]         # three orientations
    phases = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    fixed = 12.0 * ((xx.astype(int) % 2) == 0)              # column-fixed pattern

    frames = [None] * 9
    for o, (ky, kx) in enumerate(dirs):
        for p, phi in enumerate(phases):
            i = o * 3 + p if order == "canonical" else p * 3 + o
            patt = 1.0 + modulation * np.cos(2 * np.pi * (ky * yy + kx * xx) + phi)
            frames[i] = rng.poisson(obj * patt) + fixed
    return np.stack(frames).astype(np.float64)


@pytest.mark.parametrize("order,expected", [("canonical", CANONICAL),
                                            ("stride", STRIDE)])
def test_decide_grouping_recovers_the_acquisition_order(order, expected) -> None:
    """Both orders must be identified from the frames themselves.

    BioSR does not use one convention throughout -- CCPs reads as
    orientation-slow while Microtubules, F-actin and ER read as phase-slow -- so
    assuming either one silently mis-averages part of the dataset. An earlier
    version inferred the order from each frame's illumination *direction* and
    got it wrong for every anisotropic specimen; this pins the replacement,
    which tests which grouping actually cancels the pattern.
    """
    frames = synthetic_sim_frames(order, seed=7)
    groups, audit = decide_grouping(frames, WIDEFIELD_PIXEL_UM)
    assert groups == expected
    assert audit["chosen"] == order
    assert audit["ratio"] > 1.15


def test_phase_averaging_cancels_the_pattern_only_for_the_right_grouping() -> None:
    """The physical premise the whole real-widefield construction rests on."""
    from scripts.realism_check import score_grouping
    frames = synthetic_sim_frames("canonical", seed=8)
    right = score_grouping(frames, CANONICAL, WIDEFIELD_PIXEL_UM)
    wrong = score_grouping(frames, STRIDE, WIDEFIELD_PIXEL_UM)
    assert right < wrong


def test_real_widefield_sums_the_dose_and_scales_the_divisor_to_match() -> None:
    """The dose bookkeeping: summing nine frames, not averaging them.

    Averaging gives an image whose variance is a ninth of its mean, which reads
    as a gain of 1/9 and looks like the noise model has collapsed. This pins the
    sum, and pins that the divisor recovers a gain of 1 on Poisson input.
    """
    frames = synthetic_sim_frames("canonical", seed=9, modulation=0.0)
    out = real_widefield(frames, WIDEFIELD_PIXEL_UM, CANONICAL)
    ped, m_c = out["pedestal"], float(frames.mean())
    # Nine frames summed, pedestal removed, converted to photoelectrons.
    assert out["y"].mean() == pytest.approx(9.0 * (m_c - ped) * 0.6026, rel=0.02)
    # The frames are Poisson in *counts*, so per tile the photoelectron variance
    # is 0.6026^2 * 9 * m while the mean is 0.6026 * 9 * (m - pedestal). The
    # estimator is mean-squared-weighted, so the gain it must reproduce is that
    # ratio weighted the same way -- a global average would misweight it.
    # Getting the sum or the divisor wrong moves this by a factor of 9 or more.
    m = block_reduce(frames.mean(axis=0), 16, "mean").ravel()
    expected = 0.6026 * float((m * (m - ped)).sum() / ((m - ped) ** 2).sum())
    r = variance_vs_mean(out["y"], *_oob(out["diff"], out["divisor"]), tile=16)
    assert r["gain_fixed"] == pytest.approx(expected, rel=0.10)


def test_background_for_holds_the_haze_fraction_across_the_photon_range() -> None:
    """A fixed background rate is correct at exactly one photon level.

    BioSR spans 15 to 600 photons. A flat background of 45 gives a haze fraction
    of 75% at the dim end and 7% at the bright end, against an observed 19-68%.
    Holding the rate fixed while sweeping the budget was a real mistake here --
    it collapsed a mismatch correlation to -0.151 and looked like a result.
    """
    from src.optics.forward_model import BIOSR_PHOTON_RANGE, background_for

    for photons in (BIOSR_PHOTON_RANGE[0], 100.0, BIOSR_PHOTON_RANGE[1]):
        b = background_for(photons, 0.31)
        assert b / (photons + b) == pytest.approx(0.31, abs=1e-9)
    # The observed median at 100 photons is what the earlier sweep used.
    assert background_for(100.0, 0.31) == pytest.approx(44.93, abs=0.01)
    assert background_for(15.0, 0.31) == pytest.approx(6.74, abs=0.01)


def test_background_for_is_zero_at_zero_fraction_and_rejects_nonsense() -> None:
    from src.optics.forward_model import background_for

    assert background_for(100.0, 0.0) == 0.0
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            background_for(100.0, bad)


def test_a_fixed_background_does_not_hold_the_fraction() -> None:
    """The regression this parameterisation exists to prevent."""
    from src.optics.forward_model import BIOSR_PHOTON_RANGE

    lo, hi = BIOSR_PHOTON_RANGE
    assert 45.0 / (lo + 45.0) > 0.68     # 75% at 15 photons, above observed max
    assert 45.0 / (hi + 45.0) < 0.19     # 7% at 600 photons, below observed min

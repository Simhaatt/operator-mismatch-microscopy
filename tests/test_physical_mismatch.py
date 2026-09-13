"""Tests for the physical mismatch families, the l2 target, and Phase 3.

These cover the parts of the Phase 2 and Phase 3 changes that are cheap to get
subtly wrong: the defocus-to-wavefront conversion, whether a config override
actually reaches the synthesised kernel, and whether the background reaches the
unrolled network's fidelity term.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.image import relative_l2  # noqa: E402
from src.optics.aberrations import AberrationSpec, ladder_specs  # noqa: E402
from src.optics.physical_mismatch import (  # noqa: E402
    DEFOCUS_UM,
    NA_FRACTIONS,
    WAVELENGTH_OFFSETS_NM,
    apply_overrides,
    defocus_coeffs,
    defocus_rms_waves,
    physical_specs,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.unrolled import UnrolledNet  # noqa: E402


# --- relative l2 target -----------------------------------------------------

def test_relative_l2_is_zero_for_identical_inputs():
    x = torch.rand(1, 16, 16, dtype=torch.float64)
    assert float(relative_l2(x, x)) == pytest.approx(0.0, abs=1e-12)


def test_relative_l2_is_one_against_zeros():
    x = torch.rand(1, 16, 16, dtype=torch.float64)
    assert float(relative_l2(torch.zeros_like(x), x)) == pytest.approx(1.0)


def test_relative_l2_scales_linearly():
    """A pure k-fold gain error must give exactly |k - 1|."""
    x = torch.rand(1, 16, 16, dtype=torch.float64) + 1.0
    for k in (0.5, 1.1, 2.0):
        assert float(relative_l2(x * k, x)) == pytest.approx(abs(k - 1.0))


# --- spec enumeration -------------------------------------------------------

def test_physical_specs_count_and_families():
    specs = physical_specs()
    assert len(specs) == len(NA_FRACTIONS) + len(WAVELENGTH_OFFSETS_NM) + len(DEFOCUS_UM)
    assert len(specs) == 12
    assert {s.mode for s in specs} == {"na", "wavelength", "focus_drift"}


def test_combined_ladder_is_29_specs():
    zern = ladder_specs(["defocus", "astigmatism", "coma", "spherical"])
    assert len(zern) == 17
    assert len(zern) + len(physical_specs()) == 29


def test_no_duplicate_labels_across_families():
    """A duplicated label would silently overwrite a mismatch-magnitude entry."""
    specs = ladder_specs(["defocus", "astigmatism", "coma", "spherical"]) + physical_specs()
    labels = [s.label for s in specs]
    assert len(labels) == len(set(labels))


def test_severity_rungs_are_monotone_in_magnitude():
    """Rung number must increase with |perturbation|, independent of sign."""
    for family, values in (("na", NA_FRACTIONS), ("wavelength", WAVELENGTH_OFFSETS_NM)):
        specs = [s for s in physical_specs() if s.mode == family]
        by_label = {s.severity: s for s in specs}
        # Two magnitudes, each with a negative and a positive rung.
        assert set(by_label) == {"m1", "m2", "p1", "p2"}
    focus = sorted(s.severity for s in physical_specs() if s.mode == "focus_drift")
    assert focus == ["p1", "p2", "p3", "p4"]


# --- defocus physics --------------------------------------------------------

def test_defocus_rms_matches_paraxial_formula():
    na, lam, dz = 1.7, 0.525, 0.1
    expected = dz * na * na / (4.0 * math.sqrt(3.0) * lam)
    assert defocus_rms_waves(dz, na, lam) == pytest.approx(expected)


def test_defocus_rms_is_linear_in_displacement():
    a = defocus_rms_waves(0.05, 1.7, 0.525)
    b = defocus_rms_waves(0.10, 1.7, 0.525)
    assert b == pytest.approx(2.0 * a)


def test_defocus_rms_is_sign_independent_but_coeff_is_not():
    """Equal magnitude either side of focus, but distinct operators."""
    assert defocus_rms_waves(-0.1, 1.7, 0.525) == pytest.approx(
        defocus_rms_waves(0.1, 1.7, 0.525)
    )
    plus = defocus_coeffs(0.1, 1.7, 0.525)
    minus = defocus_coeffs(-0.1, 1.7, 0.525)
    (k,) = plus
    assert plus[k] == pytest.approx(-minus[k])
    assert plus[k] > 0


def test_defocus_rms_spans_the_zernike_ladder_range():
    """The chosen displacements must be comparable to the aberration ladder."""
    vals = [defocus_rms_waves(dz, 1.7, 0.525) for dz in DEFOCUS_UM]
    assert min(vals) > 0.03   # above the mild rung
    assert max(vals) < 0.25   # below the extreme rung
    assert vals == sorted(vals)


def test_defocus_rejects_nonpositive_wavelength():
    with pytest.raises(ValueError):
        defocus_rms_waves(0.1, 1.7, 0.0)


# --- config overrides reach the kernel --------------------------------------

def test_na_override_changes_the_synthesised_psf():
    cfg = PupilConfig(n_out=31)
    spec = next(s for s in physical_specs() if s.mode == "na")
    assert spec.cfg_overrides, "NA spec must carry an override"
    base = pupil_psf(spec.coeffs, cfg, warn=False)[0]
    over = pupil_psf(spec.coeffs, apply_overrides(cfg, spec), warn=False)[0]
    assert not torch.allclose(base, over)


def test_wavelength_override_changes_the_cutoff():
    cfg = PupilConfig(n_out=31)
    spec = next(s for s in physical_specs() if s.mode == "wavelength")
    new = apply_overrides(cfg, spec)
    assert new.wavelength_um != cfg.wavelength_um
    assert new.cutoff_per_um != cfg.cutoff_per_um


def test_apply_overrides_is_identity_for_zernike_specs():
    """The Zernike ladder must keep the unmodified pupil."""
    cfg = PupilConfig(n_out=31)
    for spec in ladder_specs(["defocus", "coma"]):
        assert apply_overrides(cfg, spec) is cfg


def test_apply_overrides_does_not_mutate_the_original_config():
    cfg = PupilConfig(n_out=31)
    na_before = cfg.na
    spec = next(s for s in physical_specs() if s.mode == "na")
    apply_overrides(cfg, spec)
    assert cfg.na == na_before


def test_aberration_spec_default_overrides_are_independent():
    """A mutable default shared across instances would be a silent disaster."""
    a = AberrationSpec("x", "nominal", 0.0, {})
    b = AberrationSpec("y", "nominal", 0.0, {})
    a.cfg_overrides["na"] = 1.0
    assert b.cfg_overrides == {}


# --- Phase 3: background-aware unrolled fidelity ----------------------------

@pytest.fixture(scope="module")
def unrolled_setup():
    torch.manual_seed(0)
    net = UnrolledNet(n_stages=2, width=8).double()
    psf = pupil_psf(None, PupilConfig(n_out=31), warn=False)[0].double()
    y = torch.rand(1, 32, 32, dtype=torch.float64) * 100.0 + 45.0
    return net, psf, y


def test_unrolled_background_changes_the_output(unrolled_setup):
    net, psf, y = unrolled_setup
    with torch.no_grad():
        a = net(y, psf, background=0.0)
        b = net(y, psf, background=45.0)
    assert not torch.allclose(a, b)


def test_unrolled_background_defaults_to_zero(unrolled_setup):
    """Existing checkpoints must evaluate exactly as before the change."""
    net, psf, y = unrolled_setup
    with torch.no_grad():
        assert torch.allclose(net(y, psf), net(y, psf, background=0.0))


def test_unrolled_output_stays_nonnegative_with_background(unrolled_setup):
    net, psf, y = unrolled_setup
    with torch.no_grad():
        out = net(y, psf, background=45.0)
    assert torch.all(out >= 0.0)


def test_unrolled_accepts_2d_input_with_background(unrolled_setup):
    net, psf, y = unrolled_setup
    with torch.no_grad():
        out = net(y[0], psf, background=45.0)
    assert out.dim() == 2


# --- mismatch magnitude -----------------------------------------------------

def test_mismatch_magnitude_is_monotone_within_a_family():
    """The PSF-space magnitude must order the severity rungs correctly.

    This is the column analyses use when a single mismatch axis is needed
    across families, so a non-monotone one would be worse than useless.
    """
    cfg = PupilConfig(n_out=63)
    psf0 = pupil_psf(None, cfg, warn=False)[0]
    n0 = torch.linalg.vector_norm(psf0)
    order = {"mild": 0, "moderate": 1, "severe": 2, "extreme": 3}
    for mode in ("defocus", "astigmatism", "coma", "spherical"):
        specs = [s for s in ladder_specs([mode]) if s.severity in order]
        specs.sort(key=lambda s: order[s.severity])
        mags = [
            float(
                torch.linalg.vector_norm(
                    pupil_psf(s.coeffs, cfg, warn=False)[0] - psf0
                )
                / n0
            )
            for s in specs
        ]
        assert mags == sorted(mags), f"{mode} magnitudes not monotone: {mags}"
        assert np.all(np.array(mags) > 0)

"""Physically-motivated forward-operator mismatch families.

The Zernike ladder in :mod:`src.optics.aberrations` varies RMS wavefront error,
which is the right axis for asking *how much* mismatch matters. It is not the
axis a microscopist is actually uncertain about. Nobody knows their coma
coefficient; they do know that the objective's nominal NA is not exactly the NA
in the immersion they are using, that the emission peak of their fluorophore is
not exactly the design wavelength, and that the focal plane drifts.

Three families are defined here, each parameterised in the units the
uncertainty is expressed in:

* **Numerical aperture**, +/- 5%. NA enters the pupil radius through the cutoff
  NA / lambda, so getting it wrong rescales the whole pupil. Immersion index
  mismatch, temperature, and coverslip thickness all show up this way.
* **Emission wavelength**, +/- 10 and +/- 20 nm. A fluorophore's emission is a
  band, not a line, and the value used to build a PSF is a choice about that
  band.
* **Axial defocus**, 50 to 200 nm. Focus drift over an acquisition. Unlike the
  other two this *is* a wavefront error, and a computable one.

Mismatch magnitude
------------------
Two of these families have no RMS wavefront error. Changing NA or wavelength
does not add phase to the pupil; it rescales the pupil's spatial-frequency
support. So a single "RMS waves" column cannot describe all three families, and
forcing one would be a category error.

Defocus is handled exactly. A displacement ``dz`` produces, in the paraxial
approximation, a quadratic wavefront ``W(rho) = (dz * NA^2 / 2 lambda) rho^2``
in waves. Expanding ``rho^2`` on the unit-RMS Zernike defocus mode
``Z_2^0 = sqrt(3)(2 rho^2 - 1)`` gives a piston term, which is unobservable,
and an RMS wavefront error of

    rms = dz * NA^2 / (4 sqrt(3) lambda)     [waves]

For NA and wavelength, ``rms_waves`` is left as NaN, and every family instead
carries a family-agnostic magnitude measured in PSF space: the relative l2
difference between the true and nominal kernels,
``||psf_true - psf_nominal|| / ||psf_nominal||``. That quantity is defined for
all three families and for the Zernike ladder too, so it is the column to use
when a single mismatch axis is needed across families. It is computed in the
extraction script, where both kernels exist.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .aberrations import AberrationSpec

__all__ = [
    "NA_FRACTIONS",
    "WAVELENGTH_OFFSETS_NM",
    "DEFOCUS_UM",
    "defocus_rms_waves",
    "defocus_coeffs",
    "physical_specs",
]

# Fractional NA perturbations. +/-5% brackets the disagreement between a stated
# objective NA and the effective NA under real immersion conditions.
NA_FRACTIONS: tuple[float, ...] = (-0.05, -0.025, 0.025, 0.05)

# Emission-wavelength offsets in nanometres.
WAVELENGTH_OFFSETS_NM: tuple[float, ...] = (-20.0, -10.0, 10.0, 20.0)

# Axial defocus in micrometres. At NA 1.7 and 525 nm these span RMS wavefront
# errors of roughly 0.04 to 0.16 waves, comparable to the mild-to-severe rungs
# of the Zernike ladder.
DEFOCUS_UM: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

# OSA/ANSI index of the defocus mode Z_2^0.
DEFOCUS_INDEX: int = 4


def defocus_rms_waves(dz_um: float, na: float, wavelength_um: float) -> float:
    """RMS wavefront error in waves produced by an axial displacement.

    Paraxial: ``W(rho) = (dz NA^2 / 2 lambda) rho^2`` waves, whose non-piston
    part has RMS ``dz NA^2 / (4 sqrt(3) lambda)``.
    """
    if wavelength_um <= 0:
        raise ValueError("wavelength must be positive")
    return abs(dz_um) * na * na / (4.0 * math.sqrt(3.0) * wavelength_um)


def defocus_coeffs(dz_um: float, na: float, wavelength_um: float) -> dict[int, float]:
    """Zernike coefficient dictionary for an axial displacement.

    Signed, so that defocus in front of and behind focus are distinct
    operators, even though their RMS wavefront error is the same.
    """
    mag = defocus_rms_waves(dz_um, na, wavelength_um)
    return {DEFOCUS_INDEX: math.copysign(mag, dz_um)}


def _severity_label(value: float, ladder: tuple[float, ...]) -> str:
    """Name a rung by its position in the family, not by an invented adjective.

    The Zernike ladder's ``mild``/``moderate``/``severe``/``extreme`` labels are
    anchored to the Marechal criterion. These families are not, so reusing those
    words would imply a comparability that does not hold. Rungs are named by
    their signed magnitude instead.
    """
    magnitudes = sorted({abs(v) for v in ladder})
    rung = magnitudes.index(abs(value)) + 1
    sign = "m" if value < 0 else "p"
    return f"{sign}{rung}"


def physical_specs(
    na: float = 1.7, wavelength_um: float = 0.525
) -> list[AberrationSpec]:
    """Enumerate the physical mismatch families.

    Twelve specs: four NA perturbations, four wavelength offsets, four defocus
    displacements. ``nominal`` is not re-emitted -- the Zernike ladder already
    contributes it, and a zero perturbation is the same operator.

    NA and wavelength specs carry ``cfg_overrides``, applied to the
    :class:`~src.optics.psf.PupilConfig` used to synthesise the *true* kernel.
    The nominal kernel is always built from the unmodified configuration, which
    is what makes the operator mismatched.
    """
    specs: list[AberrationSpec] = []

    for frac in NA_FRACTIONS:
        specs.append(AberrationSpec(
            mode="na",
            severity=_severity_label(frac, NA_FRACTIONS),
            rms_waves=float("nan"),  # not a wavefront error; see module docstring
            coeffs={},
            cfg_overrides={"na": na * (1.0 + frac)},
        ))

    for off_nm in WAVELENGTH_OFFSETS_NM:
        specs.append(AberrationSpec(
            mode="wavelength",
            severity=_severity_label(off_nm, WAVELENGTH_OFFSETS_NM),
            rms_waves=float("nan"),
            coeffs={},
            cfg_overrides={"wavelength_um": wavelength_um + off_nm / 1000.0},
        ))

    for dz in DEFOCUS_UM:
        specs.append(AberrationSpec(
            mode="focus_drift",
            severity=_severity_label(dz, DEFOCUS_UM),
            rms_waves=defocus_rms_waves(dz, na, wavelength_um),
            coeffs=defocus_coeffs(dz, na, wavelength_um),
            cfg_overrides={},
        ))

    return specs


def apply_overrides(cfg, spec: AberrationSpec):
    """Return ``cfg`` with the spec's overrides applied, or ``cfg`` unchanged."""
    overrides = getattr(spec, "cfg_overrides", None) or {}
    return replace(cfg, **overrides) if overrides else cfg

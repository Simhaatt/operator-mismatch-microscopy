"""Aberration severity ladder and operator probe sets.

The severity axis is anchored to the **Marechal criterion**: an RMS wavefront
error of lambda/14 (~0.0714 waves) corresponds to a Strehl ratio of ~0.8, the
conventional boundary of diffraction-limited performance. Severity levels are
therefore expressed as multiples of that limit rather than as invented numbers,
so "moderate aberration" has a defensible optical meaning.

Reference for the criterion: Marechal's condition as standardly quoted, RMS
wavefront error <= lambda/14 <=> Strehl >= 0.8. Verified numerically against our
own PSF synthesis in ``tests/test_optics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .zernike import NAMED_MODES

__all__ = [
    "MARECHAL_LIMIT_WAVES",
    "SEVERITY_LADDER",
    "ABERRATION_MODES",
    "AberrationSpec",
    "severity_coeffs",
    "ladder_specs",
]

# RMS wavefront error at the diffraction limit, in waves.
MARECHAL_LIMIT_WAVES: float = 1.0 / 14.0

# Severity levels as multiples of the Marechal limit.
SEVERITY_LADDER: dict[str, float] = {
    "nominal": 0.0,
    "mild": 0.5,      # comfortably diffraction limited
    "moderate": 1.0,  # exactly at the limit, Strehl ~0.81
    "severe": 2.0,    # Strehl ~0.43
    "extreme": 3.5,   # Strehl ~0.09, well beyond usable optics
}

# Single Zernike modes swept one factor at a time. Tilt and piston are excluded:
# piston is unobservable and tilt only translates the image, which is a
# registration problem rather than a resolution-limiting aberration.
ABERRATION_MODES: dict[str, int] = {
    "defocus": NAMED_MODES["defocus"],
    "astigmatism": NAMED_MODES["astigmatism_vertical"],
    "coma": NAMED_MODES["coma_x"],
    "spherical": NAMED_MODES["spherical"],
    "trefoil": NAMED_MODES["trefoil_x"],
}


@dataclass(frozen=True)
class AberrationSpec:
    """One cell of the aberration axis."""

    mode: str
    severity: str
    rms_waves: float
    coeffs: dict[int, float]
    # Overrides applied to the PupilConfig used for the *true* kernel. Empty for
    # the Zernike ladder, where mismatch is phase added to a fixed pupil; used by
    # src.optics.physical_mismatch, where mismatch is a wrong NA or wavelength
    # and so changes the pupil itself. The nominal kernel never sees these.
    cfg_overrides: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.mode}:{self.severity}"


def severity_coeffs(mode: str, severity: str) -> dict[int, float]:
    """Zernike coefficients for one (mode, severity) cell, in waves.

    Because the basis is RMS-normalised, putting all the amplitude in a single
    mode makes the coefficient *equal* to the total RMS wavefront error -- so the
    severity level is exactly the requested multiple of the Marechal limit.
    """
    if mode not in ABERRATION_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(ABERRATION_MODES)}")
    if severity not in SEVERITY_LADDER:
        raise ValueError(
            f"unknown severity {severity!r}; choose from {sorted(SEVERITY_LADDER)}"
        )
    rms = SEVERITY_LADDER[severity] * MARECHAL_LIMIT_WAVES
    if rms == 0.0:
        return {}
    return {ABERRATION_MODES[mode]: rms}


def ladder_specs(
    modes: list[str] | None = None, severities: list[str] | None = None
) -> list[AberrationSpec]:
    """Enumerate the one-factor-at-a-time aberration axis.

    ``nominal`` is emitted once rather than once per mode, since a zero-amplitude
    aberration is the same operator regardless of which mode it nominally names.
    """
    modes = modes or list(ABERRATION_MODES)
    severities = severities or list(SEVERITY_LADDER)
    specs: list[AberrationSpec] = []
    if "nominal" in severities:
        specs.append(AberrationSpec("none", "nominal", 0.0, {}))
    for mode in modes:
        for sev in severities:
            if sev == "nominal":
                continue
            coeffs = severity_coeffs(mode, sev)
            specs.append(
                AberrationSpec(mode, sev, SEVERITY_LADDER[sev] * MARECHAL_LIMIT_WAVES, coeffs)
            )
    return specs

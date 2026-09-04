"""Zernike polynomials on the unit disk.

Convention (fixed for this project; state it in every paper and figure caption)
-------------------------------------------------------------------------------
* Ordering:      **OSA/ANSI** single index ``j = (n(n + 2) + m) // 2``.
* Normalisation: **RMS-normalised** (orthonormal), i.e. each ``Z_j`` has unit RMS
  over the unit disk, so a coefficient vector in waves has total RMS wavefront
  error ``sqrt(sum(a_j**2))`` waves once piston is excluded.
* Units:         coefficients are in **waves** (not radians, not nm), so the
  Marechal diffraction limit lambda/14 is literally ``0.0714``.

Six mutually incompatible indexing families are in circulation (Noll, OSA/ANSI,
Fringe/Wyant, ISO 14999, Born-Wolf, Malacara) -- see "Zernike polynomials and
their applications", J. Opt. 25 (2023) 123001, doi:10.1088/2040-8986/ac9e08.
Mixing them silently is a classic source of irreproducible aberration studies,
so :func:`noll_to_osa` / :func:`osa_to_noll` are provided rather than leaving
callers to guess.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

__all__ = [
    "osa_index",
    "osa_to_nm",
    "noll_order_ms",
    "noll_to_osa",
    "osa_to_noll",
    "zernike_basis",
    "wavefront",
    "rms_wavefront_error",
    "NAMED_MODES",
]


# Human-readable names for the low-order modes, keyed by OSA/ANSI index.
NAMED_MODES: dict[str, int] = {
    "piston": 0,
    "tilt_y": 1,
    "tilt_x": 2,
    "astigmatism_oblique": 3,
    "defocus": 4,
    "astigmatism_vertical": 5,
    "trefoil_y": 6,
    "coma_y": 7,
    "coma_x": 8,
    "trefoil_x": 9,
    "spherical": 12,
}


def osa_index(n: int, m: int) -> int:
    """OSA/ANSI single index for radial order ``n``, azimuthal frequency ``m``."""
    if n < 0 or abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"invalid Zernike (n, m) = ({n}, {m})")
    return (n * (n + 2) + m) // 2


def osa_to_nm(j: int) -> tuple[int, int]:
    """Inverse of :func:`osa_index`."""
    if j < 0:
        raise ValueError("OSA index must be non-negative")
    n = 0
    while osa_index(n, n) < j:
        n += 1
    return n, 2 * j - n * (n + 2)


def noll_order_ms(n: int) -> list[int]:
    """Signed ``m`` values for radial order ``n`` in Noll's within-order sequence.

    Noll ascends in ``|m|``; for each non-zero ``|m|`` the cosine (``+m``) and
    sine (``-m``) terms form a pair. The cosine/sine assignment is then fixed by
    the parity rule applied in :func:`noll_to_osa` (even Noll index -> cosine).
    """
    ms: list[int] = []
    for k in range(n % 2, n + 1, 2):
        if k == 0:
            ms.append(0)
        else:
            ms.extend([k, -k])
    return ms


def _noll_signed_m(j_noll: int, n: int, rem: int) -> int:
    """Resolve the signed ``m`` for Noll index ``j_noll`` at position ``rem``."""
    m = noll_order_ms(n)[rem]
    if m == 0:
        return 0
    # Noll: even index -> cosine term (m > 0), odd index -> sine term (m < 0).
    return abs(m) if j_noll % 2 == 0 else -abs(m)


def noll_to_osa(j_noll: int) -> int:
    """Convert a 1-based Noll index to the 0-based OSA/ANSI index."""
    if j_noll < 1:
        raise ValueError("Noll indices are 1-based")
    n = 0
    while (n + 1) * (n + 2) // 2 < j_noll:
        n += 1
    rem = j_noll - n * (n + 1) // 2 - 1
    return osa_index(n, _noll_signed_m(j_noll, n, rem))


def osa_to_noll(j_osa: int) -> int:
    """Convert a 0-based OSA/ANSI index to the 1-based Noll index."""
    n, m = osa_to_nm(j_osa)
    base = n * (n + 1) // 2 + 1
    for rem in range(len(noll_order_ms(n))):
        cand = base + rem
        if _noll_signed_m(cand, n, rem) == m:
            return cand
    raise AssertionError(f"no Noll index found for OSA {j_osa}")


@lru_cache(maxsize=512)
def _radial_terms(n: int, m_abs: int) -> tuple[tuple[int, float], ...]:
    """Radial polynomial ``R_n^{|m|}`` as a tuple of ``(power, coefficient)``."""
    terms: list[tuple[int, float]] = []
    for k in range((n - m_abs) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        terms.append((n - 2 * k, float(coeff)))
    return tuple(terms)


def _radial(n: int, m_abs: int, rho: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(rho)
    for power, coeff in _radial_terms(n, m_abs):
        out = out + coeff * rho.pow(power)
    return out


def zernike_basis(
    indices: list[int],
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Evaluate RMS-normalised Zernike modes on a polar grid.

    Parameters
    ----------
    indices:
        OSA/ANSI indices to evaluate.
    rho, theta:
        Normalised radius (0 at centre, 1 at the pupil edge) and azimuth, of
        identical shape. Samples with ``rho > 1`` are set to zero -- the
        polynomials are orthogonal only on the unit disk.

    Returns
    -------
    Tensor of shape ``(len(indices), *rho.shape)``.
    """
    if rho.shape != theta.shape:
        raise ValueError("rho and theta must have the same shape")
    inside = rho <= 1.0
    rho_c = rho.clamp(max=1.0)
    zero = torch.zeros_like(rho)
    modes = []
    for j in indices:
        n, m = osa_to_nm(j)
        m_abs = abs(m)
        norm = math.sqrt((n + 1) if m == 0 else 2.0 * (n + 1))
        radial = _radial(n, m_abs, rho_c)
        if m == 0:
            angular = torch.ones_like(theta)
        elif m > 0:
            angular = torch.cos(m_abs * theta)
        else:
            angular = torch.sin(m_abs * theta)
        modes.append(torch.where(inside, norm * radial * angular, zero))
    return torch.stack(modes, dim=0)


def wavefront(
    coeffs: dict[int, float] | torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Sum RMS-normalised Zernike modes into a wavefront map, in **waves**.

    ``coeffs`` may be a ``{osa_index: waves}`` mapping (sparse, the usual case)
    or a dense tensor indexed by OSA index.
    """
    if isinstance(coeffs, torch.Tensor):
        items = [(j, float(v)) for j, v in enumerate(coeffs.flatten().tolist()) if v != 0.0]
    else:
        items = [(int(j), float(v)) for j, v in coeffs.items() if v != 0.0]
    if not items:
        return torch.zeros_like(rho)
    amps = torch.tensor(
        [v for _, v in items], dtype=rho.dtype, device=rho.device
    )
    basis = zernike_basis([j for j, _ in items], rho, theta)
    return (amps.view(-1, *([1] * rho.dim())) * basis).sum(dim=0)


def rms_wavefront_error(
    coeffs: dict[int, float], exclude_piston: bool = True
) -> float:
    """RMS wavefront error in waves.

    The basis is orthonormal, so this is the l2 norm of the coefficients -- no
    grid evaluation required. Piston (``j=0``) carries no image information and
    is excluded by default. Tilt is *kept*: a tilted PSF genuinely shifts the
    image, which matters for a reconstruction-error study.
    """
    total = 0.0
    for j, v in coeffs.items():
        if exclude_piston and int(j) == 0:
            continue
        total += float(v) ** 2
    return math.sqrt(total)

"""Scalar pupil-function PSF synthesis with explicit sampling control.

The forward operator of this project is a 2D convolution with an incoherent
intensity PSF

    h = |F{ P(rho, theta) * exp(i * 2 * pi * W(rho, theta)) }|^2

where W is the wavefront error in waves (see :mod:`src.optics.zernike`) and P
is the circular pupil indicator. Three sampling traps make naive
implementations quietly wrong, and all three are handled here:

1. **Intensity aliasing.** The *amplitude* PSF is band-limited to the pupil
   cutoff f_c = NA / lambda, but squaring it doubles the bandwidth to 2 f_c.
   Computing |FFT(pupil)|^2 on a grid that critically samples the amplitude
   therefore aliases the intensity. We synthesise on a grid ``oversample``
   times finer and bin down afterwards.
2. **Coarse pupil sampling.** Accuracy of a high-order Zernike phase depends on
   how many samples span the pupil, which is *coupled* to the image sampling by
   R = N * dx_fine * NA / lambda. We therefore solve for the grid size that
   delivers a requested pupil radius rather than accepting whatever falls out.
3. **Energy loss on crop.** Aberrated PSFs have long tails. Cropping to a
   tractable kernel discards energy, so the discarded fraction is reported and
   normalisation is applied *after* the crop -- the returned kernel is exactly
   the operator that will be applied.

Convention: physical lengths in micrometres, wavefronts in waves.

A scalar (not vectorial) model is the fast path used for Gate 1 and Gate 2. For
the manuscript, the vectorial Richards-Wolf propagator of ``psf-generator``
(Liu et al., arXiv:2502.03170) should drive the final PSF library: its authors
report the scalar model is only 1.5-3x faster, so the accuracy is nearly free.
This function and that library take the same (coeffs, NA, wavelength, pixel
size) inputs, so the swap stays local to this module.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from .zernike import rms_wavefront_error, wavefront

__all__ = [
    "PupilConfig",
    "PsfDiagnostics",
    "pupil_psf",
    "airy_psf",
    "nyquist_pixel_um",
    "marechal_strehl",
    "psf_fwhm_um",
]


@dataclass(frozen=True)
class PupilConfig:
    """Optical and sampling parameters for PSF synthesis.

    Defaults are the **verified** BioSR acquisition parameters, read from the
    Methods of Qiao et al., Nature Methods 18, 194-202 (2021):

    * ``na = 1.7`` -- Olympus APON100XHOTIRF, stated in "Multimodality
      structured illumination microscopy".
    * ``wavelength_um = 0.525`` -- emission for the 488 nm line (mEmerald/GFP).
      The system also has 405 and 560 nm excitation; set this per channel.
    * ``pixel_um = 0.0313`` -- the **GT-SIM grid**, which is what this project
      uses as the object. The Hamamatsu Orca Flash 4.0 v3 gives a ~62.6-65 nm
      widefield pixel at 100x, and SIM reconstruction upsamples 2x (512 -> 1024),
      so the SIM grid is half that. Three independent estimates agree to ~10%:
      65/2 = 32.5 nm from camera and magnification, 62.6/2 = 31.3 nm from the
      commonly quoted widefield pixel, and 29.3 nm inferred from the measured
      spectral band edge of the objects themselves.

    Note the widefield pixel size is *not* the right value for these objects --
    using it would understate every spatial frequency by a factor of two.
    """

    wavelength_um: float = 0.525
    na: float = 1.7
    pixel_um: float = 0.0313
    n_out: int = 127
    oversample: int = 3  # MUST be odd -- see _center_bin
    pupil_samples: int = 64  # requested pupil *radius*, in fine-grid samples

    def __post_init__(self) -> None:
        if self.n_out % 2 == 0:
            raise ValueError("n_out must be odd so the kernel has a true centre")
        if self.oversample < 1 or self.oversample % 2 == 0:
            raise ValueError(
                "oversample must be odd: an even-width binning box cannot be "
                "centred on the peak sample, which shifts the PSF by half an "
                "output pixel and destroys its symmetry."
            )
        if min(self.wavelength_um, self.na, self.pixel_um) <= 0:
            raise ValueError("wavelength, NA and pixel size must be positive")

    @property
    def cutoff_per_um(self) -> float:
        """Amplitude (pupil) cutoff frequency NA / lambda, in 1/um."""
        return self.na / self.wavelength_um

    @property
    def fine_pixel_um(self) -> float:
        return self.pixel_um / self.oversample

    @property
    def grid(self) -> int:
        """Fine-grid size delivering ``pupil_samples`` of pupil radius.

        From R = N * dx_fine * NA / lambda, rounded up to a multiple of
        ``2 * oversample`` so binning is exact and the centre stays a sample.
        """
        n = self.pupil_samples / (self.fine_pixel_um * self.cutoff_per_um)
        n = int(math.ceil(n / (2 * self.oversample))) * 2 * self.oversample
        # Must hold the centred binning window: n_out * oversample samples
        # plus a full output pixel of margin on each side.
        return max(n, (self.n_out + 2) * self.oversample)

    @property
    def pupil_radius_px(self) -> float:
        """Actual pupil radius on the fine grid, in samples."""
        return self.grid * self.fine_pixel_um * self.cutoff_per_um


@dataclass(frozen=True)
class PsfDiagnostics:
    """Quantities needed to defend a synthesised PSF as physically correct."""

    rms_waves: float
    strehl: float
    strehl_marechal: float
    pupil_radius_px: float
    grid: int
    energy_kept: float
    nyquist_ok: bool
    pixel_um: float
    nyquist_pixel_um: float

    def summary(self) -> str:
        flag = "ok" if self.nyquist_ok else "VIOLATED"
        return (
            f"rms={self.rms_waves:.4f}w strehl={self.strehl:.4f} "
            f"(marechal {self.strehl_marechal:.4f}) pupilR={self.pupil_radius_px:.1f}px "
            f"grid={self.grid} energy_kept={self.energy_kept:.5f} nyquist={flag}"
        )


def nyquist_pixel_um(wavelength_um: float, na: float) -> float:
    """Largest pixel size that samples the *intensity* PSF without aliasing.

    The incoherent OTF is the pupil autocorrelation, so its support radius is
    2 NA / lambda and the Nyquist pixel is lambda / (4 NA).
    """
    return wavelength_um / (4.0 * na)


def marechal_strehl(rms_waves: float) -> float:
    """Extended Marechal approximation S ~ exp(-(2 pi sigma)^2).

    At the Marechal diffraction limit sigma = 1/14 waves this gives S = 0.818,
    the familiar "Strehl >= 0.8 is diffraction limited" rule that anchors the
    mild / moderate / severe aberration axis of this project.
    """
    return math.exp(-((2.0 * math.pi * rms_waves) ** 2))


def _polar_grid(
    n: int, radius_px: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised pupil radius and azimuth on an n x n centred grid."""
    ax = torch.arange(n, device=device, dtype=dtype) - n // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    rho = torch.sqrt(xx**2 + yy**2) / radius_px
    theta = torch.atan2(yy, xx)
    return rho, theta


def _center_bin(img: torch.Tensor, oversample: int, n_out: int) -> torch.Tensor:
    """Centre-crop then sum-bin, keeping the peak sample at the centre of a bin.

    A detector pixel integrates over its area, so binning (not point sampling)
    is the physically correct reduction from the fine synthesis grid. But the
    box must be *centred* on the optical axis: output pixel ``k`` averages fine
    samples ``C + oversample * (k - m) + [-oversample // 2 ... + oversample // 2]``
    where ``C`` is the fine-grid centre and ``m = n_out // 2``.

    This is only possible for **odd** ``oversample``; with an even width the
    centre pixel would straddle the axis, displacing the PSF by half an output
    pixel and breaking the symmetry that an on-axis PSF must have.
    """
    if oversample % 2 == 0:
        raise ValueError("oversample must be odd for centred binning")
    n = img.shape[-1]
    centre = n // 2
    half = oversample // 2
    lo = centre - (n_out // 2) * oversample - half
    hi = lo + n_out * oversample
    if lo < 0 or hi > n:
        raise ValueError(
            f"fine grid {n} too small for a centred {n_out}x{oversample} window"
        )
    win = img[lo:hi, lo:hi]
    if oversample > 1:
        win = win.reshape(n_out, oversample, n_out, oversample).sum(dim=(1, 3))
    return win


def _intensity(pupil: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    amp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil)))
    return (amp.real**2 + amp.imag**2).to(dtype)


def pupil_psf(
    coeffs: dict[int, float] | None = None,
    cfg: PupilConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    warn: bool = True,
) -> tuple[torch.Tensor, PsfDiagnostics]:
    """Synthesise a normalised 2D intensity PSF from Zernike aberrations.

    Parameters
    ----------
    coeffs:
        ``{OSA/ANSI index: waves}``. ``None`` or empty gives the unaberrated
        (Airy) PSF.
    cfg:
        Optical and sampling configuration.

    Returns
    -------
    ``(psf, diagnostics)`` where ``psf`` is ``(n_out, n_out)``, non-negative and
    sums to 1, and ``diagnostics`` carries the values the physics tests assert.
    """
    cfg = cfg or PupilConfig()
    coeffs = {int(k): float(v) for k, v in (coeffs or {}).items()}
    device = torch.device(device)

    n = cfg.grid
    radius = cfg.pupil_radius_px
    rho, theta = _polar_grid(n, radius, device, dtype)
    aperture = rho <= 1.0

    w = wavefront(coeffs, rho, theta) if coeffs else torch.zeros_like(rho)
    phase = (2.0 * math.pi * w).to(torch.float64)
    zeros = torch.zeros_like(phase)
    pupil = torch.where(aperture, torch.polar(torch.ones_like(phase), phase),
                        torch.complex(zeros, zeros))

    intensity = _intensity(pupil, dtype)
    total_fine = intensity.sum()

    # Strehl from the *fine, uncropped* field: binning smears the peak and
    # cropping changes the normalisation, both of which would bias the ratio.
    ideal = torch.where(aperture, torch.complex(torch.ones_like(phase), zeros),
                        torch.complex(zeros, zeros))
    i0 = _intensity(ideal, dtype)
    strehl = float((intensity / total_fine).max() / (i0 / i0.sum()).max())

    psf = _center_bin(intensity, cfg.oversample, cfg.n_out)
    energy_kept = float(psf.sum() / total_fine)
    psf = psf / psf.sum()

    rms = rms_wavefront_error(coeffs) if coeffs else 0.0
    nyq = nyquist_pixel_um(cfg.wavelength_um, cfg.na)
    diag = PsfDiagnostics(
        rms_waves=rms,
        strehl=strehl,
        strehl_marechal=marechal_strehl(rms),
        pupil_radius_px=radius,
        grid=n,
        energy_kept=energy_kept,
        nyquist_ok=cfg.pixel_um <= nyq,
        pixel_um=cfg.pixel_um,
        nyquist_pixel_um=nyq,
    )

    if warn:
        if not diag.nyquist_ok:
            warnings.warn(
                f"pixel_um={cfg.pixel_um:.4f} exceeds the intensity-PSF Nyquist "
                f"limit {nyq:.4f} um (lambda/4NA); the PSF is aliased.",
                stacklevel=2,
            )
        if energy_kept < 0.99:
            warnings.warn(
                f"cropping to n_out={cfg.n_out} keeps only {energy_kept:.3%} of the "
                "PSF energy; raise n_out or the tails are silently discarded.",
                stacklevel=2,
            )
        if radius < 32:
            warnings.warn(
                f"pupil radius is only {radius:.1f} samples; high-order Zernike "
                "phases will be poorly resolved. Raise pupil_samples.",
                stacklevel=2,
            )
    return psf, diag


def airy_psf(
    cfg: PupilConfig | None = None,
    *,
    integrate: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Analytic Airy intensity ``(2 J1(v) / v)^2``, normalised to sum 1.

    Computed from the Bessel function without an FFT, so it is a genuine
    independent check on :func:`pupil_psf` rather than a restatement of the same
    numerics.

    ``integrate=True`` (default) box-averages the analytic profile over each
    detector pixel using the same odd, centred window as :func:`_center_bin`, so
    it is directly comparable to :func:`pupil_psf` output. ``integrate=False``
    point-samples at pixel centres instead.
    """
    import numpy as np
    from scipy.special import j1

    cfg = cfg or PupilConfig()
    sub = cfg.oversample if integrate else 1
    n_fine = cfg.n_out * sub
    step = cfg.pixel_um / sub
    # Fine sample centres, symmetric about 0 for odd `sub`.
    ax = (torch.arange(n_fine, device=device, dtype=dtype) - (n_fine - 1) / 2.0) * step
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    v = (2.0 * math.pi * cfg.na * r / cfg.wavelength_um).cpu().numpy()
    safe = np.where(v == 0, 1.0, v)
    amp = np.where(v == 0, 1.0, 2.0 * j1(safe) / safe)
    out = torch.from_numpy(amp**2).to(device=device, dtype=dtype)
    if sub > 1:
        out = out.reshape(cfg.n_out, sub, cfg.n_out, sub).sum(dim=(1, 3))
    return out / out.sum()


def psf_fwhm_um(psf: torch.Tensor, pixel_um: float) -> float:
    """FWHM of a PSF along its central row, in um.

    Linear interpolation between the samples bracketing the half maximum. Used
    to assert the physically required FWHM ~ 0.51 lambda / NA scaling.
    """
    row = psf[psf.shape[0] // 2].to(torch.float64)
    peak = float(row.max())
    centre = int(torch.argmax(row))
    half = peak / 2.0

    def crossing(direction: int) -> float:
        i = centre
        while 0 < i < len(row) - 1:
            nxt = i + direction
            a, b = float(row[i]), float(row[nxt])
            if b <= half:
                frac = 0.0 if a == b else (a - half) / (a - b)
                return abs(i - centre) + frac
            i = nxt
        raise ValueError("PSF does not fall to half maximum inside the kernel")

    return (crossing(-1) + crossing(+1)) * pixel_um

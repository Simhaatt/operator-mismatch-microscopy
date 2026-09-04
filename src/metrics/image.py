"""Full-reference image metrics and spectral diagnostics.

Torch implementations so the same code runs on CPU locally and on a Kaggle T4
without a NumPy round trip. :func:`ssim` follows Wang et al. (2004) with the
standard 11x11 Gaussian window, sigma = 1.5.

The error target for this project is ``1 - SSIM`` computed against the surrogate
object, so the SSIM ``data_range`` must be fixed *globally* and never inferred
per image: a per-image range makes the error incomparable across the shift
matrix, which would silently corrupt every coverage number.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "gaussian_window",
    "ssim",
    "psnr",
    "radial_power_spectrum",
    "scale_match",
]


def gaussian_window(size: int = 11, sigma: float = 1.5, dtype=torch.float64) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Mean SSIM between ``a`` and ``b``.

    Accepts ``(H, W)`` or ``(B, H, W)``; returns a scalar or ``(B,)``.
    ``data_range`` is required, not inferred -- see the module docstring.
    """
    if a.shape != b.shape:
        raise ValueError("inputs must have identical shape")
    squeeze = a.dim() == 2
    x = a[None, None] if squeeze else a[:, None]
    y = b[None, None] if squeeze else b[:, None]

    win = gaussian_window(window_size, sigma, dtype=x.dtype).to(x.device)[None, None]
    pad = window_size // 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(t, (pad,) * 4, mode="reflect"), win
        )

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sig_x = filt(x * x) - mu_x2
    sig_y = filt(y * y) - mu_y2
    sig_xy = filt(x * y) - mu_xy

    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    m = (num / den).mean(dim=(1, 2, 3))
    return m[0] if squeeze else m


def psnr(a: torch.Tensor, b: torch.Tensor, *, data_range: float) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB."""
    dims = tuple(range(a.dim()))[-2:]
    mse = (a - b).pow(2).mean(dim=dims).clamp_min(1e-20)
    return 10.0 * torch.log10(torch.as_tensor(data_range**2, dtype=mse.dtype) / mse)


def radial_power_spectrum(
    x: torch.Tensor, n_bins: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Azimuthally averaged power spectrum.

    Returns ``(freq_cycles_per_pixel, power)``. Frequencies run from 0 to the
    pixel Nyquist of 0.5 cycles/pixel, so the result is independent of the
    (currently unconfirmed) physical pixel size.

    Used to answer a question the paper must answer: does the surrogate object
    carry genuine content *above* the simulated optical cutoff? If it does not,
    the null-space hallucination story is an artefact of the data pipeline
    rather than a property of microscopy.
    """
    if x.dim() == 2:
        x = x[None]
    n = min(x.shape[-2:])
    n_bins = n_bins or n // 2

    f = torch.fft.fftshift(torch.fft.fft2(x - x.mean(dim=(-2, -1), keepdim=True)))
    power = (f.real**2 + f.imag**2).mean(dim=0)

    h, w = power.shape
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, dtype=power.dtype))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, dtype=power.dtype))
    ry, rx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(ry**2 + rx**2)

    edges = torch.linspace(0, 0.5, n_bins + 1, dtype=power.dtype)
    idx = torch.bucketize(r.flatten(), edges) - 1
    valid = (idx >= 0) & (idx < n_bins)

    total = torch.zeros(n_bins, dtype=power.dtype).index_add_(
        0, idx[valid], power.flatten()[valid]
    )
    count = torch.zeros(n_bins, dtype=power.dtype).index_add_(
        0, idx[valid], torch.ones_like(power.flatten()[valid])
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, total / count.clamp_min(1.0)


def scale_match(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale ``estimate`` by the least-squares scalar that best fits ``reference``.

    Deconvolution recovers the object only up to a global intensity scale (the
    photon budget is an acquisition parameter, not a property of the specimen),
    so a full-reference metric must remove that one degree of freedom or it
    measures gain error rather than structural error.

    Scale only -- no offset. Allowing an additive term would also absorb
    background bias, which *is* a reconstruction error we want to be charged for.

    This uses the reference, so it is legitimate only for computing the
    *evaluation target* ``e``. It must never appear inside the reliability
    predictor, which sees no ground truth.
    """
    dims = tuple(range(estimate.dim()))[-2:]
    num = (estimate * reference).sum(dim=dims, keepdim=True)
    den = (estimate * estimate).sum(dim=dims, keepdim=True).clamp_min(1e-20)
    return estimate * (num / den)

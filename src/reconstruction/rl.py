"""Richardson-Lucy deconvolution on GPU or CPU.

Richardson-Lucy is the maximum-likelihood iteration for *Poisson* data, which
makes it the natural classical baseline for fluorescence microscopy where shot
noise dominates. The multiplicative update is

    x_{k+1} = x_k * (H^T [y / (H x_k + b)]) / (H^T 1)

with ``H`` the blur, ``H^T`` its adjoint (correlation with the flipped kernel --
which is *not* the same as ``H`` for an asymmetric aberration such as coma) and
``b`` the background rate.

Two departures from textbook RL, both deliberate:

* ``H^T 1`` is computed explicitly rather than assumed to be 1. It equals
  ``sum(psf) = 1`` only in the interior; near the boundary the padded operator
  is not normalised, and ignoring that biases the edges of every patch.
* ``read_variance`` implements the standard variance-stabilising correction for
  Poisson+Gaussian data (add sigma^2 to both the observation and the model, so
  the sum behaves like a Poisson variate of the shifted mean). Without it, RL
  on read-noise-limited low-photon data amplifies negative excursions.

RL has no regularisation, so iteration count *is* the regularisation parameter.
It is a required input, never a silent default.
"""

from __future__ import annotations

import torch

from ..optics.forward_model import fft_convolve, fft_convolve_adjoint

__all__ = ["richardson_lucy", "rl_transpose"]


def rl_transpose(
    z: torch.Tensor, psf: torch.Tensor, mode: str = "reflect"
) -> torch.Tensor:
    """Exact adjoint of the blur operator used in the forward model.

    Delegates to :func:`fft_convolve_adjoint`. Note this is *not* simply a
    convolution with the flipped kernel: that shortcut is only the true adjoint
    for zero or circular padding, and silently violates ``<Hx, z> = <x, H^T z>``
    for the reflect padding this project uses.
    """
    return fft_convolve_adjoint(z, psf, mode=mode)


def richardson_lucy(
    y: torch.Tensor,
    psf: torch.Tensor,
    n_iter: int,
    *,
    background: float = 0.0,
    read_variance: float = 0.0,
    mode: str = "reflect",
    eps: float = 1e-9,
    init: torch.Tensor | None = None,
    return_history: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
    """Run ``n_iter`` Richardson-Lucy iterations.

    Parameters
    ----------
    y:
        Measurement in photoelectrons, ``(H, W)`` or ``(B, H, W)``. Negative
        values (read-noise excursions) are clamped to zero, since the Poisson
        likelihood is undefined there.
    psf:
        Normalised, odd-sided kernel.
    n_iter:
        Number of iterations. This is the regularisation parameter: too few
        under-resolves, too many amplifies noise into false structure. Fix it by
        a documented rule (e.g. validation SSIM on the nominal operator) and
        keep it constant across shift conditions, or the reliability study
        confounds mismatch with iteration count.
    background:
        Known constant background rate ``b`` added to the model.
    read_variance:
        Camera read noise variance ``sigma_r^2``; enables the Poisson+Gaussian
        variance-stabilising correction.
    init:
        Starting estimate. Defaults to a flat image at the mean of ``y``, which
        is the standard uninformative choice and keeps the first update stable.
    return_history:
        Also return the per-iteration mean relative update, a cheap convergence
        diagnostic.

    Returns
    -------
    Non-negative estimate of the same shape as ``y``.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if read_variance < 0:
        raise ValueError("read_variance must be non-negative")

    obs = y.clamp_min(0.0) + read_variance
    shift = background + read_variance

    if init is None:
        mean = obs.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        x = torch.ones_like(obs) * mean
    else:
        x = init.clamp_min(eps).to(dtype=obs.dtype, device=obs.device)

    # H^T 1 -- the true normaliser, which departs from 1 at the boundary.
    norm = rl_transpose(torch.ones_like(obs), psf, mode=mode).clamp_min(eps)

    history: list[float] = []
    for _ in range(n_iter):
        model = fft_convolve(x, psf, mode=mode).clamp_min(0.0) + shift
        ratio = obs / model.clamp_min(eps)
        update = rl_transpose(ratio, psf, mode=mode) / norm
        x_new = (x * update).clamp_min(0.0)
        if return_history:
            denom = x.abs().mean().clamp_min(eps)
            history.append(float((x_new - x).abs().mean() / denom))
        x = x_new

    return (x, history) if return_history else x

"""Unrolled proximal-gradient network -- data consistency made explicit.

The opposite pole from :mod:`src.reconstruction.unet`. Each stage takes a
gradient step on the data-fidelity term using the *assumed* operator and then
applies a learned proximal operator:

    x_{k+1} = D_theta( x_k - eta_k * H_0^T (H_0 x_k - y) )

so the forward model is baked into the architecture. If the mechanism holds --
that an explicit fidelity term drives the residual onto the noise floor
regardless of whether ``H_0`` is right -- this network should behave like
Richardson-Lucy and Poisson-TV, not like the U-Net, despite being learned. That
contrast is what separates "learned versus classical" from "fidelity-minimising
versus not", which are otherwise confounded.

Step sizes are learned per stage rather than fixed, and the denoiser is shared
across stages by default so that depth costs parameters only through the step
sizes -- the usual unrolled-network convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..optics.forward_model import apply_psf, fft_convolve_adjoint

__all__ = ["UnrolledNet"]


class _Denoiser(nn.Module):
    """Small residual CNN standing in for a proximal operator."""

    def __init__(self, width: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.net(x)).clamp_min(0.0)


class UnrolledNet(nn.Module):
    """``n_stages`` of gradient descent on data fidelity plus a learned prox.

    The PSF is supplied at call time rather than stored, so the same trained
    network can be evaluated under a *different* operator than it was trained
    with -- which is precisely the shift this project studies.
    """

    def __init__(
        self, n_stages: int = 5, width: int = 32, n_layers: int = 4, shared: bool = True
    ) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.shared = shared
        if shared:
            self.denoiser = _Denoiser(width, n_layers)
        else:
            self.denoisers = nn.ModuleList(
                _Denoiser(width, n_layers) for _ in range(n_stages)
            )
        # Learned per-stage step sizes, positive by construction.
        self.log_step = nn.Parameter(torch.zeros(n_stages))

    def _prox(self, k: int) -> nn.Module:
        return self.denoiser if self.shared else self.denoisers[k]

    def forward(
        self, y: torch.Tensor, psf: torch.Tensor, mode: str = "reflect"
    ) -> torch.Tensor:
        squeeze = y.dim() == 2
        if squeeze:
            y = y[None]
        x = y.clone()
        for k in range(self.n_stages):
            resid = apply_psf(x, psf, mode=mode) - y
            grad = fft_convolve_adjoint(resid, psf, mode=mode)
            x = (x - torch.exp(self.log_step[k]) * grad).clamp_min(0.0)
            x = self._prox(k)(x[:, None])[:, 0]
        return x[0] if squeeze else x

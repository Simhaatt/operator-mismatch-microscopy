"""Poisson total-variation deconvolution.

The second classical baseline. Where Richardson-Lucy is unregularised maximum
likelihood, this minimises

    D_Poisson(H x, y) + lambda * TV(x),   x >= 0

with the Poisson data term

    D(u, y) = sum( u - y * log u )

and isotropic total variation. Solved by accelerated projected gradient (FISTA)
with the TV proximal operator computed by Chambolle's dual iteration.

Why it matters here: TV also has an explicit data-fidelity term, so it should
show the same behaviour as Richardson-Lucy -- driving its own residual toward the
noise floor whether or not the assumed operator is right. If it does, that
mechanism is a property of fidelity-minimising reconstruction rather than of RL
specifically, which is what the paper needs to claim. A learned reconstructor
with no fidelity term is the case that could break it.
"""

from __future__ import annotations

import torch

from ..optics.forward_model import apply_psf, fft_convolve_adjoint

__all__ = ["tv_prox", "poisson_tv"]


def _grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gy = torch.zeros_like(x)
    gx = torch.zeros_like(x)
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return gy, gx


def _div(py: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    """Negative adjoint of :func:`_grad`, so that <grad u, p> = -<u, div p>."""
    dy = torch.zeros_like(py)
    dx = torch.zeros_like(px)
    dy[..., 0, :] = py[..., 0, :]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., -1, :] = -py[..., -2, :]
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, -1] = -px[..., :, -2]
    return dy + dx


def tv_prox(v: torch.Tensor, weight: float, n_iter: int = 20) -> torch.Tensor:
    """Proximal operator of isotropic TV, by Chambolle's dual algorithm."""
    if weight <= 0:
        return v
    py = torch.zeros_like(v)
    px = torch.zeros_like(v)
    tau = 0.245  # < 1/8 for the 2D gradient's operator norm, with margin
    for _ in range(n_iter):
        gy, gx = _grad(_div(py, px) - v / weight)
        norm = torch.sqrt(gy**2 + gx**2).clamp_min(1e-12)
        py = (py + tau * gy) / (1 + tau * norm)
        px = (px + tau * gx) / (1 + tau * norm)
    return v - weight * _div(py, px)


def poisson_tv(
    y: torch.Tensor,
    psf: torch.Tensor,
    n_iter: int,
    *,
    weight: float = 0.01,
    background: float = 0.0,
    step: float | None = None,
    mode: str = "reflect",
    eps: float = 1e-6,
    prox_iter: int = 20,
) -> torch.Tensor:
    """FISTA on the Poisson likelihood with a TV prior, constrained to x >= 0.

    ``weight`` is the regularisation strength and, like Richardson-Lucy's
    iteration count, must be selected once on the nominal operator and then held
    fixed across shift conditions -- otherwise operator mismatch is confounded
    with regularisation strength.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    obs = y.clamp_min(0.0)
    scale = obs.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    x = torch.ones_like(obs) * scale
    z = x.clone()
    t = 1.0
    # Step size. The Poisson Hessian is H^T diag(y / u^2) H, and with a
    # normalised PSF (||H|| ~ 1) and u ~ y the curvature is about 1 / mean(y).
    # The Lipschitz constant is therefore ~1/mean(y) and the FISTA step is its
    # reciprocal, i.e. proportional to mean(y) -- NOT its inverse. Getting this
    # backwards leaves the estimate frozen near its constant initialisation.
    # Per image, not per batch: a scalar batch mean would make one image's
    # reconstruction depend on what else happened to be batched with it.
    lr = (
        torch.full_like(scale, float(step))
        if step is not None
        else 0.5 * obs.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    )

    for _ in range(n_iter):
        model = apply_psf(z, psf, mode=mode) + background
        grad = fft_convolve_adjoint(1.0 - obs / model.clamp_min(eps), psf, mode=mode)
        # tv_prox takes a scalar weight, so apply it per image.
        cand = (z - lr * grad).clamp_min(0.0)
        x_new = torch.stack([
            tv_prox(cand[i], float(lr.reshape(-1)[i]) * weight, prox_iter)
            for i in range(cand.shape[0])
        ]) if cand.dim() == 3 else tv_prox(cand, float(lr) * weight, prox_iter)
        x_new = x_new.clamp_min(0.0)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x

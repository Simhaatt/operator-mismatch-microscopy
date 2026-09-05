"""Reconstruction algorithm tests.

Both baselines minimise an explicit data-fidelity term, which is the property the
paper's mechanism rests on, so both must actually converge -- a reconstructor
frozen near its initialisation would still produce plausible-looking output and a
small residual, and would silently corrupt every reliability number downstream.
"""

from __future__ import annotations

import pytest
import torch

from src.metrics.image import scale_match, ssim
from src.optics.forward_model import (
    NoiseConfig, apply_psf, sample_seed, simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf
from src.reconstruction.rl import richardson_lucy, rl_transpose
from src.reconstruction.tv import _div, _grad, poisson_tv, tv_prox

F64 = torch.float64


@pytest.fixture(scope="module")
def setup():
    cfg = PupilConfig(n_out=63)
    psf = pupil_psf(None, cfg, warn=False)[0].float()
    obj = torch.zeros(128, 128)
    obj[30:36, 20:70] = 1.0
    obj[60, 30:60] = 1.0
    obj[70:74, 70:74] = 1.0
    noise = NoiseConfig(photons=100.0, background=5.0, read_noise_e=1.5)
    y = simulate_measurement(obj, psf, noise, seed=sample_seed("test", 0))
    return psf, obj, y, noise


def test_gradient_and_divergence_are_adjoint() -> None:
    """<grad u, p> == -<u, div p>, which the TV prox depends on."""
    torch.manual_seed(0)
    u = torch.rand(16, 20, dtype=F64)
    py = torch.rand(16, 20, dtype=F64)
    px = torch.rand(16, 20, dtype=F64)
    gy, gx = _grad(u)
    lhs = float((gy * py).sum() + (gx * px).sum())
    rhs = -float((u * _div(py, px)).sum())
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-12


def test_tv_prox_preserves_constants_and_reduces_variation() -> None:
    c = torch.full((32, 32), 3.0, dtype=F64)
    assert float((tv_prox(c, 0.1) - c).abs().max()) < 1e-9

    torch.manual_seed(0)
    noisy = c + 0.5 * torch.randn(32, 32, dtype=F64)

    def tv(v: torch.Tensor) -> float:
        return float(
            (v[1:, :] - v[:-1, :]).abs().sum() + (v[:, 1:] - v[:, :-1]).abs().sum()
        )

    assert tv(tv_prox(noisy, 0.3)) < 0.5 * tv(noisy)


def test_tv_actually_deconvolves(setup) -> None:
    """Regression test for an inverted FISTA step size.

    The Poisson Hessian is ``H^T diag(y/u^2) H``, so curvature is about
    ``1/mean(y)`` and the step is proportional to ``mean(y)``. Using its
    reciprocal leaves the estimate frozen near its constant initialisation: it
    still looks like an image and still has a small residual, but SSIM collapses.
    """
    psf, obj, y, noise = setup
    blurred = float(ssim(scale_match(apply_psf(obj, psf), obj), obj, data_range=1.0))
    est = poisson_tv(y[None], psf, 100, weight=0.005, background=noise.background)
    got = float(ssim(scale_match(est[0], obj), obj, data_range=1.0))
    assert got > blurred, f"TV ({got:.4f}) failed to improve on the blur ({blurred:.4f})"
    assert float(est.min()) >= 0.0
    assert float(est.std()) > 0.1 * float(y.std()), "estimate is nearly constant"


def test_rl_actually_deconvolves(setup) -> None:
    psf, obj, y, noise = setup
    blurred = float(ssim(scale_match(apply_psf(obj, psf), obj), obj, data_range=1.0))
    est = richardson_lucy(y[None], psf, 75, background=noise.background)
    assert float(ssim(scale_match(est[0], obj), obj, data_range=1.0)) > blurred


@pytest.mark.parametrize("weight", [0.0, 0.005, 0.05])
def test_tv_is_deterministic_and_batched(setup, weight: float) -> None:
    psf, _, y, noise = setup
    yb = torch.stack([y, y * 1.01])
    a = poisson_tv(yb, psf, 20, weight=weight, background=noise.background)
    b = poisson_tv(yb, psf, 20, weight=weight, background=noise.background)
    assert torch.equal(a, b)
    single = poisson_tv(yb[0][None], psf, 20, weight=weight, background=noise.background)
    assert torch.allclose(a[0], single[0], atol=1e-5)


def test_stronger_tv_weight_gives_smoother_estimate(setup) -> None:
    psf, _, y, noise = setup
    def roughness(x):
        return float((x[..., 1:, :] - x[..., :-1, :]).abs().mean())
    low = poisson_tv(y[None], psf, 60, weight=0.005, background=noise.background)
    high = poisson_tv(y[None], psf, 60, weight=0.2, background=noise.background)
    assert roughness(high) < roughness(low)


# ------------------------------------------------------- learned reconstructors


def test_unet_shape_nonnegativity_and_divisibility_guard() -> None:
    from src.reconstruction.unet import UNet, normalise_measurement

    net = UNet(base=8, depth=3)
    y = torch.rand(2, 32, 32)
    yn, scale = normalise_measurement(y)
    assert torch.allclose(yn.mean(dim=(-2, -1)), torch.ones(2), atol=1e-5)
    out = net(yn) * scale
    assert out.shape == y.shape
    assert float(out.min()) >= 0.0
    with pytest.raises(ValueError, match="divisible"):
        net(torch.rand(1, 30, 30))


def test_unrolled_takes_the_operator_as_an_argument(setup) -> None:
    """The PSF must be an argument, not baked in.

    The whole experiment consists of evaluating a network under an operator that
    differs from the one it was trained with, so a model storing its PSF could
    not be tested for operator shift at all.
    """
    from src.optics.psf import PupilConfig, pupil_psf
    from src.optics.zernike import NAMED_MODES
    from src.reconstruction.unrolled import UnrolledNet

    psf, _, y, _ = setup
    net = UnrolledNet(n_stages=3, width=8, n_layers=3)
    shifted = pupil_psf({NAMED_MODES["spherical"]: 0.14}, PupilConfig(n_out=63),
                        warn=False)[0].float()
    a = net(y[None], psf)
    b = net(y[None], shifted)
    assert a.shape == y[None].shape
    assert float(a.min()) >= 0.0
    assert float((a - b).abs().mean()) > 1e-6, "output ignores the operator"


def test_unrolled_gradients_flow_through_physics_and_denoiser(setup) -> None:
    from src.reconstruction.unrolled import UnrolledNet

    psf, _, y, _ = setup
    net = UnrolledNet(n_stages=3, width=8, n_layers=3)
    net(y[None], psf).mean().backward()
    assert net.log_step.grad is not None
    assert bool(torch.isfinite(net.log_step.grad).all())
    assert net.denoiser.net[0].weight.grad is not None


def test_learned_reconstructors_are_deterministic_in_eval() -> None:
    from src.reconstruction.unet import UNet

    torch.manual_seed(0)
    net = UNet(base=8, depth=2).eval()
    y = torch.rand(2, 32, 32)
    with torch.no_grad():
        assert torch.equal(net(y), net(y))

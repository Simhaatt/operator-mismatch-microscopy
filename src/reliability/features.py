"""Reliability features computed from a measurement and a reconstruction.

Everything here is **blind**: it sees only ``y`` and ``x_hat`` plus the operator
family the analyst chose. No ground truth enters, so these features are legal to
use at deployment time.

The central object is the *physics-consistency profile*: re-image the
reconstruction under each of K candidate operators and record how well each
explains the measurement,

    r_k = || y - s_k * H_k x_hat || / || y ||

with ``s_k`` a per-probe least-squares gain, since deconvolution fixes the object
only up to a global scale. The profile's *shape* -- where it bottoms out and how
sharply -- is what could carry information about which operator actually formed
the data, over and above the single nominal residual.

Note the profile is a deterministic function of ``(y, x_hat)``. It therefore adds
no information a sufficiently expressive learner could not extract from those two
images directly; the claim it has to earn is that it is a useful *inductive
bias*, cheap and interpretable, not that it is an information gain. That is
exactly what the Gate 3 experiment tests.
"""

from __future__ import annotations

import torch

from ..optics.forward_model import apply_psf

__all__ = ["relative_residual", "band_residuals", "profile", "oscr_features",
           "FEATURE_NAMES"]


def relative_residual(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """Scale-invariant relative residual between a measurement and a re-imaging."""
    dims = tuple(range(y.dim()))[-2:]
    num = (y_hat * y).sum(dim=dims, keepdim=True)
    den = (y_hat * y_hat).sum(dim=dims, keepdim=True).clamp_min(1e-20)
    scaled = y_hat * (num / den)
    return (scaled - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1).clamp_min(1e-20)


def band_residuals(resid: torch.Tensor, n_bands: int = 4) -> torch.Tensor:
    """Energy of the residual in ``n_bands`` radial spatial-frequency rings.

    Operator error is frequency-structured -- a wrong PSF misestimates contrast
    near the cutoff far more than at low frequency -- so the spectrum of the
    residual can separate model error from white shot noise, which spreads its
    energy evenly across rings.
    """
    f = torch.fft.fftshift(torch.fft.fft2(resid), dim=(-2, -1))
    power = f.real**2 + f.imag**2
    h, w = power.shape[-2:]
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=resid.device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=resid.device))
    ry, rx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(ry**2 + rx**2)
    edges = torch.linspace(0, 0.5, n_bands + 1, device=resid.device)
    total = power.flatten(1).sum(dim=1).clamp_min(1e-20)
    out = []
    for i in range(n_bands):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        out.append((power * mask).flatten(1).sum(dim=1) / total)
    return torch.stack(out, dim=1)


def profile(
    y: torch.Tensor, x_hat: torch.Tensor, probes: list[torch.Tensor]
) -> torch.Tensor:
    """Physics-consistency profile ``(B, K)`` over the operator family."""
    return torch.stack(
        [relative_residual(y, apply_psf(x_hat, h)) for h in probes], dim=1
    )


FEATURE_NAMES = [
    "r_nominal",      # the plain baseline: residual under the assumed operator
    "r_min",          # best achievable consistency over the probe family
    "r_mean",
    "r_std",          # sensitivity of consistency to the operator choice
    "r_gap",          # r_nominal - r_min: how much a better operator would help
    "r_argmin_frac",  # where in the family the optimum sits
    "r_curvature",    # sharpness of the optimum
    "band_0", "band_1", "band_2", "band_3",   # residual spectrum, nominal operator
    "y_mean", "y_std",   # measurement statistics: photon level and content proxy
    "xhat_std",          # reconstruction contrast
]


def oscr_features(
    y: torch.Tensor,
    x_hat: torch.Tensor,
    psf_nominal: torch.Tensor,
    probes: list[torch.Tensor],
    n_bands: int = 4,
) -> torch.Tensor:
    """Full blind feature vector, shape ``(B, len(FEATURE_NAMES))``."""
    if y.dim() == 2:
        y, x_hat = y[None], x_hat[None]
    prof = profile(y, x_hat, probes)
    r_nom = relative_residual(y, apply_psf(x_hat, psf_nominal))

    r_min, r_arg = prof.min(dim=1)
    k = prof.shape[1]
    # Discrete curvature at the minimum, clamped away from the ends.
    idx = r_arg.clamp(1, max(k - 2, 1))
    gather = lambda off: prof.gather(1, (idx + off).clamp(0, k - 1)[:, None]).squeeze(1)
    curv = gather(-1) - 2 * gather(0) + gather(1)

    resid = apply_psf(x_hat, psf_nominal) - y
    bands = band_residuals(resid, n_bands)

    dims = (-2, -1)
    feats = [
        r_nom, r_min, prof.mean(dim=1), prof.std(dim=1),
        r_nom - r_min, r_arg.to(y.dtype) / max(k - 1, 1), curv,
    ]
    feats += [bands[:, i] for i in range(n_bands)]
    feats += [y.mean(dim=dims), y.std(dim=dims), x_hat.std(dim=dims)]
    return torch.stack(feats, dim=1)

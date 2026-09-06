"""Poisson-Gaussian fluorescence forward model with reproducible sampling.

Implements the measurement model of the project,

    y ~ Poisson(eta * (h * x) + b) + Normal(0, sigma_r^2)

where ``h * x`` is a 2D convolution, ``eta`` sets the photon budget, ``b`` is a
constant background rate and ``sigma_r`` is camera read noise. Units are
photoelectrons throughout.

Two properties matter more than speed here:

* **Linear, not circular, convolution.** An FFT convolution is circular by
  construction; applied naively it wraps structure from one edge of a cell onto
  the other. We reflect-pad by the kernel half-width first, so the interior is
  a true linear convolution with a plausible boundary extension.
* **Bit-reproducible noise, independent of device.** Measurements are generated
  on the fly from a deterministic per-sample seed rather than pre-stored (the
  full shift matrix would be hundreds of GB, against a 20 GB Kaggle working
  disk). Noise is therefore drawn on the CPU and moved to the target device:
  CUDA and CPU generators produce *different* streams from the same seed, so
  drawing on device would make results depend on where they ran.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

__all__ = [
    "NoiseConfig",
    "sample_seed",
    "fft_convolve",
    "fft_convolve_adjoint",
    "apply_psf",
    "simulate_measurement",
    "BIOSR_PHOTON_RANGE",
    "BIOSR_HAZE_FRACTION",
    "background_for",
]


# BioSR spans nine signal levels quoted as 15-600 average photon count
# (figshare 13264793, verified). Photon budgets outside this range are not
# supported by the source data and should be justified explicitly.
BIOSR_PHOTON_RANGE: tuple[float, float] = (15.0, 600.0)

#: Fraction of a real BioSR widefield measurement's mean that is out-of-focus
#: light, measured by fitting ``a * blur(GT) + b`` to the summed raw SIM frames:
#: 19-68% across structures, median 31%. See docs/sim2real.md.
BIOSR_HAZE_FRACTION: float = 0.31


def background_for(photons: float, fraction: float = BIOSR_HAZE_FRACTION) -> float:
    """Background rate giving a fixed haze *fraction* at a given photon budget.

    The measurement mean is ``photons + background``, so a haze fraction
    ``f = b / (photons + b)`` requires ``b = photons * f / (1 - f)``.

    This exists because a fixed background is only correct at one photon level.
    At 45 photons of background the haze fraction is 31% at 100 photons -- the
    observed median -- but **75% at 15 and 7% at 600**, and the BioSR range is
    15 to 600. Holding the absolute rate fixed while sweeping the budget was a
    real mistake here: it collapsed a mismatch correlation to -0.151 and looked
    like a finding until the arithmetic was checked.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"haze fraction must be in [0, 1), got {fraction}")
    return float(photons) * fraction / (1.0 - fraction)



@dataclass(frozen=True)
class NoiseConfig:
    """Acquisition noise parameters, in photoelectrons.

    Attributes
    ----------
    photons:
        Target *mean* signal level per pixel over the field, after blurring.
        Anchor to :data:`BIOSR_PHOTON_RANGE`.
    background:
        Constant additive rate (autofluorescence, stray light, camera offset
        expressed as a rate) entering the Poisson draw.
    read_noise_e:
        Gaussian read noise standard deviation. sCMOS cameras used in SIM are
        typically ~1-2 e-.
    clip_negative:
        Real cameras report non-negative counts after offset subtraction only
        approximately; keeping small negatives is more faithful to sCMOS data
        and avoids biasing low-photon estimators. Default False.
    """

    photons: float = 100.0
    background: float = 5.0
    read_noise_e: float = 1.5
    clip_negative: bool = False

    def __post_init__(self) -> None:
        if self.photons <= 0:
            raise ValueError("photons must be positive")
        if self.background < 0 or self.read_noise_e < 0:
            raise ValueError("background and read noise must be non-negative")


def sample_seed(*parts: object, base: int = 0) -> int:
    """Derive a stable 63-bit seed from arbitrary identifying parts.

    Uses BLAKE2b rather than :func:`hash`, whose string hashing is randomised
    per interpreter run (``PYTHONHASHSEED``) and would silently break
    reproducibility across sessions and across Kaggle shards.
    """
    key = "|".join(str(p) for p in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


def fft_convolve(
    x: torch.Tensor, kernel: torch.Tensor, mode: str = "reflect"
) -> torch.Tensor:
    """Linear 2D convolution of ``x`` with a centred ``kernel`` via FFT.

    Parameters
    ----------
    x:
        ``(..., H, W)`` real tensor.
    kernel:
        ``(kh, kw)`` real tensor with **odd** side lengths, centre at
        ``(kh // 2, kw // 2)``.
    mode:
        Padding used to extend ``x`` before the circular transform:
        ``reflect`` (default), ``replicate``, or ``circular`` for genuinely
        periodic data.

    Returns
    -------
    Tensor of the same shape as ``x``.
    """
    if kernel.dim() != 2:
        raise ValueError("kernel must be 2D")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sides must be odd so the centre is unambiguous")

    squeeze = x.dim() == 2
    xb = x[None, None] if squeeze else x.reshape(-1, 1, *x.shape[-2:])
    ph, pw = kh // 2, kw // 2
    xb = torch.nn.functional.pad(xb, (pw, pw, ph, ph), mode=mode)
    ph_h, ph_w = xb.shape[-2:]

    ker = torch.zeros(ph_h, ph_w, dtype=x.dtype, device=x.device)
    ker[:kh, :kw] = kernel.to(dtype=x.dtype, device=x.device)
    # Move the kernel centre to index (0, 0) so the convolution introduces no shift.
    ker = torch.roll(ker, shifts=(-ph, -pw), dims=(0, 1))

    out = torch.fft.irfft2(
        torch.fft.rfft2(xb) * torch.fft.rfft2(ker)[None, None], s=(ph_h, ph_w)
    )
    out = out[..., ph : ph + x.shape[-2], pw : pw + x.shape[-1]]
    return out[0, 0] if squeeze else out.reshape(x.shape)


def apply_psf(
    x: torch.Tensor, psf: torch.Tensor, mode: str = "reflect"
) -> torch.Tensor:
    """Apply the imaging operator ``H`` (blur only, no noise, no scaling).

    This is the operator re-applied to a reconstruction when building the
    physics-consistency profile, so it must be *exactly* the blur used in
    :func:`simulate_measurement` -- same kernel, same padding.
    """
    return fft_convolve(x, psf, mode=mode).clamp_min(0.0)


def simulate_measurement(
    x: torch.Tensor,
    psf: torch.Tensor,
    noise: NoiseConfig,
    *,
    seed: int,
    mode: str = "reflect",
    rng_device: str = "cpu",
    return_clean: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Draw one measurement ``y`` from the Poisson-Gaussian model.

    Parameters
    ----------
    x:
        Non-negative object, ``(H, W)`` or ``(B, H, W)``. Scale is irrelevant:
        the blurred field is renormalised to ``noise.photons`` mean, so the
        photon budget -- not the arbitrary units of ``x`` -- sets the SNR.
    psf:
        Normalised kernel from :func:`src.optics.psf.pupil_psf`.
    seed:
        Deterministic seed; use :func:`sample_seed` to derive it from
        ``(dataset, index, psf_id, photon_level)`` so every cell of the shift
        matrix is independently but reproducibly noised.
    return_clean:
        Also return the noise-free expected signal, which is the correct
        reference for an oracle and for sanity-checking the photon budget.

    Returns
    -------
    ``y`` in photoelectrons, or ``(y, clean_rate)`` if ``return_clean``.
    """
    if torch.any(x < 0):
        raise ValueError("object must be non-negative (it is a photon density)")

    blurred = apply_psf(x, psf, mode=mode)
    mean = blurred.mean()
    if float(mean) <= 0:
        raise ValueError("blurred object has non-positive mean; cannot set a photon budget")
    rate = blurred * (noise.photons / mean) + noise.background

    gen = torch.Generator(device=rng_device)
    gen.manual_seed(int(seed))
    rate_cpu = rate.to(rng_device, dtype=torch.float32)
    counts = torch.poisson(rate_cpu, generator=gen)
    if noise.read_noise_e > 0:
        counts = counts + noise.read_noise_e * torch.randn(
            rate_cpu.shape, generator=gen, device=rng_device, dtype=torch.float32
        )
    y = counts.to(device=x.device, dtype=x.dtype)
    if noise.clip_negative:
        y = y.clamp_min(0.0)
    return (y, rate) if return_clean else y


def _pad_source_indices(n: int, pad: int, mode: str) -> torch.Tensor:
    """Map each padded index to the source index it copies, or -1 if it is zero.

    This makes the padding operator explicit as a linear gather, which is what
    lets :func:`fft_convolve_adjoint` build its exact transpose as a scatter-add.
    """
    j = torch.arange(n + 2 * pad)
    if mode == "circular":
        return (j - pad) % n
    if mode in ("constant", "zeros"):
        src = j - pad
        return torch.where((src >= 0) & (src < n), src, torch.full_like(src, -1))
    if mode == "replicate":
        return (j - pad).clamp(0, n - 1)
    if mode == "reflect":
        src = j - pad
        # Reflect without repeating the edge sample (torch's "reflect").
        src = torch.where(src < 0, -src, src)
        return torch.where(src >= n, 2 * n - 2 - src, src)
    raise ValueError(f"unsupported padding mode {mode!r}")


def fft_convolve_adjoint(
    z: torch.Tensor, kernel: torch.Tensor, mode: str = "reflect"
) -> torch.Tensor:
    """Exact adjoint of :func:`fft_convolve` for the same ``kernel`` and ``mode``.

    ``fft_convolve`` is the composition ``Crop . Conv . Pad``, so its transpose
    is ``Pad^T . Conv^T . Crop^T``:

    * ``Crop^T`` zero-embeds into the padded field,
    * ``Conv^T`` is circular correlation (conjugate spectrum),
    * ``Pad^T`` *folds* the padded margins back onto the samples they were
      copied from -- a scatter-add, **not** another pad.

    Getting ``Pad^T`` wrong is easy and quiet: substituting another reflect-pad
    breaks the adjoint identity by ~1e-3 relative, which is invisible in a
    reconstruction but makes Richardson-Lucy converge to the wrong fixed point
    and corrupts any gradient-based unrolled network built on the same operator.
    """
    if kernel.dim() != 2:
        raise ValueError("kernel must be 2D")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel sides must be odd so the centre is unambiguous")

    squeeze = z.dim() == 2
    zb = z[None, None] if squeeze else z.reshape(-1, 1, *z.shape[-2:])
    n_h, n_w = zb.shape[-2:]
    ph, pw = kh // 2, kw // 2
    N_h, N_w = n_h + 2 * ph, n_w + 2 * pw

    # Crop^T : zero-embed.
    zp = zb.new_zeros(zb.shape[0], 1, N_h, N_w)
    zp[..., ph : ph + n_h, pw : pw + n_w] = zb

    # Conv^T : circular correlation via the conjugate spectrum.
    ker = zb.new_zeros(N_h, N_w)
    ker[:kh, :kw] = kernel.to(dtype=zb.dtype, device=zb.device)
    ker = torch.roll(ker, shifts=(-ph, -pw), dims=(0, 1))
    out = torch.fft.irfft2(
        torch.fft.rfft2(zp) * torch.fft.rfft2(ker).conj()[None, None], s=(N_h, N_w)
    )

    # Pad^T : fold the margins back onto their source samples.
    idx_h = _pad_source_indices(n_h, ph, mode).to(out.device)
    idx_w = _pad_source_indices(n_w, pw, mode).to(out.device)
    keep_h, keep_w = idx_h >= 0, idx_w >= 0

    folded = out.new_zeros(out.shape[0], 1, n_h, N_w)
    folded.index_add_(2, idx_h[keep_h], out[..., keep_h, :])
    final = out.new_zeros(out.shape[0], 1, n_h, n_w)
    final.index_add_(3, idx_w[keep_w], folded[..., :, keep_w])

    return final[0, 0] if squeeze else final.reshape(z.shape)

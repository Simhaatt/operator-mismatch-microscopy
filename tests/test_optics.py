"""Physics validation for the optical forward model.

These are not smoke tests. Each pins a property that, if violated, would
invalidate the scientific claims downstream:

* Zernike orthonormality and index conventions -- otherwise the aberration
  severity axis (anchored to the Marechal limit) means nothing.
* PSF agreement with the analytic Airy pattern and FWHM ~ 0.51 lambda / NA --
  otherwise the "operator" is not the optical system we claim it is.
* Exact adjointness of the blur -- otherwise Richardson-Lucy converges to the
  wrong fixed point and unrolled-network gradients are wrong.
* Poisson-Gaussian statistics and seed determinism -- otherwise measurements
  are not reproducible across Kaggle shards.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.optics.forward_model import (
    BIOSR_PHOTON_RANGE,
    NoiseConfig,
    apply_psf,
    fft_convolve,
    fft_convolve_adjoint,
    sample_seed,
    simulate_measurement,
)
from src.optics.psf import (
    PupilConfig,
    airy_psf,
    marechal_strehl,
    nyquist_pixel_um,
    psf_fwhm_um,
    pupil_psf,
)
from src.optics.zernike import (
    NAMED_MODES,
    noll_to_osa,
    osa_index,
    osa_to_nm,
    osa_to_noll,
    rms_wavefront_error,
    zernike_basis,
)

F64 = torch.float64


@pytest.mark.parametrize("j", range(45))
def test_osa_index_round_trip(j: int) -> None:
    n, m = osa_to_nm(j)
    assert osa_index(n, m) == j


@pytest.mark.parametrize("j_noll", range(1, 46))
def test_noll_osa_round_trip(j_noll: int) -> None:
    assert osa_to_noll(noll_to_osa(j_noll)) == j_noll


def test_noll_literature_anchors() -> None:
    """Noll 4 is defocus (OSA 4); Noll 11 is primary spherical (OSA 12)."""
    assert noll_to_osa(1) == NAMED_MODES["piston"]
    assert noll_to_osa(4) == NAMED_MODES["defocus"]
    assert noll_to_osa(11) == NAMED_MODES["spherical"]


def test_zernike_orthonormal_on_unit_disk() -> None:
    n = 1200
    ax = torch.linspace(-1, 1, n, dtype=F64)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    rho, theta = torch.sqrt(xx**2 + yy**2), torch.atan2(yy, xx)
    inside = rho <= 1.0
    idx = list(range(15))
    basis = zernike_basis(idx, rho, theta) * inside
    cell = (2.0 / (n - 1)) ** 2
    area = float(inside.sum()) * cell
    gram = torch.einsum("aij,bij->ab", basis, basis) * cell / area
    assert abs(area - math.pi) < 1e-3
    assert float((gram - torch.eye(len(idx), dtype=F64)).abs().max()) < 5e-3


def test_rms_is_coefficient_norm_and_ignores_piston() -> None:
    assert rms_wavefront_error({NAMED_MODES["spherical"]: 1 / 14}) == pytest.approx(1 / 14)
    coeffs = {NAMED_MODES["spherical"]: 0.05, NAMED_MODES["coma_x"]: 0.05, 0: 9.9}
    assert rms_wavefront_error(coeffs) == pytest.approx(math.sqrt(2) * 0.05)


def test_pupil_config_rejects_even_oversample() -> None:
    """Even binning cannot be centred on the axis; it shifts the PSF by dx/2."""
    with pytest.raises(ValueError, match="odd"):
        PupilConfig(oversample=2)
    with pytest.raises(ValueError, match="odd"):
        PupilConfig(n_out=64)


def test_unaberrated_psf_matches_analytic_airy() -> None:
    cfg = PupilConfig()
    h, diag = pupil_psf(None, cfg, warn=False)
    a = airy_psf(cfg, integrate=True)
    cos = float((h * a).sum() / (h.norm() * a.norm()))
    assert cos > 0.9999, f"cosine similarity to Airy only {cos}"
    assert float((h - a).abs().sum()) < 0.02
    assert float(h.sum()) == pytest.approx(1.0)
    assert float(h.min()) >= 0.0
    assert diag.strehl == pytest.approx(1.0, abs=1e-6)


def test_psf_is_symmetric_for_symmetric_aberrations() -> None:
    """Regression test for the centred-binning bug: on-axis PSF equals its flip."""
    cfg = PupilConfig()
    for coeffs in ({}, {NAMED_MODES["spherical"]: 0.1}, {NAMED_MODES["defocus"]: 0.1}):
        h, _ = pupil_psf(coeffs, cfg, warn=False)
        assert float((h - torch.flip(h, [0, 1])).abs().max()) < 1e-12
        peak = torch.unravel_index(torch.argmax(h), h.shape)
        assert int(peak[0]) == int(peak[1]) == cfg.n_out // 2


@pytest.mark.parametrize(
    "wavelength_um,na",
    [(0.488, 1.4), (0.525, 1.4), (0.640, 1.4), (0.525, 1.27), (0.525, 0.9)],
)
def test_fwhm_follows_lambda_over_na(wavelength_um: float, na: float) -> None:
    cfg = PupilConfig(wavelength_um=wavelength_um, na=na)
    h, _ = pupil_psf(None, cfg, warn=False)
    got = psf_fwhm_um(h, cfg.pixel_um)
    theory = 0.5145 * wavelength_um / na
    # Detector-pixel integration broadens the measured FWHM, so the bias is
    # one-sided and small.
    assert 1.0 <= got / theory < 1.06


@pytest.mark.parametrize("rms", [0.02, 0.05])
def test_strehl_matches_marechal_for_small_aberrations(rms: float) -> None:
    _, diag = pupil_psf({NAMED_MODES["spherical"]: rms}, PupilConfig(), warn=False)
    assert diag.strehl == pytest.approx(marechal_strehl(rms), rel=0.05)


def test_marechal_limit_is_near_strehl_point_eight() -> None:
    """lambda/14 RMS is the diffraction limit -- the anchor for the severity axis."""
    _, diag = pupil_psf({NAMED_MODES["spherical"]: 1 / 14}, PupilConfig(), warn=False)
    assert 0.75 < diag.strehl < 0.88


def test_peak_intensity_falls_monotonically_with_aberration() -> None:
    cfg = PupilConfig()
    peaks = [
        float(pupil_psf({NAMED_MODES["spherical"]: r}, cfg, warn=False)[0].max())
        for r in (0.0, 1 / 14, 1 / 7, 0.25)
    ]
    assert peaks == sorted(peaks, reverse=True)


def test_nyquist_violation_is_reported_not_silent() -> None:
    assert nyquist_pixel_um(0.525, 1.4) == pytest.approx(0.525 / 5.6)
    with pytest.warns(UserWarning, match="Nyquist"):
        _, diag = pupil_psf(None, PupilConfig(pixel_um=0.20))
    assert diag.nyquist_ok is False


def test_psf_synthesis_is_deterministic() -> None:
    coeffs = {NAMED_MODES["spherical"]: 0.1, NAMED_MODES["coma_x"]: 0.05}
    a, _ = pupil_psf(coeffs, PupilConfig(), warn=False)
    b, _ = pupil_psf(coeffs, PupilConfig(), warn=False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular", "constant"])
@pytest.mark.parametrize("shape", [(48, 52), (33, 33), (16, 64)])
def test_blur_adjoint_is_exact(mode: str, shape: tuple[int, int]) -> None:
    """<Hx, z> == <x, H^T z>. Breaks at ~1e-3 if Pad^T is not a fold."""
    torch.manual_seed(0)
    x = torch.rand(*shape, dtype=F64)
    z = torch.rand(*shape, dtype=F64)
    k = torch.rand(7, 5, dtype=F64)  # asymmetric, so H^T != H
    lhs = float((fft_convolve(x, k, mode) * z).sum())
    rhs = float((x * fft_convolve_adjoint(z, k, mode)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-12


def test_fft_convolution_matches_spatial_convolution() -> None:
    torch.manual_seed(0)
    x = torch.rand(64, 80, dtype=F64)
    k = torch.rand(9, 7, dtype=F64)
    k /= k.sum()
    pad = torch.nn.functional.pad(x[None, None], (3, 3, 4, 4), mode="reflect")
    want = torch.nn.functional.conv2d(pad, torch.flip(k, [0, 1])[None, None])[0, 0]
    assert torch.allclose(fft_convolve(x, k, "reflect"), want, atol=1e-10)


def test_delta_kernel_is_identity_and_offset_delta_shifts_exactly() -> None:
    torch.manual_seed(0)
    x = torch.rand(40, 44, dtype=F64)
    d = torch.zeros(9, 9, dtype=F64)
    d[4, 4] = 1.0
    assert torch.allclose(fft_convolve(x, d), x, atol=1e-10)
    off = torch.zeros(9, 9, dtype=F64)
    off[4, 6] = 1.0
    assert torch.allclose(fft_convolve(x, off)[:, 2:], x[:, :-2], atol=1e-10)


def test_blur_conserves_energy() -> None:
    h, _ = pupil_psf(None, PupilConfig(n_out=63), warn=False)
    obj = torch.zeros(128, 128, dtype=F64)
    obj[40:80, 40:80] = 1.0
    assert float(apply_psf(obj, h).sum() / obj.sum()) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("photons", [BIOSR_PHOTON_RANGE[0], 100.0, BIOSR_PHOTON_RANGE[1]])
def test_photon_budget_is_honoured(photons: float) -> None:
    h, _ = pupil_psf(None, PupilConfig(n_out=63), warn=False)
    obj = torch.zeros(128, 128, dtype=F64)
    obj[40:80, 40:80] = 1.0
    noise = NoiseConfig(photons=photons, background=5.0)
    _, clean = simulate_measurement(obj, h, noise, seed=1, return_clean=True)
    assert float(clean.mean()) == pytest.approx(photons + noise.background, rel=0.02)


def test_poisson_gaussian_variance_is_rate_plus_read_variance() -> None:
    delta = torch.zeros(9, 9, dtype=F64)
    delta[4, 4] = 1.0
    flat = torch.ones(512, 512, dtype=F64)
    noise = NoiseConfig(photons=50.0, background=0.0, read_noise_e=2.0)
    y = torch.stack(
        [simulate_measurement(flat, delta, noise, seed=sample_seed("var", i)) for i in range(24)]
    )
    assert float(y.mean()) == pytest.approx(50.0, abs=0.2)
    assert float(y.var()) == pytest.approx(50.0 + 2.0**2, abs=1.0)


def test_measurements_are_reproducible_and_seed_separated() -> None:
    h, _ = pupil_psf(None, PupilConfig(n_out=31), warn=False)
    obj = torch.rand(64, 64, dtype=F64)
    noise = NoiseConfig(photons=100.0)
    s = sample_seed("biosr", 17, "defocus_mild", 100)
    assert torch.equal(
        simulate_measurement(obj, h, noise, seed=s),
        simulate_measurement(obj, h, noise, seed=s),
    )
    other = sample_seed("biosr", 18, "defocus_mild", 100)
    assert not torch.equal(
        simulate_measurement(obj, h, noise, seed=s),
        simulate_measurement(obj, h, noise, seed=other),
    )


def test_sample_seed_is_stable_across_interpreter_runs() -> None:
    """Pinned value: blake2b, not the PYTHONHASHSEED-randomised builtin hash()."""
    assert sample_seed("biosr", 17, "defocus_mild", 100) == 5174345927025672357


def test_negative_object_is_rejected() -> None:
    h, _ = pupil_psf(None, PupilConfig(n_out=31), warn=False)
    with pytest.raises(ValueError, match="non-negative"):
        simulate_measurement(-torch.ones(16, 16, dtype=F64), h, NoiseConfig(), seed=0)


def test_batched_blur_matches_per_sample() -> None:
    h, _ = pupil_psf(None, PupilConfig(n_out=31), warn=False)
    x = torch.rand(4, 32, 32, dtype=F64)
    looped = torch.stack([apply_psf(x[i], h) for i in range(4)])
    assert torch.allclose(apply_psf(x, h), looped, atol=1e-12)


def test_measurement_batch_is_seedable_per_image() -> None:
    """Per-image seeding must make results independent of batch composition.

    The gate scripts process objects in chunks to bound memory. If noise were
    drawn per batch rather than per image, results would depend on how work
    happened to be divided -- breaking reproducibility across machines, chunk
    sizes and Kaggle shards.
    """
    h, _ = pupil_psf(None, PupilConfig(n_out=31), warn=False)
    noise = NoiseConfig(photons=100.0)
    x = torch.rand(6, 32, 32, dtype=F64)

    def draw(indices: list[int]) -> torch.Tensor:
        return torch.stack(
            [simulate_measurement(x[j], h, noise, seed=sample_seed("cell", j)) for j in indices]
        )

    full = draw(list(range(6)))
    chunked = torch.cat([draw([0, 1, 2]), draw([3, 4, 5])])
    assert torch.equal(full, chunked)


def _energy_spread(n_out: int) -> float:
    """Spread in retained PSF energy across the aberration severity ladder."""
    cfg = PupilConfig(n_out=n_out)
    kept = [
        pupil_psf(c, cfg, warn=False)[1].energy_kept
        for c in (
            {},
            {NAMED_MODES["spherical"]: 1 / 7},
            {NAMED_MODES["spherical"]: 0.25},
            {NAMED_MODES["coma_x"]: 0.25},
        )
    ]
    return max(kept) - min(kept)


@pytest.mark.parametrize("n_out", [191, 255])
def test_adequate_kernel_truncation_is_not_severity_correlated(n_out: int) -> None:
    """Aberrated PSFs must not lose materially more energy to cropping.

    Their tails are heavier, so a short kernel discards more of them. Because the
    kernel is renormalised to sum 1, that turns the aberrated operator into an
    artificially compact PSF -- a severity-correlated distortion that can
    manufacture a spurious residual/severity correlation, which is exactly what
    Gate 2 measures. These sizes keep the effect negligible.
    """
    assert _energy_spread(n_out) < 0.006


def test_short_kernel_truncation_is_severity_correlated() -> None:
    """Regression record: why the kernel default cannot be small.

    At the verified BioSR sampling of 31.3 nm, a 63-pixel kernel spans under
    2 um and discards 13 percentage points more energy at the extreme
    aberration than at nominal. A sweep run at that size produced an apparent
    residual/severity correlation that vanished once the kernel was widened.
    Pinned here so the failure mode stays visible rather than being rediscovered.
    """
    assert _energy_spread(63) > 0.10

"""Step 4: is our simulated measurement realistic? A defence of decision D2.

D2 is the most attackable design choice in the project: BioSR's "ground truth"
is itself a SIM *reconstruction*, and every measurement we deconvolve is
simulated from it by our own forward model. A reviewer will ask whether that
forward model resembles a real widefield acquisition at all, or whether the
whole reliability layer is calibrated against a fiction.

BioSR answers this for free, from data already on disk. Each cell ships the
**raw SIM frames**, ``RawSIMData_level_01..09.mrc``: 9 frames of 502x502, three
illumination orientations by three phases. Two facts make them usable as real
widefield measurements:

1. The illumination is ``1 + m cos(k.r + phi)`` with the three phases at
   ``0, 2pi/3, 4pi/3``. Those sum to zero, so **averaging the three phases of
   one orientation cancels the pattern exactly** and leaves a plain widefield
   image. Averaging all nine gives a widefield image at the full dose.
2. The three orientation-averages are therefore three *independent* widefield
   images of the *same* object at one third of the dose each. Their pairwise
   differences contain **no specimen structure at all** -- only noise. That is
   an assumption-free noise measurement on real data, which is what makes the
   Poisson check below a test rather than a restatement of our own model.

The frame ordering is not assumed. :func:`decide_grouping` picks it per cell by
testing which of the two possible acquisition orders actually *cancels* the
pattern, and a cell where the two are indistinguishable is skipped rather than
guessed. This matters more than it sounds: the orders **disagree across BioSR**
-- CCPs reads as orientation-slow, Microtubules and F-actin as phase-slow.

**The factor-of-two trap.** Raw frames are on the camera grid at 62.6 nm;
``SIM_gt`` is 2x upsampled at 31.3 nm. Section 5 of STATUS.md records this bug
being made once already. Everything here is compared on the *camera* grid, and
the two simulation variants differ precisely in how they get there:

* ``camera`` -- PSF sampled at 62.6 nm, object = 2x2 block-mean of ``SIM_gt``.
  One noise draw, on the grid the real camera actually samples.
* ``fine``   -- the pipeline's own configuration: PSF at 31.3 nm, object =
  ``SIM_gt``, then the measurement is block-*summed* 2x2 onto the camera grid,
  because a pixel of twice the width collects four times the photons. Read
  noise is added once afterwards: summing four fine-grid read events would
  inflate its sigma by 2x, which is not what a camera does.

Reporting both is a stronger defence than either alone -- it separates "the
noise model is right" from "the sampling representation is right".
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mrc import read_mrc  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    NoiseConfig, apply_psf, sample_seed, simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402

# Verified acquisition constants; see STATUS.md section 5 and PupilConfig.
PHOTONS_PER_COUNT = 0.6026
WIDEFIELD_PIXEL_UM = 0.0626
GT_PIXEL_UM = 0.0313


def block_reduce(a: np.ndarray, factor: int = 2, how: str = "mean") -> np.ndarray:
    """Reduce ``a`` by ``factor`` in both axes, by block mean or block sum."""
    h, w = a.shape[-2:]
    h, w = h - h % factor, w - w % factor
    b = a[..., :h, :w].reshape(*a.shape[:-2], h // factor, factor, w // factor, factor)
    return b.mean(axis=(-3, -1)) if how == "mean" else b.sum(axis=(-3, -1))


def peak_relative_power(img: np.ndarray, pixel_um: float, na: float = 1.7,
                        wavelength_um: float = 0.525) -> float:
    """Strongest single-bin power excess over the azimuthal mean, near the cutoff.

    Dividing by the azimuthal mean at each radius removes the specimen's radial
    falloff, so a residual illumination pattern -- which is a pair of narrow
    lobes -- shows up as a large maximum while pure noise gives a small one.
    Taking the maximum rather than an average is what makes it sensitive: the
    lobes occupy a handful of Fourier bins out of thousands in the annulus, and
    an angular average dilutes them below the specimen's own anisotropy. An
    earlier version of this check averaged, and was fooled by every anisotropic
    specimen in the dataset.
    """
    img = np.asarray(img, dtype=np.float64)
    n = img.shape[-1]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.hypot(fy, fx)
    cutoff = 2 * na / wavelength_um * pixel_um
    ann = (rad > 0.3 * cutoff) & (rad < 1.1 * cutoff)
    power = np.abs(np.fft.fft2(img - img.mean())) ** 2
    rb = np.clip((rad * n).astype(int), 0, n // 2)
    cnt = np.maximum(np.bincount(rb.ravel(), minlength=n // 2 + 1), 1)
    prof = np.bincount(rb.ravel(), power.ravel(), minlength=n // 2 + 1) / cnt
    rel = power / np.maximum(prof[rb], 1e-30)
    rel[~ann] = 0.0
    return float(rel.max())


# With 9 frames as 3 orientations x 3 phases there are exactly two natural
# acquisition orders: orientation slow (CANONICAL) or phase slow (STRIDE).
CANONICAL = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
STRIDE = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
GROUPINGS = {"canonical": CANONICAL, "stride": STRIDE}


def _triple_averages(frames: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Three phase-averaged images, rescaled to a common mean.

    Diffraction efficiency differs slightly between the three orientations, so
    without the rescaling their differences would inherit a multiplicative copy
    of the specimen rather than being pure noise.
    """
    o = np.stack([frames[g].mean(axis=0) for g in groups])
    scale = o.mean(axis=(1, 2), keepdims=True)
    return o * (scale.mean() / scale)


def score_grouping(frames: np.ndarray, groups: list[list[int]],
                   pixel_um: float) -> float:
    """Median peak relative power over the three pairwise triple-differences.

    If ``groups`` really are the phase-triples, each average is pattern-free and
    every difference is pure noise, so this is small. Under the wrong grouping
    each average retains all three directions at a third of their amplitude and
    the differences keep a modulation spike.
    """
    o = _triple_averages(frames, groups)
    return float(np.median([peak_relative_power(o[i] - o[j], pixel_um)
                            for i, j in ((0, 1), (1, 2), (0, 2))]))


def decide_grouping(frames_bright: np.ndarray, pixel_um: float,
                    margin: float = 1.15) -> tuple[list[list[int]], dict]:
    """Pick the phase-triples for one cell, from its brightest signal level.

    The acquisition order is a property of the microscope software, not of the
    exposure, so the decision is made once per cell on the brightest level --
    where the pattern has the most contrast against shot noise -- and then
    reused for every level of that cell. Deciding per level instead lets photon
    noise flip the answer at level 1.

    This is *not* a formality. The two orders do not agree across BioSR: CCPs
    reads as orientation-slow while Microtubules and F-actin read as phase-slow,
    by factors of 3-5x at the bright levels. The dataset already has one known
    per-structure inconsistency (ER's archive layout), so assuming a single
    convention would have silently mis-averaged two structures out of four.

    Raises when the two orders are within ``margin`` of each other, so an
    unidentifiable cell is skipped rather than guessed: a wrong grouping leaves
    residual pattern at exactly the frequencies this script is measuring.
    """
    scores = {k: score_grouping(frames_bright, g, pixel_um)
              for k, g in GROUPINGS.items()}
    lo, hi = sorted(scores, key=lambda k: scores[k])
    if scores[hi] < margin * scores[lo]:
        raise ValueError(
            f"frame ordering not identifiable: residual pattern after phase "
            f"averaging is {scores['canonical']:.1f} (canonical) vs "
            f"{scores['stride']:.1f} (stride), inside the {margin:.2f}x margin")
    return GROUPINGS[lo], {"chosen": lo, "scores": scores,
                           "ratio": scores[hi] / max(scores[lo], 1e-30)}

def radial_power(img: np.ndarray, pixel_um: float, n_bins: int = 80):
    """Azimuthally averaged power spectrum, returned in cycles per micron."""
    img = img - img.mean()
    n = img.shape[-1]
    power = np.abs(np.fft.fft2(img)) ** 2 / img.size
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.hypot(fy, fx)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    b = np.clip(np.digitize(rad.ravel(), edges) - 1, 0, n_bins - 1)
    cnt = np.maximum(np.bincount(b, minlength=n_bins), 1)
    prof = np.bincount(b, power.ravel(), minlength=n_bins) / cnt
    centres = 0.5 * (edges[:-1] + edges[1:]) / pixel_um
    return centres, prof


def spectral_flatness(freq: np.ndarray, prof: np.ndarray, cutoff: float) -> dict:
    """How flat is a power spectrum? Two numbers, both 1.0 for white noise.

    ``hf_lf`` is the ratio of mean power above the widefield cutoff to the mean
    power below it. ``slope`` is the least-squares tilt of log power against
    frequency, in decades per (cycle/um). White noise gives 1.0 and 0.0.
    """
    lo, hi = (freq > 0) & (freq < cutoff), freq >= cutoff
    if lo.sum() < 3 or hi.sum() < 3:
        return {"hf_lf": float("nan"), "slope": float("nan")}
    keep = (freq > 0) & (prof > 0)
    A = np.stack([freq[keep], np.ones(int(keep.sum()))], axis=1)
    slope = float(np.linalg.lstsq(A, np.log10(prof[keep]), rcond=None)[0][0])
    return {"hf_lf": float(prof[hi].mean() / max(prof[lo].mean(), 1e-30)),
            "slope": slope}


def out_of_band(img: np.ndarray, pixel_um: float, na: float = 1.7,
                wavelength_um: float = 0.525, f_min: float = 1.05,
                axis_guard: int = 3) -> tuple[np.ndarray, float]:
    """Keep only the frequencies a widefield measurement cannot carry signal at.

    The OTF is strictly zero beyond ``2 NA / lambda``, so above that a widefield
    image contains **only** noise -- shot noise and read noise, both white. That
    makes the out-of-band region an unbiased noise reference that no amount of
    specimen structure can contaminate. It is the same reasoning the project's
    ``otf_noise_ref`` feature uses.

    This is needed because the real orientation-difference is *not* pure noise:
    the three illumination orientations have their own spatially varying
    envelopes, so their difference keeps a smooth multiplicative copy of the
    specimen. A single global scale match cannot remove a varying field, and the
    leak is overwhelmingly low-frequency -- it showed up as a noise flatness of
    0.01 instead of 1.0. Filtering it out costs nothing, because the discarded
    band is exactly where the leak lives.

    Bins within ``axis_guard`` of ``fx = 0`` or ``fy = 0`` are dropped too: sCMOS
    fixed-pattern noise puts a deterministic ridge there which is not part of
    the stochastic noise being measured.

    Returns the filtered image and the fraction of Fourier bins retained. For
    white noise the filtered variance is that fraction of the true variance, so
    dividing by it recovers the full-band noise variance.
    """
    img = np.asarray(img, dtype=np.float64)
    n = img.shape[-1]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    rad = np.hypot(fy, fx)
    cutoff = 2 * na / wavelength_um * pixel_um
    iy = np.minimum(np.arange(n), n - np.arange(n))[:, None]
    ix = np.minimum(np.arange(n), n - np.arange(n))[None, :]
    keep = (rad > f_min * cutoff) & (iy >= axis_guard) & (ix >= axis_guard)
    frac = float(keep.mean())
    if frac <= 0:
        raise ValueError("out-of-band region is empty; check the pixel size")
    F = np.fft.fft2(img - img.mean())
    return np.real(np.fft.ifft2(F * keep)), frac


def variance_vs_mean(level: np.ndarray, diff: np.ndarray, divisor: float,
                     tile: int = 16, read_var: float = 2.25) -> dict:
    """Fit ``var = g * mu + c`` from a level image and a structure-free difference.

    ``diff`` must be a difference of two independent measurements of the same
    object, so it carries noise only; ``divisor`` converts its variance to the
    per-pixel noise variance of one full-dose image (2 for two equal-variance
    replicates, 6 for two of the one-third-dose orientation averages).

    In photoelectrons a Poisson-Gaussian sensor must give ``g = 1`` and
    ``c = read_noise^2``. Any other slope means the noise model is wrong, which
    is the whole question. Tiles are used rather than pixels because a single
    pixel gives one noisy sample of both quantities; the tile mean is a good
    estimate of the local rate and the tile variance of ``diff`` is unbiased
    regardless of how much structure the specimen has, since ``diff`` has none.
    """
    mu = block_reduce(level, tile, "mean").ravel()
    d = diff[: diff.shape[0] // tile * tile, : diff.shape[1] // tile * tile]
    d = d.reshape(d.shape[0] // tile, tile, d.shape[1] // tile, tile)
    var = d.var(axis=(1, 3)).ravel() / divisor
    keep = np.isfinite(mu) & np.isfinite(var) & (mu > 0)
    mu, var = mu[keep], var[keep]
    if mu.size < 8:
        return {"gain": float("nan"), "read_var": float("nan"), "n_tiles": 0}
    A = np.stack([mu, np.ones_like(mu)], axis=1)
    g, c = np.linalg.lstsq(A, var, rcond=None)[0]
    pred = A @ np.array([g, c])
    ss = float(((var - pred) ** 2).sum())
    tot = float(((var - var.mean()) ** 2).sum())
    # The free intercept is badly conditioned: read variance is ~2 e-^2 against
    # signal levels in the hundreds, so the fit extrapolates a long way to reach
    # zero and the intercept absorbs most of the scatter. Fixing it at the known
    # sensor value leaves gain as the one well-determined parameter, and gain is
    # what the Poisson question is about. Validated against synthetic
    # Poisson-Gaussian data with known truth, where it recovers 1.00 while the
    # free-intercept gain wanders over 0.97-1.06 with the tile size.
    g_fix = float(((var - read_var) * mu).sum() / max((mu * mu).sum(), 1e-30))
    return {"gain": float(g), "read_var": float(c), "gain_fixed": g_fix,
            "n_tiles": int(mu.size),
            "r2": float(1.0 - ss / tot) if tot > 0 else float("nan"),
            "mean_level": float(mu.mean())}


def real_widefield(frames: np.ndarray, pixel_um: float,
                   groups: list[list[int]]) -> dict:
    """Turn 9 raw SIM frames into a real widefield measurement plus its noise.

    The 9-frame combination needs no grouping at all -- it contains all three
    phases of all three orientations, so the pattern cancels however the frames
    are ordered. Only the noise difference depends on ``groups``.

    The frames are **summed**, not averaged. This is the whole dose bookkeeping
    and getting it wrong is quiet: an average of nine Poisson frames of mean
    ``mu`` has variance ``mu/9``, so comparing it against ``var = mu`` reports a
    gain of 1/9 and looks like a catastrophic failure of the noise model when
    the data are in fact perfectly Poisson. The sum is what a single exposure of
    the same total dose would have produced -- mean ``9mu``, variance ``9mu`` --
    and it is what the simulation is matched against.

    The camera pedestal is removed as the 0.1st percentile over all frames --
    raw BioSR counts bottom out around 59-90 ADU, which is offset, not signal --
    and counts are converted to photoelectrons.

    ``divisor`` turns the difference's variance into the summed image's own. A
    triple sum has variance ``3 v``; a difference of two of them ``6 v``; the
    nine-frame sum has ``9 v``. So ``var(y) = var(diff) * 9/6``, i.e. a divisor
    of 2/3.
    """
    pedestal = float(np.percentile(frames, 0.1))
    e = (np.asarray(frames, dtype=np.float64) - pedestal) * PHOTONS_PER_COUNT
    orient = 3.0 * _triple_averages(e, groups)          # per-orientation sums
    diffs = [orient[0] - orient[1], orient[1] - orient[2], orient[0] - orient[2]]
    pick = int(np.argsort([d.var() for d in diffs])[1])   # median of the three
    return {"y": e.sum(axis=0), "diff": diffs[pick], "divisor": 2.0 / 3.0,
            "groups": groups, "pedestal": pedestal}


def sim_widefield(gt: np.ndarray, variant: str, mean_e: float, noise: NoiseConfig,
                  seed_key: str, kernel_camera: int = 95, kernel_fine: int = 191):
    """Simulate a widefield measurement on the camera grid, two ways.

    ``camera`` samples the PSF at the camera pixel and feeds it the 2x-reduced
    object: one noise draw, on the grid the sensor reads out. ``fine`` is the
    project's own configuration -- PSF and object at 31.3 nm -- with the
    measurement block-summed onto the camera grid afterwards, since a pixel of
    twice the width collects four times the light. Read noise is applied once,
    after summing, because the sensor reads each camera pixel once.

    Returns two independent noise realisations of the same clean rate, so the
    caller can form a structure-free difference exactly as it does for the real
    frames.
    """
    tgt = max(mean_e - noise.background, 1e-3)
    if variant == "camera":
        cfg = PupilConfig(pixel_um=WIDEFIELD_PIXEL_UM, n_out=kernel_camera)
        psf = pupil_psf(None, cfg, warn=False)[0].to(torch.float32)
        x = torch.from_numpy(block_reduce(gt, 2, "mean")).to(torch.float32)
        nc = NoiseConfig(photons=tgt, background=noise.background,
                         read_noise_e=noise.read_noise_e)
        ys = [simulate_measurement(x, psf, nc, seed=sample_seed(seed_key, variant, r))
              for r in (0, 1)]
        clean = apply_psf(x, psf)
        return [y.numpy() for y in ys], clean.numpy(), psf, x.numpy()

    cfg = PupilConfig(pixel_um=GT_PIXEL_UM, n_out=kernel_fine)
    psf = pupil_psf(None, cfg, warn=False)[0].to(torch.float32)
    x = torch.from_numpy(gt).to(torch.float32)
    # Quarter the per-pixel budget on the fine grid so the 2x2 sum lands on the
    # target camera-grid mean; suppress read noise here and add it once below.
    nc = NoiseConfig(photons=tgt / 4.0, background=noise.background / 4.0,
                     read_noise_e=0.0)
    out = []
    for r in (0, 1):
        y = simulate_measurement(x, psf, nc, seed=sample_seed(seed_key, variant, r))
        cam = block_reduce(y.numpy(), 2, "sum")
        rng = np.random.default_rng(sample_seed(seed_key, variant, "read", r) % 2**32)
        out.append(cam + noise.read_noise_e * rng.standard_normal(cam.shape))
    clean = block_reduce(apply_psf(x, psf).numpy(), 2, "sum")
    return out, clean, psf, gt


def fit_residual(y: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Residual of ``y`` against ``clean`` after a least-squares scale and offset.

    The real measurement and the blurred ground truth live in different units --
    BioSR's GT is a SIM reconstruction with its own arbitrary scaling -- so a
    raw subtraction would report a residual dominated by that mismatch. Two free
    parameters is the least that makes the comparison meaningful, and it is the
    same convention ``scale_match`` uses for the error metric.
    """
    A = np.stack([clean.ravel(), np.ones(clean.size)], axis=1)
    coef = np.linalg.lstsq(A, y.ravel(), rcond=None)[0]
    return y - (A @ coef).reshape(y.shape)


def _oob(diff: np.ndarray, divisor: float) -> tuple[np.ndarray, float]:
    """``(filtered diff, adjusted divisor)`` for :func:`variance_vs_mean`."""
    filt, frac = out_of_band(diff, WIDEFIELD_PIXEL_UM)
    return filt, divisor * frac


def compare_cell(zf: zipfile.ZipFile, member: str, gt_member: str,
                 structure: str, cell: str, level: int, noise: NoiseConfig,
                 tile: int, groups: list[list[int]], audit: dict) -> dict:
    """One (cell, signal level) comparison: real widefield versus both variants."""
    frames, _ = read_mrc(zf.read(member))
    gt, _ = read_mrc(zf.read(gt_member))
    gt = np.asarray(gt[0], dtype=np.float64)
    frames = np.asarray(frames)
    # The whole comparison lives on the camera grid and reaches it by halving
    # the ground-truth sampling, so the 2x relation is load-bearing. It holds
    # for every layout in BioSR -- including F-actin/Cell_024's wider 1516/758
    # field and ER's nested per-level GTSIM -- but a silent mismatch would
    # compare two different frequency axes and look like a physics result, so
    # it is checked rather than trusted.
    if gt.shape[-1] != 2 * frames.shape[-1] or gt.shape[-2] != 2 * frames.shape[-2]:
        raise ValueError(
            f"ground truth {gt.shape} is not 2x the raw frames "
            f"{frames.shape[-2:]}; the 62.6/31.3 nm pairing does not hold for "
            f"{member}")
    lo, hi = np.percentile(gt, [0.1, 99.9])
    gt = np.clip((gt - lo) / max(hi - lo, 1e-12), 0.0, None)

    real = real_widefield(frames, WIDEFIELD_PIXEL_UM, groups)
    y_real = real["y"]
    mean_e = float(y_real.mean())
    cutoff = 2 * 1.7 / 0.525  # widefield OTF cutoff, cyc/um

    f, p_real = radial_power(y_real, WIDEFIELD_PIXEL_UM)
    _, p_rdiff = radial_power(real["diff"], WIDEFIELD_PIXEL_UM)
    out = {
        "structure": structure, "cell": cell, "level": level,
        "mean_photoelectrons": mean_e, "pedestal_adu": real["pedestal"],
        "orientation_groups": real["groups"], "grouping_audit": audit,
        "real": {
            # Raw difference: contaminated by the per-orientation envelope, kept
            # only so the contamination is visible rather than hidden.
            "poisson_raw": variance_vs_mean(y_real, real["diff"],
                                            real["divisor"], tile),
            "poisson": variance_vs_mean(y_real, *_oob(real["diff"],
                                        real["divisor"]), tile),
            "noise_flatness": spectral_flatness(f, p_rdiff, cutoff),
        },
        "variants": {},
    }
    clean_cam = None
    for variant in ("camera", "fine"):
        ys, clean, _, _ = sim_widefield(gt, variant, mean_e, noise,
                                        f"{structure}/{cell}/{level}")
        if variant == "camera":
            clean_cam = clean
        y_sim = 0.5 * (ys[0] + ys[1])
        _, p_sim = radial_power(y_sim, WIDEFIELD_PIXEL_UM)
        _, p_sdiff = radial_power(ys[0] - ys[1], WIDEFIELD_PIXEL_UM)
        # Both spectra are normalised to their own in-band power, so the
        # comparison is of spectral SHAPE, not of the arbitrary GT scaling.
        band = (f > 0) & (f < cutoff)
        nr = p_real / max(p_real[band].mean(), 1e-30)
        ns = p_sim / max(p_sim[band].mean(), 1e-30)
        ok = band & (nr > 0) & (ns > 0)
        out["variants"][variant] = {
            # ys[0]-ys[1] has twice one draw's variance, hence divisor 2.
            # Filtered exactly as the real difference is, so the comparison is
            # of the same estimator applied to both.
            "poisson_raw": variance_vs_mean(y_sim, ys[0] - ys[1], 2.0, tile),
            "poisson": variance_vs_mean(y_sim, *_oob(ys[0] - ys[1], 2.0), tile),
            "noise_flatness": spectral_flatness(f, p_sdiff, cutoff),
            "resid_flatness": spectral_flatness(
                f, radial_power(fit_residual(y_sim, clean), WIDEFIELD_PIXEL_UM)[1],
                cutoff),
            "spec_logdiff_inband": float(np.abs(
                np.log10(ns[ok]) - np.log10(nr[ok])).mean()),
            "spec_logdiff_at_cut": float(np.abs(np.log10(ns[ok]) - np.log10(nr[ok]))[
                (f[ok] > 0.7 * cutoff)].mean()) if (f[ok] > 0.7 * cutoff).any()
                else float("nan"),
        }
    # Real residual whiteness needs the blurred GT on the camera grid; the
    # camera variant's clean rate is exactly that, up to scale and offset.
    out["real"]["resid_flatness"] = spectral_flatness(
        f, radial_power(fit_residual(y_real, clean_cam), WIDEFIELD_PIXEL_UM)[1], cutoff)
    return out


def index_cell(names: set[str], structure: str, cell: str) -> dict[int, tuple[str, str]]:
    """Map signal level -> (raw frames member, paired ground-truth member).

    Two layouts, both present in BioSR. The fixed-cell structures put the raw
    frames at ``<S>/<Cell>/RawSIMData_level_NN.mrc`` against a single
    ``SIM_gt.mrc``. ER is live-cell and nests them under ``RawSIMData/`` with a
    *per-level* ``GTSIM/GTSIM_level_NN.mrc``, which is the better pairing -- the
    specimen moves between levels, so one shared ground truth would be wrong.
    Missing this layout silently drops 68 of 228 cells (STATUS.md section 5).
    """
    out: dict[int, tuple[str, str]] = {}
    for level in range(1, 10):
        flat = f"{structure}/{cell}/RawSIMData_level_{level:02d}.mrc"
        nest = f"{structure}/{cell}/RawSIMData/RawSIMData_level_{level:02d}.mrc"
        gt_flat = f"{structure}/{cell}/SIM_gt.mrc"
        gt_nest = f"{structure}/{cell}/GTSIM/GTSIM_level_{level:02d}.mrc"
        if flat in names and gt_flat in names:
            out[level] = (flat, gt_flat)
        elif nest in names and gt_nest in names:
            out[level] = (nest, gt_nest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", nargs="+",
                    default=["CCPs", "Microtubules", "F-actin", "ER"])
    ap.add_argument("--cells", type=int, default=3,
                    help="cells per structure, taken in archive order")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 5, 9],
                    help="RawSIMData_level_NN indices; 1 is dimmest, 9 brightest")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--background", type=float, default=5.0)
    ap.add_argument("--read-noise-e", type=float, default=1.5)
    ap.add_argument("--archives", default="data/biosr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    noise = NoiseConfig(photons=100.0, background=args.background,
                        read_noise_e=args.read_noise_e)
    print("FORWARD-MODEL REALISM CHECK  (defends D2)")
    print(f"  real widefield = SUM of 9 raw SIM frames (total dose), "
          f"pedestal-subtracted, x{PHOTONS_PER_COUNT} e-/count")
    print(f"  Poisson-Gaussian truth: gain = 1.000, read_var = "
          f"{args.read_noise_e**2:.2f} e-^2")
    print(f"  widefield OTF cutoff = {2*1.7/0.525:.3f} cyc/um; "
          f"camera pixel {WIDEFIELD_PIXEL_UM*1000:.1f} nm, GT {GT_PIXEL_UM*1000:.1f} nm")
    print()

    rows, skipped = [], []
    for structure in args.structures:
        path = Path(args.archives) / f"{structure}.zip"
        if not path.exists():
            skipped.append(f"{structure}: no archive at {path}")
            continue
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            cells = sorted({n.split("/")[1] for n in names
                            if n.count("/") >= 2 and n.split("/")[1].startswith("Cell_")})
            for cell in cells[: args.cells]:
                avail = index_cell(names, structure, cell)
                if not avail:
                    skipped.append(f"{structure}/{cell}: no raw/GT pair found")
                    continue
                # Decide the frame ordering once, on the brightest level, then
                # reuse it: the acquisition order is a property of the run, and
                # at level 1 photon noise can flip the test.
                bright = max(avail)
                try:
                    frames_b, _ = read_mrc(zf.read(avail[bright][0]))
                    groups, audit = decide_grouping(np.asarray(frames_b),
                                                    WIDEFIELD_PIXEL_UM)
                except ValueError as exc:
                    skipped.append(f"{structure}/{cell}: {exc}")
                    continue
                print(f"  {structure}/{cell}: ordering {audit['chosen']!r} "
                      f"({audit['ratio']:.2f}x, decided on level {bright:02d}, "
                      f"levels available {sorted(avail)})")
                for level in args.levels:
                    if level not in avail:
                        skipped.append(f"{structure}/{cell} level {level}: absent")
                        continue
                    try:
                        rows.append(compare_cell(
                            zf, avail[level][0], avail[level][1], structure,
                            cell, level, noise, args.tile, groups, audit))
                    except ValueError as exc:
                        skipped.append(f"{structure}/{cell} level {level}: {exc}")
                        continue
                    r = rows[-1]
                    print(f"  {structure}/{cell} L{level:02d}  "
                          f"mean={r['mean_photoelectrons']:8.2f} e-")
                    print(f"    {'source':10s} {'gain':>7s} {'gain_raw':>9s} "
                          f"{'noise hf/lf':>12s} {'resid hf/lf':>12s} {'dlog(spec)':>11s}")
                    pr, fr = r["real"]["poisson"], r["real"]["noise_flatness"]
                    print(f"    {'REAL':10s} {pr['gain_fixed']:7.3f} "
                          f"{r['real']['poisson_raw']['gain_fixed']:9.3f} "
                          f"{fr['hf_lf']:12.4f} "
                          f"{r['real']['resid_flatness']['hf_lf']:12.3f} {'--':>11s}")
                    for v, d in r["variants"].items():
                        print(f"    {'SIM ' + v:10s} {d['poisson']['gain_fixed']:7.3f} "
                              f"{d['poisson_raw']['gain_fixed']:9.3f} "
                              f"{d['noise_flatness']['hf_lf']:12.4f} "
                              f"{d['resid_flatness']['hf_lf']:12.3f} "
                              f"{d['spec_logdiff_inband']:11.3f}")
                    print()

    if not rows:
        print("  nothing compared." + ("\n  " + "\n  ".join(skipped) if skipped else ""))
        return
    summarise(rows, args)
    if skipped:
        print("\n  skipped:")
        for s in skipped:
            print(f"    {s}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "cells": rows, "skipped": skipped}, indent=2),
            encoding="utf-8")
        print(f"\n  wrote {args.out}")


def summarise(rows: list[dict], args) -> None:
    """Aggregate verdicts. Every cell of the grid is in before anything is claimed."""
    def med(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        return float(np.median(v)) if v.size else float("nan")

    print(f"  SUMMARY over {len(rows)} (cell, level) pairs")
    print(f"    {'source':12s} {'gain':>8s} {'gain_raw':>9s} {'noise hf/lf':>12s} "
          f"{'resid hf/lf':>12s} {'dlog inband':>12s} {'dlog at cut':>12s}")
    print(f"    {'REAL':12s} {med(r['real']['poisson']['gain_fixed'] for r in rows):8.3f} "
          f"{med(r['real']['poisson_raw']['gain_fixed'] for r in rows):9.3f} "
          f"{med(r['real']['noise_flatness']['hf_lf'] for r in rows):12.4f} "
          f"{med(r['real']['resid_flatness']['hf_lf'] for r in rows):12.3f} "
          f"{'--':>12s} {'--':>12s}")
    for v in ("camera", "fine"):
        g = [r["variants"][v] for r in rows]
        print(f"    {'SIM ' + v:12s} {med(d['poisson']['gain_fixed'] for d in g):8.3f} "
              f"{med(d['poisson_raw']['gain_fixed'] for d in g):9.3f} "
              f"{med(d['noise_flatness']['hf_lf'] for d in g):12.4f} "
              f"{med(d['resid_flatness']['hf_lf'] for d in g):12.3f} "
              f"{med(d['spec_logdiff_inband'] for d in g):12.3f} "
              f"{med(d['spec_logdiff_at_cut'] for d in g):12.3f}")
    print()
    real_gain = med(r["real"]["poisson"]["gain_fixed"] for r in rows)
    real_flat = med(r["real"]["noise_flatness"]["hf_lf"] for r in rows)
    print("  Per structure, since the camera settings need not have been shared:")
    print(f"    {'structure':14s} {'n':>3s} {'gain':>7s} {'implied e-/count':>17s} "
          f"{'free-fit R2':>12s} {'dlog inband':>12s}")
    for st in sorted({r["structure"] for r in rows}):
        g = [r for r in rows if r["structure"] == st]
        gn = med(r["real"]["poisson"]["gain_fixed"] for r in g)
        print(f"    {st:14s} {len(g):3d} {gn:7.3f} "
              f"{PHOTONS_PER_COUNT / max(gn, 1e-9):17.4f} "
              f"{med(r['real']['poisson']['r2'] for r in g):12.3f} "
              f"{med(r['variants']['fine']['spec_logdiff_inband'] for r in g):12.3f}")
    print()
    print(f"    A gain of g means the measured noise variance is g times the mean")
    print(f"    under the assumed {PHOTONS_PER_COUNT} photons/count. Because gain")
    print("    scales linearly with that constant, g > 1 is equally consistent")
    print("    with the constant being g times too large -- the 'implied e-/count'")
    print("    column is that reading. Which of the two it is cannot be settled")
    print("    from these data alone; either way the noise is Poisson in SHAPE,")
    print("    which is the part the forward model asserts.")
    print()
    print("  Reading these numbers:")
    print(f"    gain: measured from the out-of-band band, where a widefield")
    print("      measurement can carry no signal, with the read variance fixed")
    print("      at its known value. Recovers 0.998 +- 0.003 on synthetic")
    print(f"      Poisson-Gaussian data. 1.000 is Poisson. Real data gives "
          f"{real_gain:.3f}.")
    print("    gain_raw: the same fit on the unfiltered difference, for contrast.")
    print("      Above 1 means the sensor is noisier than Poisson (excess noise,")
    print("      or residual structure leaking into the difference); below 1 means")
    print("      the photons-per-count constant is off, since gain scales with it.")
    print(f"    noise hf/lf: 1.000 is white. Real data gives {real_flat:.3f}; a")
    print("      value well below 1 would mean the difference is NOT pure noise")
    print("      and the variance fit above is contaminated.")
    print("    resid hf/lf: y minus the blurred GT. This one is EXPECTED to differ")
    print("      between real and simulated -- the simulation's residual is white")
    print("      by construction, while the real residual also carries the")
    print("      mismatch between our PSF and the true optics, plus whatever the")
    print("      SIM reconstruction put into the GT. Its size is the honest")
    print("      measure of how much the forward model leaves unexplained.")
    print("    dlog(spec): mean |log10| gap between simulated and real radial")
    print("      power, both normalised to their own in-band mean, so it measures")
    print("      spectral SHAPE. Small in-band and at the cutoff is the pass.")


if __name__ == "__main__":
    main()

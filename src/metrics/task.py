"""Downstream task measurements on reconstructions.

SSIM and relative $\\ell_2$ both score a reconstruction against ground truth
pixel by pixel. Neither is what a biologist reports. They report a *number*
extracted from the image -- how many clathrin-coated pits, how much filament
length, how connected the ER network is -- and the error that matters is the
error in that number.

This matters here more than it usually would. Phase 2 established that
$1-\\mathrm{SSIM}$ and relative $\\ell_2$ disagree about which specimen class is
hard: the held-out punctate structure is 0.648 times the calibration error
under SSIM and 1.867 times it under $\\ell_2$. When two full-reference metrics
disagree that strongly, neither can be assumed to stand in for a task
measurement. So the task error is computed directly.

Three measurements, chosen to match what the four BioSR structures are actually
imaged for:

* ``puncta_count`` -- local maxima above threshold. The natural readout for
  CCPs, which are counted, not measured.
* ``skeleton_length`` -- skeleton pixel count after binarisation, proportional
  to total filament length. The readout for microtubules and F-actin.
* ``branch_points`` -- skeleton pixels with three or more skeleton neighbours,
  a connectivity readout for the ER network.

All three are computed on every image regardless of structure, so the analysis
can select per structure without a second extraction pass.

Thresholding
------------
Each image is thresholded **independently**, by Otsu on its own intensities.
This is deliberate and it is the realistic choice: a biologist thresholds the
image in front of them, not the ground truth they do not have. Using a
GT-derived threshold on the reconstruction would leak the answer into the
measurement and flatter every method. The cost is that threshold variability
becomes part of the measured task error, which is correct -- it is part of the
real task error too.

Inputs are expected to be scale-matched already, as for SSIM: deconvolution
recovers the object only up to a global intensity scale.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "TASK_NAMES",
    "task_measurements",
    "relative_task_error",
]

TASK_NAMES = ("puncta_count", "skeleton_length", "branch_points")

# Minimum separation between counted maxima, in pixels. Below the diffraction
# limit two maxima are not resolvable and counting both would be an artefact of
# noise, not a detection. At the BioSR GT sampling of 31.3 nm and an emission
# PSF FWHM of roughly 160 nm, three pixels is comfortably sub-resolution.
MIN_PEAK_SEPARATION_PX = 3


def _to_numpy(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    a = np.asarray(img, dtype=np.float64)
    while a.ndim > 2:
        a = a[0]
    return a


def _binarise(a: np.ndarray) -> np.ndarray:
    """Otsu threshold, computed on the image itself.

    Returns an all-false mask for a degenerate image rather than raising: a
    reconstruction that has collapsed to a constant is a legitimate (very bad)
    outcome, and it should score as zero structure, not crash the extraction.
    """
    from skimage.filters import threshold_otsu

    finite = a[np.isfinite(a)]
    if finite.size == 0 or finite.max() <= finite.min():
        return np.zeros(a.shape, dtype=bool)
    try:
        t = threshold_otsu(finite)
    except ValueError:  # pragma: no cover - degenerate histogram
        return np.zeros(a.shape, dtype=bool)
    return a > t


def task_measurements(img) -> dict[str, float]:
    """Compute all three task readouts on one image."""
    from skimage.feature import peak_local_max
    from skimage.morphology import skeletonize

    a = _to_numpy(img)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mask = _binarise(a)

    if not mask.any():
        return {k: 0.0 for k in TASK_NAMES}

    peaks = peak_local_max(
        a,
        min_distance=MIN_PEAK_SEPARATION_PX,
        labels=mask,
        exclude_border=False,
    )
    skel = skeletonize(mask)

    # A skeleton pixel with >= 3 skeleton neighbours is a junction. Counting
    # neighbours by convolution with a 3x3 ones kernel includes the pixel
    # itself, hence the >= 4 test.
    from scipy.ndimage import convolve

    neigh = convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                     mode="constant", cval=0)
    branches = int(np.count_nonzero(skel & (neigh >= 4)))

    return {
        "puncta_count": float(len(peaks)),
        "skeleton_length": float(np.count_nonzero(skel)),
        "branch_points": float(branches),
    }


def relative_task_error(estimate, reference) -> dict[str, float]:
    """Relative error in each task measurement, ``|T(xhat) - T(x)| / T(x)``.

    When the reference measurement is zero the relative error is undefined. It
    is reported as NaN rather than as zero or infinity, so downstream analysis
    can drop those cells explicitly instead of silently averaging in a number
    that means nothing.
    """
    te = task_measurements(estimate)
    tr = task_measurements(reference)
    out: dict[str, float] = {}
    for k in TASK_NAMES:
        denom = tr[k]
        out[k] = abs(te[k] - denom) / denom if denom > 0 else float("nan")
    return out

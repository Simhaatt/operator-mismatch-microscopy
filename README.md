# Calibrated Reliability of Microscopy Reconstruction under Forward-Operator Shift

A fluorescence-microscopy restoration model can return an image that looks
convincing even when the microscope's true optical response differs from the one
the model assumed. This repository asks whether we can estimate, **without clean
ground truth**, how unreliable that reconstruction is -- with calibrated bounds
that survive optical-model shift.

The contribution is not a better reconstructor. It is a post-hoc reliability
layer that attaches to any reconstructor, plus an honest account of when
calibration breaks.

## Why this framing and not a broader one

Calibrated no-reference error bounds for image reconstruction is a **crowded**
field (Wen/Ahmad/Schniter TMLR 2025 and ECCV 2024; QUTCC, which already covers
microscopy; CHEM; conformal failure masks for endoscopic super-resolution). Two
parts of the obvious framing are already taken:

* The **null-space theory** is published. Bhadra, Kelkar, Brooks & Anastasio,
  *On Hallucinations in Tomographic Image Reconstruction*, IEEE TMI 40(11)
  3249-3260 (2021) formally decomposes reconstruction error into measurement and
  null-space components and states the null component cannot be stably estimated
  from measurement data alone. We cite it as our foundation.
* **Kernel-weighted shift-adaptive calibration** is published, and the
  deterministic form is known to be *broken*: Hore & Barber
  ([arXiv:2310.07850](https://arxiv.org/abs/2310.07850)) note that deterministic
  localized conformal prediction has marginal coverage that "may be arbitrarily
  high or arbitrarily low". Their randomized variant (RLCP) fixes it, and its
  Theorem 3 already bounds coverage loss under covariate shift.

What is *not* taken: **every one of those methods assumes the forward operator is
known and fixed.** That is the gap this project targets.

## Status

Physics foundation complete and validated. Gates 1 and 2 **pass**.

### Verified physics

| Property | Result |
|---|---|
| PSF vs analytic Airy pattern | cosine similarity 1.000000, total L1 error 0.4% |
| FWHM scaling | tracks 0.5145 lambda/NA within 0.8-3.3%, lambda 488-640 nm, NA 0.9-1.4 |
| Strehl at the Marechal limit (lambda/14) | 0.8148, matching exp(-(2 pi sigma)^2) to 0.3% |
| Blur adjoint `<Hx,z> = <x,H^T z>` | exact to 3.7e-16 across four padding modes |
| Poisson-Gaussian variance | 53.991 vs theoretical `rate + read^2` = 54.000 |
| Seed reproducibility | stable across `PYTHONHASHSEED` (blake2b, not `hash()`) |

Two real bugs the tests caught, both silent in an image but fatal downstream:

1. **Even oversampling shifted every PSF by half a pixel.** A 2x2 binning box
   cannot be centred on the peak sample, so the PSF lost its symmetry (asymmetry
   0.07, cosine similarity to Airy only 0.83). Odd oversampling is now required.
2. **The blur adjoint was wrong for reflect padding.** `H = Crop . Conv . Pad`,
   so `H^T = Pad^T . Conv^T . Crop^T`, and `Pad^T` is a scatter-add *fold*, not
   another pad. The shortcut broke the adjoint identity by 1.5e-3 -- enough to
   move Richardson-Lucy's fixed point and corrupt unrolled-network gradients.

### The operator is derived from the data, not chosen

BioSR's high-resolution targets are themselves SIM reconstructions, hence already
band-limited. If the simulated cutoff sat above the object's band limit, the
hallucination story would be an artefact of the data pipeline. Measured by
`scripts/inspect_objects.py`:

* Object band edge: **0.379 cycles/pixel** (76% of pixel Nyquist).
* Matching the real ~2x linear-SIM resolution gain gives **NA = 0.794** (gain
  1.99x), leaving **11.6% of object energy in the operator's null space**.
* Had we used NA 1.4, only **1.5%** would be -- the study would have been nearly
  circular.

The operator is **incoherent**: its transfer function is the OTF (pupil
autocorrelation), so the support radius is `2 NA / lambda`, not the amplitude
cutoff `NA / lambda`. Confusing the two overstates the null-space band by exactly
a factor of two.

### Gate 1 and Gate 2 results

At severe aberration (lambda/7 RMS, Strehl ~0.42), with the reconstructor always
assuming the nominal PSF:

* Reconstruction error **doubles** (1.97x-2.05x, consistent across defocus,
  astigmatism, coma and spherical) while the nominal residual moves only 2-4%.
* Per-image discriminability of nominal vs severe: **d(error) 2.35-2.62** against
  **d(residual) 0.13-0.24** -- an **11-18x gap**.

The *mechanism* is stronger than the correlation:

* Predicted shot-noise floor of the relative residual: **0.0810**
* Observed residual, nominal operator: **0.0876**
* Observed residual, worst operator: **0.0938**
* The entire operator-mismatch signal is **7.7% of the noise floor.**

So the measurement residual is not merely weakly informative: the model-error
projection into the measurement domain falls *below the photon noise floor*.
This is the quantitative form of Bhadra et al.'s null-space argument, and it
predicts the residual should become informative only at high photon counts --
tested across BioSR's real 15-600 photon range in `experiments/gate1/`.

## Layout

```text
src/optics/         zernike.py (OSA/ANSI, RMS-normalised), psf.py (pupil
                    synthesis with sampling guards), forward_model.py
                    (Poisson-Gaussian, exact adjoint), aberrations.py
                    (Marechal-anchored severity ladder)
src/reconstruction/ rl.py (Richardson-Lucy with the true H^T 1 normaliser)
src/metrics/        image.py (SSIM, PSNR, radial power spectrum, scale match)
scripts/            inspect_objects.py, gate1_phenomenon.py
tests/              test_optics.py -- 130 tests pinning the physics
```

## Reproducing

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/inspect_objects.py
python scripts/gate1_phenomenon.py --n-objects 50
```

Data is not committed; see [data/README.md](data/README.md) for fetch commands
and licensing. Measurements are regenerated from deterministic seeds rather than
stored, because the full operator-shift matrix would run to hundreds of GB
against a 20 GB Kaggle working disk.

## Open items

* **Gate 0** (still owed): a systematic novelty sweep, including the astronomy
  and adaptive-optics vocabulary, plus full-text reads of CHEM, QUTCC and
  BayesDL-SIM. The audit lane assigned to this returned nothing.
* **Gate 3** (the likely kill shot): `r_k` is computable from `y` and `x_hat`
  alone, so the PSF-family profile is a hand-crafted feature of data a learner
  already sees. It must beat a predictor trained on raw `(y, x_hat)`, or be
  reframed as an interpretable inductive bias rather than an information gain.
* **Unconfirmed:** BioSR's NA, emission wavelengths and camera pixel size. The
  commonly cited 62.6 nm pixel is the current default and must be confirmed
  before publication.
* **W2S licensing** must be resolved before mirroring that dataset.
* The final PSF library should move to the vectorial Richards-Wolf propagator of
  `psf-generator` (Liu et al., [arXiv:2502.03170](https://arxiv.org/abs/2502.03170)),
  whose authors report the scalar model is only 1.5-3x faster than scalar.

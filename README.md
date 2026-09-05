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

So the measurement residual is not merely weakly informative: at BioSR's mid
photon levels the model-error projection into the measurement domain falls
*below the photon noise floor*. That is the quantitative form of Bhadra et al.'s
null-space argument.

### Photon sweep: the mechanism is real but incomplete

The noise-floor explanation makes a falsifiable prediction -- raise the photon
budget, the floor drops as 1/sqrt(N), and the residual must become informative.
Swept over BioSR's real 15-600 range plus one probe beyond it (50 objects per
level, `experiments/gate1/`):

| photons | RL iters | noise floor | signal/floor | d(error) | d(residual) | ratio |
|---|---|---|---|---|---|---|
| 15   | 30  | 0.2058 | 0.0015 | 2.47 | 0.01 | 176x |
| 50   | 50  | 0.1164 | 0.0557 | 2.83 | 0.08 | 34x |
| 100  | 100 | 0.0826 | 0.0849 | 3.04 | 0.14 | 21x |
| 300  | 150 | 0.0478 | 0.1033 | 3.11 | 0.24 | 13x |
| 600  | 150 | 0.0338 | 0.0982 | 3.09 | 0.27 | 12x |
| 3000 | 150 | 0.0151 | 0.0923 | 3.07 | 0.24 | 13x |

**Confirmed:** the absolute error increase from severe aberration is essentially
constant across the whole range (+0.208, +0.224, +0.231, +0.234, +0.233, +0.233).
The aberration adds a fixed amount of structural error; the error *ratio* is
compressed at low photon counts only because shot noise inflates the baseline.
This is why Gate 1 tests effect size rather than a ratio.

**Partly refuted:** the residual does become more informative as the floor drops
(d(residual) 0.01 -> 0.27 from 15 to 600 photons), so shot noise is genuinely
*a* mechanism -- but it then **saturates**. Even at a 13.6x lower noise floor,
d(residual) never exceeds ~0.3 while d(error) sits at ~3.1. Shot noise alone
therefore does **not** explain the blindness; a second, noise-independent
component is present.

**Confound resolved -- and the mechanism is stronger than expected.** `n_iter`
hit the selection grid's ceiling of 150 at 300 photons and above, so those
reconstructions were under-converged and the saturation coincided exactly with
that cap. Forcing the iteration count at 3000 photons settles it
(`experiments/convergence/`):

| n_iter | err(nom) | resid(nom) | resid / noise floor | d(error) | d(residual) | ratio |
|---|---|---|---|---|---|---|
| 50   | 0.1905 | 0.0190 | 1.284 | 2.15 | 0.70 | 3.1x |
| 150  | 0.1603 | 0.0159 | 1.074 | 2.46 | 0.23 | 10.9x |
| 400  | 0.1448 | 0.0152 | 1.027 | 2.67 | 0.38 | 7.1x |
| 1000 | 0.1374 | 0.0150 | **1.014** | 2.83 | 0.40 | 7.1x |

Under-convergence does **not** explain the blindness: at `n_iter = 1000`, where
the reconstruction is well converged, d(residual) is still only 0.40 against
d(error) 2.83.

The decisive quantity is the last column of the third block: **Richardson-Lucy
drives its own residual asymptotically onto the shot-noise floor**, reaching
1.014x it. That is structural rather than incidental -- RL's update *is* a
data-fidelity minimisation, so it reduces `||H_0 x_hat - y||` toward the noise
floor whether or not `H_0` is the right operator, absorbing the operator error
into `x_hat` where it lands in the null space and becomes invisible.

So the residual is **doubly blind**: bounded below by shot noise, and actively
driven onto that bound by the reconstructor's own objective. This argument
extends to any method with an explicit data-fidelity term -- Richardson-Lucy,
Poisson-TV, and unrolled networks with data consistency -- which covers three of
the four planned reconstructors. Whether a purely learned reconstructor (a U-Net
with no explicit fidelity term) behaves the same way is **untested and worth
checking**, since it is the one case where the residual might retain signal.

Caveat: d(residual) is non-monotone in `n_iter` (0.70, 0.23, 0.38, 0.40). With 20
objects those estimates are noisy; a 50-object confirmation is in
`experiments/convergence/n50_*`. The `resid / floor` trend and the size of the
d(error) : d(residual) gap are stable regardless.

**Side finding:** coma is the most residual-visible aberration
(d(residual) 0.53 at 3000 photons, against 0.02-0.30 for the others) and is the
single cell that fails Gate 1 there. It is the only *asymmetric* mode of the
four, so it leaves a measurement-domain signature the residual can partly see.
Symmetric aberrations stay near-invisible.

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

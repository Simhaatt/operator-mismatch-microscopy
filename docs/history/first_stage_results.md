# First-stage results and research narrative

> Archived from the repository README of release v1.0.0. The numbers below are
> the **first BioSR stage** (228 cells, Richardson-Lucy, 14/19-feature caches)
> and predate the manuscript; they are kept because the retractions and the
> verified-physics record are informative. For the published numbers, see the
> top-level README and `paper/results/claims_report.md`.

A fluorescence-microscopy restoration model can return an image that looks
convincing even when the microscope's true optical response differs from the one
the model assumed. This project asks whether we can bound how wrong that
reconstruction is, **without clean ground truth**, with calibration that survives
optical and specimen shift.

The contribution is not a better reconstructor. It is a post-hoc reliability
layer, plus an honest account of when calibration breaks.

## Headline results

The numbers below are the first BioSR stage (228 cells, four structures),
Richardson-Lucy reconstruction, verified optics, CPU only. The manuscript extends
this to a second instrument and four reconstruction methods.

**1. Physics-derived features beat the obvious baseline, and are far more robust.**
Predicting reconstruction error from a single `(y, x_hat)` observation, with
identical model class and conformal calibration:

| photons | predictor | test rho | bound width | worst-group coverage |
|---|---|---|---|---|
| 15 | plain residual | +0.267 | 0.6405 | 0.486 |
| 15 | **PSF-family profile** | **+0.971** | **0.4749** | **0.800** |
| 100 | plain residual | +0.748 | 0.5125 | 0.600 |
| 100 | **PSF-family profile** | **+0.961** | **0.3924** | **0.807** |

Bounds are **23-26% tighter at equal coverage**. The sharper point is stability:
the plain residual's usefulness collapses as photons drop (rho 0.748 -> 0.267)
while the profile is flat (0.961 -> 0.971). Fifteen photons is where live-cell
imaging actually operates, so the naive baseline fails exactly where it matters.

This also settles the strongest objection to the method: `r_k` is computable from
`(y, x_hat)` alone, so the profile could have been redundant with what a learner
extracts directly. It is not.

**2. Conformal calibration collapses under shift.** On a held-out biological
structure, coverage falls to **0.27-0.41 against a nominal 0.90**, worst-group
coverage near zero, for both predictors. This is standard methodology applied
exactly as the literature applies it, and it is the gap the novelty audit
identified as open: every conformal imaging paper assumes calibration and test
are exchangeable.

Whether shift-aware calibration repairs this is the question the manuscript
answers: conformity-score shift is a reliable *post-hoc* diagnostic of coverage
degradation, while the tested label-free observables alone are not sufficient to
anticipate it dependably on real measurements.

## What was tried and did not survive

Recorded because the retractions are as informative as the results.

* **"The measurement residual is blind to operator mismatch."** Withdrawn. Holding
  a specimen fixed and varying the operator, the residual tracks error at
  rho 0.89-0.92 above 100 photons. The blindness is real but **photon-dependent**,
  holding only in the photon-starved regime (rho -0.39 at 15 photons).
* **"The residual is doubly blind -- floored by shot noise and driven onto that
  floor by the reconstructor's own objective."** The mechanism is real (RL drives
  its residual to 0.994x the shot-noise floor) but does not imply blindness, and
  the claim was too strong.
* **"The Gate 2 failure is a kernel-truncation artifact."** Wrong. Truncation was
  a genuine artifact and is now guarded against, but the failure survived fixing it.
* **"The correlation is an artifact of pooling across severity cells."** Wrong.
  Within-cell rho is +0.77.

## Verified physics

134 tests. Each pins a property that would otherwise invalidate downstream claims.

| Property | Result |
|---|---|
| PSF vs analytic Airy pattern | cosine similarity 1.000000, L1 error 0.4% |
| FWHM scaling | 0.5145 lambda/NA within 0.8-3.3%, lambda 488-640 nm, NA 0.9-1.4 |
| Strehl at the Marechal limit | 0.8148 at lambda/14, matching exp(-(2 pi sigma)^2) to 0.3% |
| Blur adjoint `<Hx,z> = <x,H^T z>` | exact to 3.7e-16, all four padding modes |
| Poisson-Gaussian variance | 53.991 vs theoretical `rate + read^2` = 54.000 |
| Seed reproducibility | stable across `PYTHONHASHSEED` and chunk size |

Four real bugs caught, all silent in a rendered image:

1. **Even oversampling shifted every PSF by half a pixel.** A 2x2 binning box
   cannot be centred on the peak sample, so PSFs lost their symmetry (cosine
   similarity to Airy 0.83, not 1.0). Odd oversampling is now required.
2. **The blur adjoint was wrong under reflect padding.** `H = Crop . Conv . Pad`,
   so `H^T = Pad^T . Conv^T . Crop^T`, and `Pad^T` is a scatter-add fold, not
   another pad. The shortcut broke the adjoint identity by 1.5e-3 -- enough to
   move Richardson-Lucy's fixed point.
3. **A factor-of-two pixel size.** The widely quoted 62.6 nm BioSR pixel is the
   *widefield* pixel; SIM ground truth is 2x upsampled, so objects are at 31.3 nm.
4. **Severity-correlated PSF truncation.** Aberrated PSFs have heavier tails, so a
   short kernel discards more of them; after renormalisation the aberrated
   operator became artificially compact, manufacturing a residual/severity
   correlation. At 31.3 nm a 63-px kernel loses 13 points of energy at extreme
   aberration; 191 px keeps the spread under 0.6.

## The operator is derived from the data, then confirmed against the paper

BioSR's ground truth is itself a SIM reconstruction, hence already band-limited.
If the simulated cutoff sat above the object's band limit there would be no
genuine structure for a deconvolver to invent, and the whole study would be
circular. Measured (`scripts/inspect_objects.py`):

* Object band edge **0.379 cycles/pixel**, 76% of pixel Nyquist.
* At the verified optics (NA 1.7, 31.3 nm, 525 nm) the widefield OTF cutoff is
  **0.2027 cyc/px**, leaving **9.05%** of object energy in the operator's null
  space, with a measured SIM gain of 1.87x against the theoretical 2x.

Acquisition parameters are verified, not assumed: NA 1.7 (Olympus
APON100XHOTIRF) and 405/488/560 nm from the Methods of Qiao et al., Nature
Methods 18, 194-202 (2021); pixel sizes read directly from the MRC headers.

Note the operator is **incoherent**: its transfer function is the OTF, the pupil
autocorrelation, so support is `2 NA / lambda`, not the amplitude cutoff
`NA / lambda`. Confusing the two overstates the null-space band by exactly 2x.

## Positioning

Three parts of the obvious framing are already taken, and the paper must not
claim them:

| Piece | Owner |
|---|---|
| Null-space theory of hallucination | Bhadra et al., IEEE TMI 40(11) 2021 |
| Re-blur-and-compare in microscopy | SQUIRREL, Nature Methods 2018 |
| Kernel-weighted shift-adaptive calibration | Guan's LCP -- and the deterministic form is provably broken; use Hore & Barber's RLCP |
| PSF error propagation | Astronomy, since Paulin-Henriksson 2008 |

What survives: **every existing calibrated-UQ method for imaging assumes the
forward operator is known and fixed.** See [docs/gate0_novelty.md](../gate0_novelty.md)
for the full audit, including a named competitive risk.

## Workflow

```text
Gate 0  Novelty audit ........................... PASS (narrowed)
Gate 1  Does mismatch cause plausible-but-wrong output? .. PASS
Gate 2  Is the trivial residual already enough? ... FAILED above 100 photons
Gate 3  Does the PSF-family profile beat it? ...... PASS (23-26% tighter)
Gate 4  Does shift-aware calibration repair coverage? .... COMPLETE
        [generalised beyond Richardson-Lucy; second instrument added]
```

The ordering is the point. Gates 1-3 cost CPU-hours; Gate 4 and the reconstructor
sweep cost ~95 T4-hours. Establishing the phenomenon first is what prevents
spending those hours on a dead hypothesis -- and Gate 2 did kill the original one.

## Still open

* **Anticipating coverage loss before it happens.** Conformity-score shift
  diagnoses degradation after the fact; the label-free observables tested here do
  not reliably predict it in advance on real measurements.
* **A limitation to report:** the profile degrades *more* than the plain residual
  under content shift (ood rho 0.49 vs 0.76), so richer features are more tied to
  training-structure content.


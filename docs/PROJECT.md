# Calibrated Reliability of Microscopy Reconstruction under Forward-Operator Shift

A complete description of the project: what it asks, why the question is hard,
what has been established, and what remains.

---

## 1. The question

A fluorescence microscope does not record your specimen. It records the specimen
blurred by the optics and corrupted by photon noise:

```
y  ~  Poisson( eta * (h * x) + b )  +  Normal(0, sigma_r^2)
```

where `x` is the underlying structure, `h` the point-spread function (PSF), `*`
convolution, and `y` the camera image.

Recovering `x` from `y` is deconvolution, and every deconvolution method must
**assume** a PSF, call it `h_0`. In practice the true PSF `h*` drifts away from
it: refractive-index mismatch between immersion oil and specimen, coverslip
thickness variation, focus drift, temperature, wavelength, depth into the sample.

The failure mode this creates is the dangerous kind. The reconstruction still
looks like a cell -- sharp filaments, plausible texture, publication-quality. It
is simply partly invented.

> **Can we bound how wrong a reconstruction is, without ever seeing the ground
> truth, with calibration that survives the optics having drifted?**

The deliverable is not a better reconstructor. It is a post-hoc reliability
layer that attaches to any reconstructor, plus an honest account of when its
guarantees break.

---

## 2. Why the obvious approach fails

A microscope is a low-pass filter. Fluorescence imaging is **incoherent**, so the
measurement operator is the optical transfer function -- the autocorrelation of
the pupil -- with support radius `2 NA / lambda`. Above that frequency the
instrument transmits *literally nothing*. That band is the operator's **null
space**.

Deconvolution methods invent content in exactly that band. And because the
measurement carries no information there, **no measurement-domain check can see
the invention**. This is the formal content of Bhadra et al. (IEEE TMI 2021),
who decompose the estimate into measurement and null-space components and show
the null part "cannot be stably estimated from the measurement data alone".

For our data that band is large, and we measured it rather than assuming it: at
the verified BioSR optics, **9.05% of object energy lies above the widefield
cutoff**. Not a corner case.

The obvious baseline is to re-blur the reconstruction and compare against the
measurement -- the residual `||H_0 x_hat - y||`. Two things compromise it:

1. **It is floored by shot noise.** At BioSR's real photon levels the entire
   operator-mismatch signal is 0.2-8% of the noise floor.
2. **The reconstructor drives it to that floor.** Richardson-Lucy *is* a
   data-fidelity minimisation, so it shrinks the residual whether or not `H_0` is
   right, absorbing operator error into `x_hat`. Measured: RL reaches **0.994x**
   the predicted shot-noise floor -- it fits straight through it.

Point 2 is a claim about a *class* of algorithms, which is why the project tests
four reconstructors spanning classical/learned and fidelity-minimising/not.

---

## 3. Where this sits in the literature

Three parts of the obvious framing are already taken. They are background to
cite, not contributions to claim.

| Piece | Owner |
|---|---|
| Null-space theory of hallucination | Bhadra, Kelkar, Brooks & Anastasio, IEEE TMI 40(11) 2021 |
| Re-blur-and-compare in microscopy | SQUIRREL, Culley et al., Nature Methods 2018 |
| Kernel-weighted shift-adaptive calibration | Guan's localised conformal prediction -- and the deterministic form is *provably broken*; the randomised repair is Hore & Barber, arXiv:2310.07850 |
| PSF error propagation | Astronomy, since Paulin-Henriksson (2008); reviewed by Liaudat, Starck & Kilbinger (2023) |

Calibrated UQ for imaging inverse problems is a **crowded** field: Wen, Ahmad &
Schniter (TMLR 2025, ECCV 2024, and a 2025 minimax follow-up), QUTCC (which
already covers microscopy), CHEM, conformal failure masks for endoscopic
super-resolution, conformalised generative Bayesian imaging, and more.

**What survives:** every one of those methods assumes the forward operator is
**known and fixed**. That, plus microscopy, plus per-image bounds, is the wedge.

A named competitive risk, recorded during the audit: Starck co-authors CHEM
(conformal + hallucination), and Liaudat co-authors both the astronomy
PSF-modelling review and a self-supervised conformal imaging paper. One research
orbit holds all three ingredients and is one step from this project, in the
astronomy domain.

Full audit: [gate0_novelty.md](gate0_novelty.md).

---

## 4. Method

The reliability layer sits **after** any reconstructor and requires no retraining.

1. **Probe with an operator family.** Instead of a single assumed PSF, define
   candidates `{H_1 ... H_K}` spanning plausible microscope states and re-image
   the reconstruction under each: `r_k = ||y - H_k x_hat|| / ||y||`, with a
   per-probe least-squares gain since deconvolution fixes the object only up to
   scale.
2. **Extract a signature** from that profile: minimum, location and sharpness of
   the optimum, spread, and the residual's spectrum by frequency ring.
3. **Add OTF-consistency features** -- the physics-specific part. With the
   correct operator and only Poisson-Gaussian noise the residual is
   approximately white. Under mismatch it also contains `(H* - H_0) x`, whose
   power concentrates near the cutoff. Crucially, **above** the cutoff `H_0 x_hat`
   contributes nothing, so the residual there is pure noise and serves as a noise
   reference *derived from the same image* -- no calibration set, no ground
   truth. In-band excess measured against it is insensitive to photon level.
4. **Predict the error** `e = 1 - SSIM(x_hat, x)` with a small quantile model.
5. **Calibrate** into a bound with `P[e <= U] ~ 0.90`.

An objection this design must answer: `r_k` is computable from `(y, x_hat)`
alone, so the profile is a hand-crafted feature of data a learner already sees.
It has to earn its place as a useful inductive bias, not an information gain.
Gate 3 tests exactly that.

---

## 5. Experimental design

Deliberately **gated**, cheap first, so the project can die before expensive work.

```
Gate 0  Novelty audit .............................. PASS (narrowed)
Gate 1  Does mismatch cause plausible-but-wrong output? .. PASS
Gate 2  Is the trivial residual already enough? ..... FAILED above 100 photons
Gate 3  Does the operator profile beat it? .......... PASS (23-26% tighter)
Gate 4  Does shift-aware calibration repair coverage? . FAILED, diagnosed
Gate 4b Does the diagnosis suggest a fix? ........... PARTIAL (operator yes, content no)
```

Gates 1-4 cost CPU-hours. The reconstructor sweep costs GPU-hours. Running them
in this order is what prevented months spent on a hypothesis that Gate 2 killed.

**Splits are blocked by cell, never by patch**, and one biological structure is
held out entirely as a content-shift axis. Patches from one field of view share
structure, background and noise realisation, so splitting after cropping would
leak the test set into training.

---

## 6. Results

All on real BioSR data -- 228 cells, four structures -- with Richardson-Lucy
reconstruction at verified optics. No GPU used.

### 6.1 The phenomenon exists (Gate 1)

At severe aberration (lambda/7 RMS, Strehl 0.42), reconstruction error roughly
doubles while the nominal residual moves 2-4%. Per-image discriminability of
nominal versus severe: **d(error) 2.4-3.2** against **d(residual) 0.03-0.46**.

### 6.2 The trivial residual is not enough -- but less cleanly than hoped (Gate 2)

Holding a specimen fixed and varying only the operator, the residual tracks error
at **rho 0.89-0.92 above 100 photons**. Kill criterion B genuinely fires there.
The blindness is real but **photon-dependent**, holding only in the
photon-starved regime (rho -0.39 at 15 photons).

### 6.3 The operator profile earns its place (Gate 3)

Single-observation prediction, identical model class and conformal calibration:

| photons | predictor | test rho | bound width | worst-group coverage |
|---|---|---|---|---|
| 15 | plain residual | +0.267 | 0.6405 | 0.486 |
| 15 | **profile** | **+0.971** | **0.4749** | **0.800** |
| 100 | plain residual | +0.748 | 0.5125 | 0.600 |
| 100 | **profile** | **+0.961** | **0.3924** | **0.807** |

Bounds **23-26% tighter at equal coverage**. The sharper point is stability: the
plain residual collapses as photons drop (0.748 -> 0.267) while the profile is
flat (0.961 -> 0.971). Fifteen photons is where live-cell imaging operates, so
the naive baseline fails exactly where it matters.

### 6.4 Calibration collapses under shift (Gate 4)

Nominal coverage 0.90:

| shift axis | split conformal | Mondrian | RLCP |
|---|---|---|---|
| none | 0.911 | 0.896 | 0.908 |
| operator | **0.850** | 0.736 | **0.325** |
| content | 0.474 | 0.478 | 0.474 |
| combined | 0.060 | 0.060 | 0.042 |

**The shift-aware schemes are worse than doing nothing.** Diagnosis: under
operator shift **93.6% of test points lie beyond the calibration 95th percentile**
of the mismatch score, so a kernel localiser has almost no neighbours to borrow
from. It concentrates weight on a handful of extreme points and returns a
*narrower* quantile exactly where it should widen. Under content shift the
failure inverts -- novel specimens score as *benign* -- because the localiser was
trained on operator severity, and severity is not novelty.

### 6.5 The diagnosis suggests a partial fix (Gate 4b)

Replacing the severity score with a **novelty** score (distance from the
calibration distribution):

| axis | severity localiser | novelty localiser |
|---|---|---|
| operator shift, 100 ph | 0.325 | **0.750** |
| operator shift, 15 ph | 0.332 | **0.768** |
| content shift, 100 ph | 0.474 | **0.696** |

And operator shift turns out repairable almost free: a stricter quantile
(alpha 0.02) restores **0.900 coverage for +4% width**.

Two negatives that are results in their own right:

* **Content shift is not repairable from calibration data.** At alpha = 0.001 it
  still reaches only 0.821. The calibration set does not contain the failure
  mode, and no quantile of it can conjure one.
* **Abstaining on novelty backfires**, monotonically: keeping only the *most
  familiar-looking* held-out inputs gives 0.052 coverage against 0.474 for
  keeping everything. Sparse punctate structures resemble easy training cases, so
  the model issues narrow bounds while true error is high. **The dangerous cases
  are the ones that look familiar** -- a direct warning about OOD detection as a
  safety mechanism, and one that generalises beyond microscopy.

Full detail: [gate4_results.md](gate4_results.md).

### 6.6 Coverage loss can be bounded, one-sidedly (Gate 5)

Split conformal fixes a threshold at the (1-alpha) quantile of the calibration
conformity scores. For the acceptance event `A = {s <= q}`, the definition of
total variation gives `|Q(A) - P(A)| <= d_TV(P, Q)`, hence

    coverage_test >= 1 - alpha - d_TV(P, Q).

Elementary, but the right statement for this setting: the shift that matters is
in the *conformity score* distribution, not in the images or the features. It
predicts that a large content shift which happens not to move the score
distribution is harmless -- which is exactly what holding out ER does.

The contribution is that **the symmetric version is useless and the one-sided
version is near-tight**. Only mass crossing *above* the threshold costs
coverage; a shift that moves scores down makes the threshold conservative and
coverage rises. So take `d+ = sup_t [Q(s>t) - P(s>t)]_+`. Over 10 partition
seeds on 16 cells:

| quantity | value |
|---|---|
| `rho(coverage, -d+)` | **+0.952 +- 0.025** |
| `rho(coverage, -d_TV)` | -0.064 +- 0.090 |
| one-sided bound holds | **16/16 on every seed** |
| mean slack | +0.046 +- 0.006 |

The symmetric statistic is indistinguishable from zero -- not merely weaker,
*uninformative*. One-sidedness is the whole result.

The bound also survives conditions far outside the simulated grid: it holds on
real measurements where realised coverage collapses to 0.150 (section 6.8), and
across all four reconstructors including two learned ones (section 6.9).

### 6.7 The blind estimator is what limits the contribution (Gate 6)

The bound needs labels. The deployable question is whether coverage loss can be
*anticipated*, so `d+` is replaced by a label-free proxy: the divergence between
calibration and test distributions of some blind statistic.

Seven candidates, 16 cells, 10 seeds, at the measured 31% out-of-focus fraction:

| blind statistic | `rho(coverage, -blind d+)` |
|---|---|
| `qpred` -- the fitted model's own predicted quantile | **+0.616 +- 0.044** |
| all 19 features (Mahalanobis) | +0.450 +- 0.065 |
| the 14 OSCR features | +0.375 +- 0.078 |
| the 5 OTF features | +0.276 +- 0.079 |
| `r_nominal` alone | +0.083 +- 0.061 |

Two findings. **More features make a novelty distance worse** -- every
uninformative direction adds variance without signal, so the useful move is a
low-dimensional statistic rather than a richer one. And **the winner is
learned**: `qpred` beats every hand-crafted alternative, clear of seed noise.

The OTF-consistency block does not win, which was the hope. It does improve the
mismatch score's off-distribution correlation with the oracle in 3/3 caches
(+0.466 -> +0.544, +0.645 -> +0.672, +0.444 -> +0.664), but it does not make a
better blind estimator and has no detectable effect on content-shift coverage.
Full detail: [otf_evaluation.md](otf_evaluation.md).

### 6.8 The layer does not transfer to real measurements (Gate 7)

BioSR ships the raw SIM frames, so a real widefield measurement can be built
alongside the ground truth the error metric uses: the three illumination phases
of one orientation sum to zero, so averaging them cancels the pattern exactly.

A predictor calibrated entirely on simulation, applied to real measurements,
gives **coverage 0.150-0.550 against a nominal 0.900** across three forward
models, and the blind warning signal *inverts* (`rho(r_nominal, error)` -0.098
to -0.663). The Gate 5 bound holds in all three cases -- slack +0.199, +0.112,
+0.050 -- so it correctly announces the collapse rather than promising coverage
it cannot deliver. That is the bound's strongest validation, on data the forward
model never produced.

**The diagnosis is a missing physical term.** Fitting each real measurement as
`a * blur(GT) + b` (R^2 0.72-0.96 over 182 cells) puts the constant at **19-68%
of the measurement mean, median 33%**, against the 5% originally simulated. SIM
reconstruction removes out-of-focus light, so a ground truth derived from it
carries none and blurring cannot put it back. Real widefield contrast is
`std/mean = 0.61`; simulated was `2.30`.

This unifies three separate anomalies. CCPs has the highest haze fraction
(0.676) and is independently the structure with the worst spectral realism, the
worst conformal coverage, and the only one at 0.000 sim-to-real coverage. One
mechanism, not three oddities. Full detail: [sim2real.md](sim2real.md),
[realism_check.md](realism_check.md).

### 6.9 A reconstructor with no fidelity term does not break the mechanism (Gate 8)

The standing objection to sections 6.2 and 6.3 was that residual suppression
might be an artefact of a data-fidelity term rather than a property of the
inverse problem. The U-Net has no such term, which made it the decisive case.

It behaves identically to Richardson-Lucy. On the held-out structure, against
the oracle RMS wavefront error: `r_nominal` +0.727 for both, `r_mean` -0.902 vs
-0.908, `r_std` -0.890 vs -0.895, `r_curvature` -0.868 vs -0.871. The scope
does not narrow. The U-Net is also the best of the four reconstructors (mean
error 0.280 against RL 0.309, TV 0.327).

The **unrolled** network turns out to be the mechanism's extreme case. Its
forward pass takes explicit gradient steps on `||H x - y||^2` against the
*nominal* operator, so it minimises precisely what `r_nominal` measures under an
operator that may be wrong. `r_nominal` collapses to +0.168 and `r_min` flips
sign to -0.687 -- while `r_gap` more than doubles to +0.860 and `otf_slope`
reaches +0.868, the highest of any reconstructor. Gate 3's claim demonstrated at
the limit: **the operator profile earns most exactly where the plain residual is
most blind.** Full detail: [gpu_reconstructors.md](gpu_reconstructors.md).

---

## 7. Claims made and withdrawn

Recorded because the retractions are as informative as the results, and because
each was caught by a deliberate check rather than by luck.

* **"The measurement residual is blind to operator mismatch."** Withdrawn. True
  only below ~100 photons.
* **"The residual is doubly blind -- floored by shot noise and driven onto that
  floor by the reconstructor."** The mechanism is real (RL reaches 0.994x the
  floor) but does not imply blindness. Overstated.
* **"The Gate 2 failure is a kernel-truncation artifact."** Wrong -- truncation
  *was* a real artifact, now guarded, but the failure survived fixing it.
* **"The correlation is an artifact of pooling across severity cells."** Wrong;
  within-cell rho is +0.77.
* **"TV is a stronger case than RL."** True at 100 photons, reverses at 15.
* **"Conformal calibration collapses to 0.29 under content shift."** That is the
  worst of four rotations; ER gives 0.97.
* **"The OTF features change content-shift coverage."** Withdrawn in *both*
  directions. Over 10 partition seeds every 14-vs-19 difference is inside the
  seed noise (|diff| <= 0.052 against a pooled sd of 0.10-0.13). The claim and
  its own retraction were each a single-seed reading.
* **"`otf_core` is a better blind estimator."** +0.602 on seed 0, +0.340 +- 0.112
  over ten -- last place. It was selected on the oracle, and this is what that
  costs.
* **"The plain residual is the best blind estimator."** True only at the
  unrealistic 5% background. At the measured 31% it *changes sign*, +0.604 to
  -0.398, from joint best to active anti-predictor. The real data agrees with
  the realistic setting, not the clean one.

**Two standing rules came out of this.** Do not characterise a result until
every cell of the grid is in -- that pattern cost seven corrections. And report
every number over at least 10 partition seeds: conformal coverage is
split-sensitive, and single-seed differences of up to 0.2 on the worst cell are
pure noise.

---

## 8. Correctness

The physics is pinned by tests rather than asserted. **147 tests.**

| Property | Result |
|---|---|
| PSF vs analytic Airy pattern | cosine similarity 1.000000, L1 error 0.4% |
| FWHM scaling | 0.5145 lambda/NA within 0.8-3.3%, lambda 488-640 nm, NA 0.9-1.4 |
| Strehl at the Marechal limit | 0.8148 at lambda/14, matching exp(-(2 pi sigma)^2) to 0.3% |
| Blur adjoint `<Hx,z> = <x,H^T z>` | exact to 3.7e-16, four padding modes |
| Poisson-Gaussian variance | 53.991 vs theoretical 54.000 |
| Seed reproducibility | stable across `PYTHONHASHSEED`, chunk size, and device |

Six real bugs caught, every one of which produced plausible-looking output:

1. **Even oversampling shifted every PSF half a pixel.** A 2x2 binning box cannot
   be centred on the peak sample; cosine similarity to Airy was 0.83, not 1.0.
2. **The blur adjoint was wrong under reflect padding.** `Pad^T` is a scatter-add
   fold, not another pad; the shortcut broke the adjoint identity by 1.5e-3.
3. **A factor-of-two pixel size.** The widely quoted 62.6 nm BioSR pixel is the
   *widefield* pixel; SIM ground truth is 2x upsampled, so objects are 31.3 nm.
4. **Severity-correlated PSF truncation.** Aberrated PSFs have heavier tails, so a
   short kernel discarded 13 percentage points more energy at extreme aberration
   than at nominal -- fabricating a residual/severity correlation.
5. **An inverted FISTA step size** in TV, leaving the estimate frozen at its
   constant initialisation: SSIM 0.043 against 0.862 for the blurred input.
6. **A tuple-unpacking slip** that made the error and residual arrays identical,
   yielding a perfect correlation of 1.000 that looked like a dramatic finding.

Each now has a regression test.

---

## 9. Data

**BioSR** (Qiao & Li, figshare `10.6084/m9.figshare.13264793`, CC BY 4.0).
2,200+ paired images, four structures, nine signal levels quoted as **15-600
average photon count**.

Acquisition parameters are **verified**, from the Methods of Qiao et al. (Nature
Methods 18, 194-202, 2021) and cross-checked against the MRC headers:

| Parameter | Value | Source |
|---|---|---|
| Objective NA | 1.7 | Olympus APON100XHOTIRF, Methods |
| Excitation | 405 / 488 / 560 nm | Methods |
| Camera | Hamamatsu Orca Flash 4.0 v3 | Methods |
| Photon conversion | 0.6026 photons/count | Methods |
| GT-SIM pixel | **0.0313 um** | MRC header |
| Raw/widefield pixel | 0.0626 um | MRC header |

Two layout traps worth knowing:

* **ER is a live-cell dataset** with a different structure -- per-level ground
  truth under `Cell_XXX/GTSIM/` rather than a single `SIM_gt.mrc`. Missing that
  silently drops 68 of 228 cells, 30% of the data.
* **One cell has a wider field of view** (`F-actin/Cell_024`, raw 758 px rather
  than 502) at the *same* pixel size. Cropping to the common size is safe;
  mixing pixel *scales* would not be, and the exporter refuses it.

**The operator is derived from the data, not assumed.** BioSR's ground truth is
itself a SIM reconstruction, hence already band-limited. If the simulated cutoff
sat above the object's own band limit there would be no genuine structure for a
deconvolver to invent, and the study would be circular. Measured band edge is
**0.379 cyc/px**; the widefield OTF cutoff at verified optics is **0.2027**,
leaving 9.05% of object energy in the null space, at a measured SIM gain of 1.87x
against a theoretical 2x.

Only 441 MB of the 8 GB of archives is ever read, so
`scripts/export_compact.py` writes a **303 MB** portable `.npz`; loading from it
is bit-identical to loading from the archives.

---

## 10. Status

**Complete: Gates 0-8.** Physics foundation, data pipeline, four
reconstructors (Richardson-Lucy, Poisson-TV, U-Net, unrolled) all profiled with
the same 19 blind features, the coverage bound, the blind-estimator comparison,
the forward-model realism check, and the sim-to-real transfer test. Everything
runs over 10 partition seeds. The GPU session is done -- one A100, ~45 minutes.

**Nothing is GPU-blocked any more.**

**Remaining:**

| work | where | effort |
|---|---|---|
| Conformal calibrated **on real measurements** -- does it work in-domain? | CPU | ~2.5 h |
| Per-structure haze sensitivity (the spread is 0.404) | CPU | ~1 h |
| `otf_excess_at_cut` split dependence, currently unexplained | CPU | ~1 h |
| Structured haze term (the fit leaves 4-28% unexplained) | CPU | ~5 h |
| Photon sweep to pin Gate 2's crossover | CPU | ~1.75 h |
| Self-supervised / test-time calibration | CPU | ~1 day, **may not work** |

The first is the one open scientific question. Sim-to-real showed the layer does
not transfer *from simulation*; calibrating in-domain either recovers coverage --
a usable result -- or establishes a limit no better simulator would fix.
| Writing | -- | 2-3 weeks |

The U-Net is the scientifically decisive remaining experiment: it has **no
data-fidelity term**, so it is the single case that could overturn the mechanism
established for RL and TV. The unrolled network is the control that separates
"learned versus classical" from "fidelity-minimising versus not", which are
otherwise confounded.

**Target venue: TMLR.** Free, no page limit, and its stated criteria are whether
claims are supported by evidence and whether the findings interest someone --
explicitly *not* novelty or impact. That matches a project whose novelty is
narrow and whose strongest results are negative. SIAM Journal on Imaging Sciences
becomes viable if the coverage bound works; IEEE TCI is the fallback but charges
$175/page beyond ~9 pages.

---

## 11. Reproducing

```bash
pip install -e ".[dev]"
python -m pytest -q                                   # 147 physics and analysis tests
python scripts/inspect_objects.py                     # derive the operator from the data
python scripts/export_compact.py --out data/biosr_gt.npz
python scripts/extract_features.py --photons 100 --n-iter 75 \
    --source data/biosr_gt.npz --out experiments/features/photons_100.npz
python scripts/gate4_calibration.py --features experiments/features/photons_100.npz
python scripts/gate4b_repair.py    --features experiments/features/photons_100.npz
```

Data is not committed; see [../data/README.md](../data/README.md) for fetch
commands and licensing, and [colab.md](colab.md) for the hosted-runtime path.
Measurements are regenerated from deterministic per-image seeds rather than
stored, because the full shift matrix would run to hundreds of GB against a
20 GB Kaggle working disk -- and because per-image seeding makes results
invariant to batch composition, chunk size and device.

## 12. Layout

```text
src/optics/         zernike (OSA/ANSI, RMS-normalised), psf (sampling guards),
                    forward_model (Poisson-Gaussian, exact adjoint),
                    aberrations (Marechal-anchored severity ladder)
src/reconstruction/ rl, tv (classical), unet, unrolled (learned)
src/reliability/    features (operator profile, spectra, OTF consistency)
src/data/           mrc (dependency-free reader), biosr (cell-blocked splits)
src/metrics/        image (SSIM, PSNR, radial spectrum, scale match)
scripts/            inspect_objects, gate1_phenomenon, gate3_deployment,
                    gate4_calibration, gate4b_repair, extract_features,
                    train_reconstructor, export_compact, summarize_gate1
docs/               PROJECT.md, gate0_novelty.md, gate4_results.md, colab.md
```

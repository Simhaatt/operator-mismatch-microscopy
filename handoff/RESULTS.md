# Complete results compendium

Everything measured, with the number, the condition it was measured under, and
what it does and does not support. Written to be read by someone drafting the
manuscript who needs the value and its caveat in the same place.

Provenance: commit `822927b`, test suite 183 passing. Numbers here are quoted
from committed runs under `experiments/`; that directory is gitignored, so this
file and the `docs/` set are the record.

---

## 1. What the project is

A microscope blurs the specimen with a point-spread function (PSF) the
deconvolution software must **assume**. When the true PSF drifts away from the
assumed one, deconvolution still returns a convincing image that is partly
invented. This project builds a post-hoc **reliability layer** that bounds
reconstruction error without ground truth at deployment, and characterises when
its statistical guarantees break.

The distinctive axis is **forward-operator mismatch**. The published literature
on conformal prediction for imaging treats measurement noise and uses simulated
denoising or deblurring benchmarks. Nothing found in the 2025 literature
addresses operator misspecification, and nothing validates on real measurements
with paired ground truth.

## 2. Setup

### Data

- **BioSR**, 228 cells, four structures: CCPs (clathrin-coated pits),
  ER (endoplasmic reticulum), F-actin, Microtubules.
- Archives `data/biosr/*.zip`, 8 GB, MD5-verified. A 303 MB compact export
  `data/biosr_gt.npz` holds the 228 ground-truth images at 1004x1004 uint16;
  loading from it is bit-identical to loading from the archives.
- **ER has a different archive layout** (live-cell: per-level ground truth under
  `Cell_XXX/GTSIM/`). Missing it silently drops 68 of 228 cells.
- `F-actin/Cell_024` has a wider field (1516 px) at the same pixel size. The
  exporter centre-crops to the common 1004 px and says so.

### Verified optics

| quantity | value | source |
|---|---|---|
| numerical aperture | 1.7 | Methods, Qiao et al., Nat Methods 18, 194-202 (2021) |
| emission wavelength | 525 nm (assumed for the 488 line) | same |
| GT-SIM pixel | **0.0313 um** | same, and MRC headers |
| widefield pixel | 0.0626 um | same |
| conversion | 0.6026 photons/count | same |
| widefield OTF cutoff | 6.476 cycles/um | 2 NA / lambda |

**The widely quoted "62.6 nm BioSR pixel" is the WIDEFIELD pixel.** SIM ground
truth is 2x upsampled, so objects sit at 31.3 nm. Using 62.6 understates every
spatial frequency by a factor of two. This error was made once and fixed.

### Splits

Blocked **by cell**, never by patch, 60/20/20, with one structure removed
entirely from train and calibration and returned as the content-shift axis.
Each simulated cache holds 3876 samples: 1768 train, 595 calibration, 595 test,
918 out-of-distribution.

### The 19 blind features

Fourteen operator-consistency (OSCR) features:
`r_nominal`, `r_min`, `r_mean`, `r_std`, `r_gap`, `r_argmin_frac`,
`r_curvature`, `band_0`..`band_3`, `y_mean`, `y_std`, `xhat_std`.

Five OTF-consistency features, added later and evaluated in Gate 6:
`otf_white_chi2`, `otf_excess_inband`, `otf_excess_at_cut`, `otf_slope`,
`otf_noise_ref`.

`r_nominal` is the *relative* residual under the assumed operator. That the
denominator is the measurement matters for retractions 10 and 11.

### Reconstructors

| name | fidelity term | setting |
|---|---|---|
| Richardson-Lucy | implicit, Poisson likelihood | 75 iterations (30 in the original low-photon caches) |
| Poisson-TV | explicit, plus TV prior | 100 iterations, weight 0.02 selected once on the nominal operator |
| U-Net | **none** | 1.95M parameters, val SSIM 0.9245 |
| unrolled | explicit, optimised against the nominal PSF | 20k parameters, 5 stages, val SSIM 0.8888 |

Both networks were trained at haze 0.31, 32 patches per cell, 60 epochs, on one
A100 in about 45 minutes.

---

## 3. The eleven gates

| # | question | verdict |
|---|---|---|
| 0 | Is the novelty gap real? | PASS, narrowed. Competitive risk named and since confirmed |
| 1 | Does mismatch cause plausible-but-wrong output? | PASS |
| 2 | Is the trivial residual already enough? | FAILED. Superseded by Gate 11 |
| 3 | Does the operator profile beat the residual? | **PASS, 23-26% tighter bounds** |
| 4 | Does shift-aware calibration repair coverage? | FAILED, diagnosed |
| 4b | Does the diagnosis suggest a fix? | PARTIAL. Operator shift yes, content shift no |
| 5 | Can coverage loss be bounded? | **PASS, one-sided, near-tight** |
| 6 | Does the OTF block beat the plain residual blind? | FAILED, and the ranking depends on background |
| 7 | Does the layer transfer from simulation to real data? | **FAILED, and diagnosed** |
| 8 | Does a reconstructor with no fidelity term break the mechanism? | **PASS, scope holds** |
| 9 | Does the layer work calibrated on real measurements? | **PASS, coverage is nominal** |
| 10 | Can coverage loss be anticipated blindly on real data? | **FAILED, nothing works** |
| 11 | Is Gate 2's photon dependence real? | **NO, it is background dependence** |

### Gate 1: the phenomenon exists

At severe aberration (lambda/7 RMS wavefront error, Strehl 0.42), reconstruction
error roughly **doubles** while the nominal residual moves only **2-4%**. The
reconstruction remains visually convincing. This is the phenomenon the whole
project exists to detect.

### Gate 3: the operator profile earns its place

Physics-derived features give **23-26% tighter bounds at equal coverage**, and
are stable across photon levels (rho 0.961 to 0.971) where the plain residual
collapses (0.748 to 0.267).

### Gate 4: calibration collapses under shift

Nominal coverage 90%. Split conformal, Mondrian (group-conditional on an
estimated mismatch score) and RLCP (randomised localised conformal prediction,
Hore and Barber, arXiv:2310.07850) on three shift axes. Representative numbers,
RL at 100 photons, 19 features:

| axis | split conformal | Mondrian | RLCP |
|---|---|---|---|
| in-distribution | 0.914 | 0.909 | 0.914 |
| operator shift | 0.850 | 0.668 | 0.346 |
| content shift | 0.521 | 0.521 | 0.515 |
| combined | 0.083 | 0.076 | 0.067 |

**Worst-group coverage under content shift is 0.019 to 0.060 across every cache
and every scheme.** Marginal coverage can look tolerable while the worst
severity stratum is near zero.

Localisation actively harms under operator shift: RLCP at 0.346 against split
conformal's 0.850. Deterministic localisation was not used, because Guan's LCP
has marginal coverage that "may be arbitrarily high or arbitrarily low" and the
randomisation is what restores a guarantee.

### Gate 4b: what a diagnosis buys

Replacing the severity score with a **novelty** score (distance from the
calibration distribution) helps localisation substantially:

| axis | severity localiser | novelty localiser |
|---|---|---|
| operator shift, 100 photons | 0.325 | **0.750** |
| operator shift, 15 photons | 0.332 | **0.768** |
| content shift, 100 photons | 0.474 | **0.696** |

Operator shift is repairable almost free: a stricter quantile (alpha 0.02)
restores **0.900 coverage for +4% width**.

Two negatives that are results in their own right:

- **Content shift is not repairable from calibration data.** At alpha = 0.001 it
  still reaches only 0.869, marked unreachable. No quantile of a calibration set
  can conjure a failure mode the set does not contain.
- **Abstaining on novelty backfires, monotonically.** Keeping only the *most
  familiar-looking* held-out inputs gives 0.079 coverage against 0.521 for
  keeping everything. Sparse punctate structures resemble easy training cases,
  so the model issues narrow bounds while true error is high. **The dangerous
  cases are the ones that look familiar.** This is the most portable finding in
  the project and a direct warning about out-of-distribution detection as a
  safety mechanism.

### Gate 5: the coverage bound

Split conformal fixes a threshold at the (1-alpha) quantile of the calibration
conformity scores. For the acceptance event A = {s <= q}, the definition of
total variation gives |Q(A) - P(A)| <= d_TV(P, Q), and since P(A) >= 1 - alpha
by the usual exchangeable argument,

    coverage_test  >=  1 - alpha - d_TV(P, Q).

Elementary, and the project says so. The **contribution is that the symmetric
form is useless and the one-sided form is near-tight.** Only mass crossing
*above* the threshold costs coverage; a shift moving scores down makes the
threshold conservative and coverage rises. So define

    d+  =  sup_t [ Q(s > t) - P(s > t) ]_+

which is zero for any benign shift however large.

Over 10 partition seeds, 16 cells, the corrected forward model:

| quantity | value |
|---|---|
| rho(coverage, -d+) | **+0.952 +- 0.025** |
| rho(coverage, -d_TV) | **-0.064 +- 0.090** |
| one-sided bound holds | **16/16 on every seed** |
| mean slack | +0.046 +- 0.006 |

The symmetric statistic is indistinguishable from zero. Not weaker,
**uninformative**. That is the whole result: one-sidedness is not a refinement,
it is the difference between a bound that explains the data and one that does
not.

Holding across every setting tested:

| setting | rho(cov, -d+) | bound | cells |
|---|---|---|---|
| simulated, haze 0.048 | +0.908 +- 0.056 | 8/8 | 8 |
| simulated, haze 0.310 | +0.959 +- 0.034 | 8/8 | 8 |
| simulated, haze 0.310, both photon levels | +0.952 +- 0.025 | 16/16 | 16 |
| four reconstructors, haze 0.310 | +0.933 +- 0.027 | 16/16 | 16 |
| **real measurements** | +0.933 +- 0.053 | 16/16 | 16 |

**48 of 48 cells across five settings, on every one of ten seeds.** It also
holds on real data where realised coverage collapses to 0.150 under
simulation-calibration, with slack +0.199, +0.112 and +0.050 in the three
forward models tried. The bound correctly announces the collapse rather than
promising coverage it cannot deliver, which is its strongest validation because
it happens on data the forward model never produced.

### Gate 6: the blind estimator, and why it is the limiting factor

The bound needs labels. The deployable question replaces d+ with a label-free
proxy: the divergence between calibration and test distributions of some blind
statistic. Seven candidates, 16 cells, 10 seeds, at the measured 31% haze:

| blind statistic | rho(coverage, -blind d+) |
|---|---|
| `qpred`, the fitted model's own predicted quantile | **+0.616 +- 0.044** |
| all 19 features, Mahalanobis | +0.450 +- 0.065 |
| the 14 OSCR features | +0.375 +- 0.078 |
| `otf_core`, 3 selected OTF features | +0.366 +- 0.099 |
| the 5 OTF features | +0.276 +- 0.079 |
| `blind_score` = z(r_nominal) - z(qpred) | +0.101 +- 0.071 |
| `r_nominal` alone | +0.083 +- 0.061 |

Two findings. **More features make a novelty distance worse** -- every
uninformative direction adds variance without signal, so the useful move is a
low-dimensional statistic, not a richer one. And **the winner is learned**:
`qpred` beats every hand-crafted alternative, clear of seed noise (leading by
0.166 against a pooled sd of 0.079).

**The OTF block does not win, which was the hope.** It does improve the mismatch
score's off-distribution correlation with the oracle in 3 of 3 caches
(+0.466 to +0.544, +0.645 to +0.672, +0.444 to +0.664), measured over thousands
of samples. But it makes no better blind estimator, and it has **no detectable
effect on content-shift coverage** -- see retraction 8.

Per-feature, the block is not uniform. Against the oracle RMS on RL at 100
photons:

| feature | test | ood | reading |
|---|---|---|---|
| `otf_slope` | +0.769 | +0.551 | strong |
| `otf_excess_at_cut` | -0.678 | -0.580 | strong |
| `otf_white_chi2` | +0.043 | +0.390 | ood only |
| `otf_excess_inband` | +0.002 | -0.084 | **no signal** |
| `otf_noise_ref` | -0.029 | +0.014 | **no signal** |

The two dead features are dead by construction, which is worth saying because it
is predictable rather than empirical: `otf_noise_ref` is the out-of-band
*normaliser* and tracks the photon budget, not the operator; `otf_excess_inband`
is driven by specimen content.

### Gate 7: the layer does not transfer from simulation to real data

BioSR ships the raw SIM frames. The illumination is 1 + m cos(k.r + phi) with
the three phases at 0, 2pi/3, 4pi/3, which sum to zero, so **averaging the three
phases of one orientation cancels the pattern exactly** and leaves a plain
widefield image. Two consequences:

1. Combining all nine frames gives a real widefield measurement of a real
   specimen at a real signal level, paired with the exact ground truth the error
   metric uses. This needs no grouping -- all three phases of all three
   orientations are present however the frames are ordered.
2. The three per-orientation combinations are three **independent** widefield
   images of the **same** object, so their pairwise differences contain no
   specimen structure. That is an assumption-free noise measurement on real data.

A predictor calibrated entirely on simulation, applied to real measurements:

| calibration | real coverage | mean true error | mean width | d+ | slack |
|---|---|---|---|---|---|
| background 5 (the original) | 0.550 | 0.702 | 0.593 | 0.549 | **+0.199** |
| background 45 (realistic constant) | 0.400 | 0.677 | 0.567 | 0.612 | **+0.112** |
| background 45 + **oracle** background | **0.150** | 0.513 | 0.334 | 0.800 | **+0.050** |

Nominal is 0.900. Nothing gets close. The blind warning signal **inverts**:
rho(r_nominal, true error) is -0.098 to -0.663 on real measurements.

Correcting the background reduces real error everywhere (0.702 to 0.513) and
fixes CCPs almost completely (0.868 to 0.165, a five-fold improvement), but
**coverage gets worse** because the intervals narrow faster than the error
falls. A better forward model makes a more confident predictor, and on real data
that confidence is misplaced.

#### The diagnosis: the forward model has no out-of-focus light

Fitting each real widefield measurement as `a * blur(GT) + b` over 182 cells:

| structure | cells | median haze fraction | fit R^2 |
|---|---|---|---|
| CCPs | 20 | **0.676** | 0.886 |
| Microtubules | 54 | 0.371 | 0.938 |
| F-actin | 45 | 0.322 | 0.938 |
| ER | 63 | 0.271 | 0.970 |
| **pooled** | **182** | **0.332** | -- |

R^2 of 0.89 to 0.97 says the two-term model describes the measurement well. The
constant is **19-68% of the measurement mean, median 33%**, against the **5%**
originally simulated -- four to fourteen times too small.

The cause is structural. **SIM reconstruction removes out-of-focus light** --
optical sectioning is one of its selling points -- so a ground truth derived
from a SIM reconstruction carries no haze, and no amount of blurring puts it
back. Real widefield contrast is std/mean = 0.61; simulated was 2.30.

**This unifies three separate anomalies.** CCPs has the highest haze fraction
and is independently the structure with the worst spectral realism (dlog inband
0.725 against ~0.28 for the other three), the worst conformal coverage, and the
only one at 0.000 sim-to-real coverage. One mechanism, not three oddities.

46 of 228 cells were skipped, almost all CCPs, because their frame ordering was
not identifiable. **High haze lowers illumination contrast, which is exactly
what makes ordering ambiguous**, so the skips are probably not random with
respect to the quantity being measured. The direction of that bias cannot be
settled from these data and should be declared.

#### Forward-model realism, separately measured

44 (cell, level) pairs, comparing real widefield against both simulation
representations:

| source | Poisson gain | noise flatness | dlog inband |
|---|---|---|---|
| simulated, camera-grid PSF | 0.993 | 1.000 | 0.298 |
| simulated, pipeline's fine grid | 0.992 | 1.004 | 0.312 |
| **real** | **1.248** | 0.267 | -- |

Both simulations recover Poisson to within 1%, which is the point of the row --
it shows the estimator is not producing the real-data number. Real noise is
Poisson **in shape** (free-fit R^2 0.911 to 0.954) with gain 1.08 to 1.29,
stable when the out-of-band threshold is tightened from 1.05x to 1.35x the
cutoff, so the excess is a real property and not structure leaking in.

The absolute calibration is **not** established: gain scales linearly with the
assumed photons-per-count, so 1.25 is equally consistent with a sensor ~25%
noisier than Poisson or with 0.6026 photons/count being ~25% too large. These
data cannot separate the two. The consequence is bounded -- it shifts what a
nominal "100 photon" simulation means in real dose, and every conclusion is a
comparison at fixed nominal level -- but photon-level labels should not be read
as absolute.

Spectral shape differs by a factor of ~1.9 for three structures (dlog inband
0.277 to 0.285) and **~5 for CCPs** (0.725).

#### Two traps found in the raw frames, both now guarded and tested

**BioSR's frame ordering is not consistent across structures.** With nine frames
as three orientations by three phases there are two natural acquisition orders:
orientation slow, or phase slow. Measured by which order actually cancels the
pattern, on the brightest level of each cell:

| structure | order | margin over the alternative |
|---|---|---|
| CCPs | orientation slow | 1.21x |
| Microtubules | phase slow | 5.20x |
| F-actin | phase slow | 3.35x |
| ER | phase slow | 8.50x |

Across the full dataset: all 20 usable CCPs cells read orientation-slow, all 162
others read phase-slow. Assuming a single convention silently mis-averages three
structures of four.

An earlier implementation tried to *recover* the order from each frame's
illumination direction. It does not work, for two reasons worth keeping: sCMOS
**fixed-pattern noise** lays a deterministic ridge on the frequency axes, and
being fixed it is identical in all nine frames, so an argmax picks it in every
frame at once (it returned 90 degrees for six of nine frames at level 5 and for
all nine at level 9); and a genuinely **anisotropic specimen** is
indistinguishable from illumination in a single frame, which mis-grouped every
Microtubules and F-actin cell. Cancellation is testable; direction estimation is
not.

**The per-orientation difference is not pure noise.** It should be, and it is
not: measured spectral flatness of the raw difference is 0.02 to 0.37 where
white noise gives 1.0, with power piled at low frequency. The three illumination
orientations have their own spatially varying envelopes, so their difference
keeps a smooth multiplicative copy of the specimen, and a global scale match
cannot remove a varying field. The fix is physical: **the OTF is strictly zero
beyond 2 NA / lambda**, so above that a widefield measurement carries only
noise. Restricting the variance estimate there makes it immune to specimen
structure. Unfiltered, the same fit gives gains of 1.3, 4.3 and 15.6 at levels
3, 5 and 7 of one cell -- diverging with signal, as a multiplicative leak must.
Filtered, the same cell gives 0.73, 1.02 and 1.17.

### Gate 8: no data-fidelity term, same mechanism

The standing objection to Gates 2 and 3 was that residual suppression might be
an artefact of a data-fidelity term rather than a property of the inverse
problem. The U-Net has no such term, which made it the decisive case.

On the held-out structure, against the oracle RMS wavefront error:

| feature | RL | U-Net |
|---|---|---|
| `r_nominal` | +0.727 | **+0.727** |
| `r_min` | +0.679 | +0.683 |
| `r_mean` | -0.902 | -0.908 |
| `r_std` | -0.890 | -0.895 |
| `r_curvature` | -0.868 | -0.871 |
| `xhat_std` | -0.817 | -0.811 |

Agreement to three decimal places on the feature that matters most. **The
mechanism is a property of the inverse problem and the paper's scope does not
narrow.** The U-Net is also the best of the four reconstructors (mean error
0.280 against RL 0.309, TV 0.327, unrolled 0.329).

#### The unrolled network is the mechanism's extreme case, not an anomaly

Its off-distribution mismatch correlation is **-0.159**, against +0.644 (RL),
+0.684 (TV), +0.506 (U-Net). That looked like a failure until the per-feature
breakdown:

| feature | RL | unrolled | |
|---|---|---|---|
| `r_nominal` | +0.727 | **+0.168** | collapsed |
| `r_min` | +0.679 | **-0.687** | sign flipped |
| `r_curvature` | -0.868 | -0.072 | collapsed |
| `r_gap` = `r_nominal` - `r_min` | +0.383 | **+0.860** | far stronger |
| `otf_slope` | +0.596 | **+0.868** | strongest of the four |

`UnrolledNet.forward` takes explicit gradient steps on the least-squares
residual and feature extraction always calls it with the **nominal** PSF, so it
minimises precisely what `r_nominal` measures under an operator that may be
wrong. It drives the nominal residual onto the noise floor whatever the true
operator was -- the mechanism, taken to its limit by an explicit optimised
fidelity term rather than a fixed iteration count.

**And the physics contribution is strongest exactly where the plain residual is
most blind.** `r_gap` more than doubles and `otf_slope` reaches its highest
value of any reconstructor. Gate 3's claim demonstrated at the extreme rather
than argued in the average.

The -0.159 is then a property of **gate 4's fitted score**, not of the physics:
the score is fit on one split and applied to the held-out structure, and several
unrolled features change sign between them. A fitted score cannot survive a sign
flip in its inputs.

#### A learned reconstructor can disable a physics feature

`otf_excess_at_cut` per specimen type, pooling all splits:

| structure | RL | TV | U-Net | unrolled |
|---|---|---|---|---|
| F-actin | -0.687 | -0.719 | **+0.085** | -0.585 |
| Microtubules | -0.710 | -0.689 | -0.338 | -0.535 |
| ER | -0.724 | -0.728 | -0.415 | -0.710 |
| CCPs *(held out in training)* | -0.618 | -0.456 | **-0.824** | -0.845 |
| **range** | 0.106 | 0.273 | **0.909** | 0.311 |

All four vary somewhat, so the accurate claim is comparative: the U-Net's range
is three times any other's, and **it is the only one that changes sign**.

The direction is specific: worthless on F-actin, moderate on Microtubules and
ER, strongest by far on CCPs -- the one structure the network never saw. With no
fidelity term, what the U-Net's residual does near the OTF cutoff is *learned
from the training distribution* rather than enforced by physics. On familiar
content it reproduces the specimen well and the residual carries no mismatch
signal; it has been learned away. On unseen content the fit is looser and the
residual reflects genuine mismatch again.

That is residual suppression in a **content-dependent** form, because the
suppression is itself learned. It is a concrete warning: a physics-derived
feature can be silently disabled by training, and only for the specimens the
network knows -- which is precisely where a practitioner would deploy it and
least expect a blind spot. The other 18 features do not show it.

### Gate 9: calibrating on real measurements works

784 samples from 157 cells at five signal levels, blocked by cell, oracle
background, 10 seeds:

| reconstructor | in-distribution | held-out structure |
|---|---|---|
| Richardson-Lucy | 0.884 +- 0.017 | 0.843 +- 0.151 |
| Poisson-TV | 0.871 +- 0.012 | 0.781 +- 0.151 |
| U-Net | **0.895 +- 0.009** | 0.691 +- 0.289 |
| unrolled | 0.883 +- 0.009 | 0.578 +- 0.341 |

In-distribution coverage is 0.871 to 0.895 against nominal 0.900 for all four --
nominal to within the seed spread. **The reliability layer works on real
measurements; Gate 7's collapse was the forward model.** The prescription is
simply to calibrate on real data.

**The learned reconstructors degrade far more under real content shift** --
0.691 and 0.578 against 0.843 and 0.781, with roughly twice the spread across
structures and a worst cell of 0.105. Consistent with the content-dependence
above: a learned reconstructor's behaviour on unfamiliar specimens is itself
unfamiliar, so a reliability layer built on one inherits two content shifts.

Per structure, RL, against the simulated equivalent:

| held out | real | simulated |
|---|---|---|
| CCPs | **0.999** | 0.427 |
| Microtubules | 0.937 | 0.892 |
| ER | 0.833 | 0.848 |
| F-actin | **0.604** | 0.936 |

CCPs reverses from worst to best, which the haze diagnosis predicts since CCPs
is the structure whose *simulated* features were most distorted. F-actin becomes
the worst. **Content shift has relocated, not gone.** CCPs' 0.999 carries the
widest intervals of the four, so it is conservative rather than sharp.

The signal-level axis gives a clean monotone real dose-response, 182 samples per
level: median true error 0.608 (L01), 0.383 (L03), 0.287 (L05), 0.275 (L07),
0.248 (L09).

### Gate 10: coverage cannot be anticipated blindly on real data

Same seven statistics, 16 real cells, 10 seeds:

| blind statistic | real | simulated |
|---|---|---|
| `qpred` | +0.222 +- 0.153 | **+0.616 +- 0.044** |
| 14 OSCR features | +0.197 +- 0.055 | +0.375 +- 0.078 |
| 5 OTF features | +0.182 +- 0.041 | +0.276 +- 0.079 |
| all 19 features | +0.166 +- 0.044 | +0.450 +- 0.065 |
| `otf_core` | -0.123 +- 0.067 | +0.366 +- 0.099 |
| `r_nominal` | -0.162 +- 0.084 | +0.083 +- 0.061 |
| `blind_score` | -0.310 +- 0.092 | +0.101 +- 0.071 |

Nothing reaches +0.25, and the leader beats the runner-up by 0.025 against a
pooled standard deviation of 0.163 -- **not separated**. On simulated data
`qpred` won clear of the noise; on real data it does not win at all.

**This is the project's sharpest limitation, and the three-way split is the
honest way to state the whole contribution:**

- coverage can be **achieved** on real measurements by calibrating in-domain;
- coverage loss can be **bounded** post-hoc by d+, which holds everywhere tested;
- coverage loss **cannot be anticipated** blindly, by any of seven statistics.

That third clause is what a self-supervised method would have to fix, which
makes it a well-posed target rather than a hopeful one.

### Gate 11: Gate 2's photon dependence is background dependence

Gate 2 reported that the plain residual is blind below ~100 photons and tracks
error above it, rho rising +0.332 at 15 photons to +0.647 at 100. That
comparison **moved three things at once**: the original caches fixed
background = 5.0, so haze was 0.250 at 15 photons and 0.048 at 100, and n_iter
was 30 versus 75.

Holding haze at 0.31 and n_iter at 75, the residual gets monotonically **worse**
with dose:

| photons | rho(r_nominal, error) |
|---|---|
| 15 | +0.038 |
| 30 | -0.087 |
| 60 | -0.197 |
| 100 | -0.251 |
| 200 | -0.289 |

And the decisive comparison needs no new run. Two caches share photon count
**and** iteration count and differ only in haze:

| photons | n_iter | haze | rho(r_nominal, error) |
|---|---|---|---|
| 100 | 75 | 0.048 | **+0.647** |
| 100 | 75 | 0.310 | **-0.251** |

Same dose, same iterations, opposite sign. **The residual's informativeness is
governed by the out-of-focus fraction, not the photon count** -- the same
mechanism as the blind-estimator inversion, since `r_nominal` is a *relative*
residual and a flat pedestal inflates its denominator until the signal reverses.

This matters beyond bookkeeping. "Photon-dependent" implied the residual becomes
usable in bright, low-noise conditions, the easy regime. It actually becomes
usable in **low-background** conditions, and real widefield carries 19-68%
background -- so the regime where the plain residual works is the one real data
is not in.

---

## 4. Sensitivity: how much depends on the haze fraction?

A single global out-of-focus fraction is a simplification -- the per-structure
medians span 0.271 (ER) to 0.676 (CCPs), a spread of 0.404. Five settings span
haze 0.048 to a real-data median of 0.40:

| setting | rho(cov, -d+) | bound | held-out coverage | best blind |
|---|---|---|---|---|
| simulated, 0.048 | +0.908 +- 0.056 | 8/8 | 0.747-0.792 | `residual` +0.604 |
| simulated, 0.310 | +0.959 +- 0.034 | 8/8 | 0.752-0.779 | `residual` **-0.398** |
| simulated, 0.310, 16 cells | +0.952 +- 0.025 | 16/16 | 0.752-0.779 | `qpred` +0.616 |
| four reconstructors, 0.310 | +0.933 +- 0.027 | 16/16 | 0.731-0.776 | `qpred` +0.700 |
| **real**, median 0.40 | +0.933 +- 0.053 | 16/16 | 0.578-0.843 | `qpred` **+0.222** |

- **The bound is insensitive.** rho stays in +0.908 to +0.959 and the bound
  holds every cell on every seed across a 6.5x change in haze and across the
  simulation-to-real boundary. Nothing about it rests on the parameter.
- **Content-shift coverage is insensitive in simulation** (0.731 to 0.792) and
  widens only on real measurements (0.578 to 0.843), where the cause is the
  reconstructor.
- **The blind estimator is extremely sensitive.** The residual goes +0.604 to
  -0.398 between 5% and 31% haze. Any blind-estimator number quoted without its
  haze fraction is uninterpretable.

A per-structure haze extraction was **considered and rejected**: the two
conclusions that could plausibly move are flat across a 6.5x global change, so a
refinement inside that range is unlikely to shift them, and the third already
fails on real data at any fraction. The single global fraction is a
simplification whose consequences have been *measured*, not one assumed away.

---

## 5. The eleven retractions

Recorded because the corrections are as informative as the results, and because
each was caught by a deliberate check rather than by luck. **Do not re-assert
any of these.**

1. ~~"The measurement residual is blind to operator mismatch."~~ Withdrawn, and
   its replacement withdrawn too -- see 11. Not a photon effect at all.
2. ~~"The residual is doubly blind: floored by shot noise and driven onto that
   floor by the reconstructor."~~ The mechanism is real (RL reaches 0.994x the
   floor) but does not imply blindness. Overstated.
3. ~~"The Gate 2 failure is a kernel-truncation artifact."~~ Truncation was a
   real artifact and is now guarded, but the failure survived fixing it.
4. ~~"The correlation is an artifact of pooling across severity cells."~~
   Within-cell rho is +0.77.
5. ~~"TV is a stronger case than RL."~~ True at 100 photons, reverses at 15.
6. ~~"Conformal calibration collapses to 0.29 under content shift."~~ That is
   the worst of four rotations; ER gives 0.97.
7. ~~"The OTF features help exactly where the failure is worst."~~ Asserted from
   the RL/100 cell alone, where CCPs coverage went 0.294 to 0.402. The next cell
   reverses it.
8. ~~"The OTF features change content-shift coverage."~~ Withdrawn in **both**
   directions. Over 10 seeds every 14-versus-19 difference is inside the seed
   noise (|diff| <= 0.052 against a pooled sd of 0.10-0.13). The claim and its
   own retraction were each a single-seed reading.
9. ~~"`otf_core` is a better blind estimator."~~ +0.602 on seed 0, +0.340 +-
   0.112 over ten -- last place. It was selected on the oracle, and this is what
   that costs.
10. ~~"The plain residual is the best blind estimator."~~ True only at the
    unrealistic 5% background. At the measured 31% it *changes sign*, +0.604 to
    -0.398, from joint best to active anti-predictor.
11. ~~"The residual is blind below ~100 photons and tracks error above."~~ The
    comparison confounded dose with haze and iteration count. At matched dose
    and iterations, haze 0.048 gives +0.647 and haze 0.310 gives -0.251.

### The four standing rules those produced

1. **Do not characterise a result until every cell of the grid is in.** This
   pattern caused seven of the eleven.
2. **Report every number over at least 10 partition seeds.** Conformal coverage
   is split-sensitive; single-seed differences of up to 0.2 on the worst cell
   are pure noise.
3. **State the haze fraction with every blind-estimator number.** The ranking
   inverts between 5% and 31%.
4. **Vary one thing.** Three retractions trace to a sweep that moved two
   parameters together. `--haze-frac` and a fixed `--n-iter` prevent it.

---

## 6. The five figures

All rebuild in seconds from cached results via `python scripts/make_figures.py`.
Each panel is stamped with the commit and the sources behind it -- not
decoration, since several retractions happened because a number could not be
traced to the run that produced it.

| figure | what it carries |
|---|---|
| `fig1_bound` | coverage against d+ across five settings, 48/48 cells above the bound, with the symmetric version beside it showing d_TV is valid but uninformative |
| `fig2_blind` | seven blind statistics, simulated against real. The gap between panels is the limitation |
| `fig3_haze` | measured haze on 182 cells; haze flipping the residual's sign at matched dose and iterations; dose at constant haze running the other way |
| `fig4_transfer` | the same held-out-structure experiment on simulated and real measurements, plus in-domain coverage for four reconstructors |
| `fig5_reconstructors` | blind features by reconstructor, and the one feature that is content-dependent |

Two errors the figures caught that the tables had not:

- Figure 4's first draft plotted sim-to-real **transfer** coverage against real
  **in-domain** coverage and labelled it "CCPs reverses". Those are different
  experiments, and the reversal would have read as a property of transfer rather
  than of the measurement.
- Figure 5 disproved a sentence in `docs/gpu_reconstructors.md` claiming RL, TV
  and the unrolled net were "uniform between -0.46 and -0.73" on
  `otf_excess_at_cut`. The unrolled net reaches -0.845 on CCPs.

Building a figure is a check on a claim, not a rendering of it.

---

## 7. What is not established

Stated plainly so the manuscript can carry them as limitations rather than have
them found in review.

1. **No blind warning signal works on real data.** Seven statistics tried. This
   is the headline limitation.
2. **Absolute photon calibration.** Real noise gain 1.25 is equally consistent
   with a sensor 25% noisier than Poisson or with 0.6026 photons/count being 25%
   too high. Photon-level labels are nominal, not absolute.
3. **Spectral realism.** Simulated spectra differ from real by a factor ~1.9 for
   three structures and ~5 for CCPs, even after the background correction.
4. **The haze model is a constant.** `a * blur(GT) + b` reaches R^2 0.72 to 0.96,
   so 4-28% of the real measurement is structured out-of-focus light the model
   does not describe.
5. **A single global haze fraction** is a simplification; the per-structure
   spread is 0.404.
6. **Frame-ordering skips may be biased.** 46 of 228 cells skipped, almost all
   CCPs, and high haze is what makes ordering ambiguous.
7. **Calibration needs labels.** Standard for conformal prediction (labels once,
   not per image), but for microscopy it means paired SIM ground truth for a
   calibration set. How small that set can be was not measured.
8. **The bound is elementary.** Its value is empirical -- that one-sidedness is
   the difference between informative and useless -- not mathematical depth.
9. **One dataset, one modality, one microscope.**
10. **The unrolled net's fidelity term omits background**, assuming zero. It is
    self-consistent between training and evaluation but biased at haze 0.31.

---

## 8. Reproducing

```
python -m pytest -q                    # expect 183 passed
python scripts/make_figures.py         # all five figures, seconds
```

Environment traps that have already cost time:

- **RAM is 7.35 GB.** One extraction peaks at 784 MB resident / 1.5 GB commit
  and already saturates all 6 physical cores, so running two costs memory and
  buys no throughput. Never run two heavy jobs at once.
- **Python cannot resolve `/tmp`** on this machine; use paths inside the repo.
- **Heredocs truncate at ~150 lines and eat backslashes.** A `\n` in Python
  source written through a heredoc arrives as a real newline. Use a file-writing
  tool for anything containing escapes or LaTeX.
- **`pytest` promotes `UserWarning` to an error** deliberately -- it is what
  makes the PSF Nyquist guard testable.
- **`experiments/` is gitignored.** The `docs/` set and this file are the record.

Timings measured on this machine: RL extraction 30 min at 100 photons (75
iterations), 16 min at 15 photons (30 iterations), 45 min at fixed 75
iterations; TV extraction 87-91 min regardless of photon level, since `--tv-iter`
is fixed; 10-seed gate5 on 16 cells 6.5 min; 10-seed rotation 1.7 min per cache;
real-measurement extraction 40 min for 784 samples. The GPU session was 45
minutes on an A100 against the 3-4 hours predicted for a T4.

# Forward-model realism, from BioSR's own raw frames

**What this defends.** Decision D2, the most attackable choice in the project:
BioSR's "ground truth" is itself a SIM *reconstruction*, and every measurement
we deconvolve is simulated from it by our own forward model. If that model does
not resemble a real widefield acquisition, the reliability layer is calibrated
against a fiction and nothing downstream survives.

**Why it costs nothing.** BioSR ships the raw SIM frames next to the ground
truth it derives from them: `RawSIMData_level_01..09.mrc`, nine frames of
502x502, already on disk. No new download, no second dataset, no confound from
comparing across instruments. This replaced the W2S real-optics validation
(40.7 GB) for exactly that reason.

Run with `python scripts/realism_check.py`.

---

## 1. The construction

Structured illumination is `1 + m cos(k.r + phi)`. The three phases are at
`0, 2pi/3, 4pi/3`, which sum to zero, so **averaging the three phases of one
orientation cancels the pattern exactly** and leaves a plain widefield image.
Two consequences, and the second is what makes this a test rather than a
restatement of our own assumptions:

1. Combining all nine frames gives a real widefield measurement of a real
   specimen at a real signal level, paired with the exact ground truth the error
   metric uses. No grouping is needed for this -- all three phases of all three
   orientations are present however the frames are ordered.
2. The three per-orientation combinations are three **independent** widefield
   images of the **same** object. Their pairwise differences therefore contain
   no specimen structure -- only noise. That is an assumption-free noise
   measurement on real data.

### The dose bookkeeping

The frames are **summed**, not averaged. An average of nine Poisson frames of
mean `mu` has variance `mu/9`; comparing that against `var = mu` reports a gain
of 1/9 and looks like a total collapse of the noise model when the data are in
fact perfectly Poisson. This error was made and caught here -- the first run
reported a real-data gain of 0.134, which is 1/9 to two figures.

The sum is what one exposure of the same total dose would have produced: mean
`9mu`, variance `9mu`. Downstream, a per-orientation sum has variance `3v`, a
difference of two of them `6v`, and the nine-frame sum `9v`, so the difference's
variance converts by a factor of `9/6`.

### The factor of two

Raw frames are on the camera grid at **62.6 nm**; `SIM_gt` is 2x upsampled at
**31.3 nm**. Every comparison here is made on the camera grid, and the ratio is
asserted rather than assumed -- `compare_cell` refuses any cell whose ground
truth is not exactly 2x its raw frames. This is the same trap as
STATUS.md section 5, which cost a bug once already.

Two simulation variants reach the camera grid differently, and reporting both
separates "the noise model is right" from "the sampling representation is right":

| variant | PSF sampled at | object | to the camera grid |
|---|---|---|---|
| `camera` | 62.6 nm | 2x2 block-**mean** of `SIM_gt` | one noise draw, natively |
| `fine` | 31.3 nm (the pipeline's own) | `SIM_gt` | measurement block-**summed** 2x2 |

`fine` sums rather than averages because a pixel of twice the width collects
four times the light. Its read noise is added once *after* summing: summing four
fine-grid read events would inflate its sigma by 2x, which is not what a sensor
does.

---

## 2. Two traps that changed the implementation

Both are recorded because both produced a plausible wrong answer first.

### 2.1 The frame ordering is not the same across BioSR

With nine frames as three orientations by three phases there are two natural
acquisition orders: orientation slow (`[[0,1,2],[3,4,5],[6,7,8]]`) or phase slow
(`[[0,3,6],[1,4,7],[2,5,8]]`). **BioSR does not use one throughout.**

Measured by which order actually cancels the pattern, on the brightest level of
each cell:

| structure | order | margin over the alternative |
|---|---|---|
| CCPs | orientation slow | 1.21x |
| Microtubules | phase slow | 5.20x |
| F-actin | phase slow | 3.35x |
| ER | phase slow | 8.50x |

Assuming a single convention would have silently mis-averaged three structures
out of four, leaving residual illumination pattern at precisely the frequencies
this script measures. The dataset already has one known per-structure
inconsistency -- ER's archive layout -- so this is in character.

The first implementation tried to *recover* the order from each frame's
illumination direction. It does not work, for two reasons worth keeping:

* sCMOS **fixed-pattern noise** lays a deterministic ridge on the `fx = 0` and
  `fy = 0` axes. Being fixed, it is identical in all nine frames, so an argmax
  picks it in every frame at once -- returning 90 degrees for six of nine frames
  at level 5 and for all nine at level 9.
* A genuinely **anisotropic specimen** is indistinguishable from illumination in
  a single frame. It mis-grouped every Microtubules and F-actin cell.

The replacement tests the property the analysis actually needs -- cancellation --
rather than estimating a direction, and is decided once per cell on the
brightest level, because at level 1 photon noise flips the answer. Cells where
the two orders are within 1.15x are skipped, not guessed.

### 2.2 The orientation difference is not pure noise

It should be, and it is not. Measured spectral flatness of the raw difference is
**0.02-0.37** where white noise gives 1.0, with the power piled up at low
frequency. Cause: the three illumination orientations have their own spatially
varying envelopes, so their difference keeps a smooth multiplicative copy of the
specimen. A single global scale match cannot remove a varying field.

The fix is physical rather than statistical. **The OTF is strictly zero beyond
`2 NA / lambda`**, so above that a widefield measurement carries only noise --
shot noise and read noise, both white. Restricting the variance estimate to that
out-of-band region makes it immune to any amount of specimen structure, and
costs nothing because the discarded band is exactly where the leak lives. It is
the same reasoning as the project's own `otf_noise_ref` feature.

How much it matters, on real data: the unfiltered fit gives a gain of 1.3, 4.3
and 15.6 at levels 3, 5 and 7 of one cell -- diverging with signal, as a
multiplicative leak must. Filtered, the same cell gives 0.73, 1.02 and 1.17.

---

## 3. The estimator is validated before it is trusted

A number about real data means nothing until the estimator recovers a known
truth. On synthetic Poisson-Gaussian fields with a structured specimen:

| estimator | recovered gain (truth 1.000) |
|---|---|
| free intercept | 1.005 +- 0.041 (tile 16), 0.983 +- 0.048 (tile 32) |
| intercept fixed at the known read variance | **0.998 +- 0.003** |

The free intercept is badly conditioned -- read variance is ~2 e-^2 against
signal levels in the hundreds, so the fit extrapolates a long way to reach zero
and the intercept absorbs the scatter. Fixing it leaves gain as the one
well-determined parameter, 13x tighter, and gain is what the Poisson question is
about. The reported read variance should not be quoted.

`tests/test_realism.py` pins this, along with the dose bookkeeping (a wrong sum
or divisor moves the gain by a factor of 9), the 2x block reduction, the
out-of-band fraction, and both frame orderings on synthetic SIM frames built
with a known pattern, a known order and a fixed-pattern ridge.

---

## 4. What the real data says

44 (cell, level) pairs: 3 cells per structure, levels 1/3/5/7/9 (ER has only 6).
Two cells were **skipped**, not guessed, because their frame ordering was not
identifiable within the 1.15x margin (CCPs/Cell_003, F-actin/Cell_003).

| source | gain | noise hf/lf | resid hf/lf | dlog inband |
|---|---|---|---|---|
| SIM `camera` | 0.993 | 1.000 | 1.003 | 0.298 |
| SIM `fine` | 0.992 | 1.004 | 1.004 | 0.312 |
| **REAL** | **1.248** | 0.267 | 0.004 | -- |

Both simulation variants recover Poisson to within 1%, as they must -- they are
our own model measured by our own estimator, and the value of the row is that it
shows the estimator is not the thing producing the real-data number.

### Per structure

| structure | pairs | gain | implied e-/count | free-fit R2 | dlog inband |
|---|---|---|---|---|---|
| CCPs | 10 | 1.083 | 0.557 | 0.938 | **0.725** |
| ER | 9 | 1.082 | 0.557 | 0.954 | 0.277 |
| F-actin | 10 | 1.293 | 0.466 | 0.911 | 0.285 |
| Microtubules | 15 | 1.263 | 0.477 | 0.948 | 0.278 |

**The noise is Poisson in shape.** Free-fit R2 of 0.91-0.95 says variance tracks
mean linearly across the whole signal range, which is the property the forward
model asserts. Raising the out-of-band threshold from 1.05x to 1.35x the cutoff
leaves the gain essentially unchanged (Microtubules 1.293 -> 1.265), so the
excess is a real property of the data and not residual structure leaking in.

**The absolute calibration is not established.** Gain scales linearly with the
assumed photons-per-count, so a measured 1.25 is equally consistent with the
sensor being ~25% noisier than Poisson (sCMOS excess noise, photo-response
non-uniformity) or with the assumed **0.6026 photons/count** being ~25% too
large. These data cannot separate the two. The consequence is bounded: it shifts
what a nominal "100 photon" simulation corresponds to in real dose, and every
conclusion in the project is a comparison at a fixed nominal level, so no result
moves -- but the photon-level *labels* should not be read as absolute.

**Two clean groups, unexplained.** CCPs and ER sit at 1.08; F-actin and
Microtubules at 1.26-1.29. A single camera constant would not do that, so either
the four structures were acquired under different settings, or something
structure-dependent remains. Worth noting the pilot run on one cell put ER at
1.27; with nine pairs it is 1.082. That is the standing rule earning its keep --
one cell was not enough to characterise it, and would have produced a third
group that does not exist.

### The honest gap

`dlog inband` is the mean `|log10|` gap between simulated and real radial power,
each normalised to its own in-band mean, so it measures spectral **shape** and
not the arbitrary ground-truth scaling. Three structures agree at **0.28**, a
factor of ~1.9 in power. **CCPs is the outlier at 0.725**, a factor of ~5.

That is the real limitation this check exposes, and the likely cause is D2
itself: BioSR's ground truth is a SIM reconstruction whose spectrum has been
Wiener-filtered and apodised, so blurring it with a widefield PSF does not
reproduce the spectrum of a widefield image of the actual specimen. The effect
being worst for CCPs is consistent with that -- sparse punctate objects are
where SIM reconstruction re-weighting bites hardest, and CCPs is also the
structure on which conformal coverage collapses furthest. Whether those two
facts share a cause is not established here and is worth a look.

**Verdict on D2.** The forward model's *noise* is right in shape and uncertain
in absolute scale by ~25%. Its *spectral content* is off by a factor of ~1.9 for
three structures and ~5 for CCPs. That is a real and quantified limitation, not
a fiction -- and it is now a measured number a reviewer can be pointed at rather
than an assumption.


---

## 5. How much does any of this depend on the haze fraction?

A single global out-of-focus fraction is a simplification: the per-structure
medians span **0.271 (ER) to 0.676 (CCPs)**, a spread of 0.404 over 182 cells.
That raises the obvious worry that the project's conclusions are artefacts of
one parameter choice. Five settings, spanning haze 0.048 to a real-data median
of 0.40, say otherwise for two of the three conclusions and confirm it for the
third:

| setting | `rho(cov, −d+)` | bound | held-out coverage | best blind |
|---|---|---|---|---|
| simulated, 0.048 | +0.908 ± 0.056 | 8/8 | 0.747-0.792 | `residual` +0.604 |
| simulated, 0.310 | +0.959 ± 0.034 | 8/8 | 0.752-0.779 | `residual` **−0.398** |
| simulated, 0.310, 16 cells | +0.952 ± 0.025 | 16/16 | 0.752-0.779 | `qpred` +0.616 |
| four reconstructors, 0.310 | +0.933 ± 0.027 | 16/16 | 0.731-0.776 | `qpred` +0.700 |
| **real**, median 0.40 | +0.933 ± 0.053 | 16/16 | 0.578-0.843 | `qpred` **+0.222** |

**The bound is insensitive.** `rho(coverage, −d+)` stays in +0.908 to +0.959 and
the one-sided bound holds every cell on every seed across a 6.5x change in haze
and across the simulation-to-real boundary. Nothing about it rests on the
parameter.

**Content-shift coverage is insensitive in simulation** -- 0.731 to 0.792
everywhere -- and only widens on real measurements (0.578 to 0.843), where the
cause is the reconstructor rather than the haze (see
[sim2real.md](sim2real.md)).

**The blind estimator is extremely sensitive**, which is the whole content of
retraction 10: the residual goes +0.604 to −0.398 between 5% and 31% haze, and
`qpred` falls from +0.616 simulated to +0.222 real. Any blind-estimator result
quoted without its haze fraction is uninterpretable, which is why that is now a
standing rule.

### Why the per-structure variant was not run

The natural next step would be to extract at a per-structure fraction rather
than a pooled median. It was considered and rejected: the two conclusions that
could plausibly move are demonstrably flat across a 6.5x global change, so a
refinement within that range is very unlikely to shift them, and the third
conclusion -- the blind estimator -- already fails on real data at any fraction.
The honest statement is that a single global fraction is a simplification whose
consequences have been measured, not one that has been assumed away. Recorded
here rather than left as an untried option.

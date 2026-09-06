# Evaluating the OTF-consistency block

The OTF block (5 features, commit 23) was implemented and then left unevaluated,
because evaluating it needed fresh 19-feature caches. Those now exist for all
four configurations -- RL and TV, at 100 and 15 photons -- so the whole 16-cell
grid is in and the block can be characterised.

**The hope**, as recorded in STATUS.md section 6: that the physics-derived OTF
features would make a better *blind* estimator of coverage loss than a
Mahalanobis distance over all features, closing two open improvements into one
result.

**The answer is no**, with one genuine success alongside it. The three questions
separate cleanly and give three different answers.

---

## 1. Does the block track true operator mismatch? YES, consistently

The mismatch score's rank correlation with the oracle RMS wavefront error, on
the held-out structure -- the case that matters, since in-distribution it was
already good:

| cache | 14 features | 19 features |
|---|---|---|
| RL, 100 photons | +0.466 | **+0.544** |
| RL, 15 photons | +0.645 | **+0.672** |
| TV, 100 photons | +0.444 | **+0.664** |

Three of three improve, TV/100 by +0.22. In-distribution the two are
indistinguishable (+0.864 vs +0.864, +0.870 vs +0.870), so the block adds
nothing where the plain residual already works and adds real signal where it
does not. This is the block doing exactly what it was designed for.

Per-feature, the block is **not uniform**. Against the oracle RMS on RL/100:

| feature | test | ood | reading |
|---|---|---|---|
| `otf_slope` | +0.769 | +0.551 | strong |
| `otf_excess_at_cut` | -0.678 | -0.580 | strong |
| `otf_white_chi2` | +0.043 | +0.390 | ood only |
| `otf_excess_inband` | +0.002 | -0.084 | **no signal** |
| `otf_noise_ref` | -0.029 | +0.014 | **no signal** |

The two dead features are dead by construction, which is worth stating because
it is predictable rather than empirical: `otf_noise_ref` is the out-of-band
*normaliser* and tracks the photon budget, not the operator; `otf_excess_inband`
is driven by specimen content. Neither is a mismatch discriminator.

## 2. Does it repair coverage under content shift? NO -- inconsistently

Held-out-structure coverage with CCPs held out, the worst cell of the rotation:

| cache | 14 features | 19 features | |
|---|---|---|---|
| RL, 100 photons | 0.294 | 0.402 | better |
| RL, 15 photons | 0.412 | **0.203** | **much worse** |
| TV, 100 photons | 0.381 | 0.462 | better |

Two improve, one degrades by more than either improves. Mean held-out coverage
across the four rotations tells the same story: 0.747 -> 0.774, 0.781 -> 0.734,
0.769 -> 0.792.

**This one was nearly mis-stated.** From the RL/100 cell alone the pattern read
as "the physics features help exactly where the failure is worst", which is a
clean and appealing claim. It does not survive the second cell. The standing
rule -- do not characterise a result until every cell of the grid is in -- caught
it, and it is the fifth time on this project that a one-cell reading would have
been wrong.

Nothing here repairs the underlying failure in any case: worst-group coverage
under content shift stays at **0.019-0.060** against a nominal 0.90 across every
cache and every scheme, and abstaining on novelty still backfires monotonically.

## 3. Does it make a better blind estimator? NO -- the plain residual wins

The full 16-cell grid, 19-feature caches throughout. `rho(coverage, -blind d+)`
is the question: can a practitioner see the coverage loss coming with no ground
truth?

| block | features | rho(cov, -blind d+) | rho(d+, blind d+) |
|---|---|---|---|
| `residual` (`r_nominal` alone) | 1 | **+0.740** | **+0.753** |
| `otf_core` | 3 | +0.602 | +0.504 |
| `all` | 19 | +0.456 | +0.421 |
| `oscr` | 14 | +0.437 | +0.383 |
| `otf` | 5 | +0.406 | +0.368 |

The single plainest feature in the project beats everything, including the full
19-feature distance, by a wide margin.

**The ordering is the finding.** Performance falls monotonically with dimension:
1 > 3 > 19 > 14 > 5 is not quite monotone in raw count, but within each family it
is exact -- `otf_core` (3) beats `otf` (5), and `residual` (1) beats `oscr` (14)
and `all` (19). This is Mahalanobis dilution: every uninformative direction adds
variance to the distance without adding signal, and the OTF block carries two
such directions out of five.

So this is **not** evidence that the physics features are worthless. Gate 3
already showed they give 23-26% tighter bounds when used as *regression inputs*.
It is evidence that a full-rank novelty distance is the wrong estimator for
anticipating coverage loss, and that the right move is a low-dimensional
projection rather than more features.

`otf_core` is reported for the mechanism only. Its three features were chosen by
their correlation with the oracle, so its score is optimistic and it is a
hypothesis for the GPU caches, not a result.

---

## What this does to the project's claims

* Gate 5's bound still holds **16/16** on the new caches, mean slack +0.044.
* `rho(coverage, -d+)` is **+0.938** here, against +0.985 on the older caches.
  Adding features changed the fitted quantile model and so the conformity
  scores; the bound is slightly less tight but the conclusion is unchanged.
* The symmetric total-variation version is now **-0.003** -- indistinguishable
  from zero. That *strengthens* the one-sidedness argument, which is the actual
  contribution: TV is not merely weaker, it is uninformative.
* The blind proxy remains the limiting factor, at +0.740 rather than the +0.615
  previously recorded. Better than thought, but from the trivial feature.

---

## 4. Across ten partition seeds — two of the above are wrong

Everything above, and everything in the project before it, was computed on a
**single** cell-blocked partition (`seed=0`). Conformal coverage is
split-sensitive, so that was never enough. Repeating on ten independent seeds:

### The content-shift story survives

Spread **across seeds** is 0.024-0.037; spread **across structures** is
0.200-0.252. Structure-to-structure variation is roughly **7x** larger than
partition noise, so content shift is a real effect and not a lucky draw.
In-distribution coverage is 0.902-0.917 on every cache.

### But the OTF coverage comparison was noise, in both directions

CCPs held out -- the worst cell, and the one section 2 above drew conclusions
from:

| cache | 14 features | 19 features | difference | pooled sd |
|---|---|---|---|---|
| RL, 100 photons | 0.353 ± 0.076 | 0.388 ± 0.107 | +0.035 | 0.131 |
| RL, 15 photons | 0.440 ± 0.046 | 0.429 ± 0.090 | −0.011 | 0.101 |
| TV, 100 photons | 0.328 ± 0.064 | 0.380 ± 0.089 | +0.052 | 0.110 |

Every difference is well inside the seed noise. **The OTF block has no
detectable effect on content-shift coverage, in either direction.** Section 2's
"two improve, one degrades" was reading partition noise -- the single seed that
produced 0.203 at RL/15 is an outlier against a ten-seed mean of 0.429, and the
single seed that produced 0.294 at RL/100 is an outlier against 0.353.

Note what this does to the earlier retraction: the *original* claim was wrong,
and the *retraction of it* was wrong too. Both were single-seed readings. The
correct statement is that the comparison is null.

The one OTF result that does survive is section 1's -- the mismatch score's
correlation with the oracle improves in 3/3 caches -- because that is measured
over thousands of samples rather than four cells.

### Gate 5 is robust; `otf_core` was a lucky draw

| quantity | 10-seed mean ± sd |
|---|---|
| `rho(coverage, -d+)` | **+0.940 ± 0.026** |
| `rho(coverage, -d_TV)` | +0.115 ± 0.085 |
| one-sided bound holds | **16.0 ± 0.0 of 16** — every cell, every seed |
| mean slack | +0.048 ± 0.007 |

The bound never fails, on any of 160 cells. The symmetric version sits near zero
with a wide error bar, so "TV is uninformative" holds, though the single-seed
−0.003 was itself noise around ≈ +0.1.

Blind estimators, 10 seeds:

| estimator | rho(cov, −blind d+) |
|---|---|
| `residual` (`r_nominal` alone) | **+0.728 ± 0.049** |
| `qpred` (the model's own predicted quantile) | +0.488 ± 0.062 |
| `all` (19 features) | +0.397 ± 0.053 |
| `oscr` (14) | +0.381 ± 0.045 |
| `otf` (5) | +0.352 ± 0.082 |
| `otf_core` (3) | +0.340 ± 0.112 |
| `blind_score` (`z(r) − z(q)`) | **−0.442 ± 0.109** |

`residual` leads the runner-up by +0.240 against a pooled sd of 0.079 — clear of
the noise. **`otf_core` collapses from +0.602 to +0.340 and finishes last**,
which is exactly the optimism its oracle-based selection was flagged for. It is
withdrawn.

Two new estimators were tried and both lose. `qpred` is the model's own
predicted quantile, a genuinely low-dimensional statistic, and it beats every
novelty distance -- consistent with the dilution story -- but not the residual.
`blind_score` mirrors the labelled conformity score term for term, replacing the
unobservable error with the standardised residual; it is *consistently
anti-correlated*, meaning the predicted quantile moves with the residual under
shift and subtracting it destroys the signal rather than sharpening it.

**So the plain measurement residual, tested against five alternatives over ten
seeds, remains the best available deployment-time warning.** That is a firmer
negative than the single-seed version, and it is what makes self-supervised
calibration the right next step rather than more feature engineering.

---

## 5. The blind-estimator ranking is an artefact of the forward model

Section 4 concluded that the plain `r_nominal` residual is the best blind
estimator, at +0.728 ± 0.049, having beaten five alternatives over ten seeds.
**That conclusion does not survive a realistic forward model.**

[sim2real.md](sim2real.md) established that real BioSR widefield carries 19-68%
of its mean as out-of-focus light (median 31%) against the 5% these caches were
built with. Re-extracting at a realistic background and comparing at a **matched
photon level** -- 100 photons, both reconstructors, 8 cells, 10 seeds, where
`background=45` gives exactly the observed median 31%:

| blind estimator | 5% haze | 31% haze |
|---|---|---|
| `qpred` (the model's predicted quantile) | +0.613 ± 0.091 | **+0.716 ± 0.083** |
| `blind_score` (`z(r) − z(q)`) | −0.280 ± 0.216 | **+0.635 ± 0.065** |
| `all` (19 features) | +0.347 ± 0.085 | +0.492 ± 0.075 |
| `oscr` (14) | +0.344 ± 0.058 | +0.407 ± 0.081 |
| `otf_core` (3) | +0.328 ± 0.090 | +0.314 ± 0.064 |
| `otf` (5) | +0.430 ± 0.143 | +0.244 ± 0.060 |
| `residual` (`r_nominal`) | **+0.604 ± 0.081** | **−0.398 ± 0.100** |

The residual does not merely lose its lead, it **changes sign**: from the joint
best predictor of coverage loss to an active anti-predictor. `blind_score`
flips the other way, −0.280 to +0.635. `qpred` and `blind_score` are
statistically tied at the top (gap +0.082, pooled sd 0.105).

The mechanism is straightforward. `r_nominal` is a *relative* residual, so a
flat pedestal carrying no information about operator mismatch inflates its
denominator and dilutes the signal. At 31% haze the dilution is enough to
invert the relationship. The two winners are both built on the fitted quantile
model, which learns around the pedestal.

**The real data agrees with the realistic model, not the clean one.** On real
measurements `rho(r_nominal, true error)` is −0.098 to −0.663
([sim2real.md](sim2real.md)) -- anti-correlated, exactly as the 31%-haze
simulation predicts and exactly opposite to what the 5% simulation predicts.
That is independent evidence that the realistic background is the right setting
and that this reversal is a property of the physics rather than of the sweep.

So the deployable warning signal is the **model's own predicted quantile**, not
the measurement residual. This matters more than the OTF question it was raised
to answer: it says the blind estimator must be *learned*, because the one
hand-crafted quantity that looked sufficient was sufficient only under a
forward model too clean to be real.

### Caveat on the sweep itself

`--background 45` was applied at both photon levels, which is right at 100
photons (45/145 = 31%) and **wrong at 15** (45/60 = 75%, outside the observed
range). That mis-specification collapsed the RL/15 mismatch correlation to
−0.151, which is an artefact of the choice and not a result. Haze is a fraction
of the measurement, so the background has to scale with the photon budget;
`0.45 × photons` holds the fraction at 31%. The comparison above therefore uses
the 100-photon caches only, where the fraction is correct, and the 15-photon
level is being re-extracted at `background=6.75`.

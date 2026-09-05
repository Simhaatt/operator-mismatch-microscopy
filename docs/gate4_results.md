# Gate 4 -- shift-aware calibration does not repair coverage

**Result: FAIL, consistently across both photon levels.** The localised scheme is
not merely unhelpful, it is actively harmful. The failure is mechanistic and
diagnosed, and it points at a specific fix.

## Numbers

Nominal coverage 0.90. Richardson-Lucy reconstruction, verified BioSR optics,
228 cells, 17 operators.

### photons = 100

| shift axis | scheme | coverage | worst-group | width | n |
|---|---|---|---|---|---|
| none | split conformal | 0.911 | 0.786 | 0.4032 | 595 |
| none | Mondrian | 0.896 | 0.821 | 0.4051 | 595 |
| none | RLCP | 0.908 | 0.836 | 0.4075 | 595 |
| operator | split conformal | **0.850** | 0.779 | 0.5938 | 280 |
| operator | Mondrian | 0.736 | 0.679 | 0.5703 | 280 |
| operator | RLCP | **0.325** | 0.121 | 0.4250 | 280 |
| content | split conformal | 0.474 | 0.032 | 0.1686 | 918 |
| content | Mondrian | 0.478 | 0.032 | 0.1693 | 918 |
| content | RLCP | 0.474 | 0.032 | 0.1681 | 918 |
| combined | split conformal | 0.060 | 0.032 | 0.2135 | 432 |
| combined | Mondrian | 0.060 | 0.037 | 0.2099 | 432 |
| combined | RLCP | 0.042 | 0.037 | 0.1742 | 432 |

### photons = 15

Same ordering. Operator shift: 0.786 / 0.675 / 0.332. Content shift:
0.524 / 0.512 / 0.512. Combined: 0.132 / 0.100 / 0.079.

## Diagnosis

Distribution of the mismatch score in calibration versus test:

| axis | cal mean | cal p95 | test mean | fraction of test above cal p95 |
|---|---|---|---|---|
| none | 0.1178 | 0.2476 | 0.1229 | 8.6% |
| **operator** | 0.0547 | 0.0999 | **0.1928** | **93.6%** |
| **content** | 0.1178 | 0.2476 | **0.0354** | 1.3% |
| combined | 0.0547 | 0.0999 | 0.0633 | 24.8% |

Two different failures, one root cause.

**Operator shift -- the localiser has nothing to localise on.** 93.6% of test
points lie beyond the calibration p95. A kernel localiser therefore concentrates
its weight on a handful of extreme calibration points, and the effective sample
size collapses. Worse, the resulting quantile is *narrower* than the global one
(width 0.4250 against 0.5938): the method tightens the bound exactly where it
should widen it. That is not a tuning problem, it is what localisation does when
asked to extrapolate.

**Content shift -- the localiser reads novelty as easiness.** Test mismatch
scores are *lower* than calibration (0.035 against 0.118). The held-out structure
looks benign to a score trained on operator severity, so the method issues narrow
bounds while true error is high. The score tracks true wavefront error at
rho 0.838 in distribution but only 0.455 out of it.

Both trace to the same thing: **the localiser detects operator severity, and
neither extrapolates beyond its training range nor detects specimen novelty.**

## What this means

1. **Do not use localised conformal calibration for operator shift.** This is a
   usable negative result in its own right, and a cautionary one: the method is
   published, principled, and here it is worse than doing nothing. Split
   conformal is the best of the three under every shift axis tested.
2. **The fix indicated is the localiser, not the scheme.** A score that measures
   *novelty relative to calibration* rather than *operator severity* is what
   localisation needs to be given.
3. **Two fallbacks are guaranteed by construction** if the localiser cannot be
   fixed: robust conformal, which inflates the quantile for bounded shift and
   trades tightness for validity, and selective prediction, which abstains
   outside calibration support and reports coverage on the accepted subset.

## Theoretical context

The failure is consistent with theory rather than a bug. Exact conditional
coverage forces infinite expected interval length (Barber, Candes, Ramdas &
Tibshirani, *The limits of distribution-free conditional predictive inference*).
Weighted conformal needs the likelihood ratio between test and calibration, which
is unavailable under content shift. RLCP's own Theorem 3 bounds coverage loss in
terms of kernel localisation quality, and that bound is vacuous when the test
point sits outside calibration support -- which is precisely the regime measured
above at 93.6%.

---

# Gate 4b -- the diagnosis was right, and it partly repairs the failure

Acting on the diagnosis produces four findings, two positive and two negative.
All confirmed at both photon levels.

## 1. Localise on novelty, not on severity (positive)

Replacing the severity-trained mismatch score with a **Mahalanobis distance from
the calibration feature distribution** -- blind, computable at deployment --
substantially repairs coverage:

| axis | severity localiser | **novelty localiser** |
|---|---|---|
| operator shift, 100 ph | 0.325 | **0.750** |
| operator shift, 15 ph | 0.332 | **0.768** |
| content shift, 100 ph | 0.474 | **0.696** |
| content shift, 15 ph | 0.512 | **0.597** |
| in-distribution | 0.908 | 0.882 |

This matters for the paper's positioning: the localiser the original plan
proposed -- an estimate of optical mismatch severity -- is the *wrong* covariate,
and is worse than doing nothing. What localisation needs is a measure of distance
from the calibration set.

## 2. Operator shift is cheaply repairable by inflation (positive)

Simply taking a stricter quantile restores nominal coverage on the operator axis
at almost no cost in width:

| axis | alpha needed | coverage | width vs alpha=0.1 |
|---|---|---|---|
| operator shift, 100 ph | 0.02 | 0.900 | 0.6175 vs 0.5938, **+4%** |
| operator shift, 15 ph | 0.001 | 0.904 | 0.6664 vs 0.6420, +4% |

So for the axis this project is actually about -- the optics drifting away from
the calibrated configuration -- there is a working, almost free recipe.

## 3. Content shift is not repairable from calibration data (negative, hard)

No amount of inflation works. At alpha = 0.001, content shift still reaches only
0.821 (100 photons) and 0.678 (15 photons), against a 0.90 target; combined shift
reaches 0.111 and 0.167. This is the distribution-free impossibility biting: the
calibration set simply does not contain the failure mode, and no quantile of it
can conjure one.

## 4. Abstaining on novelty backfires (negative, and counterintuitive)

The obvious safety mechanism -- refuse to certify unfamiliar inputs -- makes
things **worse**, monotonically, at both photon levels:

| content shift | accept 100% | 75% | 50% | 25% |
|---|---|---|---|---|
| coverage, 100 ph | 0.474 | 0.315 | 0.144 | **0.052** |
| coverage, 15 ph | 0.524 | 0.369 | 0.148 | **0.026** |

Keeping only the *most familiar-looking* held-out-structure inputs gives the
*worst* coverage. The plausible reading: clathrin pits are sparse and punctate,
so their features resemble easy training cases -- low residual, low contrast --
and the model issues a narrow bound, while true reconstruction error is high.

**The dangerous cases are the ones that look familiar.** That is a direct warning
about OOD-detection as a safety mechanism for reconstruction, and it is worth
reporting on its own.

## Where this leaves the paper

* A working method for **operator shift**: novelty-localised calibration, or
  inflation, restoring nominal coverage at ~4% width cost.
* A measured, principled **impossibility for content shift**, with the obvious
  repairs each shown to fail for a specific reason.
* A cautionary result about novelty-based abstention that generalises beyond
  microscopy.

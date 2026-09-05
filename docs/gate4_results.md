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

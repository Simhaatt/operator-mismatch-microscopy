# Sim-to-real transfer, and the missing background

Every bound in this project is calibrated on measurements our own forward model
produced. [realism_check.md](realism_check.md) showed the model's *noise* is
right in shape. This asks the question one step further on, and the one that
actually decides whether any of this is deployable:

> Take the conformal predictor exactly as calibrated on simulated data, feed it
> a **real** widefield measurement, and does its 90% bound still cover the true
> error?

Run with `python scripts/sim2real.py`.

## The construction

Summing the nine raw SIM frames of one level gives a real widefield measurement
paired with the exact ground truth the error metric uses (see
[realism_check.md](realism_check.md) for the frame-ordering and
illumination-envelope traps). It is upsampled 2x by zero-padding in the Fourier
domain onto the pipeline's 31.3 nm grid -- **exact**, because the widefield band
ends at 6.48 cyc/um, inside the 62.6 nm Nyquist of 7.99, so nothing is aliased
and nothing invented. It changes the noise from white to band-limited, which is
reported rather than hidden.

Two structural caveats: there is **no controlled operator mismatch** on real data
(the true PSF is whatever the microscope's was), and the measurement is rescaled
to the calibration photon budget by a single global factor, chosen after picking
the signal level whose native dose is closest.

## The result: transfer fails in every configuration tried

Three forward models, from the project's original to the most favourable one
that could be built:

| calibration | real coverage | mean true error | mean width | d+ | slack |
|---|---|---|---|---|---|
| `background=5` (the project's) | 0.550 | 0.702 | 0.593 | 0.549 | **+0.199** |
| `background=45` (realistic constant) | 0.400 | 0.677 | 0.567 | 0.612 | **+0.112** |
| `background=45` + **oracle** background on real | **0.150** | 0.513 | 0.334 | 0.800 | **+0.050** |

Nominal is 0.900. Nothing gets close.

Per structure, with the oracle background:

| structure | coverage | mean error | was, at `background=5` |
|---|---|---|---|
| CCPs | 1.000 | **0.165** | 0.868 |
| ER | 0.000 | 0.468 | 0.646 |
| Microtubules | 0.000 | 0.609 | 0.674 |
| F-actin | 0.000 | 0.660 | 0.702 |

Correcting the background **reduces the real error** everywhere (0.702 to 0.513
overall) and fixes CCPs almost completely -- 0.868 to 0.165, a five-fold
improvement, which is strong confirmation that haze was CCPs' dominant problem.
But **coverage gets worse**, because the intervals narrow faster than the error
falls. A better forward model produces a more confident predictor, and on real
data that confidence is misplaced.

The blind warning signal is worse than useless: `rho(r_nominal, true error)` is
**-0.098 to -0.663** across the three configurations. On real measurements the
residual points the wrong way, against +0.728 for the blind coverage proxy on
simulated data.

### The one thing that does survive: the bound

`coverage >= 1 - alpha - d+` holds in **all three** configurations, with slack
+0.199, +0.112 and +0.050. In the worst case it predicts coverage could fall to
0.100 and the realised value is 0.150. So Gate 5's bound remains honest under a
distribution shift far more severe than any in the simulated grid -- it does not
promise coverage it cannot deliver, it correctly announces the collapse.

That is the strongest validation the bound has had, precisely because it is on
data the forward model never generated. The catch is unchanged and now sharper:
computing `d+` needs labels, and the blind proxy that substitutes for it is the
part that inverts on real data.

## This unifies three separate anomalies

CCPs has the **highest haze fraction (0.65)**, and CCPs is independently:

* the structure with the worst spectral mismatch (`dlog inband` 0.725 against
  ~0.28 for the other three, [realism_check.md](realism_check.md)),
* the structure where conformal coverage collapses furthest (0.33-0.50 held out
  against 0.92-0.96 for ER and F-actin),
* the only structure with 0.000 sim-to-real coverage here.

One mechanism explains all three. CCPs is sparse and punctate, so its specimen
signal is weak relative to the diffuse background; the forward model omits that
background; so CCPs is where simulation departs furthest from reality. What
looked like three unrelated oddities is one.

---

## Calibrating on real measurements instead: it works

The transfer failure above leaves one question. Is the problem the *forward
model*, or the reliability layer itself? Calibrating on real measurements and
testing on held-out real measurements separates them.

784 samples from 157 cells at five signal levels, blocked by cell, oracle
background, 10 partition seeds (`scripts/real_calibration.py`).

| | calibrated on simulation | calibrated on real |
|---|---|---|
| in-distribution coverage | -- | **0.884 ± 0.017** |
| held-out structure | 0.150-0.550 | **0.843 ± 0.151** |

**In-distribution coverage is essentially nominal** -- 0.884 against 0.900, a
gap smaller than the seed spread. So the reliability layer works on real
measurements. The sim-to-real collapse was the forward model's missing
out-of-focus term, not a defect in the method. The practical prescription is
correspondingly simple: **calibrate on real data.**

Per held-out structure, and note what happened to CCPs:

| held out | real coverage | in simulation |
|---|---|---|
| CCPs | **0.999** | 0.245-0.497 |
| Microtubules | 0.937 | 0.842-0.884 |
| ER | 0.833 | 0.848-0.972 |
| F-actin | **0.604** | 0.918-0.931 |

CCPs reverses from the worst structure to the best, which is what the haze
diagnosis predicts: CCPs has the highest out-of-focus fraction (0.676), so its
*simulated* features were the most distorted, and on real data that distortion
is simply absent. F-actin becomes the worst instead. **Content shift still
bites; it has relocated, not gone.** And CCPs' 0.999 comes with the widest
intervals of the four (0.585), so it is conservative rather than sharp.

The signal-level axis gives a clean real dose-response, monotone across 182
samples per level:

| level | median true error |
|---|---|
| L01 | 0.608 |
| L03 | 0.383 |
| L05 | 0.287 |
| L07 | 0.275 |
| L09 | 0.248 |

**The bound holds here too**: 4/4 cells on every one of 10 seeds,
`rho(coverage, -d+)` = +0.920 ± 0.098, mean slack +0.053. The symmetric version
is again uninformative (-0.400 ± 0.219).

### What cannot be claimed from this run

The blind estimators are **not assessable** on these data. One cache gives only
four rotations, and every blind statistic comes back between -0.72 and +0.14
with a seed standard deviation of 0.32 to 0.51 -- the comparison that separated
`qpred` from the rest over 16 cells is simply not resolvable over four. The run
says nothing about which blind statistic to prefer on real data, and the
standing rule applies: do not characterise until the grid is in. Getting there
needs more real caches -- other reconstructors, or the levels split into
separate cells -- not more seeds.

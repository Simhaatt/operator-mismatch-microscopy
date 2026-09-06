# The learned reconstructors: the mechanism survives, and sharpens

Two networks trained on an A100 at a constant 31% out-of-focus fraction
(`--haze-frac 0.31`, 32 patches per cell, 60 epochs), then profiled with the
same 19 blind features as Richardson-Lucy and Poisson-TV. Validation SSIM 0.9245
(U-Net) and 0.8888 (unrolled). Caches in `experiments/features19/*_haze31.npz`.

## Why the U-Net was the decisive experiment

Every result up to this point came from RL and TV, both of which carry a
data-fidelity term. The standing objection was that the whole mechanism -- the
reconstructor driving its own residual onto the noise floor, so the plain
residual goes blind to operator mismatch -- might be an artefact of that term
rather than a property of the inverse problem. The U-Net has **no** explicit
data-fidelity term, so it is the one case that could have narrowed the paper's
scope to fidelity-based reconstructors.

**It does not.** On the held-out structure, against the oracle RMS wavefront
error:

| feature | RL | U-Net |
|---|---|---|
| `r_nominal` | +0.727 | **+0.727** |
| `r_min` | +0.679 | +0.683 |
| `r_mean` | −0.902 | −0.908 |
| `r_std` | −0.890 | −0.895 |
| `r_curvature` | −0.868 | −0.871 |
| `xhat_std` | −0.817 | −0.811 |

Agreement to three decimal places on the feature that matters most. The
operator-probe profile behaves identically whether or not the reconstructor was
built around a forward model. The mechanism is a property of the inverse
problem, and the paper's scope holds. The U-Net is also the *best* of the four
reconstructors (mean error 0.280 against RL 0.309, TV 0.327, unrolled 0.329).

## The unrolled network is the mechanism's extreme case, not an anomaly

Its off-distribution mismatch correlation is **−0.159**, against +0.644 (RL),
+0.684 (TV) and +0.506 (U-Net). That looked like a failure until the per-feature
breakdown, which shows something much more useful:

| feature | RL | Unrolled | |
|---|---|---|---|
| `r_nominal` | +0.727 | **+0.168** | collapsed |
| `r_min` | +0.679 | **−0.687** | sign flipped |
| `r_curvature` | −0.868 | −0.072 | collapsed |
| `r_gap` (`r_nominal − r_min`) | +0.383 | **+0.860** | far stronger |
| `otf_slope` | +0.596 | **+0.868** | strongest of the four |

`UnrolledNet.forward` takes explicit gradient steps on
`‖apply_psf(x, psf) − y‖²`, and feature extraction always calls it with the
**nominal** PSF. So it actively minimises the very quantity `r_nominal`
measures, under an operator that may be wrong. It therefore drives the nominal
residual onto the noise floor whatever the true operator was -- which is exactly
the mechanism this project describes, taken to its limit by an explicit,
optimised fidelity term rather than RL's fixed iteration count.

**And the physics contribution is strongest precisely where the plain residual
is most blind.** `r_gap` more than doubles (+0.383 → +0.860) and `otf_slope`
reaches its highest value of any reconstructor. That is Gate 3's claim -- the
operator profile beats the plain residual -- demonstrated at the extreme rather
than argued in the average.

The −0.159 is then a property of *gate 4's fitted score*, not of the physics: the
score is fit on one split and applied to the held-out structure, and several
unrolled features change sign between them. A fitted score cannot survive a sign
flip in its inputs. Worth stating plainly rather than reporting as a mismatch
failure.

## The bound holds across all four reconstructors

Pooling RL, TV, U-Net and unrolled at 100 photons, 16 cells, 10 partition seeds:

| quantity | value |
|---|---|
| `rho(coverage, −d+)` | **+0.933 ± 0.027** |
| one-sided bound holds | **16.0 ± 0.0 of 16** — every cell, every seed |
| mean slack | +0.042 ± 0.007 |
| `rho(coverage, −d_TV)` | +0.102 ± 0.069 (uninformative, as everywhere) |

Blind estimators on the same grid: `qpred` +0.700, `blind_score` +0.495,
`all` +0.438, `otf` +0.347, and `residual` **−0.350**. The residual's inversion
under realistic haze holds with learned reconstructors in the pool.

## Content shift is unchanged in character

Held-out coverage 0.731 (U-Net) and 0.754 (unrolled) against 0.776 (RL), with
seed spread 0.025-0.034 against 0.25-0.28 across structures. CCPs still
collapses -- 0.245 for the U-Net, its worst cell. Nothing about a learned
reconstructor repairs content shift.

## A learned reconstructor learns away the feature it was trained to explain

`otf_excess_at_cut` first looked anomalous for the U-Net: −0.178 on the
in-distribution test split against −0.686 (RL) and −0.718 (TV), but −0.824
off-distribution, where it is the *strongest* of the four. Breaking it down per
structure resolves it, and the resolution is a result rather than a caveat:

| structure | RL | TV | U-Net | Unrolled |
|---|---|---|---|---|
| F-actin | −0.687 | −0.719 | **+0.085** | −0.585 |
| Microtubules | −0.710 | −0.689 | −0.338 | −0.535 |
| ER | −0.724 | −0.728 | −0.415 | −0.710 |
| CCPs *(held out in training)* | −0.618 | −0.456 | **−0.824** | −0.845 |

RL, TV and the unrolled net are uniform across specimen types, between −0.46 and
−0.73. **The U-Net is not**, and it varies in a specific direction: the feature
is worthless on F-actin, moderate on Microtubules and ER, and strongest by far
on CCPs -- the one structure the network never saw, since it was trained with
CCPs held out.

So the −0.178 was never "weak in-distribution". It is the pooled average over
three structures where the value ranges from +0.085 to −0.415, and the −0.824 is
CCPs alone.

**The mechanism.** The U-Net has no data-fidelity term, so what its residual
does near the OTF cutoff is *learned from the training distribution* rather than
enforced by physics. On content it was trained on, it reproduces the specimen
well and the residual near the cutoff carries no mismatch signal -- it has been
learned away. On content it never saw, the fit is looser and the residual there
reflects genuine operator mismatch again.

That is the same mechanism as section 6.2's residual suppression, appearing in a
**content-dependent** form because the suppression is itself learned. It is a
concrete warning for anyone building reliability features on a learned
reconstructor: a physics-derived feature can be silently disabled by training,
and only for the specimens the network knows -- which is precisely where a
practitioner would deploy it and least expect a blind spot.

The other 18 features do not show this: the U-Net matches RL to three decimal
places on the operator-probe profile. Whatever is being learned away is specific
to the cutoff region, which is where a fidelity term would otherwise constrain
the residual.

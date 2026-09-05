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

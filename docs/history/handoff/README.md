# Manuscript handoff package

Everything needed to draft the paper, in four documents. Prepared from the
project at commit `822927b`, test suite 183 passing.

| file | what it is | read when |
|---|---|---|
| `README.md` | this file, plus the one-page summary | first |
| `RESULTS.md` | every result, with its condition and its caveat | writing any section |
| `VENUE.md` | how to assess a venue, and where this paper goes | before writing a word |
| `WRITING.md` | structure, sentences, and voice | while writing |

Read `VENUE.md` first. It changes how the paper opens.

---

## The paper in one page

### The problem

Deconvolution software must **assume** a point-spread function. When the true
PSF drifts from the assumed one, the reconstruction stays visually convincing
while its error roughly doubles -- and the obvious warning signal, the
measurement residual, moves by 2-4%. A biologist is handed an image that is
partly invented, with nothing to indicate it.

### The claim

> Reconstruction error under forward-operator mismatch can be **bounded**
> without ground truth and **achieved** at nominal coverage by in-domain
> calibration, but **cannot be anticipated** blindly.

Three clauses, one claim. Everything else supports or qualifies it.

### The evidence, briefly

- **Bounded.** `coverage >= 1 - alpha - d+`, where d+ is the *one-sided* excess
  of test conformity-score mass above calibration. Holds on **48 of 48 cells
  across five settings, on every one of ten seeds**. The symmetric
  total-variation version is indistinguishable from zero (-0.064 +- 0.090):
  one-sidedness is not a refinement, it is the whole result.
- **Achieved.** Calibrating on real BioSR widefield measurements gives
  in-distribution coverage **0.871 to 0.895** against nominal 0.900, for all
  four reconstructors.
- **Not anticipated.** Seven label-free statistics -- physics-derived, learned
  and trivial -- all fail on real data. Nothing reaches +0.25, and the leader is
  not separated from seed noise.

### The two results that will interest people most

**"The dangerous cases look familiar."** Abstaining on novelty backfires
monotonically: keeping only the most familiar-looking inputs drops coverage from
0.521 to 0.079. Sparse structures resemble easy training cases, so the model
issues narrow bounds where true error is high. This is a direct warning about
out-of-distribution detection as a safety mechanism and it generalises well
beyond microscopy.

**The forward model was missing a physical term.** Real widefield measurements
carry **19-68% of their mean as out-of-focus light** (median 33% over 182 cells,
fit R^2 0.89-0.97) against the 5% simulated. SIM reconstruction removes
out-of-focus light, so a ground truth derived from it carries none and blurring
cannot put it back. This one mechanism explains why CCPs was simultaneously the
worst structure for spectral realism, for conformal coverage, and for
simulation-to-real transfer -- three anomalies, one cause.

### What is distinctive against the 2025 literature

Self-supervised conformal prediction for imaging is crowded. **Operator
mismatch is not.** No published work found addresses PSF misspecification as the
shift axis, and none validates on real measurements with paired ground truth.
Frame the paper there. See `VENUE.md`.

### Where it goes

**TMLR**, on evidence rather than assumption: its two acceptance criteria are
whether claims are supported by evidence and whether some audience would find
the findings interesting, and it explicitly declines to reject for lack of
novelty. Three headline negatives and eleven documented retractions are assets
there. **Post to arXiv immediately** -- three competing papers appeared in 2025.

---

## Honesty ledger

The project retracted **eleven** claims, seven of them because a result was
characterised before the full grid was in. Four standing rules came out of that,
and they are worth carrying into the writing:

1. Do not characterise a result until every cell of the grid is in.
2. Report every number over at least ten partition seeds.
3. State the background fraction with every blind-estimator number.
4. Vary one thing.

The retractions belong **in the paper**. At TMLR they demonstrate the exact
standard the first acceptance criterion asks about. Concealing them would be
both dishonest and strategically worse.

---

## What is not in this package

- The code, data and figures -- those are the repository.
- A draft. These documents are the raw material for one, not a substitute.
- The `docs/` set in the repository, which carries per-experiment detail:
  `PROJECT.md` (the full science, gates 0-11), `sim2real.md`,
  `realism_check.md`, `gpu_reconstructors.md`, `otf_evaluation.md`,
  `gate0_novelty.md`, `gate4_results.md`, `colab.md`.

`RESULTS.md` here is the consolidated version. Where it and a `docs/` file
disagree, `RESULTS.md` is newer.

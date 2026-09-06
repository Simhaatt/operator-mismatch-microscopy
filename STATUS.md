# STATUS — operational handoff

**Read [docs/PROJECT.md](docs/PROJECT.md) first** for the science: the question, the
physics, prior art, the method, and every result with numbers. This file is the
operational state: what is done, what to run next, and the traps that have
already cost time.

Last updated after commit 24. `git log --oneline` is a reliable narrative — every
commit message states what was found and what it means.

---

## 1. One-paragraph orientation

A microscope blurs the specimen with a point-spread function (PSF) the software
must *assume*. When the true PSF drifts, deconvolution still returns a
convincing-looking image that is partly invented. This project builds a post-hoc
**reliability layer** that bounds reconstruction error without ground truth, and
characterises when its statistical guarantees break. Everything runs on real
BioSR data (228 cells, 4 structures). No GPU has been used yet.

---

## 2. Where the project stands

**All gates are answered. The project cannot now produce nothing.**

| Gate | Question | Result |
|---|---|---|
| 0 | Is the novelty gap real? | PASS, narrowed. Competitive risk named |
| 1 | Does mismatch cause plausible-but-wrong output? | PASS |
| 2 | Is the trivial residual already enough? | FAILED above 100 photons (photon-dependent) |
| 3 | Does the operator profile beat the residual? | **PASS — 23–26% tighter bounds** |
| 4 | Does shift-aware calibration repair coverage? | FAILED, diagnosed |
| 4b | Does the diagnosis suggest a fix? | PARTIAL — operator shift yes, content shift no |
| 5 | Can coverage loss be bounded? | **PASS — one-sided bound, rho 0.938, holds 16/16** |
| 6 | Does the OTF block beat it blind? | FAILED — but so does the residual once the background is realistic |
| 7 | Does the layer transfer to REAL measurements? | **FAILED — coverage 0.15–0.55 vs 0.90. The bound still holds** |
| 8 | Does a reconstructor with NO fidelity term break the mechanism? | **PASS — U-Net matches RL to 3 d.p. The scope holds** |
| 9 | Does the layer work calibrated ON real measurements? | **PASS — 0.871–0.895 in-distribution, all four reconstructors** |
| 10 | Can coverage loss be anticipated BLINDLY on real data? | **FAILED — nothing beats +0.222, none separated from seed noise** |

Headline results, all reproducible from cached features:

- Physics-derived features give **23–26% tighter bounds** at equal coverage, and
  are stable across photon levels (rho 0.961→0.971) where the plain residual
  collapses (0.748→0.267).
- **Coverage bound**: `coverage >= 1 - alpha - d+` where `d+` is the *one-sided*
  excess of test conformity-score mass above calibration. On the 19-feature
  caches it holds **16/16**, mean slack +0.044, rho(coverage, -d+) = **+0.938**
  (it was +0.985 on the 14-feature caches; adding features changed the fitted
  quantile model). The symmetric total-variation version is now **-0.003** —
  indistinguishable from zero, which *strengthens* the point: TV is not merely
  weaker, it is uninformative. One-sidedness is the contribution.
- **The blind proxy is the limiting factor, and the trivial feature wins it.**
  `r_nominal` alone predicts coverage loss at rho **+0.740**; all 19 features
  give +0.456 and the OTF block +0.406. Adding dimensions to a Mahalanobis
  novelty distance makes it worse. See [docs/otf_evaluation.md](docs/otf_evaluation.md).
- **Content-shift failure is conditional**, not universal: held-out coverage
  ranges 0.20 (CCPs, RL/15) to 0.97 (ER, TV/15). But **worst-group coverage is
  below nominal in every rotation**, and under content shift it sits at
  **0.019–0.060** against a nominal 0.90 even where marginal coverage looks fine.
- **Abstaining on novelty backfires** (0.521 → 0.079 as you keep only the most
  familiar inputs). The dangerous cases look familiar.
- **The forward model is defended, with a measured limitation.** Real BioSR
  widefield noise is Poisson in shape (R² 0.91–0.95); simulated spectra differ
  from real by a factor ~1.9 for three structures and ~5 for CCPs. See
  [docs/realism_check.md](docs/realism_check.md).

---

## 3. Claims that were made and RETRACTED

**Do not re-assert these.** Each was stated from partial data and corrected by a
fuller run. If you find yourself about to write one, check the grid is complete.

1. ~~"The measurement residual is blind to operator mismatch."~~ Withdrawn, then
   its *replacement* withdrawn too — see retraction 11. It is not a photon
   effect at all.
2. ~~"The residual is doubly blind — floored by shot noise AND driven onto that
   floor by the reconstructor."~~ Mechanism is real (RL reaches 0.994x the floor)
   but does not imply blindness.
3. ~~"The Gate 2 failure is a kernel-truncation artifact."~~ Truncation was a real
   artifact and is now guarded, but the failure survived fixing it.
4. ~~"The correlation is an artifact of pooling across severity cells."~~
   Within-cell rho is +0.77.
5. ~~"TV is a stronger case than RL."~~ True at 100 photons, reverses at 15.
6. ~~"Conformal calibration collapses to 0.29 under content shift."~~ That is the
   worst of four rotations; ER gives 0.97.
7. ~~"The OTF features help exactly where the failure is worst."~~ Nearly
   asserted from the RL/100 cell alone, where CCPs coverage went 0.294 → 0.402.
   The next cell reverses it: at RL/15 the same number goes 0.412 → **0.203**.
   The block's effect on content-shift coverage is inconsistent, and only its
   effect on the *mismatch score* is uniform (3/3 caches).

8. ~~"The OTF features change content-shift coverage."~~ Withdrawn in BOTH
   directions. Over 10 partition seeds every 14-vs-19 difference is inside the
   seed noise (|diff| <= 0.052, pooled sd 0.10-0.13). The claim and its own
   retraction were each a single-seed reading.
9. ~~"otf_core is a better blind estimator."~~ +0.602 on seed 0, +0.340 +- 0.112
   over ten -- last place. It was selected on the oracle, and this is what that
   costs.

10. ~~"The plain residual is the best blind estimator."~~ True only at the
   unrealistic 5% background. At the observed 31% it **changes sign**, +0.604
   to -0.398, from joint best to active anti-predictor, while `qpred` (the
   model's own predicted quantile) leads at +0.716. The real data agrees with
   the realistic setting: `rho(r_nominal, true error)` is -0.098 to -0.663 on
   real measurements. The deployable signal is *learned*, not hand-crafted.

11. ~~"The residual is blind below ~100 photons and tracks error above."~~ The
   original caches fixed `background = 5.0`, so haze was 0.250 at 15 photons and
   0.048 at 100, and `n_iter` was 30 vs 75 — dose, haze and iterations all moved
   together. At **matched dose and iterations**, haze 0.048 gives +0.647 and
   haze 0.310 gives **−0.251**: opposite sign. Holding haze fixed, the residual
   gets monotonically *worse* with dose (+0.038 at 15 ph to −0.289 at 200).
   It is a **background** effect. This matters because "photon-dependent"
   implied the residual works in the easy bright regime; it actually works in
   the low-background regime, which real widefield is not in.

**Fourth standing rule: vary one thing.** Three of eleven retractions came from
a sweep that moved two parameters at once — `--background` fixed while
`--photons` varied, and `--n-iter` chosen per photon level. Use `--haze-frac`
and hold `--n-iter` fixed when the point is to isolate an effect.

**Standing rule adopted: do not characterise a result until every cell of the
grid is in.** This pattern has now cost seven corrections.

**Third standing rule: state the background fraction with any blind-estimator
result.** The ranking inverts between 5% and 31% haze, so a result quoted
without it is not interpretable. Haze is a *fraction* of the measurement, so
`--background` must scale with `--photons` (0.45x holds it at the observed 31%);
a flat 45 is correct at 100 photons and 75% -- out of range -- at 15.

**Second standing rule: report every number over >= 10 partition seeds.**
Conformal coverage is split-sensitive; single-seed differences of up to 0.2 on
the CCPs cell are pure noise. `rotate_structures.py` and `gate5_bound.py` both
take `--seeds`.

---

## 4. Environment traps (these have already cost time)

- **RAM is 7.35 GB total.** Do not run two heavy jobs at once; a previous run was
  OOM-killed. Check with `Get-CimInstance Win32_OperatingSystem`. Extraction
  peaks at 784 MB RSS / 1.5 GB commit and already saturates all 6 physical
  cores, so running two costs memory and buys no throughput. There is **no
  `--chunk` flag**; memory is bounded by streaming one cell at a time.
- **Python cannot resolve `/tmp`.** Git Bash resolves it, Python maps it to
  `C:\tmp` which does not exist. Use the session scratchpad directory or paths
  inside the repo.
- **Heredocs truncate** around ~150 lines through this tool. Write long files in
  two or three appended chunks and verify with `ast.parse` after each.
- **Heredocs also eat backslashes.** A `
` inside Python source written through
  a `<<'EOF'` heredoc arrives as a real newline, and `` in `r"$ho$"`
  vanishes — both produce `SyntaxError: unterminated string literal`, which
  looks like a truncation bug and is not. This cost three attempts on one
  figure. **Use the Write tool for any Python containing escapes or LaTeX**,
  and reserve heredocs for plain text.
- **`pytest` promotes `UserWarning` to an error** (`pyproject.toml`,
  `filterwarnings`). That is deliberate — it is what makes the PSF Nyquist guard
  testable. Calling `float()` on a grad-tracking tensor trips it; wrap in
  `torch.no_grad()`.
- **`experiments/` is gitignored.** Results there are not committed; findings must
  be written into `docs/` to be preserved.
- Background task log files get rotated by the harness. Poll for the output
  `.npz` files rather than the log.

---

## 5. Data facts worth not rediscovering

- Archives are at `data/biosr/*.zip` (8 GB, MD5-verified). A **303 MB** compact
  export `data/biosr_gt.npz` holds everything actually used; loading from it is
  bit-identical. Prefer `--source data/biosr_gt.npz`.
- **Verified optics**: NA **1.7**, emission 525 nm assumed for the 488 line,
  GT-SIM pixel **0.0313 um**, widefield pixel 0.0626 um, 0.6026 photons/count.
  From the Methods of Qiao et al. (Nat Methods 2021) and the MRC headers.
- **The widely quoted "62.6 nm BioSR pixel" is the WIDEFIELD pixel.** SIM ground
  truth is 2x upsampled, so objects are at 31.3 nm. Using 62.6 understates every
  spatial frequency by 2x. This bug was made and fixed once.
- **ER has a different archive layout** (live-cell: per-level ground truth under
  `Cell_XXX/GTSIM/`). Missing it silently drops 68 of 228 cells.
- `F-actin/Cell_024` has a wider field (1516 px) at the *same* pixel size. Benign;
  the exporter centre-crops and says so.
- **Splits must be blocked by cell**, never by patch. `src/data/biosr.split_by_cell`
  does this and holds out one structure for the content-shift axis.

---

## 6. THE CPU WORK — DONE

All of steps 1-4 below have been run. Results are in
[docs/otf_evaluation.md](docs/otf_evaluation.md) and
[docs/realism_check.md](docs/realism_check.md); raw output is under
`experiments/` (gitignored, so the docs are the record).

**Timing correction.** Step 1 took **3h41m**, not the ~6 h estimated. The "~85
min per run" figure holds for TV (89 and 87 min) but is 3-5x too pessimistic for
RL (29.6 and 15.5 min), because RL's cost tracks `n_iter` (75, then 30) while
TV's is pinned by `--tv-iter 100` at every photon level.

**Step 1 — re-extract with 19 features.** Done; all four caches in
`experiments/features19/`. Peak RSS 784 MB, so it fits, but only one at a time:
a single run already saturates all 6 physical cores, so parallelism costs memory
and buys no throughput.

**Step 2 — does the OTF block improve the blind estimator? NO.** The plain
`r_nominal` residual alone is the best blind proxy at rho **+0.740**, beating
`otf_core` (+0.602), all 19 features (+0.456) and the OTF block (+0.406).
Performance falls as dimension rises -- Mahalanobis dilution. The improvement
STATUS hoped to close is **not** closed; the useful direction is a
low-dimensional projection, not more features.

**Step 3 — gates on 19-feature caches.** Two of three results are good and one
is not:
- The mismatch score's off-distribution correlation with the oracle improves in
  3/3 caches (+0.466->+0.544, +0.645->+0.672, +0.444->+0.664).
- Content-shift coverage is **inconsistent**: CCPs held out goes 0.294->0.402
  and 0.381->0.462, but **0.412->0.203** at RL/15. Do not claim the OTF block
  repairs content shift.
- Worst-group coverage under content shift stays at 0.019-0.060 everywhere.

**Step 4 — forward-model realism.** Written as `scripts/realism_check.py`, run
over 44 (cell, level) pairs. Real BioSR widefield noise is Poisson **in shape**
(free-fit R2 0.91-0.95) with gain 1.08-1.29, stable under filtering. Spectral
shape differs from simulation by a factor ~1.9 for three structures and ~5 for
CCPs. D2 is defended with a measured limitation attached rather than assumed.

Two traps found there, both now guarded and tested: **BioSR's raw-frame ordering
is not consistent across structures** (CCPs orientation-slow; the other three
phase-slow), and the per-orientation difference is not pure noise because the
illumination envelopes differ, which requires estimating noise only above the
OTF cutoff.

### Correctness fix applied

`spearman` was duplicated in four scripts and ranked with
`argsort(argsort(x))`, which is only a rank when values are distinct. Against a
constant it returned **+1.0**. `rms` takes five distinct values, so every
correlation reported against it ran on tied data. Now one tie-corrected
implementation in `src/metrics/stats.py`, verified against scipy. Gate 4's rho
moves +0.836 -> +0.864; **Gate 5's headline is unaffected**.

### Steps 5-8 — done since, and what they changed

**Multi-seed (`--seeds`).** Content shift survives: spread across seeds
0.024-0.037 against 0.200-0.252 across structures, ~7x. Gate 5 holds **16/16 on
every one of 10 seeds**, rho +0.940 +- 0.026. But the OTF coverage comparison is
null (retraction 8) and `otf_core` is withdrawn (retraction 9).

**Blind estimators.** Two new candidates, both lose to `r_nominal` (+0.728 +-
0.049): the model's own predicted quantile `qpred` (+0.488), and `blind_score`
= z(r) - z(q), which mirrors the labelled conformity score and comes out
*anti*-correlated (-0.442). Five alternatives tested; the trivial residual wins.

**Sim-to-real transfer (`scripts/sim2real.py`) — the important one.** The layer
calibrated on simulation does **not** transfer to real BioSR widefield
measurements: coverage 0.150-0.550 against nominal 0.900, and the blind warning
signal *inverts* (rho -0.098 to -0.663). The Gate 5 bound holds in all three
configurations (slack +0.199, +0.112, +0.050), which is its strongest validation
yet -- it correctly announces the collapse rather than promising coverage.

**Diagnosis: the forward model has no out-of-focus light.** Real widefield fits
`a*blur(GT) + b` at R^2 0.72-0.96 with `b` at **19-68% of the measurement mean**
(median 31%); the project simulated 5%. SIM reconstruction removes out-of-focus
light, so a GT derived from it carries none and blurring cannot restore it.
`extract_features.py` now takes `--background`. Correcting it cuts real error
0.702 -> 0.513 and fixes CCPs five-fold (0.868 -> 0.165), but coverage falls
because intervals narrow faster than error does. **This one mechanism explains
CCPs being simultaneously worst on spectral realism, worst on coverage, and
worst here.** See [docs/sim2real.md](docs/sim2real.md).

### Step 9 — self-supervised calibration: STRUCK

Not deferred, struck. Three 2025 papers do exactly this, one of them in the
Poisson regime -- see section 9. The remaining work is the manuscript.

---

## 7. What needs a GPU (NOT part of the CPU loop)

One Colab session, ~2 h. Do not attempt on CPU — the U-Net is ~19 h there.

1. Train U-Net and unrolled net: `scripts/train_reconstructor.py --device cuda --amp`
2. Extract their 19-feature caches on the same GPU
3. Bring back only the caches (~200 KB each); all analysis runs locally

Both models are **already written and debugged** at small scale. See
[docs/colab.md](docs/colab.md) for the upload path — 303 MB, not 8 GB.

The U-Net is the scientifically decisive experiment: it has **no data-fidelity
term**, so it is the one case that could overturn the mechanism established for
RL and TV.

---

## 8. Verification — run this first in any new session

```bash
python -m pytest -q          # expect 147 passed
git log --oneline | head -5
ls experiments/features/     # existing 14-feature caches
```

If tests fail, stop and diagnose before running experiments. The suite pins the
physics: Airy agreement, FWHM scaling, Marechal Strehl, exact blur adjoint,
Poisson-Gaussian variance, seed reproducibility, and six specific bug regressions.

---

## 9. Publication plan

**Checked against the literature on 2026-09-06, not assumed.** The earlier plan
in this file rated SIAM Journal on Imaging Sciences as newly viable and
speculated that TMLR fit best. One of those was right for the wrong reason and
the other was wrong.

### The competitive picture is worse than Gate 0 recorded

Self-supervised conformal prediction for imaging is now a crowded 2025 line,
and all of it comes from the group Gate 0 flagged:

| work | approach | validated on | operator mismatch? |
|---|---|---|---|
| arXiv 2502.05127 (SSVM 2025) | SURE-based self-calibration | denoising, deblurring | no |
| arXiv 2502.19194 | **Poisson** URE self-calibration | denoising, deblurring | no |
| arXiv 2605.18655 (MaxEnt 2025) | equivariant bootstrapping | weak lensing | no |

**This kills Step 5 as originally conceived.** "Estimate or control `d+` without
labels" is precisely what these three do, one of them in the Poisson regime this
project works in. Entering that race a year late with no differentiator is the
worst available use of a day. Step 5 is struck, not deferred.

**It also confirms the novelty gap is real and narrow, and says where it is.**
None of the three addresses forward-operator mismatch; all validate on
simulated denoising or deblurring, or on FFHQ and fastMRI. The distinctive
claims here are:

1. **operator mismatch** as the shift axis, not measurement noise;
2. **real measurements with paired ground truth** -- BioSR raw widefield against
   SIM ground truth, where the literature uses simulation;
3. **characterising when the guarantee breaks** rather than proposing a method
   that works;
4. the **out-of-focus background** finding, which explains a sim-to-real gap
   quantitatively.

The paper should be framed around 1-3, *not* as "a conformal method for
imaging", which is a crowded framing this work would lose in.

### Venue

**TMLR, confirmed on evidence.** Its two criteria are whether claims are
supported by evidence and whether the findings interest someone -- explicitly
not novelty or impact, and it exists partly to carry work "not yet accepted in
mainstream venues". It publishes conformal prediction on imaging inverse
problems and has conformal expertise on its editorial board. No page limit
matters for an eleven-gate grid; free matters while the NITC open-access
question is open. Three headline negatives and eleven recorded retractions are
an asset there rather than a liability.

**SIIMS is a legitimate second choice and the earlier downgrade was wrong.** It
publishes applied conformal work -- *Posterior-Variance-Based Error
Quantification for Inverse Problems in Imaging* (10.1137/23M1546129) uses
conformal prediction for coverage guarantees on pixelwise error bounds without
distributional assumptions. The reservation is not scope but disposition: a
mathematics-leaning venue is a harder sell for a paper whose strongest results
are failures, and this bound is elementary by its own admission.

**IEEE TCI stays the fallback** and has got more expensive, since the paper grew.

**Rule out optics and bio-imaging venues.** The haze and real-measurement
results would interest that audience, but the headline is a method whose
deployable warning signal does not work, and those venues want a working
instrument.

### Post to arXiv now

Three competing papers appeared in 2025 and one is Poisson-specific. This is no
longer cheap insurance; it is overdue.

## 10. Open questions for the human

- Colab Pro session: when? It is the only GPU-blocked work.
- Is a departmental open-access fund available at NITC? Changes whether IEEE TCI
  is affordable as a fallback.
- W2S real-optics validation (40.7 GB) was **dropped** in favour of the BioSR
  raw-frame realism check, which is free, unconfounded and uses data already on
  disk. Revisit only if targeting an optics journal later.

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
| 5 | Can coverage loss be bounded? | **PASS — near-tight one-sided bound, rho 0.985** |

Headline results, all reproducible from cached features:

- Physics-derived features give **23–26% tighter bounds** at equal coverage, and
  are stable across photon levels (rho 0.961→0.971) where the plain residual
  collapses (0.748→0.267).
- **Coverage bound**: `coverage >= 1 - alpha - d+` where `d+` is the *one-sided*
  excess of test conformity-score mass above calibration. Holds 16/16, mean slack
  +0.038, rho(coverage, -d+) = **+0.985**. The symmetric total-variation version
  is valid but useless (rho +0.076) — one-sidedness is the contribution.
- **Content-shift failure is conditional**, not universal: held-out coverage
  ranges 0.29 (CCPs) to 0.97 (ER). But **worst-group coverage is below nominal in
  all 16 rotations** (0.65–0.88) even where marginal coverage looks fine.
- **Abstaining on novelty backfires** (0.474 → 0.052 as you keep only the most
  familiar inputs). The dangerous cases look familiar.

---

## 3. Claims that were made and RETRACTED

**Do not re-assert these.** Each was stated from partial data and corrected by a
fuller run. If you find yourself about to write one, check the grid is complete.

1. ~~"The measurement residual is blind to operator mismatch."~~ True only below
   ~100 photons; above that it tracks error at rho 0.89–0.92.
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

**Standing rule adopted: do not characterise a result until every cell of the
grid is in.** This pattern cost three corrections.

---

## 4. Environment traps (these have already cost time)

- **RAM is 7.35 GB total.** Do not run two heavy jobs at once; a previous run was
  OOM-killed. Check with `Get-CimInstance Win32_OperatingSystem`. Feature
  extraction uses `--chunk` to bound memory.
- **Python cannot resolve `/tmp`.** Git Bash resolves it, Python maps it to
  `C:\tmp` which does not exist. Use the session scratchpad directory or paths
  inside the repo.
- **Heredocs truncate** around ~150 lines through this tool. Write long files in
  two or three appended chunks and verify with `ast.parse` after each.
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

## 6. THE CPU WORK TO DO — in order

Everything below runs on CPU. Nothing here needs a GPU. Run heavy steps **one at
a time** (7.35 GB RAM).

### Step 1 — Re-extract with the 19-feature set (~6 h, background)

Current caches in `experiments/features/` have **14** features. The
OTF-consistency block (5 more, added in commit 23) is implemented but **never
evaluated**, because evaluating it needs fresh caches. This is the blocker on the
highest-value remaining analysis.

```bash
for ph in 100 15; do
  python scripts/extract_features.py --photons $ph --n-iter $([ $ph = 15 ] && echo 30 || echo 75) \
    --crop 256 --source data/biosr_gt.npz --out experiments/features19/rl_photons_$ph.npz
  python scripts/extract_features.py --photons $ph --reconstructor tv --tv-iter 100 \
    --tv-prox-iter 10 --crop 256 --source data/biosr_gt.npz \
    --out experiments/features19/tv_photons_$ph.npz
done
```

Roughly 85 min per run, 4 runs. TV cost does **not** scale with photon level
(fixed `--tv-iter`), unlike RL where `n_iter` does.

### Step 2 — Does the OTF block improve the blind estimator? (instant)

This is the single highest-value analysis left. The coverage bound (Gate 5) is
tight at rho 0.985 but needs labels; the **blind proxy is only rho 0.615**, and
that is what limits the contribution. The OTF features are physics-derived and
built to detect operator mismatch from the residual spectrum alone.

```bash
python scripts/gate5_bound.py --features experiments/features19/*.npz --out experiments/gate5/bound19.json
```

Then extend `novelty()` in `scripts/gate5_bound.py` to use only the OTF block
(`otf_white_chi2, otf_excess_inband, otf_excess_at_cut, otf_slope, otf_noise_ref`)
and compare its `rho(coverage, -blind)` against the full-feature Mahalanobis.
**If the OTF block wins, that closes improvements #1 and #2 into one result:** a
near-tight bound plus a physics-derived blind estimator of it.

### Step 3 — Re-run the gates on 19-feature caches (instant)

```bash
python scripts/gate4_calibration.py --features experiments/features19/rl_photons_100.npz
python scripts/gate4b_repair.py    --features experiments/features19/rl_photons_100.npz
python scripts/rotate_structures.py --features experiments/features19/rl_photons_100.npz
```

Confirms the OTF features do not *hurt*, and refreshes every headline number.

### Step 4 — Forward-model realism check (~2 h, needs new code)

Defends **decision D2**, the most attackable design choice: BioSR's ground truth
is itself a SIM reconstruction, so are our simulated measurements realistic?

BioSR ships the **raw SIM frames** — `RawSIMData_level_01..09.mrc`, 9 frames per
level, 502x502, already on disk. Averaging the 9 gives a **real widefield
measurement** at a **real signal level**, paired with the exact ground truth the
error metric uses.

Write `scripts/realism_check.py` to, for several cells and signal levels:
1. read `RawSIMData_level_NN.mrc`, average the 9 frames → real widefield `y_real`
2. simulate `y_sim` from the paired `SIM_gt` at the matched photon level
3. compare radial power spectra, noise variance versus mean (should follow
   Poisson), and residual whiteness

Use `src/data/mrc.read_mrc`. Note raw frames are at **62.6 nm** while GT is at
**31.3 nm** — a factor of 2, so downsample the object or upsample the measurement
before comparing. Getting this wrong is the same 2x trap as section 5.

### Step 5 — Self-supervised calibration (~1 day, needs new code)

The remaining improvement. Calibrate from the observed data itself so shift
cannot bite. Now has a much sharper objective than when first proposed:
**estimate or control `d+` without labels.** Reference: Aldridge, Liaudat,
Pereyra & McEwen, arXiv:2605.18655 (equivariant bootstrapping).

Target: content shift, where inflation, Mondrian and RLCP all failed and where
CCPs coverage sits at 0.29–0.48.

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

**Target: TMLR.** Free, no page limit, and its stated criteria are whether claims
are supported by evidence and whether findings interest someone — explicitly
*not* novelty or impact. That fits a project whose novelty is narrow and whose
strongest results include negatives.

**SIAM Journal on Imaging Sciences** becomes viable now that the coverage bound
works (free via subscription route). **IEEE TCI** is the fallback but charges
$175/page beyond ~9 pages, realistically $875–1,225 for a paper this size.

Post to **arXiv as soon as the GPU session is done** — Gate 0 identified a group
(Starck, Liaudat) holding all three ingredients of this work, so priority is
cheap insurance.

---

## 10. Open questions for the human

- Colab Pro session: when? It is the only GPU-blocked work.
- Is a departmental open-access fund available at NITC? Changes whether IEEE TCI
  is affordable as a fallback.
- W2S real-optics validation (40.7 GB) was **dropped** in favour of the BioSR
  raw-frame realism check, which is free, unconfounded and uses data already on
  disk. Revisit only if targeting an optics journal later.

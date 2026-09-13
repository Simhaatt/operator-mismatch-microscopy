# Reproducing the results

Every number in the paper comes from this pipeline. Steps 1-6 are CPU only;
step 7 needs one GPU session and is only required for the two learned
reconstructors.

Timings are measured on the development machine: 6-core AMD Ryzen 5 5600H,
7.35 GB RAM, Windows 11, Python 3.14, PyTorch CPU. The GPU session was one
Colab A100.

**Run heavy steps one at a time.** A single extraction peaks at 784 MB resident
and already saturates all six physical cores, so running two costs memory and
buys no throughput.

---

## 0. Environment

```bash
python -m pip install numpy torch scikit-learn matplotlib pytest
```

Versions used: numpy 2.x, torch 2.x (CPU build is sufficient for steps 1-6),
scikit-learn 1.x, matplotlib 3.10.

## 1. Get the data

BioSR is **not** redistributed here. Download it from the authors and cite the
original publication:

> Qiao, C. et al. Evaluation and development of deep neural networks for image
> super-resolution in optical microscopy. *Nature Methods* **18**, 194-202 (2021).

Place the four archives as:

```text
data/biosr/CCPs.zip
data/biosr/ER.zip
data/biosr/F-actin.zip
data/biosr/Microtubules.zip
```

Then build the compact export. The archives total 8 GB, of which the project
reads 441 MB; the export is 303 MB and loading from it is bit-identical.

```bash
python scripts/export_compact.py --out data/biosr_gt.npz
```

**Two data facts worth not rediscovering.** ER uses a different archive layout
(live-cell, per-level ground truth under `Cell_XXX/GTSIM/`); missing it silently
drops 68 of 228 cells. And the widely quoted "62.6 nm BioSR pixel" is the
*widefield* pixel -- SIM ground truth is 2x upsampled, so objects sit at
31.3 nm. Using 62.6 understates every spatial frequency by a factor of two.

## 2. Verify the physics before trusting anything

```bash
python -m pytest -q
```

Expect **183 passed**. The suite pins Airy agreement, FWHM scaling against
lambda/NA, Marechal Strehl, the exact blur adjoint, Poisson-Gaussian variance,
seed reproducibility, the haze-fraction parameterisation, and regressions for
eleven specific bugs. If it fails, stop and diagnose; every later number depends
on it.

Note that `pyproject.toml` promotes `UserWarning` to an error. That is
deliberate -- it is what makes the PSF Nyquist guard testable.

## 3. Extract the simulated feature caches

Four configurations: two reconstructors at two photon levels, all at the
measured out-of-focus fraction.

```bash
python scripts/extract_features.py --photons 100 --n-iter 75 --crop 256 --source data/biosr_gt.npz --haze-frac 0.31 --out experiments/features19/rl_100.npz
```

```bash
python scripts/extract_features.py --photons 15 --n-iter 30 --crop 256 --source data/biosr_gt.npz --haze-frac 0.31 --out experiments/features19/rl_15.npz
```

```bash
python scripts/extract_features.py --photons 100 --reconstructor tv --tv-iter 100 --tv-prox-iter 10 --crop 256 --source data/biosr_gt.npz --haze-frac 0.31 --out experiments/features19/tv_100.npz
```

```bash
python scripts/extract_features.py --photons 15 --reconstructor tv --tv-iter 100 --tv-prox-iter 10 --crop 256 --source data/biosr_gt.npz --haze-frac 0.31 --out experiments/features19/tv_15.npz
```

Roughly 30 min, 16 min, 90 min, 90 min. Richardson-Lucy scales with `--n-iter`;
TV does not, because `--tv-iter` is fixed.

**Pass `--haze-frac`, never a fixed `--background`.** A fixed rate is correct at
one photon level only: 45 photons of background is 31% of the measurement at
100 photons but 75% at 15 and 7% at 600. Getting this wrong reversed a headline
correlation and read as a photon-dependence finding until the arithmetic was
checked.

Each cache holds 3876 samples (1768 train, 595 calibration, 595 test, 918
out-of-distribution) with 19 blind features, split **by cell**, one structure
held out entirely as the content-shift axis.

## 4. The gates on simulated data

All of these read cached features and run in seconds to minutes.

```bash
python scripts/gate5_bound.py --features experiments/features19/rl_100.npz experiments/features19/rl_15.npz experiments/features19/tv_100.npz experiments/features19/tv_15.npz --seeds 10 --out experiments/gate5/bound.json
```

Produces the coverage bound and the seven-way blind-estimator comparison.
About 6.5 min for 16 cells at 10 seeds. Expect `rho(coverage, -d+)` near
**+0.952 +- 0.025** with the one-sided bound holding **16/16 on every seed**,
and the symmetric version indistinguishable from zero.

```bash
python scripts/rotate_structures.py --features experiments/features19/rl_100.npz --seeds 10
```

Rotates the held-out structure through all four. About 1.7 min per cache.

```bash
python scripts/gate4_calibration.py --features experiments/features19/rl_100.npz
```

```bash
python scripts/gate4b_repair.py --features experiments/features19/rl_100.npz
```

Coverage under three shift axes against three calibration schemes, and the
abstention analysis.

**Always pass `--seeds 10` or more.** Conformal coverage is split-sensitive:
single-seed differences of up to 0.2 on the worst cell are pure partition noise,
and this accounts for several corrections made during the work.

## 5. Forward-model realism, on real BioSR raw frames

These read the archives directly, not the compact export, because they need the
raw SIM frames.

```bash
python scripts/haze_fraction.py --out experiments/realism/haze_fraction.json
```

Fits `a * blur(GT) + b` per cell and reports `b` as a fraction of the
measurement mean. About 25 min for 182 usable cells. Expect a pooled median near
**0.332** with per-structure medians spanning 0.271 (ER) to 0.676 (CCPs).

```bash
python scripts/realism_check.py --cells 3 --levels 1 3 5 7 9 --out experiments/realism/realism.json
```

Compares real widefield noise and spectra against both simulation
representations. About 1.5 min. Expect simulated Poisson gain near 0.99 and real
gain 1.08 to 1.29.

```bash
python scripts/sim2real.py --cache experiments/features19/rl_100.npz --cells 6 --out experiments/sim2real/transfer.json
```

Applies a simulation-calibrated predictor to real measurements. About 3 min.

## 6. Calibrating on real measurements

```bash
python scripts/real_calibration.py --levels 1 3 5 7 9 --out experiments/features19/real_rl.npz
```

About 40 min for 784 samples from 157 cells. Repeat with
`--reconstructor tv`, and with `--reconstructor unet --checkpoint <path>` or
`--reconstructor unrolled --checkpoint <path>` once step 7 has produced the
weights. Four caches give a 16-cell grid comparable to the simulated one; a
single cache gives only four cells, where no blind statistic is resolvable.

The output is written in the same layout `extract_features.py` produces, so the
gate scripts above run on it unchanged:

```bash
python scripts/gate5_bound.py --features experiments/features19/real_*.npz --seeds 10
```

```bash
python scripts/rotate_structures.py --features experiments/features19/real_rl.npz --seeds 10
```

Expect in-distribution coverage **0.871 to 0.895** against nominal 0.900, and
no blind statistic above **+0.25**.

## 7. The two learned reconstructors (GPU)

One session, about 45 minutes on an A100. `notebooks/colab_gpu_session.ipynb` is
built for **Runtime > Run all** and needs only two files in Drive:
`data/biosr_gt.npz` and a code zip from `python scripts/pack_code.py`. See
[colab.md](colab.md).

By hand, the equivalent is:

```bash
python scripts/train_reconstructor.py --model unet --device cuda --amp --haze-frac 0.31 --per-cell 32 --epochs 60 --out experiments/models/unet.pt
```

```bash
python scripts/train_reconstructor.py --model unrolled --device cuda --amp --haze-frac 0.31 --per-cell 32 --epochs 60 --out experiments/models/unrolled.pt
```

```bash
python scripts/extract_features.py --reconstructor unet --checkpoint experiments/models/unet.pt --haze-frac 0.31 --device cuda --crop 256 --photons 100 --n-iter 75 --source data/biosr_gt.npz --out experiments/features19/unet_100.npz
```

Expect validation SSIM near 0.9245 (U-Net) and 0.8888 (unrolled).

**Match the haze fraction between training and extraction.** Extraction refuses
a checkpoint trained at a different fraction, because a learned reconstructor
evaluated under a different forward model than it was trained on measures domain
shift rather than reliability. **And do not reduce `--per-cell` below 32**: at
the old default of 8 the networks saw 832 patches while Richardson-Lucy ran its
full 75 iterations, which rigs the comparison against them and makes the
unrolled net visibly overfit.

## 8. Figures

```bash
python scripts/make_figures.py
```

Rebuilds all five figures in seconds from the cached results. Each panel is
stamped with the commit and the sources behind it.

Figures 6 and 7 cover the second dataset and the task-level result and are built
separately, because they depend on steps 9-12 below:

```bash
python scripts/figure_w2s_task.py
```

---

## 9. Calibration that does not depend on the miscoverage level

`scripts/gate4_calibration.py` refits its base predictor at every alpha, as a
conditional quantile regressor. Below about alpha = 0.02 that fit degenerates --
the spread of its prediction falls from 0.242 to 0.0037 between alpha 0.1 and
0.001, so the predictor stops being conditional at all -- and the reported bound
then *narrows* as the requested confidence rises, which no valid procedure may
do. Audited over the twelve shift-axis by scheme combinations, the original
violates monotonicity nine times.

```bash
python scripts/gate4_scaled.py --features experiments/features19/rl_photons_100_bg45.npz
```

Fits mean and scale once, scores on the scale-normalised residual, and lets the
conformal quantile do all the alpha-dependent work. Expect **0 monotonicity
violations** and in-distribution coverage tracking nominal across three orders
of magnitude: 0.901, 0.948, 0.983, 0.993, 0.997, 0.998.

## 10. Per-family, ablation, and the Phase 1 analyses

No re-extraction: `mode` and `severity` are stored per row.

```bash
python scripts/family_analysis.py  --features experiments/features19/rl_photons_100_bg45.npz
python scripts/ablation_matched.py --features experiments/features19/rl_photons_100_bg45.npz --seeds 5
python scripts/phase1_analyses.py  --features experiments/features19/rl_photons_100_bg45.npz
python scripts/runtime_table.py    --features experiments/features19/rl_photons_100_bg45.npz
```

Expect coverage spread across the four aberration families of about 0.657
(spherical highest, coma lowest, in every cache), and the full feature set
21-29% narrower than the residual alone at matched coverage.

## 11. Relative l2, physical mismatch families, and task readouts

Re-extraction with 29 specs instead of 17: the twelve extra are NA at +/-5%,
emission wavelength at +/-10 and +/-20 nm, and axial defocus from 50 to 200 nm.
NA and wavelength are **not** wavefront errors -- they rescale the pupil -- so
`rms` is NaN for those eight and every spec carries `mismatch_mag`, the relative
l2 difference between the true and nominal kernel, instead.

```bash
bash ieee/run_phase2.sh            # ~6 h CPU, four caches
python scripts/phase2_analysis.py  --features experiments/features29/rl_photons_100.npz --seeds 5
python scripts/task_analysis.py    --features experiments/features29_task/rl_photons_100.npz --seeds 5
```

`--task-metrics` adds three downstream readouts at no measurable cost, because
the reference measurement depends only on the ground truth and is cached per
cell. Only two are reported: `branch_points` is non-monotone in severity and is
excluded.

**Read the task numbers against the shift ratio, not on their own.** The
held-out structure is not equally hard for every error definition, so a bound
can look good purely because the shift ran in its favour. `task_analysis.py`
prints the per-structure medians needed to check this.

## 12. Second dataset (W2S)

W2S ships 400 genuine widefield captures per field alongside a SIM
reconstruction of the same field, so the real-measurement experiment can be run
against a real acquisition rather than one summed from raw SIM frames.

Download `W2S_raw.zip` (40.7 GiB, CC BY 4.0) from
<https://datasets.epfl.ch/w2s/W2S_raw.zip> or doi `10.5281/zenodo.3895807`, and
place it at the repository root. It is gitignored.

```bash
python scripts/ingest_w2s.py                    # 52 s; SIM only, 720 MiB of 73 GiB
python scripts/haze_fraction_w2s.py             # ~15 min
bash ieee/run_w2s.sh                            # ~2 h, RL at two photon levels
python scripts/real_calibration_w2s.py --n-iter 75 \
    --out experiments/features19/real_w2s_rl.npz    # ~2 h, 2160 measurements
```

**Measure the haze before simulating anything.** It is the parameter that most
changes behaviour here -- at 0.048 the nominal residual correlates +0.647 with
reconstruction error and at 0.310 it correlates -0.251 -- so carrying BioSR's
value across instruments would be a real error. Expect a median of **0.606** on
W2S against 0.33 on BioSR, from 233 of 360 views passing R^2 >= 0.5.

Two things the ingest records so a later reader cannot be misled: `pixel_um` is
**inferred** from the Nikon N-SIM configuration (16 um iXon3 pitch behind 100x
and a 2.5x relay) because W2S does not document it, and `structure` is the
**wavelength channel**, not a specimen class. Holding out a W2S channel is a
weaker content shift than holding out a BioSR structure, and the results differ
accordingly.

Expect in-domain coverage of **0.891** against a nominal 0.900 on the real
measurements, and the one-sided bound to hold in every evaluated cell.

---

## Environment traps

- **Memory.** Never run two heavy jobs at once on a 7-8 GB machine; a previous
  run was OOM-killed.
- **`/tmp`.** Python cannot resolve it on Windows even where Git Bash can; use
  paths inside the repository.
- **`experiments/` is gitignored.** Results there are not committed, so findings
  must be written into `docs/` to be preserved.
- **Background job logs get rotated.** Poll for the output `.npz` rather than
  the log.
- **Heredocs eat backslashes and truncate.** `
` becomes a real newline and a
  long script is cut off around 150 lines. Write Python containing escapes, or
  any LaTeX, to a file instead of piping it through a heredoc.
- **Memory-mapped files stay locked on Windows.** Release a `np.load(...,
  mmap_mode="r")` handle before reopening that path for writing, or the reopen
  fails with EINVAL.
- **A 400-frame uint16 stack is 800 MiB as float64.** Accumulate in chunks
  rather than casting the whole slice.

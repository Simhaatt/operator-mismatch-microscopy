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

# Reliability of Conformal Reconstruction-Error Bounds Under Forward-Operator Mismatch in Computational Microscopy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22750126.svg)](https://doi.org/10.5281/zenodo.22750126)

This is the reproducibility package for the manuscript of the same title. It
contains the code, the derived data, and a single command that regenerates
**every figure, every table and every quantitative claim** in the paper and its
supplementary material, then checks each printed number against the
regenerated one.

> Experiments span two microscopy datasets acquired on different instruments,
> four reconstruction methods, two photon regimes, Zernike aberrations, and
> physical perturbations of focus, numerical aperture and wavelength.
> Severe-to-extreme Zernike mismatch increases reconstruction error by up to
> fivefold even as the nominal data-consistency residual decreases, with nominal
> 90% coverage falling as low as 0.31. The one-sided coverage inequality holds in
> every evaluated setting-by-content cell across ten random splits on both
> instruments. On genuine widefield measurements from the second instrument,
> in-domain calibration attains 0.891 ± 0.017 mean coverage at a nominal level of
> 0.900.

---

## Quick start: the whole paper in one command

Needs Python ≥ 3.10 and about 1 GB of disk. No raw microscopy data, no GPU.

```powershell
git clone https://github.com/Simhaatt/operator-mismatch-microscopy.git
cd operator-mismatch-microscopy
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python reproduce.py --jobs 4
```

This takes about 20 minutes on a 6-core laptop (`--jobs` = number of physical
cores you can spare). It writes:

| Output | Contents |
|---|---|
| `paper/figures/fig*.pdf` | The six figures, under the file names the manuscript uses |
| `paper/tables/*.tex`, `paper/tables/tables.md` | Table II and the nine results tables of the supplement |
| `paper/results/claims_report.md` | Every quantitative claim: value printed in the paper, regenerated value, status |
| `paper/results/*.json`, `logs/` | Raw output of each analysis step |

and ends with a line like

```text
  249 of 251 claims reproduce exactly; 2 approximate claims within tolerance; 0 differ; 0 missing results
```

Reference copies of all of these outputs are committed, so you can also just
read [paper/results/claims_report.md](paper/results/claims_report.md) and
[paper/tables/tables.md](paper/tables/tables.md) on GitHub. The same pipeline
runs in CI on every push.

`python reproduce.py --tests` runs the 275-test physics, analysis and data-integrity suite
first.

## Where each result comes from

Every paper number is produced by one analysis step in
[paper/run_analyses.py](paper/run_analyses.py) (`--list` prints them all),
reading one or more of the shipped feature caches.

| Paper element | Analysis step(s) | Script | Shipped cache(s) |
|---|---|---|---|
| Fig. 1; Sec. IV-A; supp. family table | `family_rl100`, `scaled_rl100` | `family_analysis.py`, `gate4_scaled.py` | `features19/rl_photons_100_bg45` |
| Sec. IV-B paired coverage drop | `phase1_*` | `phase1_analyses.py` | `features19/*_bg45`, `*_bgprop` |
| Fig. 2a/2b; Sec. IV-C; Table II bound rows | `gate5_biosr` | `gate5_bound.py` | the four `features19` simulated caches |
| Fig. 3; Table II coverage rows; supp. metric, cross-setting and physical tables | `phase2_*` | `phase2_analysis.py` | `features29/*`, `features29_w2s/*` |
| Sec. IV-D W2S bound | `gate5_w2s` | `gate5_bound.py` | `features29_w2s/*` |
| Fig. 4a; supp. width ablation | `ablation_*` | `ablation_matched.py` | the four `features19` simulated caches |
| Fig. 4b; supp. monitoring table; real-grid bound | `gate5_real_grid` | `gate5_bound.py` | `features19/real_{rl,tv,unet,unrolled}` |
| Real-data coverage (Sec. IV-B, IV-D; Table II; supp. real table) | `real_coverage` | `real_coverage.py` | `features19/real_*` |
| Supp. calibration audit and scale floors | `scaled_*` | `gate4_scaled.py` | the four `features19` simulated caches |

Figures are drawn by [paper/make_figures.py](paper/make_figures.py), tables by
[paper/make_tables.py](paper/make_tables.py), and the claim-by-claim comparison
by [paper/check_claims.py](paper/check_claims.py).

## Three levels of reproduction

| Level | Starts from | Needs | Time | How |
|---|---|---|---|---|
| **A. Analyses** | shipped feature caches (14 MB, in this repo) | CPU | ~20 min | `python reproduce.py --jobs 5` |
| **B. Feature caches** | raw BioSR (12 GB) and W2S (41 GB) | CPU, 8 GB RAM | ~17 CPU-hours | [docs/reproduction.md](docs/reproduction.md) |
| **C. Learned reconstructors** | `data/biosr_gt.npz` | one GPU | ~45 min on an A100 | [docs/reproduction.md §7](docs/reproduction.md#7-the-two-learned-reconstructors-gpu) |

Level A is what a reviewer needs. The analysis layer is deterministic, so with
the locked environment every number matches the paper exactly. Level B reruns
the forward model, reconstructions and feature extraction; each cache's exact
build command is listed in [experiments/README.md](experiments/README.md), and
`paper/check_claims.py` then tells you which printed numbers your rebuilt
caches move. The trained U-Net and unrolled-network weights are shipped in
`experiments/gpu/`, so level B does not depend on level C.

## Data

Neither raw dataset is redistributed here. Both are CC BY 4.0.

* **BioSR**: Qiao *et al.*, *Nature Methods* 18, 194–202 (2021);
  data DOI [10.6084/m9.figshare.13264793](https://doi.org/10.6084/m9.figshare.13264793).
* **W2S**: Zhou *et al.*, ECCV Workshops 2020;
  data DOI [10.5281/zenodo.3895807](https://doi.org/10.5281/zenodo.3895807).

The committed caches in `experiments/` hold per-sample features, reconstruction
errors and cell IDs derived from these datasets, never images. Fetch commands,
archive layouts and the verified acquisition parameters are in
[data/README.md](data/README.md). Cache contents, provenance and checksums are
in [experiments/README.md](experiments/README.md).

## Two things that are easy to misread

* **"Seeds".** In `gate5_bound.py` each seed draws a new cell-blocked
  train/calibration/test partition. In `phase1_analyses.py`,
  `phase2_analysis.py`, `ablation_matched.py` and `real_coverage.py` the
  partition is the one stored in the cache, and each seed redraws the random
  half of the training split used to fit the conditional mean and scale models
  and the gradient-boosting random state.
* **Cache names.** `*_bg45` (45 background photons at 100 photons) and
  `*_bgprop` (6.75 at 15 photons) are both an out-of-focus fraction of 0.31, the
  value measured on BioSR. `*_photons_15_bg45` (a 0.75 fraction) is not a paper
  setting and is not shipped.

## Repository layout

```text
reproduce.py           one command: analyses -> figures -> tables -> claim check
paper/                 run_analyses, make_figures, make_tables, check_claims,
                       and their committed reference outputs
src/optics/            Zernike pupils (OSA/ANSI, RMS-normalised), PSFs with sampling
                       guards, Poisson-Gaussian forward model with exact adjoint,
                       aberration ladder, physical (NA/wavelength/focus) mismatch
src/reconstruction/    Richardson-Lucy, Poisson-TV (FISTA), residual U-Net, unrolled net
src/reliability/       the 19 deployment-observable features
src/data/, src/metrics/  MRC reader, BioSR loader, SSIM/l2/task metrics, statistics
scripts/               feature extraction, calibration and bound analyses, data ingest
experiments/           shipped feature caches, trained weights, measured haze (+ README)
tests/                 275 tests pinning the physics, the analyses and the shipped data
docs/                  reproduction guide and research notes; see docs/README.md
notebooks/             Colab "Run all" notebook for the GPU step
ieee/run_*.sh          drivers that built the 29-spec and W2S caches
```

## Verified physics

The test suite pins every property that would otherwise invalidate downstream
claims, including:

| Property | Result |
|---|---|
| PSF vs analytic Airy pattern | cosine similarity 1.000000, L1 error 0.4% |
| FWHM scaling | 0.5145 λ/NA within 0.8–3.3%, λ 488–640 nm, NA 0.9–1.4 |
| Strehl at the Maréchal limit | 0.8148 at λ/14, matching exp(-(2πσ)²) to 0.3% |
| Blur adjoint ⟨Hx,z⟩ = ⟨x,Hᵀz⟩ | exact to 3.7e-16, all four padding modes |
| Poisson-Gaussian variance | 53.991 vs theoretical rate + read² = 54.000 |
| Seed reproducibility | stable across `PYTHONHASHSEED` and chunk size |

`pyproject.toml` promotes `UserWarning` to an error on purpose: that is what
makes the PSF Nyquist guard testable.

## Citation

Please cite the archived release (metadata in [CITATION.cff](CITATION.cff)) and
the two datasets above.

> T. T., S. (2026). *Reliability of Conformal Reconstruction-Error Bounds Under
> Forward-Operator Mismatch in Computational Microscopy: Code and
> Reproducibility Artifacts*. Zenodo. https://doi.org/10.5281/zenodo.22750126

## License

[CC BY 4.0](LICENSE).

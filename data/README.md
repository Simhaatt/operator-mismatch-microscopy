# Data

Nothing here is committed. Every dataset is public; fetch it with the commands
below. Measurements `y` are **never** stored -- they are regenerated on the fly
from a deterministic per-sample seed (see `src/optics/forward_model.sample_seed`),
because the full operator-shift matrix would run to hundreds of GB against a
20 GB Kaggle working disk.

## BioSR (primary objects)

Qiao & Li, *BioSR: a biological image dataset for super-resolution microscopy*,
figshare, DOI [10.6084/m9.figshare.13264793](https://doi.org/10.6084/m9.figshare.13264793) (v9).
**CC BY 4.0** -- redistribution is permitted with attribution, so a Kaggle
mirror is legal (Kaggle allows 200 GB per dataset, 50 top-level files).

* 2200+ low/high-resolution pairs; four structures (CCPs, ER, microtubules,
  F-actin); nine signal levels quoted as **15-600 average photon count**; linear
  and non-linear SIM upscaling.
* Total **12.48 GB** in 7 files: `ER.zip` 2.96, `F-actin_Nonlinear.zip` 4.48,
  `F-actin.zip` 2.23, `Microtubules.zip` 1.63, `CCPs.zip` 1.18 GB.
* The high-resolution target is a **SIM reconstruction**, not a clean object.
  This project treats it as the surrogate object `x` and re-synthesises `y` with
  a known PSF (decision D2 in the plan); the band-limit caveat is discussed
  there and must be addressed in the manuscript.

Companion paper: Qiao et al., *Evaluation and development of deep neural networks
for image super-resolution in optical microscopy*, Nature Methods 18, 194-202
(2021), doi:10.1038/s41592-020-01048-5. Code: https://github.com/qc17-THU/DL-SR.

**Acquisition parameters -- now VERIFIED** from the Methods of Qiao et al. and
cross-checked against the MRC headers in the archives themselves:

| Parameter | Value | Source |
|---|---|---|
| Objective NA | **1.7** | Olympus APON100XHOTIRF, stated in Methods |
| Excitation | 405 / 488 / 560 nm | Methods |
| Camera | Hamamatsu Orca Flash 4.0 v3 sCMOS | Methods |
| Photon conversion | 0.6026 photons per count | Methods |
| **GT-SIM pixel** | **0.0313 um (31.3 nm)** | MRC header of `SIM_gt.mrc` |
| **Raw / widefield pixel** | **0.0626 um (62.6 nm)** | MRC header of `RawSIMData_*.mrc` |
| GT-SIM image size | 1004 x 1004 | MRC header |
| Raw SIM stack | 502 x 502 x **9** frames | MRC header, 3 phases x 3 orientations |

> **Correction worth remembering.** The widely quoted "62.6 nm BioSR pixel" is the
> **widefield** pixel. The SIM ground truth is 2x upsampled, so the objects this
> project uses are at **31.3 nm**. Applying the widefield value to
> SIM-resolution objects understates every spatial frequency by a factor of two.
> Three independent estimates agree: 31.3 nm from MRC metadata (definitive),
> 32.5 nm from camera pixel over magnification, and 29.3 nm inferred from the
> measured spectral band edge of the objects.

At the true optics the widefield OTF cutoff is **0.2027 cyc/px** on the SIM grid,
leaving **9.05%** of object energy in the operator's null space, with a measured
SIM gain of 1.87x against the theoretical 2x.

### Archive layout

```text
<Structure>/GT_all.mrc
<Structure>/Cell_XXX/SIM_gt.mrc                    # 1004x1004, the object x
<Structure>/Cell_XXX/RawSIMData_gt.mrc             # raw frames at GT illumination
<Structure>/Cell_XXX/RawSIMData_level_01..09.mrc   # 502x502x9 per signal level
```

Organised **by cell**, which is exactly the unit that train/calibration/test
splits must be blocked on to avoid leakage. The raw SIM frames are included, so a
*real* widefield measurement can be formed by averaging the nine frames, as the
paper does, and compared against the simulated forward model -- a realism check
the preprocessed subset could not support.

Read them with `src/data/mrc.py` (dependency-free; `mrcfile` is not required).

Extensions: BioSR+ ([Zenodo 7115540](https://zenodo.org/records/7115540)),
BioTISR ([Zenodo 13984825](https://zenodo.org/records/13984825)).

## Pilot subset (Gate 1 / Gate 2)

[DeepTrackAI/biosr_dataset](https://github.com/DeepTrackAI/biosr_dataset) -- a
preprocessed microtubules-only slice: 128x128 32-bit TIFF, 41,040 train / 2,160
validate / 150 HR test (1,350 LR, 150 per signal level), CC BY 4.0. Small enough
to avoid the 12.48 GB download while the cheap gates run.

```bash
git clone --depth 1 https://github.com/DeepTrackAI/biosr_dataset data/biosr_pilot
```

## W2S (real-optics validation)

Zhou, El Helou, Sage, Laroche, Seitz, Susstrunk, *W2S: A Joint Denoising and
Super-Resolution Dataset for Widefield to SIM Microscopy*, ECCV Workshops 2020,
[arXiv:2003.05961](https://arxiv.org/abs/2003.05961),
[repo](https://github.com/IVRL/w2s).

Chosen over FMD because it carries a **real optical-shift axis**: 120 FOVs x 400
captures x **three emission wavelengths (488, 561, 640 nm)** = 360 sets, on a
Nikon Eclipse T1 SIM. PSF width scales as lambda/NA, so the three channels are
genuinely different operators on genuinely real data (~30% span). LR 512x512
widefield, HR 1024x1024 SIM, noise-free LR = average of 400 captures, pairs
already registered.

```bash
curl -L -o data/W2S_raw.zip https://datasets.epfl.ch/w2s/W2S_raw.zip
```

> **Blocking check:** the W2S license is not stated in the paper or the repo.
> Resolve before redistributing (mirroring to Kaggle) -- reading it locally is
> fine either way.

## FMD -- evaluated and rejected

Zhang et al., *A Poisson-Gaussian Denoising Dataset with Real Fluorescence
Microscopy Images*, CVPR 2019, [arXiv:1812.10366](https://arxiv.org/abs/1812.10366).

**Cannot support a deconvolution claim.** Its ground truth is the average of all
50 captures of the *same field under the same optics*; noise levels come from
averaging S = 1, 2, 4, 8, 16 frames. There is no PSF variation and no measured
PSF anywhere in the dataset, so it validates denoising only. Retained here as a
record of the decision, not as a dependency.

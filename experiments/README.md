# Shipped derived data

Everything in the paper is computed from the small per-sample feature caches in
this directory. They are committed (about 14 MB in total) so that every figure,
table and number can be regenerated on a laptop in minutes, without the 50 GB of
raw microscopy data and without a GPU:

```powershell
python reproduce.py --jobs 4
```

Each cache holds, for every reconstructed sample, the 19 deployment-observable
features (`*_X`), the reconstruction error `1 - SSIM` (`*_e`, plus relative
l2 `*_e_l2` in the 29-spec caches), the operator-mismatch labels (`*_mode`,
`*_severity`, `*_rms`, `*_mismatch_mag`) and the cell / structure IDs used for
blocked splitting, already partitioned into `train`, `cal`, `test` and `ood`
(held-out structure or channel). **No images are stored.** Both source datasets
are CC BY 4.0; these are derived data and carry the same attribution
requirement (see [data/README.md](../data/README.md)).

Check integrity with:

```powershell
python -m pytest tests/test_shipped_caches.py -q
```

Every other file under `experiments/` is a local output and is gitignored.

## Caches and how each was built

All builds start from `data/biosr_gt.npz` (BioSR, built by
`scripts/export_compact.py`) or `data/w2s_gt.npz` (W2S, built by
`scripts/ingest_w2s.py`); see [docs/reproduction.md](../docs/reproduction.md)
for obtaining the raw data. Times are on a 6-core laptop CPU.

### `features19/` -- 17-spec Zernike ladder, BioSR, out-of-focus fraction 0.31

The file names are historical. `bg45` means 45 background photons at 100
signal photons and `bgprop` means 6.75 at 15 photons; **both are an
out-of-focus fraction of 0.31**, the value measured on BioSR. The paper uses
only these four simulated caches from this folder.

| File | Paper setting | Build command | Rows (train/cal/test/ood) |
|---|---|---|---|
| `rl_photons_100_bg45.npz` | RL-100 | `python scripts/extract_features.py --crop 256 --source data/biosr_gt.npz --background 45 --photons 100 --n-iter 75 --out ...` | 1768/595/595/918 |
| `rl_photons_15_bgprop.npz` | RL-15 | `... --background 6.75 --photons 15 --n-iter 30` | same |
| `tv_photons_100_bg45.npz` | TV-100 | `... --background 45 --photons 100 --reconstructor tv --tv-iter 100 --tv-prox-iter 10` | same |
| `tv_photons_15_bgprop.npz` | TV-15 | `... --background 6.75 --photons 15 --reconstructor tv --tv-iter 100 --tv-prox-iter 10` | same |

Used by: Figs. 1, 2, 4a; the bound, ablation, family and calibration-audit
results.

### `features19/real_*` -- real measurements

| File | Instrument | Build command | Rows |
|---|---|---|---|
| `real_rl.npz` | BioSR, widefield formed by summing the 9 raw SIM frames | `python scripts/real_calibration.py --levels 1 3 5 7 9 --out ...` | 415/132/137/100 |
| `real_tv.npz` | BioSR | `... --reconstructor tv` | same |
| `real_unet.npz` | BioSR | `... --reconstructor unet --checkpoint experiments/gpu/unet_haze31.pt` | same |
| `real_unrolled.npz` | BioSR | `... --reconstructor unrolled --checkpoint experiments/gpu/unrolled_haze31.pt` | same |
| `real_w2s_rl.npz` | W2S, genuine widefield, 6 dose levels | `python scripts/real_calibration_w2s.py --n-iter 75 --out ...` (needs `W2S_raw.zip`) | 864/288/288/720 |

Used by: Fig. 4b, the real-data coverage numbers, and the real-grid bound.

### `features29/` and `features29_w2s/` -- 29 specs (Zernike + NA, wavelength, focus)

Built with `--include-physical`, which adds NA at +/-2.5 and +/-5 %, emission
wavelength at +/-10 and +/-20 nm, and axial defocus of 50 to 200 nm to the
Zernike ladder, and stores relative l2 error alongside 1 - SSIM.

| Folder | Build | Settings |
|---|---|---|
| `features29/` | `bash ieee/run_phase2.sh` (`--haze-frac 0.31`) | RL/TV x 100/15 photons |
| `features29_w2s/` | `bash ieee/run_w2s.sh` (`--haze-frac 0.606 --na 1.49 --wavelength-um 0.590 --pixel-um 0.032 --held-out ch2_640nm`) | RL x 100/15 photons |

Used by: Fig. 3, Table II coverage rows, the metric and physical-family tables.

### `gpu/` -- learned reconstructors

`unet_haze31.pt` and `unrolled_haze31.pt` were trained once on an A100 with
`scripts/train_reconstructor.py --haze-frac 0.31 --per-cell 32 --epochs 60`
(validation SSIM 0.9245 and 0.8888); `MANIFEST.json` records the run. They are
shipped so `real_unet.npz` and `real_unrolled.npz` can be rebuilt without a
GPU. The four `*_photons_*_haze31.npz` caches are the matching simulated
feature caches; the paper does not quote numbers from them.

### `realism/` -- measured out-of-focus fractions

`haze_fraction.json` (BioSR, `scripts/haze_fraction.py`, median 0.332) and
`w2s_haze.json` (W2S, `scripts/haze_fraction_w2s.py`, median 0.606). These are
the measurements that fix the `--background` / `--haze-frac` values above.

## Regenerating instead of trusting

Rebuilding a cache from raw data reruns the forward model, the reconstruction
and the feature extraction with deterministic per-sample seeds. The analysis
layer on top is bit-for-bit deterministic for a fixed scikit-learn version, so
`paper/check_claims.py` is the end-to-end test: run it on caches you rebuilt
yourself and it reports which printed numbers move.

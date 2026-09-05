# Running on Colab Pro

## What actually needs uploading: 303 MB, not 8 GB

The BioSR archives total 8 GB, but this project only ever reads the ground-truth
images -- 441 MB of it. The rest is raw SIM stacks (nine frames per level, nine
levels per cell) that nothing here opens.

`scripts/export_compact.py` writes every ground-truth image plus its structure
and cell labels into one compressed `.npz`:

```bash
python scripts/export_compact.py --out data/biosr_gt.npz
```

* `data/biosr_gt.npz` -- **303 MB**, 228 images at 1004x1004, uint16, 31.3 nm
* `data/microscopy_code.zip` -- **61 KB**, the source tree

Loading from the export is bit-identical to loading from the archives (verified),
so results are comparable across machines.

> One field-of-view note: `F-actin/Cell_024` was imaged over a wider camera
> region (raw 758 px rather than 502, so SIM 1516 rather than 1004) at the *same*
> 31.3 nm pixel. The exporter centre-crops to the common 1004 px and says so.
> Differing pixel *scales* would be refused rather than silently mixed.

## Upload once to Drive

Put both files in a Drive folder, e.g. `MyDrive/microscopy/`. A 303 MB upload
over a home connection is minutes, and it persists across Colab sessions --
unlike the runtime's own disk.

## Notebook

```python
# 1. Mount Drive and unpack the code
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/ms && cd /content/ms && unzip -oq /content/drive/MyDrive/microscopy/microscopy_code.zip
!cp /content/drive/MyDrive/microscopy/biosr_gt.npz /content/ms/data/
%cd /content/ms
```

```python
# 2. Check what GPU was allocated. Colab Pro gives a "premium" GPU subject to
#    availability, so this may be an A100, an L4 or a V100 -- it matters for
#    timing, not for correctness.
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())
```

```python
# 3. Confirm the physics still holds on this machine before trusting any result.
!pip -q install pytest
!python -m pytest -q
```

```python
# 4. Feature extraction, which is FFT-bound and gains ~4-5x from GPU
!python scripts/extract_features.py --photons 100 --n-iter 75 --device cuda \
    --source data/biosr_gt.npz --out experiments/features/photons_100.npz
```

```python
# 5. Calibration runs on cached features in seconds
!python scripts/gate4_calibration.py --features experiments/features/photons_100.npz
!python scripts/gate4b_repair.py   --features experiments/features/photons_100.npz
```

```python
# 6. Copy results back to Drive before the runtime is recycled
!cp -r experiments /content/drive/MyDrive/microscopy/
```

## What to spend GPU time on

The remaining compute is roughly **3-4 hours on a T4, ~1 hour on an A100**, having
collapsed from an earlier estimate of 95 T4-hours: Gates 3 and 4 are done, and
ablations, structure rotations and calibration variants are all re-partitions of
cached features rather than re-runs.

The two jobs that genuinely need the GPU:

1. **U-Net reconstructor** -- the scientifically decisive one. It has no explicit
   data-fidelity term, so it is the single case that could overturn the
   "reconstructor drives its own residual onto the noise floor" mechanism
   established for Richardson-Lucy and Poisson-TV.
2. **Unrolled network** -- data consistency made explicit, the other end of the
   spectrum.

Feature extraction only speeds up ~4-5x (it is FFT and memory-bandwidth bound,
not compute bound), so on GPU it becomes the dominant cost. Training speeds up
20-40x. Plan accordingly: batch the training jobs, and do not bother moving the
calibration work, which is already instantaneous.

## Reproducibility across machines

Measurements are drawn on the CPU from blake2b-derived per-image seeds even when
the model runs on GPU, because CUDA and CPU RNGs produce different streams from
the same seed. That keeps results bit-comparable between your laptop and a Colab
runtime, and is covered by a test.

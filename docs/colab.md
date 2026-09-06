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

## The notebook

`notebooks/colab_gpu_session.ipynb` (regenerate with
`python scripts/make_colab_notebook.py`) is built to be run with **Runtime > Run
all** and nothing else. Upload it to Colab directly -- File > Upload notebook --
it does not need to be in Drive. It locates `biosr_gt.npz` and
`microscopy_code.zip` anywhere under `MyDrive` rather than hard-coding a path,
writes results next to the data, skips work already finished so it is safe to
re-run, and routes every shell step through a runner that raises on a non-zero
exit so Run all halts at the first real failure instead of feeding a stale
checkpoint into the next step.

**Rebuild the code zip before every session** with `python scripts/pack_code.py`.
A stale zip is silent: Colab unpacks it, everything imports, and the results are
from whenever it was last built. The zip now carries a `BUILD_STAMP` with the
commit and the notebook prints it back.

The cells below are the same steps, if you would rather drive it by hand.

## Notebook — the cells, run in order

### 0. Before you start

Two things were missing until commit `a56bebe`/`HEAD` and would each have cost a
whole session:

* `train_reconstructor.py` hardcoded `background=5.0`. It now takes
  `--background`, and **you must pass 45** -- see below.
* `extract_features.py` had no way to *use* a trained network. It now takes
  `--reconstructor unet|unrolled --checkpoint <.pt>`, and refuses a checkpoint
  whose training background differs from the extraction background, because a
  learned reconstructor evaluated under a different forward model than it was
  trained on measures domain shift rather than reliability.

**Why `--haze-frac 0.31` and not the default background of 5.** Fitting real
BioSR widefield measurements as `a*blur(GT) + b` puts the constant term at
19-68% of the measurement mean (median 31%); the old default is 5%. SIM
reconstruction removes out-of-focus light, so a ground truth derived from it
carries none and blurring cannot put it back. See [sim2real.md](sim2real.md).

**Pass the fraction, not a fixed rate.** Training samples photons across 15-600
per patch, so a flat `--background 45` would give 75% haze at the dim end and 7%
at the bright end -- only one photon level lands in the observed range. Only the
*fraction* is a property of the microscope. This is not hypothetical: a flat
rate applied at two photon levels collapsed a mismatch correlation to -0.151 and
read as a finding until the arithmetic was checked. `--haze-frac` recomputes the
rate per photon budget, and `extract_features.py` refuses a checkpoint whose
training fraction differs from the extraction fraction.

```python
# 1. Mount Drive and unpack the code
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/ms && cd /content/ms && unzip -oq /content/drive/MyDrive/microscopy/microscopy_code.zip
!cp /content/drive/MyDrive/microscopy/biosr_gt.npz /content/ms/data/
%cd /content/ms
!mkdir -p /content/drive/MyDrive/microscopy/out
```

```python
# 2. What GPU did we get? Matters for timing, not correctness.
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch; print(torch.__version__, torch.cuda.is_available())
```

```python
# 3. Confirm the physics holds on this machine before trusting any result.
!pip -q install pytest
!python -m pytest -q          # expect 170 passed
```

```python
# 4. CHEAP GPU SMOKE TEST -- do this before committing 2 hours.
#    The --device cuda path in extract_features.py has only ever been exercised
#    on CPU here, so prove it works on 8 images first. ~1 minute.
import numpy as np
d = np.load('data/biosr_gt.npz', allow_pickle=False)
st = np.asarray([str(x) for x in d['structure']]); ce = np.asarray([str(x) for x in d['cell']])
sel = [i for s in sorted(set(st)) for i in np.where(st == s)[0][:2]]
np.savez_compressed('data/tiny_gt.npz', images=d['images'][sel],
                    structure=st[sel], cell=ce[sel], pixel_um=d['pixel_um'])
```

```python
!python scripts/train_reconstructor.py --model unet --device cuda --amp     --background 45 --limit-cells 4 --epochs 1 --per-cell 1 --patch 128     --out /content/tiny_unet.pt
!python scripts/extract_features.py --reconstructor unet --checkpoint /content/tiny_unet.pt     --background 45 --device cuda --crop 64 --kernel 31 --photons 100     --source data/tiny_gt.npz --out /content/tiny.npz
```

If either fails, stop and fix it -- do not train the real models first.

```python
# 5. Train both models. THE REAL JOB. --out points at Drive so a disconnect
#    does not lose the weights; Colab recycles the runtime disk without warning.
#    ~1-2 h total depending on the GPU allocated.
D = '/content/drive/MyDrive/microscopy/out'
!python scripts/train_reconstructor.py --model unet --device cuda --amp     --background 45 --out {D}/unet_bg45.pt
!python scripts/train_reconstructor.py --model unrolled --device cuda --amp     --background 45 --out {D}/unrolled_bg45.pt
```

```python
# 6. Extract their 19-feature caches, at the SAME background. Four runs.
#    Cheaper than the RL caches -- one network forward per sample instead of 75
#    Richardson-Lucy iterations -- so the 16-probe residual family dominates.
for model in ('unet', 'unrolled'):
    for ph, it in ((100, 75), (15, 30)):
        !python scripts/extract_features.py --reconstructor {model}             --checkpoint {D}/{model}_bg45.pt --background 45 --device cuda             --crop 256 --photons {ph} --n-iter {it} --source data/biosr_gt.npz             --out {D}/{model}_photons_{ph}_bg45.npz
```

```python
# 7. Copy everything back. The caches are ~300 KB each; that is all you need
#    locally. Leave the .pt weights in Drive.
!ls -la {D}
```

## Back on the laptop

Copy the four `*_bg45.npz` caches into `experiments/features19/`, then all the
analysis is instant:

```bash
python scripts/gate5_bound.py --features experiments/features19/*_bg45.npz --seeds 10
```

```bash
python scripts/rotate_structures.py --features experiments/features19/unet_photons_100_bg45.npz --seeds 10
```

**The question the U-Net answers.** It has no explicit data-fidelity term, so it
is the one reconstructor that could overturn the mechanism established for
Richardson-Lucy and Poisson-TV. If its blind features behave like theirs, the
mechanism is a property of the inverse problem; if not, it is a property of
fidelity-term reconstructors and the paper's scope narrows to those.

**What it does not answer.** The sim-to-real transfer failure
([sim2real.md](sim2real.md)) affects every claim in the project rather than one
mechanism, and no amount of GPU time touches it.

**A simplification to state, not discover in review.** `UnrolledNet.forward`
takes a least-squares residual `apply_psf(x, psf) - y` with no background term,
so it implicitly assumes zero background. It is self-consistent between training
and evaluation -- the learned step sizes and prox absorb the offset -- but at
`background=45` its data-fidelity gradient is biased in a way it must learn
around.

## Reproducibility across machines

Measurements are drawn on the CPU from blake2b-derived per-image seeds even when
the model runs on GPU, because CUDA and CPU RNGs produce different streams from
the same seed. That keeps results bit-comparable between your laptop and a Colab
runtime, and is covered by a test.

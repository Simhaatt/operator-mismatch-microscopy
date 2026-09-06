"""Generate the Colab GPU-session notebook.

Kept as a generator rather than a checked-in ``.ipynb`` edited by hand, because
the notebook has to stay in step with the CLI flags it calls. When
``extract_features.py`` gained ``--reconstructor unet`` and ``--background``,
a hand-maintained notebook would have silently kept running the old commands.

The notebook is written to be run with **Runtime > Run all** and nothing else:
every shell step goes through a runner that raises on a non-zero exit, so
execution halts at the first real failure instead of carrying a broken state
into a two-hour training job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HAZE = 0.31   # observed median out-of-focus fraction; see docs/sim2real.md
# Measured on a Colab GPU: training took 1.4 min (U-Net) and 2.3 min (unrolled)
# at the old defaults of 8 patches per cell and 30 epochs, against the ~2 h
# STATUS.md predicted for a T4. That left the learned reconstructors trained on
# 832 patches out of the ~560,000 distinct 256-crops the 104 training cells
# admit, while RL ran at its full 75 iterations and TV at a tuned weight. The
# comparison was rigged against the networks, which matters because the U-Net is
# the decisive experiment. 32 per cell and 60 epochs is 8x the work and still
# only ~30 min.
PER_CELL = 32
EPOCHS = 60


def lines(src: str) -> list[str]:
    """Split into nbformat ``source`` lines.

    Every element must carry its own trailing newline. A reader concatenates
    them with no separator, so stripping the newlines collapses the entire cell
    onto a single line -- which is how this shipped once. Validating by joining
    the elements with ``"\n"`` hides the bug perfectly, because that is the one
    thing the reader does not do; ``test_notebook.py`` joins with ``""``.
    """
    body = src.strip("\n").split("\n")
    return [line + "\n" for line in body[:-1]] + body[-1:]


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines(src)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(src)}


INTRO = """
# BioSR reliability -- GPU session

**Runtime > Change runtime type > GPU**, then **Runtime > Run all**. Nothing to
edit and nothing to click through: every step raises on failure, so Run all
halts at the first real problem rather than carrying a broken state into a
two-hour training job.

Put exactly **two files** anywhere in your Drive -- this notebook finds them:

* `biosr_gt.npz` (303 MB)
* `microscopy_code.zip` (~110 KB, rebuilt with `python scripts/pack_code.py`)

Around 45 minutes: ~30 min training, ~10 min extraction. Weights are written
straight to Drive as they improve, so a disconnect costs time but not work.

**Re-running skips anything already finished** -- delete the `.pt` and `.npz`
files under `out/` first if you want to redo them.
"""

CELL_FIND = '''
# 1 | Mount Drive and FIND the two files, wherever they happen to live
import glob, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive')

def find(name, max_depth=6):
    """Locate a file anywhere in MyDrive, shallowest match first."""
    hits = []
    for depth in range(max_depth):
        hits += glob.glob(str(DRIVE / ('*/' * depth) / name))
        if hits:
            break
    if not hits:
        raise FileNotFoundError(
            f"{name} not found under {DRIVE}. Put it anywhere in Drive; this "
            f"searches {max_depth} levels deep.")
    hits.sort(key=len)
    if len(hits) > 1:
        print(f"  {len(hits)} copies of {name}; using the shallowest")
    return Path(hits[0])

NPZ = find('biosr_gt.npz')
ZIP = find('microscopy_code.zip')
print(f"  data: {NPZ}  ({NPZ.stat().st_size / 1e6:.0f} MB)")
print(f"  code: {ZIP}  ({ZIP.stat().st_size / 1e3:.0f} KB)")

OUT = NPZ.parent / 'out'      # results land beside their inputs
OUT.mkdir(exist_ok=True)

MS = Path('/content/ms')
MS.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(MS)
    stamp = (z.read('BUILD_STAMP').decode().strip()
             if 'BUILD_STAMP' in z.namelist()
             else 'NO STAMP -- zip predates scripts/pack_code.py, so probably STALE')
(MS / 'data').mkdir(exist_ok=True)
local = MS / 'data/biosr_gt.npz'
if not local.exists():
    # Local disk, not a Drive symlink: six scripts each open this 303 MB file.
    print('  copying the data to local disk (once)...')
    shutil.copy2(NPZ, local)
os.chdir(MS)
print(f"  code {stamp}")
print(f"  out  {OUT}")
if 'dirty' in stamp or 'STALE' in stamp:
    print("  WARNING: code zip not built from a clean commit -- rerun "
          "scripts/pack_code.py. A stale zip is silent.")
'''

CELL_RUNNER = '''
# 2 | Run shell steps so that a failure stops "Run all".
#     IPython's `!cmd` swallows non-zero exits; a failed training step would be
#     followed by extraction loading a stale checkpoint and reporting numbers.
def sh(cmd, label=None):
    t0 = time.time()
    print(f"$ {cmd}\\n")
    p = subprocess.run(cmd, shell=True, cwd=MS, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    mins = (time.time() - t0) / 60
    if p.returncode != 0:
        raise RuntimeError(f"{label or cmd} failed (exit {p.returncode}) after "
                           f"{mins:.1f} min -- see the output above")
    print(f"  [ok] {label or 'done'} in {mins:.1f} min")
    return p.stdout

import torch
print(torch.__version__, 'cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime > Change runtime type > GPU, then Run all.")
print(' ', torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
'''

CELL_TESTS = '''
# 3 | The physics must hold on THIS machine before any result is trusted.
sh('pip -q install pytest', 'install pytest')
out = sh('python -m pytest -q', 'test suite')
line = [l for l in out.strip().split('\\n') if 'passed' in l]
if not line:
    raise RuntimeError('test suite reported no passes')
print('  suite:', line[-1])
'''

CELL_SMOKE = '''
# 4 | One-minute smoke test of the whole chain, before committing hours.
#     The --device cuda path has only ever been exercised on CPU.
import numpy as np
d = np.load('data/biosr_gt.npz', allow_pickle=False)
st = np.asarray([str(x) for x in d['structure']])
ce = np.asarray([str(x) for x in d['cell']])
sel = [i for s in sorted(set(st)) for i in np.where(st == s)[0][:2]]
np.savez_compressed('data/tiny_gt.npz', images=d['images'][sel],
                    structure=st[sel], cell=ce[sel], pixel_um=d['pixel_um'])
del d
print(f"  tiny source: {len(sel)} images, {sorted(set(st[sel]))}")

# Out-of-focus fraction. Pass the FRACTION, never a fixed --background: photons
# vary per patch, so a flat rate drifts from 75% haze to 7% across the range.
HAZE = __HAZE__
TAG = f'haze{int(HAZE * 100)}'
# Training scale. The old defaults (8 per cell, 30 epochs) trained the networks
# on 832 patches in under 3 min while RL ran its full 75 iterations -- a
# comparison rigged against the very models the U-Net experiment turns on.
PER_CELL = __PER_CELL__
EPOCHS = __EPOCHS__

sh('python scripts/train_reconstructor.py --model unet --device cuda --amp '
   f'--haze-frac {HAZE} --limit-cells 4 --epochs 1 --per-cell 1 --patch 128 '
   '--out /content/tiny_unet.pt', 'smoke train')
sh('python scripts/extract_features.py --reconstructor unet '
   f'--checkpoint /content/tiny_unet.pt --haze-frac {HAZE} --device cuda '
   '--crop 64 --kernel 31 --photons 100 --source data/tiny_gt.npz '
   '--out /content/tiny.npz', 'smoke extract')

n = len(np.load('/content/tiny.npz', allow_pickle=True)['feature_names'])
if n != 19:
    raise RuntimeError(f'expected 19 features, got {n}')
print('  smoke test passed: 19 features out of the trained-network path')
'''.replace('__HAZE__', str(HAZE)
      ).replace('__PER_CELL__', str(PER_CELL)
      ).replace('__EPOCHS__', str(EPOCHS))   # literal % rules out %-formatting

CELL_TRAIN = '''
# 5 | Train both reconstructors, ~30 min. Writes to Drive as it improves.
#     FINISHED MODELS ARE SKIPPED -- delete the .pt files to retrain.
for model in ('unet', 'unrolled'):
    dst = OUT / f'{model}_{TAG}.pt'
    if dst.exists():
        print(f"  {dst.name} already present -- skipping (delete it to retrain)")
        continue
    sh(f'python scripts/train_reconstructor.py --model {model} --device cuda '
       f'--amp --haze-frac {HAZE} --per-cell {PER_CELL} --epochs {EPOCHS} '
       f'--out "{dst}"', f'train {model}')
'''

CELL_EXTRACT = '''
# 6 | Extract the 19-feature caches: two models x two photon levels.
#     A haze-fraction mismatch against the checkpoint is refused, not run.
for model in ('unet', 'unrolled'):
    for ph, it in ((100, 75), (15, 30)):
        dst = OUT / f'{model}_photons_{ph}_{TAG}.npz'
        if dst.exists():
            print(f"  {dst.name} already present -- skipping")
            continue
        sh(f'python scripts/extract_features.py --reconstructor {model} '
           f'--checkpoint "{OUT}/{model}_{TAG}.pt" --haze-frac {HAZE} '
           f'--device cuda --crop 256 --photons {ph} --n-iter {it} '
           f'--source data/biosr_gt.npz --out "{dst}"',
           f'extract {model} @ {ph} photons')
'''

CELL_DONE = '''
# 7 | What to take home
caches = sorted(OUT.glob(f'*_{TAG}.npz'))
print(f"Copy these {len(caches)} files into experiments/features19/ locally:\\n")
for c in caches:
    print(f"  {c.name:34s} {c.stat().st_size / 1024:6.0f} KB")
print(f"\\nLeave the .pt weights in Drive ({OUT}) -- you do not need them locally.")
if len(caches) != 4:
    print(f"\\nWARNING: expected 4 caches, found {len(caches)}. Re-run this notebook; "
          "finished steps are skipped.")
else:
    print("\\nThen, on the laptop:")
    print(f"  python scripts/gate5_bound.py --features experiments/features19/*_{TAG}.npz --seeds 10")
    print("  python scripts/rotate_structures.py "
          f"--features experiments/features19/unet_photons_100_{TAG}.npz --seeds 10")
    print("\\nThe U-Net has no data-fidelity term, so it is the one reconstructor")
    print("that could overturn the mechanism established for RL and TV. It says")
    print("nothing about the sim-to-real transfer failure -- see docs/sim2real.md.")
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="notebooks/colab_gpu_session.ipynb")
    args = ap.parse_args()

    cells = [md(INTRO), code(CELL_FIND), code(CELL_RUNNER), code(CELL_TESTS),
             code(CELL_SMOKE), code(CELL_TRAIN), code(CELL_EXTRACT),
             code(CELL_DONE)]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {args.out}  ({len(cells)} cells, {n_code} code)")


if __name__ == "__main__":
    main()

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

BG = 45


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").split("\n")}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").split("\n")}


INTRO = """
# BioSR reliability -- GPU session

**Runtime > Change runtime type > GPU**, then **Runtime > Run all**. Nothing to
edit and nothing to click through: every step raises on failure, so Run all
halts at the first real problem rather than carrying a broken state into a
two-hour training job.

Put exactly **two files** anywhere in your Drive -- this notebook finds them:

* `biosr_gt.npz` (303 MB)
* `microscopy_code.zip` (~110 KB, rebuilt with `python scripts/pack_code.py`)

Roughly 1.5-3 h depending on the GPU allocated. Weights are written straight to
Drive as they improve, so a disconnect costs time but not work. Re-running the
notebook skips anything already finished.
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

# Results land beside their inputs rather than in a path you have to remember.
OUT = NPZ.parent / 'out'
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
    # Copy to the runtime disk rather than symlinking into Drive. The 303 MB
    # file is opened once per script invocation and there are six of them, so
    # reading it through the Drive FUSE mount every time costs far more than
    # this one copy.
    print('  copying the data to local disk (once)...')
    shutil.copy2(NPZ, local)
os.chdir(MS)
print(f"  code {stamp}")
print(f"  out  {OUT}")
if 'dirty' in stamp or 'STALE' in stamp:
    print("  WARNING: this code zip was not built from a clean commit. A stale "
          "zip is silent -- everything imports and you get last week's results.")
'''

CELL_RUNNER = '''
# 2 | A runner that makes "Run all" stop at the first real failure.
#     IPython's `!cmd` swallows non-zero exits, which is exactly wrong here:
#     a failed training step would otherwise be followed by an extraction step
#     that loads a stale checkpoint and reports plausible numbers.
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
#     The suite pins Airy agreement, FWHM scaling, the exact blur adjoint,
#     Poisson-Gaussian variance, seed reproducibility and eight bug regressions.
sh('pip -q install pytest', 'install pytest')
out = sh('python -m pytest -q', 'test suite')
line = [l for l in out.strip().split('\\n') if 'passed' in l]
if not line:
    raise RuntimeError('test suite reported no passes')
print('  suite:', line[-1])
'''

CELL_SMOKE = '''
# 4 | One-minute GPU smoke test, BEFORE committing hours to training.
#     The --device cuda path has only ever been exercised on CPU, so prove the
#     whole chain -- train, checkpoint, load, extract -- on 8 images first.
import numpy as np
d = np.load('data/biosr_gt.npz', allow_pickle=False)
st = np.asarray([str(x) for x in d['structure']])
ce = np.asarray([str(x) for x in d['cell']])
sel = [i for s in sorted(set(st)) for i in np.where(st == s)[0][:2]]
np.savez_compressed('data/tiny_gt.npz', images=d['images'][sel],
                    structure=st[sel], cell=ce[sel], pixel_um=d['pixel_um'])
del d
print(f"  tiny source: {len(sel)} images, {sorted(set(st[sel]))}")

# Real BioSR widefield carries 19-68% of its mean as out-of-focus light
# (median 31%) against this project's old default of 5%; see docs/sim2real.md.
# The RL and TV caches to compare against are the *_bg45 set, so training at
# anything else makes the comparison meaningless.
BG = __BG__

sh('python scripts/train_reconstructor.py --model unet --device cuda --amp '
   f'--background {BG} --limit-cells 4 --epochs 1 --per-cell 1 --patch 128 '
   '--out /content/tiny_unet.pt', 'smoke train')
sh('python scripts/extract_features.py --reconstructor unet '
   f'--checkpoint /content/tiny_unet.pt --background {BG} --device cuda '
   '--crop 64 --kernel 31 --photons 100 --source data/tiny_gt.npz '
   '--out /content/tiny.npz', 'smoke extract')

n = len(np.load('/content/tiny.npz', allow_pickle=True)['feature_names'])
if n != 19:
    raise RuntimeError(f'expected 19 features, got {n}')
print('  smoke test passed: 19 features out of the trained-network path')
'''.replace('__BG__', str(BG))   # literal % in the comments rules out %-formatting

CELL_TRAIN = '''
# 5 | THE REAL JOB: train both reconstructors. ~1-2 h total.
#     --out is on Drive, so every improvement survives a disconnect. Already
#     finished models are skipped, which makes the notebook safe to re-run.
for model in ('unet', 'unrolled'):
    dst = OUT / f'{model}_bg{BG}.pt'
    if dst.exists():
        print(f"  {dst.name} already present -- skipping (delete it to retrain)")
        continue
    sh(f'python scripts/train_reconstructor.py --model {model} --device cuda '
       f'--amp --background {BG} --out "{dst}"', f'train {model}')
'''

CELL_EXTRACT = '''
# 6 | Extract the 19-feature caches: two models x two photon levels.
#     extract_features.py refuses a checkpoint whose training background differs
#     from this one, because a learned reconstructor evaluated under a different
#     forward model than it was trained on measures domain shift, not
#     reliability. Cheaper than the RL caches -- one network forward per sample
#     instead of 75 Richardson-Lucy iterations -- so the 16-probe residual
#     family dominates the cost.
for model in ('unet', 'unrolled'):
    for ph, it in ((100, 75), (15, 30)):
        dst = OUT / f'{model}_photons_{ph}_bg{BG}.npz'
        if dst.exists():
            print(f"  {dst.name} already present -- skipping")
            continue
        sh(f'python scripts/extract_features.py --reconstructor {model} '
           f'--checkpoint "{OUT}/{model}_bg{BG}.pt" --background {BG} '
           f'--device cuda --crop 256 --photons {ph} --n-iter {it} '
           f'--source data/biosr_gt.npz --out "{dst}"',
           f'extract {model} @ {ph} photons')
'''

CELL_DONE = '''
# 7 | What to take home
caches = sorted(OUT.glob(f'*_bg{BG}.npz'))
print(f"Copy these {len(caches)} files into experiments/features19/ locally:\\n")
for c in caches:
    print(f"  {c.name:34s} {c.stat().st_size / 1024:6.0f} KB")
print(f"\\nLeave the .pt weights in Drive ({OUT}) -- you do not need them locally.")
if len(caches) != 4:
    print(f"\\nWARNING: expected 4 caches, found {len(caches)}. Re-run this notebook; "
          "finished steps are skipped.")
else:
    print("\\nThen, on the laptop:")
    print("  python scripts/gate5_bound.py --features experiments/features19/*_bg45.npz --seeds 10")
    print("  python scripts/rotate_structures.py "
          "--features experiments/features19/unet_photons_100_bg45.npz --seeds 10")
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

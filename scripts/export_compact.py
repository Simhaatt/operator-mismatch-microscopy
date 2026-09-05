"""Export just the ground-truth images into one portable file.

The BioSR archives total 8 GB, but this project only ever opens ``SIM_gt.mrc``
(or ER's per-level equivalent) -- 441 MB of the 8 GB. The rest is raw SIM stacks,
nine frames per level and nine levels per cell, which nothing here reads.

This writes a single ``.npz`` holding every ground-truth image as uint16 plus the
structure and cell labels needed for cell-blocked splitting, so a Colab or Kaggle
session can be fed a few hundred megabytes instead of eight gigabytes.

Images are kept as uint16 at their native size by default: normalisation is
per-image and by percentile (see :func:`src.data.biosr.load_object`), so storing
raw integers loses nothing and compresses far better than floats.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.biosr import index_biosr  # noqa: E402
from src.data.mrc import read_mrc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/biosr")
    ap.add_argument("--crop", type=int, default=None,
                    help="optional centre crop; omit to keep the native 1004 px")
    ap.add_argument("--out", default="data/biosr_gt.npz")
    args = ap.parse_args()

    entries = index_biosr(args.root)
    if not entries:
        raise SystemExit(f"no BioSR archives found under {args.root!r}")

    images, structures, cells, pixels = [], [], [], []
    handles: dict[str, zipfile.ZipFile] = {}
    try:
        for i, e in enumerate(entries):
            z = handles.setdefault(e.archive, zipfile.ZipFile(e.archive))
            arr, hdr = read_mrc(z.read(e.member))
            img = arr[0]
            if args.crop:
                h, w = img.shape
                t, l = (h - args.crop) // 2, (w - args.crop) // 2
                img = img[t : t + args.crop, l : l + args.crop]
            images.append(np.ascontiguousarray(img, dtype=np.uint16))
            structures.append(e.structure)
            cells.append(e.cell)
            pixels.append(hdr.pixel_um)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(entries)}", flush=True)
    finally:
        for z in handles.values():
            z.close()

    # Pixel size must be identical -- that is the physics. Field of view need not
    # be: BioSR contains at least one cell imaged over a wider camera region
    # (F-actin/Cell_024, raw 758 px rather than 502, still 2x upsampled at the
    # same 31.3 nm). Cropping such images to the common size is safe; silently
    # mixing *pixel* scales would not be, so that is refused.
    if len(set(pixels)) != 1:
        raise SystemExit(f"images differ in pixel size ({set(pixels)}); refusing to mix scales")

    sizes = {img.shape[0] for img in images}
    if len(sizes) > 1:
        target = min(sizes)
        print(f"  field sizes {sorted(sizes)} at a common pixel size; "
              f"centre-cropping all to {target} px")
        cropped = []
        for img in images:
            h, w = img.shape
            t, l = (h - target) // 2, (w - target) // 2
            cropped.append(np.ascontiguousarray(img[t : t + target, l : l + target]))
        images = cropped
    shapes = {img.shape for img in images}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=np.stack(images),
        structure=np.asarray(structures),
        cell=np.asarray(cells),
        pixel_um=np.asarray([pixels[0]]),
    )
    mb = out.stat().st_size / 1048576
    print(f"\nwrote {out}  ({len(images)} images, {shapes.pop()}, "
          f"pixel {pixels[0]:.4f} um, {mb:.1f} MB)")


if __name__ == "__main__":
    main()

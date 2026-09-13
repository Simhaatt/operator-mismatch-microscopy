"""Build a compact ground-truth export from the W2S raw archive.

W2S (Zhou et al., ECCVW 2020, doi 10.5281/zenodo.3895807) is the second dataset
for this project. The archive is 40.7 GiB compressed and 73.6 GiB expanded, but
the simulated arm needs only the SIM reconstructions -- 360 arrays of
1024x1024 uint16, about 720 MiB -- so nothing else is extracted here. The
400-shot widefield stacks are the other 72 GiB and are handled separately by
``ingest_w2s_widefield.py``, which never materialises them on disk either.

Output matches the layout ``src.data.biosr.index_npz`` already reads, so the
existing extraction and gate scripts run against W2S without modification:
``images`` (N, H, W) uint16, ``structure``, ``cell``, ``pixel_um``.

Optics, for the record
----------------------
Taken from the paper's supplementary material rather than assumed:

* Plan Apochromat TIRF 100x, **NA 1.49**, Andor iXon3 EMCCD.
* Excitation at 488, 561 and 640 nm; the three channels are three laser lines,
  not three specimen classes -- see the caveat below.
* Widefield 512x512, SIM 1024x1024, so the SIM grid is 2x finer.
* Pixel size is **not stated** in the paper. The Nikon N-SIM standard
  configuration places a 2.5x relay before a 16 um-pitch iXon3, giving
  16/(100*2.5) = 64 nm at the widefield grid and **32 nm** on the SIM grid.
  That is the documented configuration for this instrument, and it lands within
  2% of BioSR's 31.3 nm, but it is a reconstruction from the instrument
  specification rather than a number the authors printed.

The caveat that matters
-----------------------
``structure`` here is the **wavelength channel**, not a specimen class. BioSR
has four genuinely different structures (CCPs, ER, F-actin, microtubules) and
the content-shift experiments hold one out. W2S has one specimen type imaged in
three colours, and the paper never says what the channels label. Holding out a
W2S channel is therefore a weaker form of content shift -- the same cells under
a different fluorophore -- and must not be reported as equivalent to holding out
a structure in BioSR.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Nikon N-SIM: 16 um iXon3 pitch behind 100x objective and a 2.5x relay.
W2S_PIXEL_UM_SIM = 0.032
W2S_NA = 1.49
# Excitation lines; emission peaks are not stated, so downstream code that needs
# an emission wavelength should say what it assumed.
W2S_EXCITATION_NM = (488, 561, 640)
# The readme's own split: FOVs 001-080 train, 081-120 test.
W2S_TEST_FOV_FROM = 81


def channel_label(ch: int) -> str:
    """Name a channel by its laser line, so it can never read as a structure."""
    return f"ch{ch}_{W2S_EXCITATION_NM[ch]}nm"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="W2S_raw.zip")
    ap.add_argument("--out", default="data/w2s_gt.npz")
    ap.add_argument("--limit-fov", type=int, default=None,
                    help="use at most this many FOVs; for a quick check")
    args = ap.parse_args()

    zpath = Path(args.zip)
    if not zpath.exists():
        sys.exit(f"missing {zpath}")

    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if re.search(r"/sim_channel\d\.npy$", n)]
    if not names:
        sys.exit("no sim_channel*.npy entries found -- is this the W2S raw archive?")

    def fov_of(n: str) -> str:
        m = re.search(r"(Image\d+)/", n)
        return m.group(1) if m else "?"

    def ch_of(n: str) -> int:
        return int(re.search(r"sim_channel(\d)\.npy$", n).group(1))

    names.sort(key=lambda n: (fov_of(n), ch_of(n)))
    fovs = sorted({fov_of(n) for n in names})
    if args.limit_fov:
        keep = set(fovs[: args.limit_fov])
        names = [n for n in names if fov_of(n) in keep]
        fovs = sorted(keep)

    print(f"W2S ingest: {len(fovs)} FOVs, {len(names)} SIM reconstructions")

    images, structure, cell = [], [], []
    shapes = set()
    for i, n in enumerate(names):
        a = np.load(io.BytesIO(z.read(n)))
        if a.dtype != np.uint16:
            sys.exit(f"{n}: expected uint16, got {a.dtype}")
        shapes.add(a.shape)
        images.append(a)
        structure.append(channel_label(ch_of(n)))
        cell.append(fov_of(n))
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)

    if len(shapes) != 1:
        sys.exit(f"inconsistent SIM shapes: {shapes}")
    arr = np.stack(images)
    print(f"  stacked {arr.shape} {arr.dtype}, {arr.nbytes / 2**20:.0f} MiB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        images=arr,
        structure=np.asarray(structure),
        cell=np.asarray(cell),
        pixel_um=np.asarray([W2S_PIXEL_UM_SIM]),
        na=np.asarray([W2S_NA]),
        excitation_nm=np.asarray(W2S_EXCITATION_NM),
        # Recorded so a later reader cannot mistake the channel label for a
        # specimen class, and knows the pixel size was derived not quoted.
        structure_is_wavelength_channel=np.asarray([True]),
        pixel_um_is_inferred=np.asarray([True]),
        test_fov_from=np.asarray([W2S_TEST_FOV_FROM]),
    )
    mb = out.stat().st_size / 2**20
    print(f"wrote {out} ({mb:.0f} MiB)")

    import collections
    print("  per-channel counts:", dict(collections.Counter(structure)))
    print(f"  intensity: min {arr.min()}, max {arr.max()}, mean {arr.mean():.1f}")


if __name__ == "__main__":
    main()

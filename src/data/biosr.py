"""BioSR loading with cell-blocked splits.

Reads ``SIM_gt.mrc`` ground-truth images straight out of the distributed zip
archives -- no 8 GB extraction step, which matters on a Kaggle worker with a
20 GB working disk.

Two design points that protect the science:

* **Splits are blocked by cell, never by patch.** Every crop from one cell stays
  in one split. Patches from the same field of view share structure, background
  and noise realisation, so splitting after cropping leaks the test set into
  training and inflates every downstream number.
* **Normalisation is per image, by percentile, and fixed.** The SIM ground truth
  is uint16 spanning the full 16-bit range. Downstream SSIM needs a *global*
  ``data_range``, so images are mapped to [0, 1] here once and the metric is
  always called with ``data_range=1.0``. Inferring a range per image would make
  errors incomparable across the shift matrix.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .mrc import read_mrc

__all__ = [
    "BioSREntry",
    "index_biosr",
    "index_npz",
    "load_object",
    "split_by_cell",
    "STRUCTURES",
]

# Populated lazily when a compact .npz export is indexed, so the same loader
# serves both the 8 GB archives locally and a 300 MB export on a hosted runtime.
_NPZ_CACHE: dict[str, dict] = {}

STRUCTURES = ("CCPs", "ER", "F-actin", "Microtubules")


@dataclass(frozen=True)
class BioSREntry:
    """One ground-truth image, identified by the cell it came from."""

    structure: str
    cell: str
    archive: str
    member: str

    @property
    def key(self) -> str:
        return f"{self.structure}/{self.cell}"


def index_biosr(
    root: str | Path = "data/biosr", structures: tuple[str, ...] = STRUCTURES
) -> list[BioSREntry]:
    """List one ground-truth image per cell, across the structure archives.

    BioSR uses **two different layouts**, which is easy to miss and silently
    drops a quarter of the data:

    * Fixed-cell structures (CCPs, F-actin, Microtubules) store a single
      ``Cell_XXX/SIM_gt.mrc``.
    * ER is a *live-cell* dataset, so the specimen moves between exposures and
      every signal level carries its own reconstruction:
      ``Cell_XXX/GTSIM/GTSIM_level_NN.mrc``. We take the highest level per cell,
      which is the cleanest ground truth available for that field.
    """
    root = Path(root)
    entries: list[BioSREntry] = []
    for structure in structures:
        archive = root / f"{structure}.zip"
        if not archive.exists():
            continue
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
        direct = [n for n in names if n.endswith("/SIM_gt.mrc")]
        for name in direct:
            entries.append(
                BioSREntry(structure, name.split("/")[-2], str(archive), name)
            )
        if direct:
            continue
        # Live-cell layout: keep only the highest level for each cell.
        best: dict[str, str] = {}
        for name in names:
            part = name.split("/")
            if len(part) >= 4 and part[-2] == "GTSIM" and part[-1].startswith("GTSIM_level_"):
                cell = part[-3]
                if cell not in best or name > best[cell]:
                    best[cell] = name
        for cell, name in best.items():
            entries.append(BioSREntry(structure, cell, str(archive), name))
    return sorted(entries, key=lambda e: (e.structure, e.cell))


def index_npz(path: str | Path) -> list[BioSREntry]:
    """Index a compact export written by ``scripts/export_compact.py``.

    The BioSR archives are 8 GB, of which this project reads 441 MB. On a hosted
    runtime -- Colab, Kaggle -- moving the export instead of the archives is the
    difference between a 300 MB upload and an 8 GB one.
    """
    path = str(Path(path))
    if path not in _NPZ_CACHE:
        d = np.load(path, allow_pickle=False)
        _NPZ_CACHE[path] = {
            "images": d["images"],
            "structure": [str(x) for x in d["structure"]],
            "cell": [str(x) for x in d["cell"]],
            "pixel_um": float(d["pixel_um"][0]),
        }
    c = _NPZ_CACHE[path]
    return [
        BioSREntry(c["structure"][i], c["cell"][i], f"npz:{path}", str(i))
        for i in range(len(c["structure"]))
    ]


def load_object(
    entry: BioSREntry,
    *,
    crop: int | None = 512,
    dtype: torch.dtype = torch.float32,
    p_low: float = 0.1,
    p_high: float = 99.9,
) -> torch.Tensor:
    """Load one ground-truth image as a non-negative tensor scaled to [0, 1].

    Percentile normalisation follows the convention of the BioSR paper. The
    result is clamped at zero because it is used as a photon density, and SIM
    reconstructions ring slightly negative.
    """
    if entry.archive.startswith("npz:"):
        cache = _NPZ_CACHE[entry.archive[4:]]
        img = np.asarray(cache["images"][int(entry.member)], dtype=np.float64)
    else:
        with zipfile.ZipFile(entry.archive) as z:
            arr, _ = read_mrc(z.read(entry.member))
        img = np.asarray(arr[0], dtype=np.float64)

    if crop is not None:
        h, w = img.shape
        if crop > min(h, w):
            raise ValueError(f"crop {crop} exceeds image size {h}x{w}")
        top, left = (h - crop) // 2, (w - crop) // 2
        img = img[top : top + crop, left : left + crop]

    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        raise ValueError(f"degenerate intensity range in {entry.key}")
    img = (img - lo) / (hi - lo)
    return torch.from_numpy(img).to(dtype).clamp_(0.0, 1.0)


def split_by_cell(
    entries: list[BioSREntry],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    *,
    seed: int = 0,
    held_out_structure: str | None = None,
) -> dict[str, list[BioSREntry]]:
    """Partition entries into train / calibration / test, blocked by cell.

    ``held_out_structure`` removes one biological structure from train and
    calibration entirely and returns it as ``ood``, giving the content-shift
    axis: a structure the reliability model has never seen.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")

    ood = [e for e in entries if e.structure == held_out_structure]
    pool = [e for e in entries if e.structure != held_out_structure]

    # Shuffle at the level of cells, which is the unit that must not be split.
    keys = sorted({e.key for e in pool})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    assign = {
        **{k: "train" for k in keys[:n_train]},
        **{k: "cal" for k in keys[n_train : n_train + n_cal]},
        **{k: "test" for k in keys[n_train + n_cal :]},
    }
    out: dict[str, list[BioSREntry]] = {"train": [], "cal": [], "test": [], "ood": ood}
    for e in pool:
        out[assign[e.key]].append(e)
    return out

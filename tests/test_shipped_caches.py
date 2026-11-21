"""The shipped feature caches are intact and have the layout the analyses expect."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"
SUMS = EXP / "SHA256SUMS"


def _entries():
    for line in SUMS.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        yield digest, name.lstrip("*")


@pytest.mark.parametrize("digest,name", list(_entries()))
def test_checksum(digest, name):
    p = EXP / name
    assert p.exists(), f"missing shipped file {p}"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", [n for _, n in _entries() if n.endswith(".npz")])
def test_cache_layout(name):
    d = np.load(EXP / name, allow_pickle=True)
    nfeat = len(d["feature_names"])
    assert nfeat == 19
    for split in ("train", "cal", "test", "ood"):
        X, e = d[f"{split}_X"], d[f"{split}_e"]
        assert X.shape == (len(e), nfeat)
        assert np.isfinite(X).all() and np.isfinite(e).all()
        assert len(d[f"{split}_cell"]) == len(e)


@pytest.mark.parametrize("name", [n for _, n in _entries() if n.endswith(".npz")])
def test_splits_are_cell_blocked(name):
    """No cell appears in more than one split: the guarantee the paper relies on.

    One documented exception: in the genuine-widefield W2S cache the held-out
    split is an excitation *channel*, so its fields of view also appear, in the
    other two channels, in training. Train, calibration and test stay disjoint.
    """
    d = np.load(EXP / name, allow_pickle=True)
    splits = ["train", "cal", "test"] + ([] if "real_w2s" in name else ["ood"])
    cells = {s: set(d[f"{s}_cell"].tolist()) for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not (cells[a] & cells[b]), f"{name}: cells shared by {a} and {b}"

"""Minimal MRC/CCP4 reader for the BioSR archives.

BioSR ships ``.mrc`` files rather than TIFF. Rather than take a dependency on
``mrcfile`` -- which would have to be installed on every Kaggle worker -- this
reads the ~40 lines of header we actually need. Only the modes BioSR uses are
supported, and anything else raises rather than guessing.

Layout: a fixed 1024-byte header, then an optional extended header of ``nsymbt``
bytes, then the data. Axis order in the file is (nz, ny, nx), i.e. z-major.

Note BioSR writes the pixel size into the cell-dimension field in
**micrometres**, not the Angstroms the format nominally specifies. Values are
returned as stored, and :func:`read_mrc` reports them as micrometres because
that is what the data means: SIM ground truth is 0.0313 and raw/widefield frames
are 0.0626, exactly the 2x SIM upsampling.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

__all__ = ["MrcHeader", "read_mrc_header", "read_mrc"]

# MRC mode -> numpy dtype. Only what BioSR contains.
_MODES: dict[int, np.dtype] = {
    0: np.dtype("<i1"),
    1: np.dtype("<i2"),
    2: np.dtype("<f4"),
    6: np.dtype("<u2"),
}

_HEADER_BYTES = 1024


@dataclass(frozen=True)
class MrcHeader:
    """The subset of the MRC header this project needs."""

    nx: int
    ny: int
    nz: int
    mode: int
    nsymbt: int
    pixel_um: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Array shape in numpy order, ``(nz, ny, nx)``."""
        return (self.nz, self.ny, self.nx)

    @property
    def dtype(self) -> np.dtype:
        return _MODES[self.mode]


def read_mrc_header(buf: bytes) -> MrcHeader:
    """Parse an MRC header from the first 1024 bytes of a file."""
    if len(buf) < _HEADER_BYTES:
        raise ValueError(f"MRC header truncated: {len(buf)} < {_HEADER_BYTES} bytes")
    nx, ny, nz, mode = struct.unpack("<4i", buf[0:16])
    mx = struct.unpack("<i", buf[28:32])[0]
    xlen = struct.unpack("<f", buf[40:44])[0]
    nsymbt = struct.unpack("<i", buf[92:96])[0]
    if mode not in _MODES:
        raise ValueError(
            f"unsupported MRC mode {mode}; BioSR uses 2 (float32) and 6 (uint16)"
        )
    if min(nx, ny, nz) <= 0:
        raise ValueError(f"implausible MRC dimensions ({nx}, {ny}, {nz})")
    # BioSR stores the pixel size directly in the cell length with mx == 1.
    pixel = float(xlen / mx) if mx > 0 else float(xlen)
    return MrcHeader(nx=nx, ny=ny, nz=nz, mode=mode, nsymbt=nsymbt, pixel_um=pixel)


def read_mrc(data: bytes) -> tuple[np.ndarray, MrcHeader]:
    """Read a whole MRC file from bytes.

    Returns ``(array, header)`` with the array shaped ``(nz, ny, nx)``. For the
    single-plane ground-truth images ``nz == 1``; for raw SIM stacks ``nz == 9``
    (three phases x three orientations).
    """
    header = read_mrc_header(data)
    start = _HEADER_BYTES + max(header.nsymbt, 0)
    count = header.nx * header.ny * header.nz
    expected = count * header.dtype.itemsize
    payload = data[start : start + expected]
    if len(payload) != expected:
        raise ValueError(
            f"MRC payload truncated: got {len(payload)} bytes, expected {expected}"
        )
    arr = np.frombuffer(payload, dtype=header.dtype, count=count)
    return arr.reshape(header.shape), header

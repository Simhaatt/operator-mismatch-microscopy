"""Characterise the surrogate objects before committing to an operator.

Decision D2 treats BioSR's high-resolution SIM images as the object ``x``. But a
SIM reconstruction is itself band-limited, which raises the sharpest objection to
the whole study: if the simulated optical cutoff sits at or above the object's
own band limit, no genuine structure lies in the operator's null space, and the
"reconstructors hallucinate beyond the cutoff" story becomes an artefact of the
data pipeline rather than a property of microscopy.

This script settles that empirically. It measures the objects' radial power
spectrum, then reports how much object energy each candidate operator would push
into its null space.

The operator here is INCOHERENT: fluorescence imaging convolves with an
intensity PSF, whose transfer function (the OTF) is the pupil autocorrelation.
Its support radius is therefore ``2 NA / lambda``, not the amplitude cutoff
``NA / lambda``. Using the amplitude cutoff overstates the null-space band by a
factor of two -- the single easiest way to get this design decision wrong.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.image import radial_power_spectrum  # noqa: E402


def load_objects(pattern: str) -> torch.Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matched {pattern!r}")
    arr = np.stack([np.squeeze(tifffile.imread(f)) for f in files]).astype(np.float64)
    return torch.from_numpy(arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="data/biosr_pilot/test_gt/*.tif")
    ap.add_argument("--wavelength-um", type=float, default=0.525)
    ap.add_argument("--pixel-um", type=float, default=0.0313)
    ap.add_argument("--band-threshold", type=float, default=1e-4,
                    help="power relative to DC defining the object band edge")
    args = ap.parse_args()

    x = load_objects(args.pattern)
    print(f"objects: {tuple(x.shape)}  dtype={x.dtype}")
    print(f"  range [{float(x.min()):+.5f}, {float(x.max()):+.5f}]  mean={float(x.mean()):.5f}")
    print(f"  negative pixels: {float((x < 0).to(torch.float64).mean()):.4%}"
          "  (SIM ringing; clamped to 0 on load)")

    freq, power = radial_power_spectrum(x.clamp_min(0.0))
    power = power / power[0]
    # Energy fraction above each frequency, weighted by the annulus area (f df).
    tail = torch.flip(torch.cumsum(torch.flip(power * freq, [0]), 0), [0])
    tail = tail / tail[0]

    def energy_above(f: float) -> float:
        j = int(torch.searchsorted(freq, torch.tensor(f, dtype=freq.dtype)))
        return float(tail[min(j, len(tail) - 1)])

    print("\nradial power spectrum (normalised to the DC bin):")
    print("     f (cyc/px)      power      energy above f")
    for i in range(0, len(freq), max(1, len(freq) // 14)):
        print(f"      {float(freq[i]):.4f}      {float(power[i]):.3e}     {float(tail[i]):.4f}")

    above = (power > args.band_threshold).nonzero().flatten()
    f_edge = float(freq[above[-1]]) if len(above) else 0.0
    print(f"\nobject band edge (power > {args.band_threshold:g} of DC): "
          f"{f_edge:.4f} cyc/px ({f_edge / 0.5:.1%} of pixel Nyquist)")

    print("\ncandidate operators (lambda = "
          f"{args.wavelength_um} um, pixel = {args.pixel_um * 1000:.1f} nm):")
    print("     NA    f_c=NA/lam   OTF cut=2f_c   object energy in null space   SIM gain")
    for na in (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.27, 1.4):
        fc = na / args.wavelength_um * args.pixel_um
        otf = 2.0 * fc
        if otf >= 0.5:
            print(f"    {na:.2f}    {fc:.4f}       {otf:.4f}      -- exceeds pixel Nyquist --")
            continue
        print(f"    {na:.2f}    {fc:.4f}       {otf:.4f}           {energy_above(otf):.4f}"
              f"                {f_edge / otf:.2f}x")

    # Principled choice: reproduce the real ~2x linear-SIM resolution gain, so
    # the simulated widefield operator stands in the same relation to the object
    # as BioSR's actual widefield arm does to its SIM reconstruction.
    otf_target = f_edge / 2.0
    na_target = otf_target * args.wavelength_um / (2.0 * args.pixel_um)
    print("\nprincipled choice -- match the real ~2x linear-SIM resolution gain:")
    print(f"  object band edge {f_edge:.4f} cyc/px  ->  target OTF cutoff {otf_target:.4f} cyc/px")
    print(f"  ->  NA = {na_target:.3f}")
    print(f"  ->  {energy_above(otf_target):.1%} of object energy falls in the operator's null")
    print("      space: genuine structure a deconvolver must invent, and that a measurement")
    print("      residual is structurally blind to (Bhadra et al., IEEE TMI 2021).")
    print("\n  Derived from the data, not chosen for convenience: it reproduces the")
    print("  widefield-to-SIM relationship BioSR actually contains.")


if __name__ == "__main__":
    main()

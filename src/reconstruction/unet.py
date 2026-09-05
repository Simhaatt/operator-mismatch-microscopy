"""CARE-style U-Net reconstructor -- the case with no data-fidelity term.

Richardson-Lucy and Poisson-TV both minimise an explicit fidelity term, and both
therefore drive ``||H_0 x_hat - y||`` toward the shot-noise floor whether or not
``H_0`` is the operator that actually formed the data. That is the mechanism the
paper rests on, and it is a property of the *algorithm*, not of the physics.

This network is the control. It never sees a PSF and never evaluates a residual;
it is a learned image-to-image map. If its residual carries more information
about operator mismatch than RL's and TV's do, then the mechanism is a statement
about fidelity-minimising reconstruction rather than about reconstruction in
general, and the paper's scope narrows accordingly. That is the single most
informative experiment left.

Architecture follows Weigert et al.'s CARE: a symmetric encoder-decoder with skip
connections, which is the standard fluorescence-restoration baseline rather than
anything novel. The point is to be representative, not competitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["UNet", "normalise_measurement"]


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Encoder-decoder with skip connections.

    Parameters
    ----------
    base:
        Channel width at the finest resolution; doubles each level.
    depth:
        Number of downsampling stages. Inputs must be divisible by
        ``2 ** depth``, which is checked rather than left to produce a shape
        error deep inside the decoder.
    residual:
        Predict a correction to the input rather than the image itself. Standard
        for restoration, and it keeps the network from having to re-learn the
        identity.
    """

    def __init__(self, base: int = 32, depth: int = 3, residual: bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.residual = residual
        self.encoders = nn.ModuleList()
        ch = 1
        for d in range(depth):
            self.encoders.append(_block(ch, base * 2**d))
            ch = base * 2**d
        self.bottleneck = _block(ch, ch * 2)
        ch *= 2
        self.decoders = nn.ModuleList()
        for d in reversed(range(depth)):
            self.decoders.append(_block(ch + base * 2**d, base * 2**d))
            ch = base * 2**d
        self.head = nn.Conv2d(ch, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 3
        if squeeze:
            x = x[:, None]
        if x.shape[-1] % 2**self.depth or x.shape[-2] % 2**self.depth:
            raise ValueError(
                f"input {tuple(x.shape[-2:])} is not divisible by {2**self.depth}; "
                "pad or crop before calling"
            )
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(torch.cat([self.up(h), skip], dim=1))
        out = self.head(h)
        if self.residual:
            out = out + x
        out = out.clamp_min(0.0)  # a photon density cannot be negative
        return out[:, 0] if squeeze else out


def normalise_measurement(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a measurement to unit mean, returning the scale for inversion.

    Photon budgets span 15 to 600 across BioSR, so feeding raw counts would make
    the network's job partly about gain rather than structure. Normalising per
    image and restoring the scale afterwards keeps the learned map about
    structure, and keeps a single network valid across signal levels.
    """
    scale = y.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return y / scale, scale

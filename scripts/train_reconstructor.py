"""Train a learned reconstructor on the NOMINAL operator.

Both learned baselines are trained under the nominal PSF only. That is the
situation the paper is about: a model is built and calibrated for one optical
configuration, then meets a microscope that has drifted. Training across
aberrations would answer a different and easier question.

Training pairs are **pre-generated once with fixed seeds** rather than simulated
on the fly. On-the-fly generation would either bottleneck a GPU on CPU-side FFTs
or force CUDA RNG, which produces different streams from the same seed and would
break the cross-device reproducibility the rest of the project maintains.
Pre-generating keeps every measurement bit-reproducible.

Photon level is randomised per patch across BioSR's real 15-600 range, so one
network serves every signal level instead of needing one per level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.biosr import index_biosr, index_npz, load_object, split_by_cell  # noqa: E402
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.forward_model import (  # noqa: E402
    BIOSR_PHOTON_RANGE, NoiseConfig, sample_seed, simulate_measurement,
)
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.unet import UNet, normalise_measurement  # noqa: E402
from src.reconstruction.unrolled import UnrolledNet  # noqa: E402


def make_pairs(entries, psf0, cfg, n_per_cell, patch, device, dtype, tag, seed=0):
    """Pre-generate (measurement, target) patches under the nominal operator."""
    rng = np.random.default_rng(seed)
    ys, xs = [], []
    for i, e in enumerate(entries):
        full = load_object(e, crop=None, dtype=dtype).to(device)
        h, w = full.shape
        for j in range(n_per_cell):
            top = int(rng.integers(0, h - patch + 1))
            left = int(rng.integers(0, w - patch + 1))
            x = full[top : top + patch, left : left + patch]
            if float(x.mean()) < 1e-4:
                continue  # skip empty background crops
            photons = float(rng.uniform(*BIOSR_PHOTON_RANGE))
            noise = NoiseConfig(photons=photons, background=5.0, read_noise_e=1.5)
            y = simulate_measurement(x, psf0, noise, seed=sample_seed(tag, e.key, j))
            ys.append(y.cpu())
            xs.append(x.cpu())
        if (i + 1) % 20 == 0:
            print(f"    {tag}: {i+1}/{len(entries)} cells, {len(ys)} patches", flush=True)
    return torch.stack(ys), torch.stack(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("unet", "unrolled"), default="unet")
    ap.add_argument("--source", default="data/biosr_gt.npz")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kernel", type=int, default=191)
    ap.add_argument("--held-out", default="CCPs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--out", default="experiments/models/unet.pt")
    args = ap.parse_args()

    dtype, device = torch.float32, torch.device(args.device)
    torch.manual_seed(0)
    cfg = PupilConfig(n_out=args.kernel)
    psf0 = pupil_psf(None, cfg, warn=False)[0].to(device=device, dtype=dtype)

    idx = index_npz(args.source) if args.source else index_biosr()
    sp = split_by_cell(idx, held_out_structure=args.held_out)
    train, val = sp["train"], sp["cal"]
    if args.limit_cells:
        train, val = train[: args.limit_cells], val[: max(args.limit_cells // 4, 1)]

    print(f"model={args.model} device={args.device} patch={args.patch} "
          f"cells: train={len(train)} val={len(val)}")
    print("  pre-generating training pairs (fixed seeds, bit-reproducible)...")
    ytr, xtr = make_pairs(train, psf0, cfg, args.per_cell, args.patch, device, dtype, "train")
    yva, xva = make_pairs(val, psf0, cfg, 2, args.patch, device, dtype, "val")
    print(f"  {len(ytr)} training patches, {len(yva)} validation")

    model = (UNet(base=32, depth=3) if args.model == "unet"
             else UnrolledNet(n_stages=5, width=32)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    print(f"  {n_par/1e6:.2f}M parameters")

    def forward(y_batch):
        """Estimate on the same scale convention as RL and TV: mean(x_hat) ~ mean(y).

        A blur preserves the mean, so any estimate satisfying ``H x_hat ~ y`` has
        roughly the measurement's mean. RL and TV produce that naturally; the
        U-Net must be made to, or its ``xhat_std`` feature would live on a
        different scale from the classical reconstructors' and the comparison
        would be measuring units rather than reliability.
        """
        if args.model == "unet":
            yn, sc = normalise_measurement(y_batch)
            return model(yn) * sc
        return model(y_batch, psf0)

    def target(x_batch, y_batch):
        """Rescale the object to the measurement's scale, so the loss is well posed.

        The stored objects are percentile-normalised to [0, 1] with mean ~0.1,
        while measurements are in photoelectrons with mean ~100. Regressing one
        onto the other directly makes the MSE enormous and dominated by gain
        rather than structure.
        """
        mx = x_batch.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        my = y_batch.mean(dim=(-2, -1), keepdim=True)
        return x_batch * (my / mx)

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        tot, t0 = 0.0, time.perf_counter()
        for i in range(0, len(perm), args.batch):
            sel = perm[i : i + args.batch]
            yb, xb = ytr[sel].to(device), xtr[sel].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = nn.functional.mse_loss(forward(yb), target(xb, yb))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(sel)
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = torch.cat([forward(yva[i : i + args.batch].to(device)).cpu()
                               for i in range(0, len(yva), args.batch)])
            vs = float(ssim(scale_match(preds, xva), xva, data_range=1.0).mean())
        history.append({"epoch": epoch, "train_mse": tot / len(perm), "val_ssim": vs})
        flag = ""
        if vs > best:
            best = vs
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "val_ssim": vs, "args": vars(args)}, args.out)
            flag = "  <- saved"
        print(f"  epoch {epoch+1:3d}/{args.epochs}  mse {tot/len(perm):.5f}  "
              f"val SSIM {vs:.4f}  {time.perf_counter()-t0:.1f}s{flag}")

    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"best_val_ssim": best, "params": n_par, "history": history}, indent=2),
        encoding="utf-8")
    print(f"\n  best val SSIM {best:.4f}  ->  {args.out}")


if __name__ == "__main__":
    main()

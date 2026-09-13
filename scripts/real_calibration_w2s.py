"""In-domain conformal calibration on the W2S real widefield measurements.

The BioSR real-data result rests on a widefield that was *synthesised* by
summing raw structured-illumination frames, because BioSR ships no widefield of
its own. That is the weakest methodological point in the paper. W2S ships 400
genuine widefield captures per field alongside a SIM reconstruction of the same
field, so the same experiment can be run against a real acquisition on a second
instrument at a different numerical aperture.

Structure mirrors ``real_calibration.py`` exactly, and the output uses
``extract_features.py``'s cache layout so the gate scripts run unchanged.

Signal level
------------
BioSR provides nine acquisition levels. W2S provides one acquisition repeated
400 times, so dose is varied by averaging: 1, 2, 4, 8, 16 and 400 frames give
six noise levels spanning a 20x range in photon count. The 400-frame average is
W2S's own definition of the noise-free low-resolution image. Level labels record
the number of frames averaged, and land in ``severity`` so the existing
worst-group machinery reports coverage per dose for free.

Background
----------
Fitted against the blurred ground truth, as in the BioSR script, and for the
same reason: estimating the pedestal blindly is a separate hard problem, and an
oracle background makes the test of the reliability layer strictly harder to
pass rather than easier. Any failure that survives an oracle background is a
failure of the layer.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sim2real import centre_crop, fourier_upsample  # noqa: E402
from src.optics.aberrations import ABERRATION_MODES  # noqa: E402
from src.reliability.features import (  # noqa: E402
    FEATURE_NAMES,
    OTF_FEATURE_NAMES,
    oscr_features,
)
from src.metrics.image import scale_match, ssim  # noqa: E402
from src.optics.forward_model import apply_psf  # noqa: E402
from src.optics.psf import PupilConfig, pupil_psf  # noqa: E402
from src.reconstruction.rl import richardson_lucy  # noqa: E402

W2S_NA = 1.49
W2S_SIM_PIXEL_UM = 0.032
W2S_EMISSION_UM = {0: 0.515, 1: 0.590, 2: 0.670}
LEVELS = (1, 2, 4, 8, 16, 400)
SPLITS = ("train", "cal", "test", "ood")


def channel_label(ch: int) -> str:
    return f"ch{ch}_{(488, 561, 640)[ch]}nm"


def prepare(wf_stack, gt, level, psf0, crop, target_photons):
    """One real measurement on the pipeline's grid, with its oracle background."""
    # Accumulate in chunks: casting a 400-frame slice to float64 in one go
    # allocates 800 MiB, which an 8 GB machine will not give up while an
    # extraction is running.
    acc = np.zeros(wf_stack.shape[-2:], dtype=np.float64)
    for lo in range(0, level, 32):
        acc += wf_stack[lo:min(lo + 32, level)].astype(np.float64).sum(axis=0)
    y_cam = acc / level
    y = np.clip(centre_crop(fourier_upsample(y_cam, 2), crop), 0.0, None)

    gt_c = centre_crop(gt.astype(np.float64), crop)
    lo, hi = np.percentile(gt_c, [0.1, 99.9])
    gt_t = torch.from_numpy(
        np.clip((gt_c - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    ).to(torch.float32)

    blur = apply_psf(gt_t, psf0).numpy().ravel().astype(np.float64)
    A = np.stack([blur, np.ones_like(blur)], axis=1)
    (_, b), *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    b = float(max(b, 0.0))

    scale = target_photons / max(y.mean() - b, 1e-9)
    return (torch.from_numpy(y * scale).to(torch.float32), gt_t,
            float(b * scale), float(y_cam.mean()))


def write_cache(rows, args) -> None:
    """Cache in ``extract_features.py``'s layout, held out by channel."""
    rng = np.random.default_rng(args.seed)
    keys = sorted({r["cell"] for r in rows if r["structure"] != args.held_out})
    rng.shuffle(keys)
    n_tr, n_ca = int(0.6 * len(keys)), int(0.2 * len(keys))
    role_of = {k: ("train" if i < n_tr else "cal" if i < n_tr + n_ca else "test")
               for i, k in enumerate(keys)}
    role = np.array([
        "ood" if r["structure"] == args.held_out else role_of.get(r["cell"], "ood")
        for r in rows
    ])

    store = {}
    n_feat = len(FEATURE_NAMES) + len(OTF_FEATURE_NAMES)
    for s in SPLITS:
        sel = [i for i, x in enumerate(role) if x == s]
        store[f"{s}_X"] = (
            np.asarray([rows[i]["X"] for i in sel], dtype=np.float32)
            if sel else np.zeros((0, n_feat), dtype=np.float32)
        )
        store[f"{s}_e"] = np.asarray([rows[i]["e"] for i in sel], dtype=np.float32)
        # No injected aberration to compare against. NaN rather than zero:
        # zero would read as "nominal operator", a claim we cannot make.
        store[f"{s}_rms"] = np.full(len(sel), np.nan, dtype=np.float32)
        store[f"{s}_severity"] = np.asarray([f"L{rows[i]['level']:03d}" for i in sel])
        store[f"{s}_mode"] = np.asarray(["real"] * len(sel))
        store[f"{s}_cell"] = np.asarray([rows[i]["cell"] for i in sel])
        store[f"{s}_structure"] = np.asarray([rows[i]["structure"] for i in sel])
        print(f"  {s}: {len(sel)} samples")

    store["feature_names"] = np.asarray(FEATURE_NAMES + OTF_FEATURE_NAMES)
    store["photons"] = np.asarray([args.photons])
    bg = float(np.median([r["background"] for r in rows]))
    store["background"] = np.asarray([bg])
    store["haze_frac"] = np.asarray([bg / (args.photons + bg)])
    store["reconstructor"] = np.asarray([f"{args.reconstructor}_real_w2s"])
    store["na"] = np.asarray([W2S_NA])
    store["pixel_um"] = np.asarray([W2S_SIM_PIXEL_UM])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(f"  wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default="W2S_raw.zip")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--kernel", type=int, default=127)
    ap.add_argument("--n-iter", type=int, default=75)
    ap.add_argument("--photons", type=float, default=100.0)
    ap.add_argument("--reconstructor", default="rl", choices=("rl",))
    ap.add_argument("--levels", nargs="+", type=int, default=list(LEVELS))
    ap.add_argument("--limit-fov", type=int, default=None)
    ap.add_argument("--held-out", default="ch2_640nm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/features19/real_w2s_rl.npz")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    sims = sorted(n for n in z.namelist() if re.search(r"/sim_channel\d\.npy$", n))
    if args.limit_fov:
        keep = sorted({re.search(r"(Image\d+)/", n).group(1) for n in sims})[: args.limit_fov]
        sims = [n for n in sims if re.search(r"(Image\d+)/", n).group(1) in keep]

    psfs = {
        ch: pupil_psf(
            None,
            PupilConfig(na=W2S_NA, wavelength_um=lam,
                        pixel_um=W2S_SIM_PIXEL_UM, n_out=args.kernel),
            warn=False,
        )[0].to(torch.float32)
        for ch, lam in W2S_EMISSION_UM.items()
    }
    cutoffs = {
        ch: 2 * W2S_NA / lam * W2S_SIM_PIXEL_UM
        for ch, lam in W2S_EMISSION_UM.items()
    }

    print(f"W2S real calibration: {len(sims)} views x {len(args.levels)} levels "
          f"= {len(sims) * len(args.levels)} measurements")
    zip_names = set(z.namelist())
    scratch = Path(tempfile.gettempdir()) / "w2s_wf_scratch.npy"
    rows = []
    for i, sname in enumerate(sims):
        fov = re.search(r"(Image\d+)/", sname).group(1)
        ch = int(re.search(r"sim_channel(\d)\.npy$", sname).group(1))
        try:
            gt = np.load(io.BytesIO(z.read(sname)))
        except KeyError:
            continue
        # The widefield stack is 200 MiB. Loading it whole spikes memory and
        # fails on an 8 GB machine when anything else is running, so it is
        # spilled to a scratch file and memory-mapped: only the frames actually
        # averaged are ever paged in.
        wf_member = f"raw/{fov}/wf_channel{ch}.npy"
        if wf_member not in zip_names:
            continue
        # Release the previous memory map first: on Windows an open mapping
        # keeps the file locked and reopening it for writing fails with EINVAL.
        wf = None
        scratch.unlink(missing_ok=True)
        with z.open(wf_member) as src, open(scratch, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        wf = np.load(scratch, mmap_mode="r")
        psf0 = psfs[ch]
        probes = build_probes(ch, args)
        for level in args.levels:
            y, gt_t, bg, _ = prepare(wf, gt, level, psf0, args.crop, args.photons)
            x_hat = richardson_lucy(y[None], psf0, args.n_iter, background=bg,
                                    read_variance=1.5 ** 2)
            X = oscr_features(y[None], x_hat, psf0, probes,
                              cutoff_cyc_px=cutoffs[ch])[0].cpu().numpy()
            e = float(1.0 - ssim(scale_match(x_hat[0], gt_t), gt_t, data_range=1.0))
            rows.append({
                "X": X, "e": e, "level": level, "cell": fov,
                "structure": channel_label(ch), "background": bg,
            })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(sims)} views", flush=True)

    print(f"\n  {len(rows)} measurements")
    write_cache(rows, args)


def build_probes(ch, args):
    """Probe set identical to ``extract_features.build_probes``.

    Must match exactly: the probe set defines the OSCR feature coordinates, so a
    different one would make these features incomparable with the simulated
    caches they are meant to be evaluated alongside.
    """
    cfg = PupilConfig(na=W2S_NA, wavelength_um=W2S_EMISSION_UM[ch],
                      pixel_um=W2S_SIM_PIXEL_UM, n_out=args.kernel)
    probes = [pupil_psf(None, cfg, warn=False)[0]]
    for mode in ABERRATION_MODES.values():
        for amp in (0.05, 0.11, 0.18):
            probes.append(pupil_psf({mode: amp}, cfg, warn=False)[0])
    return [p.to(torch.float32) for p in probes]


if __name__ == "__main__":
    main()

"""Run every analysis behind the paper's numbers, from the shipped feature caches.

Each step calls one of the repository's analysis scripts with exactly the
arguments that produced the published value, and writes its JSON output to
``paper/results/`` and its console output to ``paper/results/logs/``.
Nothing here needs the raw BioSR or W2S data or a GPU.

    python paper/run_analyses.py                 # everything (~19 min with --jobs 5 on a 6-core laptop)
    python paper/run_analyses.py --list          # show the steps
    python paper/run_analyses.py --only gate5_biosr real_coverage
    python paper/run_analyses.py --jobs 4        # run independent steps in parallel

A step whose output already exists is skipped unless ``--force`` is given.
Every analysis is a single-threaded gradient-boosting fit, so ``--jobs`` up to
the number of physical cores scales almost linearly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"
LOGS = RESULTS / "logs"

F19 = "experiments/features19"
F29 = "experiments/features29"
W2S = "experiments/features29_w2s"

# The four principal simulated BioSR settings, all at out-of-focus fraction
# 0.31. The file names are historical: "bg45" is 45 background photons at 100
# signal photons and "bgprop" is 6.75 at 15, both 0.31 of the measurement.
# rl_photons_15_bg45 / tv_photons_15_bg45 (0.75 haze) are NOT the paper's setting.
BIOSR_SIM = [
    ("rl100", f"{F19}/rl_photons_100_bg45.npz"),
    ("rl15", f"{F19}/rl_photons_15_bgprop.npz"),
    ("tv100", f"{F19}/tv_photons_100_bg45.npz"),
    ("tv15", f"{F19}/tv_photons_15_bgprop.npz"),
]
BIOSR_REAL = [f"{F19}/real_{r}.npz" for r in ("rl", "tv", "unet", "unrolled")]


def _steps() -> list[tuple[str, list[str], str]]:
    """(name, argv, what in the paper it feeds)."""
    py = sys.executable
    s: list[tuple[str, list[str], str]] = []

    def add(name, script, args, feeds):
        out = f"paper/results/{name}.json"
        s.append((name, [py, f"scripts/{script}", *args, "--out", out], feeds))

    add("gate5_biosr", "gate5_bound.py",
        ["--features", *[p for _, p in BIOSR_SIM], "--seeds", "10"],
        "Fig. 2a/2b; Table II bound rows; Sec. IV-C; supp. bound table")
    add("gate5_w2s", "gate5_bound.py",
        ["--features", f"{W2S}/rl_photons_100.npz", f"{W2S}/rl_photons_15.npz",
         "--seeds", "10"],
        "Sec. IV-D W2S bound; Table II W2S bound rows")
    add("gate5_real_grid", "gate5_bound.py",
        ["--features", *BIOSR_REAL, "--seeds", "10"],
        "Fig. 4b; Sec. IV-E; supp. monitoring + bound tables")
    add("gate5_w2s_real", "gate5_bound.py",
        ["--features", f"{F19}/real_w2s_rl.npz", "--seeds", "10"],
        "supp. bound table, W2S genuine widefield row")
    add("real_coverage", "real_coverage.py", ["--seeds", "10"],
        "Sec. IV-B/IV-D real coverage; Table II; supp. real table")
    add("family_rl100", "family_analysis.py", ["--features", BIOSR_SIM[0][1]],
        "Fig. 1; Sec. IV-A; supp. family table")
    for tag, path in BIOSR_SIM:
        add(f"scaled_{tag}", "gate4_scaled.py", ["--features", path],
            "Sec. IV-A (0.901 / 0.664); supp. calibration audit + scale floors")
        add(f"phase1_{tag}", "phase1_analyses.py", ["--features", path, "--seeds", "10"],
            "Sec. IV-B paired coverage drop")
        add(f"ablation_{tag}", "ablation_matched.py", ["--features", path, "--seeds", "5"],
            "Fig. 4a; Sec. IV-E; supp. width ablation")
    for tag, path in (("biosr_rl100", f"{F29}/rl_photons_100.npz"),
                      ("biosr_rl15", f"{F29}/rl_photons_15.npz"),
                      ("biosr_tv100", f"{F29}/tv_photons_100.npz"),
                      ("biosr_tv15", f"{F29}/tv_photons_15.npz"),
                      ("w2s_rl100", f"{W2S}/rl_photons_100.npz"),
                      ("w2s_rl15", f"{W2S}/rl_photons_15.npz")):
        add(f"phase2_{tag}", "phase2_analysis.py", ["--features", path, "--seeds", "5"],
            "Fig. 3; Table II coverage rows; Sec. IV-B/IV-D; supp. tables")
    return s


STEPS = _steps()


def run_step(name: str, argv: list[str]) -> tuple[str, int, float]:
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env=_env())
    return name, rc, time.time() - t0


def _env():
    import os
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    env["PYTHONIOENCODING"] = "utf-8"
    # One thread per process: the fits are single-threaded anyway, and this
    # stops BLAS from oversubscribing the cores when --jobs > 1.
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, "1")
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, help="step names to run")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="rerun even if output exists")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, argv, feeds in STEPS:
            print(f"{name:20s} {feeds}")
        return 0

    todo = [(n, a) for n, a, _ in STEPS
            if (args.only is None or n in args.only)
            and (args.force or not (RESULTS / f"{n}.json").exists())]
    unknown = set(args.only or []) - {n for n, _, _ in STEPS}
    if unknown:
        print(f"unknown step(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    missing = sorted({p for _, a in todo for p in a
                      if p.endswith(".npz") and not (ROOT / p).exists()})
    if missing:
        print("missing feature caches (see experiments/README.md):", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    if not todo:
        print("all outputs present; nothing to do (use --force to rerun)")
        return 0

    print(f"running {len(todo)} step(s) with {args.jobs} job(s); logs in {LOGS}")
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futs = [pool.submit(run_step, n, a) for n, a in todo]
        for f in as_completed(futs):
            name, rc, dt = f.result()
            status = "ok" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  {name:20s} {status:14s} {dt / 60:5.1f} min", flush=True)
            if rc != 0:
                failed.append(name)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"failed: {', '.join(failed)} -- see paper/results/logs/", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

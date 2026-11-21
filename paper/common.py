"""Shared loaders for the paper scripts: one place that knows the JSON layouts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "results"

# Paper label -> run_analyses tag, gate5 cell tag (cache stem).
BIOSR_SETTINGS = [
    ("RL-100", "rl100", "rl_photons_100_bg45", "RL, 100 ph"),
    ("RL-15", "rl15", "rl_photons_15_bgprop", "RL, 15 ph"),
    ("TV-100", "tv100", "tv_photons_100_bg45", "TV, 100 ph"),
    ("TV-15", "tv15", "tv_photons_15_bgprop", "TV, 15 ph"),
]
PHASE2 = [
    ("BioSR", "RL-100", "biosr_rl100"),
    ("BioSR", "RL-15", "biosr_rl15"),
    ("BioSR", "TV-100", "biosr_tv100"),
    ("BioSR", "TV-15", "biosr_tv15"),
    ("W2S", "RL-100", "w2s_rl100"),
    ("W2S", "RL-15", "w2s_rl15"),
]
AXIS_IN = "none (in-distribution)"
AXIS_ZERN = "operator shift (Zernike)"
AXIS_PHYS = "operator shift (physical)"
FAMILIES = ("defocus", "astigmatism", "coma", "spherical")

# gate5 blind-block key -> label used in Fig. 4b and the supplementary table.
BLIND_LABELS = [
    ("blind_qpred", "Conditional predictor"),
    ("blind_oscr", "Physics-consistency profile"),
    ("blind_otf", "OTF feature set"),
    ("blind_all", "All observables"),
    ("blind_otf_core", "OTF core"),
    ("blind_residual", "Nominal residual"),
    ("blind_blind_score", "Composite blind score"),
]


def round_half_up(x: float, dp: int) -> float:
    """Round as the manuscript does: ties go up (0.8125 -> 0.813).

    Python's round() and format() send ties to the even digit (0.8125 -> 0.812),
    which differs from the printed value whenever a coverage is an exact tie,
    e.g. 39/48.
    """
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp), ROUND_HALF_UP))


def fmt(x: float, dp: int = 3) -> str:
    return f"{round_half_up(x, dp):.{dp}f}"


class MissingResult(FileNotFoundError):
    pass


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise MissingResult(
            f"{p.relative_to(ROOT)} not found -- run "
            f"`python paper/run_analyses.py --only {name}` first")
    return json.loads(p.read_text(encoding="utf-8"))


def phase2_cov(tag: str, axis: str, metric: str = "ssim") -> tuple[float, float]:
    m, s = load(f"phase2_{tag}")[f"coverage_{metric}"][axis]["coverage"]
    return float(m), float(s)


def family_extreme_physical(tag: str) -> dict[str, float]:
    """Coverage at the largest-magnitude rung of each physical family."""
    rows = load(f"phase2_{tag}")["families_ssim"]
    return {r["family"]: r["coverage"] for r in rows if r["kind"] == "physical"}


def seeds_stat(name: str, key: str) -> tuple[float, float]:
    m, s, _n = load(name)["across_seeds"][key]
    return float(m), float(s)


def gate5_cells(name: str) -> list[dict]:
    """Seed-0 cells of a gate5 run, flattened, with their cache tag."""
    out = []
    for tag, by_struct in load(name)["cells"].items():
        for struct, r in by_struct.items():
            out.append(dict(r, tag=tag, structure=struct))
    return out


def ablation_width(tag: str, block: str) -> tuple[float, float]:
    m, s = load(f"ablation_{tag}")["blocks"][block]["width_in"]
    return float(m), float(s)


def ablation_coverage(tag: str, block: str) -> tuple[float, float]:
    m, s = load(f"ablation_{tag}")["blocks"][block]["coverage_in"]
    return float(m), float(s)


def scaled_split(tag: str, axis: str, alpha: str = "0.1") -> dict:
    return load(f"scaled_{tag}")["results"][axis]["split"][alpha]

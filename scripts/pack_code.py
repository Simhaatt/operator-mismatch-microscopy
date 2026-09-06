"""Pack the source tree for a hosted runtime.

The Colab notebook needs the code, not the data. This writes the ~60 KB zip that
goes to Drive alongside `biosr_gt.npz`. Keeping it a script rather than a
remembered `zip` incantation matters because a **stale zip is silent**: Colab
unpacks it, everything imports, and you get results from last week's code. This
one stamps the git commit it was built from and the notebook prints it back.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("src", "scripts", "tests")
FILES = ("pyproject.toml",)


def git_stamp() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{h}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/microscopy_code.zip")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = git_stamp()
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE:
            for p in sorted((ROOT / d).rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for f in FILES:
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
                n += 1
        z.writestr("BUILD_STAMP", f"commit {stamp}\n")
    size = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(ROOT)}  {n} files, {size:.0f} KB, commit {stamp}")
    if "dirty" in stamp:
        print("  NOTE: built from a dirty tree; commit first if this run matters")


if __name__ == "__main__":
    main()

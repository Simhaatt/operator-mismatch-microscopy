"""Build the anonymised supplementary archive for journal submission.

Replaces the ad-hoc process used for the TMLR submission with something
auditable. Three things matter and are enforced rather than trusted:

1. **What goes in.** An explicit allowlist. Nothing is swept up by wildcard, so
   a stray file in the working tree cannot leak into a submission.
2. **What must never go in.** The BioSR data itself (redistribution is not
   permitted -- download instructions and a citation only), the copyrighted
   publisher PDF, Git metadata, model checkpoints, and anything under `data/`.
   These are checked by pattern *after* collection, so adding a new source
   directory cannot silently defeat them.
3. **Anonymity.** The file contents are scanned for author names, e-mail
   addresses, identifiable repository URLs and personal filesystem paths. Any
   hit is reported and the build **fails** unless ``--allow-identifying`` is
   passed. Failing closed is the point: a submission that leaks identity cannot
   be un-leaked.

Note on review model: IEEE TCI uses single-anonymous review, so anonymisation
is not strictly required there. It is applied anyway -- it costs nothing and
keeps one archive usable for any venue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- what goes in -----------------------------------------------------------
# (directory, glob) pairs. Order only affects the manifest's readability.
INCLUDE: list[tuple[str, str]] = [
    ("src", "**/*.py"),
    ("scripts", "*.py"),
    ("tests", "*.py"),
    ("docs", "*.md"),
    # Only the results document. VENUE_AND_POSITIONING.md is internal strategy
    # -- prior-submission history, page charges, how to frame the paper -- and
    # is not reviewer-facing supplementary material.
    ("ieee", "RESULTS_TCI.md"),
    ("ieee", "SUPPLEMENTARY_README.md"),
    # JOURNAL_TCI.md is excluded for the same reason as
    # VENUE_AND_POSITIONING.md: submission strategy, not reviewer-facing
    # material. It carries no identifying strings, but page-charge arithmetic
    # and notes on which editor might handle the paper do not belong in a
    # supplementary archive.
    ("experiments/features19", "*.npz"),
    ("experiments/features29", "*.npz"),
    ("experiments/features29_w2s", "*.npz"),
    ("experiments/features29_task", "*.npz"),
    ("experiments/gate5", "*.json"),
    ("experiments/gate4", "*.json"),
    ("experiments/gate4b", "*.json"),
    ("experiments/gate4_haze31", "*.json"),
    ("experiments/gate1_verified", "*.json"),
    ("experiments/logs", "*.log"),
    ("experiments/realism", "*.json"),
    ("experiments/sim2real", "*.json"),
]

INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "README.md"]

# --- what must never go in --------------------------------------------------
FORBIDDEN_PATTERNS = [
    (re.compile(r"(^|/)\.git(/|$)"), "Git metadata"),
    # Top-level ``data/`` only. ``src/data/`` is the loader package -- source
    # code that reads BioSR, which ships; the data itself does not.
    (re.compile(r"^data(/|$)"), "raw data directory"),
    (re.compile(r"\.zip$"), "nested archive"),
    (re.compile(r"\.(pt|pth|ckpt|safetensors)$"), "model checkpoint"),
    (re.compile(r"s41592-020-01048-5", re.I), "copyrighted publisher PDF"),
    # Match BioSR *data* payloads by extension, not any path mentioning the
    # dataset by name -- a module called ``biosr.py`` is code, not data.
    (
        re.compile(r"biosr.*\.(npz|npy|mrc|tif|tiff|h5|mat)$", re.I),
        "BioSR data (not redistributable)",
    ),
    (re.compile(r"\.pdf$", re.I), "PDF"),
    (re.compile(r"__pycache__"), "bytecode cache"),
]

# --- identity leaks ---------------------------------------------------------
# Substring checks are deliberately broad; a false positive costs one review,
# a false negative costs anonymity.
IDENTIFYING = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "e-mail address"),
    (re.compile(r"github\.com/[\w.-]+/[\w.-]+"), "repository URL"),
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\s\"']+"), "Windows personal path"),
    (re.compile(r"/home/[^\s\"'/]+"), "Unix personal path"),
    (re.compile(r"\bSimhaa\w*", re.I), "author name"),
    (re.compile(r"\bshihabudheen\b", re.I), "author name"),
    (re.compile(r"\bnitc\.ac\.in\b", re.I), "institutional domain"),
    (re.compile(r"openreview\.net/forum\?id=\w+"), "prior-submission link"),
]

TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".log", ".cfg", ".ini"}

# Redactions applied to text files on the way into the archive. The originals
# on disk are never modified. Run logs are evidence and worth shipping, but
# library tracebacks inside them quote absolute interpreter paths that carry a
# username; rewriting those keeps the evidence and drops the identity.
REDACTIONS = [
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\s\"']*?[\\/]site-packages"),
     "<site-packages>"),
    (re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\s\"']+"), "<path>"),
    (re.compile(r"/home/[^\s\"'/]+"), "<path>"),
]

# This module is never shipped. Its detection patterns necessarily spell out
# the author name and institutional domain, so including it would put exactly
# the strings it exists to catch inside the archive. Excluding it from
# packaging -- rather than exempting it from the scan -- keeps the scanner
# free of blind spots.
EXCLUDE_FROM_ARCHIVE = {"scripts/pack_supplementary.py"}


def redact(text: str) -> tuple[str, int]:
    """Apply :data:`REDACTIONS`, returning the text and the number of hits."""
    total = 0
    for rx, repl in REDACTIONS:
        text, n = rx.subn(repl, text)
        total += n
    return text, total


def collect() -> list[Path]:
    out: list[Path] = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            out.append(p)
    for sub, pattern in INCLUDE:
        base = ROOT / sub
        if not base.is_dir():
            continue
        out.extend(sorted(p for p in base.glob(pattern) if p.is_file()))
    # De-duplicate while keeping order, dropping anything excluded.
    seen, uniq = set(), []
    for p in out:
        r = p.relative_to(ROOT).as_posix()
        if r in EXCLUDE_FROM_ARCHIVE or r in seen:
            continue
        seen.add(r)
        uniq.append(p)
    return uniq


def check_forbidden(paths: list[Path]) -> list[str]:
    problems = []
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        for rx, why in FORBIDDEN_PATTERNS:
            if rx.search(rel):
                problems.append(f"{rel}: {why}")
    return problems


def scan_identity(paths: list[Path]) -> list[str]:
    """Scan the content that will actually ship, i.e. after redaction."""
    hits = []
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - unreadable file
            hits.append(f"{rel}: unreadable ({exc})")
            continue
        text, _ = redact(text)
        for i, line in enumerate(text.splitlines(), 1):
            for rx, why in IDENTIFYING:
                m = rx.search(line)
                if m:
                    hits.append(f"{rel}:{i}: {why}: {m.group(0)[:70]}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="ieee/supplementary_ieee.zip")
    ap.add_argument(
        "--allow-identifying",
        action="store_true",
        help="build even if identifying strings are found (do not use for a "
             "double-anonymous venue)",
    )
    ap.add_argument("--max-mb", type=float, default=100.0)
    args = ap.parse_args()

    paths = collect()
    print(f"collected {len(paths)} files")

    forbidden = check_forbidden(paths)
    if forbidden:
        print("\nFORBIDDEN CONTENT -- refusing to build:")
        for f in forbidden:
            print(f"  {f}")
        sys.exit(1)
    print("forbidden-content check: clean")

    hits = scan_identity(paths)
    if hits:
        print(f"\nIDENTIFYING STRINGS ({len(hits)}):")
        for h in hits[:60]:
            print(f"  {h}")
        if len(hits) > 60:
            print(f"  ... and {len(hits) - 60} more")
        if not args.allow_identifying:
            print("\nrefusing to build; fix these or pass --allow-identifying")
            sys.exit(1)
        print("\n--allow-identifying set: building anyway")
    else:
        print("anonymity scan: clean")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    n_redacted = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix.lower() in TEXT_SUFFIXES:
                original = p.read_text(encoding="utf-8", errors="replace")
                text, n = redact(original)
                if n:
                    n_redacted += n
                payload = text.encode("utf-8")
                z.writestr(rel, payload)
            else:
                payload = p.read_bytes()
                z.write(p, rel)
                n = 0
            manifest.append({
                "path": rel,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()[:16],
                "redactions": n,
            })
        z.writestr("MANIFEST.json", json.dumps(
            {"n_files": len(manifest), "redactions": n_redacted,
             "files": manifest}, indent=1
        ))
    if n_redacted:
        print(f"redacted {n_redacted} personal path(s) on the way in")

    size_mb = out.stat().st_size / 2**20
    print(f"\nwrote {args.out}")
    print(f"  {len(paths)} files + MANIFEST.json")
    print(f"  {size_mb:.2f} MB")
    if size_mb > args.max_mb:
        print(f"  WARNING: exceeds the {args.max_mb:.0f} MB limit")
        sys.exit(1)
    print(f"  within the {args.max_mb:.0f} MB limit")


if __name__ == "__main__":
    main()

"""Tests for the generated Colab notebook.

These exist because of a specific failure. `nbformat` stores a cell's text as a
list of strings which the reader **concatenates with no separator**, so every
element has to carry its own trailing newline. The generator split on newlines
and dropped them, and every cell in the notebook rendered as one enormous line.

It shipped because the check that was meant to catch it joined the elements with
`"\n"` before compiling -- which is the one thing a reader never does, so a
collapsed cell compiled perfectly. The tests below join with `""`, exactly as
the reader does. That distinction is the whole point of this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "colab_gpu_session.ipynb"


@pytest.fixture(scope="module")
def nb() -> dict:
    if not NB.exists():
        pytest.skip(f"{NB.name} not generated")
    return json.loads(NB.read_text(encoding="utf-8"))


def render(cell: dict) -> str:
    """The cell's text as a notebook reader reconstructs it: plain join."""
    return "".join(cell["source"])


def test_no_cell_collapses_to_a_single_line(nb) -> None:
    """The regression itself."""
    for i, cell in enumerate(nb["cells"]):
        if len(cell["source"]) > 1:
            assert "\n" in render(cell), (
                f"cell {i} has {len(cell['source'])} source elements but no "
                f"newline after joining -- the elements are missing their "
                f"trailing newlines and the cell will render as one line")


def test_every_source_element_ends_with_a_newline_except_the_last(nb) -> None:
    for i, cell in enumerate(nb["cells"]):
        src = cell["source"]
        for j, line in enumerate(src[:-1]):
            assert line.endswith("\n"), f"cell {i} element {j} lacks a newline"
        assert not src[-1].endswith("\n"), f"cell {i} has a trailing blank line"


def test_every_code_cell_compiles_as_the_reader_sees_it(nb) -> None:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile(render(cell), f"cell{i}", "exec")


def test_the_notebook_passes_the_haze_fraction_and_never_a_fixed_background(nb) -> None:
    """A fixed rate is correct at one photon level only; training samples many.

    Getting this wrong is silent -- the models train fine and produce plausible
    numbers -- and costs a two-hour GPU session to repeat.
    """
    # Comments are excluded: one of them warns against --background by name, and
    # a test that cannot tell advice from a command is worse than no test.
    text = "\n".join(
        line
        for cell in nb["cells"] if cell["cell_type"] == "code"
        for line in render(cell).split("\n")
        if not line.lstrip().startswith("#")
    )
    assert "--haze-frac" in text
    assert "--background" not in text


def test_the_notebook_requests_a_gpu_and_verifies_the_physics(nb) -> None:
    text = "\n".join(render(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "cuda.is_available" in text, "must refuse to run on CPU"
    assert "pytest" in text, "must run the suite before trusting a result"
    assert nb["metadata"].get("accelerator") == "GPU"


def test_the_generator_is_deterministic(tmp_path) -> None:
    """Regenerating must not churn the file, or 'is it current?' is unanswerable."""
    out = tmp_path / "nb.ipynb"
    for _ in range(2):
        subprocess.run([sys.executable, "scripts/make_colab_notebook.py",
                        "--out", str(out.relative_to(ROOT) if out.is_relative_to(ROOT)
                                     else out)],
                       cwd=ROOT, check=True, capture_output=True)
    first = out.read_text(encoding="utf-8")
    subprocess.run([sys.executable, "scripts/make_colab_notebook.py",
                    "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
    assert out.read_text(encoding="utf-8") == first


def test_every_interpolated_name_is_defined_before_use(nb) -> None:
    """Catch what `compile` cannot: an f-string naming a variable nobody sets.

    Adding `--per-cell {PER_CELL}` to a command string compiles perfectly and
    then raises NameError halfway through a GPU session. This walks the cells in
    order and checks that every ALL-CAPS name interpolated into an f-string was
    assigned in an earlier cell or the same one.
    """
    import re

    defined: set[str] = set()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = render(cell)
        defined |= set(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", text, re.M))
        for name in re.findall(r"\{([A-Z][A-Z0-9_]*)[\}\[\.\:]", text):
            assert name in defined, (
                f"cell {i} interpolates {{{name}}} but nothing assigns it yet -- "
                f"this compiles and then fails at runtime")


def test_nothing_is_written_to_drive(nb) -> None:
    """Drive is read-only: inputs come from it, results come back as a download.

    The trade-off is deliberate and worth not losing by accident -- writing
    checkpoints to Drive protected them from a disconnect, and this gives that
    up in exchange for not touching the user's Drive at all.
    """
    text = "\n".join(render(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "OUT = Path('/content/out')" in text, "outputs must be on local disk"
    # Checking writes rather than mentions: the notebook legitimately reads from
    # Drive (glob, stat, unzip, and one copy *out* of it), so an allowlist of
    # lines that mention Drive ends up policing prose instead of behaviour.
    assert "NPZ.parent" not in text, "must not write beside the input data"
    assert "DRIVE /" not in text.replace("glob.glob(str(DRIVE / ('*/' * depth) / name))", ""), \
        "the only DRIVE path construction should be the input search"
    # drive.mount('/content/drive') is the one legitimate use of that literal,
    # so the patterns here are write verbs, not the path.
    for bad in ("copy2(local", "copytree", "OUT = NPZ", "OUT = DRIVE"):
        assert bad not in text, f"looks like a Drive write: {bad}"


def test_results_come_back_as_one_download_with_a_manifest(nb) -> None:
    text = render(nb["cells"][-1])
    assert "files.download" in text, "the last cell must hand back the results"
    assert "MANIFEST.json" in text
    for key in ("code_commit", "haze_fraction", "per_cell", "epochs"):
        assert key in text, f"manifest must record {key}"
    assert "val_ssim" in text, "manifest must record what the models scored"


def test_the_manifest_reports_what_happened_not_what_was_asked(nb) -> None:
    """A skipped model may predate the settings this run requested.

    Cell 5 skips a checkpoint that already exists, so recording the notebook's
    constants would claim a training scale the weights never saw. The manifest
    has to read the scale back out of each checkpoint's own args and say so when
    they disagree.
    """
    text = render(nb["cells"][-1])
    assert "trained_per_cell" in text and "trained_epochs" in text,         "manifest must record the scale each checkpoint was ACTUALLY trained at"
    assert "mismatch" in text, "must detect and report a scale disagreement"
    assert "'requested'" in text,         "the notebook's own constants belong under a 'requested' key, not as fact"

"""Dry-run parity: one planner, one rendering, no writes (T-11 gate ``dryrun``).

FR-31 requires the operations printed by ``--dry-run`` to be byte-identical to
the ones a real run performs. AC-9 requires a dry run to write nothing, checked
by hashing the destination tree before and after.

Asserted three ways, because each catches a different way of cheating
---------------------------------------------------------------
1. The destination tree hash is unchanged (AC-9) -- a dry run that quietly
   created an empty directory still fails here.
2. The rendered operation lines are byte-identical between the two commands
   (FR-31) -- what the user is shown is the same either way.
3. The plan *tuples* are equal at the API level (FR-31) -- the reason (2) holds
   is that both runs came from one planner, and this asserts the cause rather
   than only the symptom. Two renderers kept in step by discipline would pass
   (2) and fail (3).

Everything is driven through ``main(argv)``, the function the console script
calls.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Import from the working tree rather than from an installed copy. The gate is
# literally `python -m pytest`, and `python` on this host is the system
# interpreter (3.14.5) where the package is NOT installed -- the installed copy
# lives in the uv venv (3.12.14). Prepending `src` makes the suite runnable
# under either, which AC-3 / NFR-5 require when the same suite must pass on
# 3.12 and 3.14.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_code_library.cli import (
    EXIT_DOMAIN_ERROR,
    EXIT_OK,
    main,
)
from rag_code_library.generator import add_component
from rag_code_library.registry import Registry

_LIBRARY_ROOT = _ROOT / "components"
_REGISTRY = Registry.load(_LIBRARY_ROOT)
#: Two components and their dependency -- enough for an ordering mistake to
#: change the output rather than merely reorder a line.
_COMPONENT = "storage-adapters"


@pytest.fixture(scope="module", autouse=True)
def _registry_is_usable() -> None:
    assert _REGISTRY.ok, f"fixture registry did not load: {_REGISTRY.issues}"


def _run(*args: str, capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _project(tmp_path: Path, name: str = "demo-client") -> Path:
    project = tmp_path / name
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    return project


def _tree_hash(root: Path) -> str:
    """Digest of every file under ``root``: names and contents together.

    Names are included because a dry run that created an empty directory has
    still changed the destination, and an empty directory hashes to nothing if
    only contents are read.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    # Directories, so an empty one still registers.
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        digest.update(b"d:" + path.relative_to(root).as_posix().encode("utf-8"))
    return digest.hexdigest()


def _operations(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip().startswith("copy ")]


# ----------------------------------------------------------------------- AC-9


def test_a_dry_run_writes_nothing(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    project = _project(tmp_path)
    before = _tree_hash(project)

    code, out, err = _run(
        "add", _COMPONENT, "--destination", str(project), "--dry-run", capsys=capsys
    )
    assert code == EXIT_OK, err
    assert _tree_hash(project) == before, "a dry run changed the destination"
    assert _operations(out), "a dry run printed no file operations"


def test_a_dry_run_prints_every_operation_the_real_run_performs(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    project = _project(tmp_path)

    code, dry, err = _run(
        "add", _COMPONENT, "--destination", str(project), "--dry-run", capsys=capsys
    )
    assert code == EXIT_OK, err

    code, _real, err = _run(
        "add", _COMPONENT, "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    assert _operations(dry), "the dry run listed nothing"
    assert len(_operations(dry)) == 12  # 3 components x 4 files


# ---------------------------------------------------------------------- FR-31


def test_dry_run_and_real_run_render_the_same_bytes(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    project = _project(tmp_path)

    code, dry, err = _run(
        "add", _COMPONENT, "--destination", str(project), "--dry-run", capsys=capsys
    )
    assert code == EXIT_OK, err
    code, real, err = _run(
        "add", _COMPONENT, "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    # Byte-identical, not merely equal-looking.
    assert "".join(_operations(dry)).encode("utf-8") == "".join(
        _operations(real)
    ).encode("utf-8")


def test_one_planner_produces_one_plan(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """FR-31's cause, asserted directly: same inputs, same plan.

    Two different destination projects with the same package name -- so any
    difference between the plans is the planner changing its mind, not the
    filesystem having been written to.
    """
    dry_run_project = _project(tmp_path / "one")
    real_project = _project(tmp_path / "two")

    dry = add_component(
        _COMPONENT,
        destination=dry_run_project,
        registry=_REGISTRY,
        library_root=_LIBRARY_ROOT,
        dry_run=True,
    )
    real = add_component(
        _COMPONENT,
        destination=real_project,
        registry=_REGISTRY,
        library_root=_LIBRARY_ROOT,
    )

    assert dry.operations == real.operations
    assert dry.written == ()
    assert len(real.written) == len(real.operations)
    assert dry.report == real.report


def test_a_dry_run_is_silent_about_writes_it_could_not_make(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """A dry run that ignored conflicts would report a plan that cannot happen."""
    project = _project(tmp_path)
    code, _out, err = _run(
        "add", _COMPONENT, "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    code, out, err = _run(
        "add", _COMPONENT, "--destination", str(project), "--dry-run", capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "--force" in err
    assert "Traceback" not in err


def test_a_dry_run_with_force_still_writes_nothing(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    code, _out, err = _run(
        "add", _COMPONENT, "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err
    before = _tree_hash(project)

    code, out, err = _run(
        "add",
        _COMPONENT,
        "--destination",
        str(project),
        "--dry-run",
        "--force",
        capsys=capsys,
    )
    assert code == EXIT_OK, err
    assert _tree_hash(project) == before, "--dry-run --force wrote something"
    assert _operations(out)

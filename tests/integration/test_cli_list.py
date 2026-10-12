"""`ragkit list` and `ragkit info` reading the canonical registry (T-10 gate).

Covers FR-13 (list: id, category, one-line description, optional category
filter), FR-14 (info: description, dependencies, requirements, example, usage
notes, trade-offs), FR-35 (lifecycle status in both), NFR-7 (readable errors
naming the fault, never a traceback) and acceptance criteria AC-7 and AC-8.

Two ways of driving the CLI, on purpose
---------------------------------------
Most tests call ``main(argv)`` directly. That is the same function the
``ragkit`` console script invokes, so the code path is identical, and a suite
that asserted 35 ``info`` outputs through a subprocess each would spend its
time in interpreter startup rather than on the behaviour. Two tests *do* run a
real process, because "the script exits 1" and "stderr went to stderr" are
properties of a process, not of a return value, and asserting them in-process
would be inferring rather than proving.

Everything is compared against one registry snapshot
---------------------------------------------------
``_REGISTRY`` is loaded once, at import time. The assertions below compare the
CLI's output against *that* snapshot. Loading a second registry inside each
test to compare against would defeat the point: if the CLI silently dropped
components, two independent loads of the same tree would still agree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import from the working tree rather than from an installed copy.
#
# The gate command is literally `python -m pytest`, and `python` on this host is
# the system interpreter (3.14.5) where the package is NOT installed -- the
# installed copy lives in the uv venv (3.12.14). Prepending `src` makes the
# suite runnable under any interpreter, which is what AC-3 / NFR-5 require when
# the same suite must pass on 3.12 and 3.14, and it guarantees the working tree
# is tested rather than a stale installed snapshot.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_code_library.cli import (  # noqa: E402
    EXIT_DOMAIN_ERROR,
    EXIT_OK,
    EXIT_USAGE_ERROR,
    main,
)
from rag_code_library.registry import Registry  # noqa: E402

_REGISTRY = Registry.load(_ROOT / "components")
_IDS = tuple(_REGISTRY.ids)
_CATEGORIES = tuple(sorted({entry.manifest.category for entry in _REGISTRY}))

# A component that exists only inside a temporary directory. Its manifest
# declares no `documentation` or `tests` because those are the two declared-file
# checks the registry enforces; keeping them out keeps the fixture to two files.
_DEMO = {
    "id": "demo-chunker",
    "name": "Demo Chunker",
    "category": "chunking",
    "source": "demo.py",
    "version": "0.1.0",
    "dependencies": [],
    "description": "A component invented by the test, not shipped by the library.",
    "destination": "src/{project_package}/rag/chunking/demo.py",
    "status": "experimental",
    "requires_python": {"min": "3.12"},
    "required_packages": [],
    "optional_packages": [],
    "external_services": [],
    "install": "No additional packages required. Copy with: ragkit add <id>",
    "missing_dependency_message": (
        "demo-chunker declares no optional dependencies. If an import fails, "
        "restore the project environment with: uv sync"
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _registry_is_usable() -> None:
    """Fail loudly rather than assert against a registry that did not load."""
    assert _REGISTRY.ok, f"fixture registry did not load: {_REGISTRY.issues}"
    assert _IDS, "fixture registry is empty"


def _run(*args: str, capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """Drive the CLI and return its exit code, stdout and stderr separately."""
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _section(text: str, heading: str) -> str:
    """The indented body under one ``info`` section heading.

    Located by matching the heading as a whole line, not by splitting on the
    word: a component's description could one day contain "Example", and a
    CLI whose section boundaries move with prose is a CLI with flaky tests.
    """
    lines = text.splitlines()
    body: list[str] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line and not line.startswith("  "):
            break
        body.append(line.strip())
    return "\n".join(body).strip()


def _rows(listing: str) -> list[list[str]]:
    """The data lines of a ``list`` table, each split into its columns."""
    return [line.split(maxsplit=3) for line in listing.splitlines()[2:]]


def _write_demo_registry(tmp_path: Path) -> Path:
    """A one-component ``components`` tree rooted at ``tmp_path``."""
    directory = tmp_path / "components" / _DEMO["category"] / _DEMO["id"]
    directory.mkdir(parents=True)
    (directory / "demo.py").write_text("# fixture\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(_DEMO), encoding="utf-8"
    )
    return tmp_path


# --------------------------------------------------------------------- list --


def test_list_exits_zero_and_prints_a_table(capsys: pytest.CaptureFixture) -> None:
    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err
    assert out, "`ragkit list` printed nothing"
    assert err == ""


def test_list_contains_every_registered_id(capsys: pytest.CaptureFixture) -> None:
    """AC-7: every registry id appears, and nothing else does."""
    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err

    listed = [row[0] for row in _rows(out)]
    assert sorted(listed) == sorted(_IDS), "ragkit list and the registry disagree"
    # The failure mode AC-7 exists to catch: an id present in one and not the
    # other. Name the difference instead of reporting a mismatch in a 35-row list.
    omitted = sorted(set(_IDS) - set(listed))
    invented = sorted(set(listed) - set(_IDS))
    assert omitted == [], f"ragkit list omitted: {omitted}"
    assert invented == [], f"ragkit list invented: {invented}"


def test_list_reports_id_category_and_description_for_every_row(
    capsys: pytest.CaptureFixture,
) -> None:
    """FR-13: id, category and one-line description -- checked row by row."""
    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err

    assert out.splitlines()[0].split() == [
        "id",
        "category",
        "status",
        "description",
    ]

    manifests = {entry.manifest.id: entry.manifest for entry in _REGISTRY}
    rows = _rows(out)
    assert len(rows) == len(_IDS)
    for row in rows:
        component_id, category, _status, description = row
        manifest = manifests[component_id]
        assert category == manifest.category, component_id
        assert description == manifest.description, component_id


def test_list_is_sorted_by_id(capsys: pytest.CaptureFixture) -> None:
    """Deterministic output, so a diff between two runs means a real change."""
    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err
    assert [row[0] for row in _rows(out)] == sorted(_IDS)


def test_category_filter_returns_exactly_that_category(
    capsys: pytest.CaptureFixture,
) -> None:
    """AC-7: `--category chunking` contains only chunking components.

    Exact set equality, not a containment check -- the observable asks that the
    filter return *only* that category, and the bug worth catching is the
    unfiltered table being printed with a flag on it.
    """
    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err
    every_category = [row[0] for row in _rows(out)]

    for category in _CATEGORIES:
        code, filtered, err = _run("list", "--category", category, capsys=capsys)
        assert code == EXIT_OK, err

        expected = sorted(
            entry.manifest.id
            for entry in _REGISTRY
            if entry.manifest.category == category
        )
        assert [row[0] for row in _rows(filtered)] == expected, category
        # And it really is a subset of what the unfiltered table printed.
        assert set(row[0] for row in _rows(filtered)) <= set(every_category), category


def test_unknown_category_exits_1_naming_the_value_and_the_alternatives(
    capsys: pytest.CaptureFixture,
) -> None:
    """NFR-7: a mistyped category must not read as "this library has none"."""
    code, out, err = _run("list", "--category", "chunkk", capsys=capsys)
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "chunkk" in err
    for category in _CATEGORIES:
        assert category in err, f"error did not offer {category!r}"
    assert "Traceback" not in err


# ---------------------------------------------------------------------- info --


def test_info_prints_description_dependencies_example_and_usage_notes(
    capsys: pytest.CaptureFixture,
) -> None:
    """AC-8, over *every* registered component rather than a hand-picked few."""
    for component_id in _IDS:
        code, out, err = _run("info", component_id, capsys=capsys)
        assert code == EXIT_OK, f"{component_id}: {err}"

        manifest = _REGISTRY.get(component_id).manifest
        assert _section(out, "Description") == manifest.description, component_id
        assert manifest.id in out.splitlines()[0], component_id

        dependencies = _section(out, "Dependencies")
        assert dependencies, f"{component_id} printed no dependencies"
        if manifest.dependencies:
            for dependency in manifest.dependencies:
                assert dependency in dependencies, component_id
        else:
            assert dependencies == "(none)", component_id

        # Usage notes carry `install`, which stores a literal `<id>` placeholder
        # (26 of 35 manifests do). The id is known here, so it is resolved.
        assert _section(out, "Usage notes") == manifest.install.replace(
            "<id>", manifest.id
        ), component_id

        # An example that names a file which does not exist is worse than none.
        example = _section(out, "Example")
        assert example, f"{component_id} printed no example"
        if not example.startswith("("):
            assert (_REGISTRY.get(component_id).directory / example).is_file(), (
                f"{component_id} names a missing example: {example}"
            )


def test_info_prints_requirements_for_every_component(
    capsys: pytest.CaptureFixture,
) -> None:
    """FR-14: requirements, as python bound plus declared packages."""
    for component_id in _IDS:
        code, out, err = _run("info", component_id, capsys=capsys)
        assert code == EXIT_OK, f"{component_id}: {err}"

        manifest = _REGISTRY.get(component_id).manifest
        requirements = _section(out, "Requires")
        assert f"python >= {manifest.requires_python.min}" in requirements, component_id
        for package in manifest.required_packages:
            assert f"{package.name}{package.specifier}" in requirements, component_id


def test_info_prints_trade_offs_and_marks_the_absent_case(
    capsys: pytest.CaptureFixture,
) -> None:
    """FR-14: trade-offs, which most manifests do not yet record."""
    for component_id in _IDS:
        code, out, err = _run("info", component_id, capsys=capsys)
        assert code == EXIT_OK, f"{component_id}: {err}"

        recorded = _REGISTRY.get(component_id).manifest.tradeoffs
        trade_offs = _section(out, "Trade-offs")
        if recorded:
            assert trade_offs == recorded, component_id
        else:
            assert trade_offs == "(not recorded in this manifest)", component_id


def test_list_and_info_agree_on_lifecycle_status(
    capsys: pytest.CaptureFixture,
) -> None:
    """FR-35: two commands must not report two statuses for one component."""
    code, listing, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err
    listed = {row[0]: row[2] for row in _rows(listing)}

    for component_id in _IDS:
        code, out, err = _run("info", component_id, capsys=capsys)
        assert code == EXIT_OK, f"{component_id}: {err}"
        header = dict(
            field.split(": ", 1)
            for field in out.splitlines()[1].split("   ")
            if ": " in field
        )
        assert header["status"] == listed[component_id], component_id


def test_unknown_id_exits_1_naming_the_id_and_offering_the_real_ones(
    capsys: pytest.CaptureFixture,
) -> None:
    """The observable's failure path, and NFR-7's demand for a next action."""
    code, out, err = _run("info", "definitely-not-a-component", capsys=capsys)
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "definitely-not-a-component" in err
    assert "known ids" in err
    assert "fixed-size-chunker" in err
    assert "Traceback" not in err


def test_missing_id_argument_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    """Argument parsing runs before any command, so exit 2 rather than 1.

    ``main`` swallows argparse's ``SystemExit`` and returns its code, which
    keeps the console script's status and an embedding caller's return value
    the same number instead of one of them being an exception.
    """
    code, out, err = _run("info", capsys=capsys)
    assert code == EXIT_USAGE_ERROR
    assert out == ""
    assert "required" in err
    assert "ID" in err


# ------------------------------------------------- registry discovery itself --


def test_list_reads_the_registry_it_discovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Prove the CLI lists the tree in front of it, not a hardcoded list.

    ``_components_root`` walks up from the current directory, so a ``components``
    directory in the temporary cwd wins over the repository's own -- which is
    what makes this test able to substitute a registry at all.
    """
    _write_demo_registry(tmp_path)
    monkeypatch.chdir(tmp_path)

    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_OK, err
    assert "demo-chunker" in out
    assert "fixed-size-chunker" not in out


def test_a_broken_manifest_is_reported_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """NFR-7 for registry data: every problem is named, and only named."""
    directory = tmp_path / "components" / "chunking"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "manifest.json" in err
    assert "schema violation" in err
    assert "Traceback" not in err


def test_running_outside_the_repository_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """NFR-7: "no components found" must name the directory it walked from."""
    if any((base / "components").is_dir() for base in (tmp_path, *tmp_path.parents)):
        pytest.skip("temp directory is not isolated from a components tree")
    monkeypatch.chdir(tmp_path)

    code, out, err = _run("list", capsys=capsys)
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "components" in err
    assert "Traceback" not in err


# ----------------------------------------------------------- real processes --


def _run_as_process(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI the way ``uv run ragkit`` does, as a real process.

    The package is not installed under the interpreter running this suite, so
    `src` is handed over on PYTHONPATH; everything else about the invocation is
    what the console script does.
    """
    return subprocess.run(
        [sys.executable, "-m", "rag_code_library.cli", *args],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": str(_SRC)},
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_the_console_entry_point_lists_every_component() -> None:
    """AC-7 end to end: exit code 0, ids on stdout, nothing on stderr."""
    process = _run_as_process("list")
    assert process.returncode == EXIT_OK, process.stderr
    assert process.stderr == ""
    listed = [line.split(maxsplit=1)[0] for line in process.stdout.splitlines()[2:]]
    assert sorted(listed) == sorted(_IDS)


def test_the_console_entry_point_exits_one_for_an_unknown_id() -> None:
    """AC-8's failure path, with the exit code a shell would actually see."""
    process = _run_as_process("info", "definitely-not-a-component")
    assert process.returncode == EXIT_DOMAIN_ERROR
    assert "definitely-not-a-component" in process.stderr
    assert "Traceback" not in process.stderr

"""`ragkit add` against a real destination project (T-11 gate ``add``).

Covers FR-15 (copy source, docs, tests and manifest; never install; never touch
a dependency file; print packages and services), FR-18 (conflicts name the file,
the owner and the remedy), FR-19 (exit codes, no raw tracebacks), AC-9 and
AC-10 (byte-identical copies) and AC-12 (conflict, ``--force``, and force still
refusing a path outside the root).

Driven through ``main(argv)`` -- the function the ``ragkit`` console script
calls -- because that is the same code path without a process per assertion.
Where the *exit code itself* is the claim, a test runs a real process instead;
see ``test_cli_add_process`` at the bottom.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
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
from rag_code_library.generator import (
    DependencyError,
    add_component,
    resolve_order,
)
from rag_code_library.registry import Registry, UnknownComponentError

_LIBRARY_ROOT = _ROOT / "components"
_REGISTRY = Registry.load(_LIBRARY_ROOT)
_DEP_FILES = ("pyproject.toml", "uv.lock", "requirements.txt")


@pytest.fixture(scope="module", autouse=True)
def _registry_is_usable() -> None:
    assert _REGISTRY.ok, f"fixture registry did not load: {_REGISTRY.issues}"


def _run(*args: str, capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _project(tmp_path: Path, name: str = "demo-client") -> Path:
    """A destination project that can be named as a package."""
    project = tmp_path / name.replace("_", "-")
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    return project


def _dependency_hashes(project: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((project / name).read_bytes()).hexdigest()
        for name in _DEP_FILES
        if (project / name).is_file()
    }


def _add_dependencies(project: Path) -> None:
    """Dependency files that must survive an `add` untouched."""
    (project / "uv.lock").write_text("# a lock file\n", encoding="utf-8")
    (project / "requirements.txt").write_text("# pinned\n", encoding="utf-8")


def _written_files(project: Path) -> list[Path]:
    return sorted(
        path
        for path in project.rglob("*")
        if path.is_file() and path.name not in _DEP_FILES
    )


# --------------------------------------------------------------- FR-15, AC-10


def test_a_real_add_copies_every_file_byte_for_byte(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """AC-10: the destination receives the component's files, as they are."""
    project = _project(tmp_path)
    code, _out, err = _run(
        "add", "fixed-size-chunker", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    entry = _REGISTRY.get("fixed-size-chunker")
    expected = {
        (entry.directory / entry.manifest.source).read_bytes(),
        (entry.directory / entry.manifest.documentation).read_bytes(),
        (entry.directory / entry.manifest.tests).read_bytes(),
        entry.path.read_bytes(),
    }
    written = _written_files(project)
    assert len(written) == 4, [p.relative_to(project).as_posix() for p in written]
    assert {path.read_bytes() for path in written} == expected

    # And the declared relative paths are kept, not flattened: a manifest that
    # says tests live in `tests/` should not have them dropped at the root.
    assert (project / "src/demo_client/rag/chunking/fixed_size_chunker.py").is_file()
    assert (
        project / "src/demo_client/rag/chunking/fixed-size-chunker/tests/test_component.py"
    ).is_file()


def test_dependencies_are_copied_before_the_component(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The whole point of transitive resolution (DECISION-fcd5b130).

    ``storage-adapters`` needs both ``core-types`` and ``vector-store-protocol``,
    and the latter needs ``core-types`` too -- so a naive breadth-first order
    would write a module importing one that is not there yet.
    """
    project = _project(tmp_path)
    code, out, err = _run(
        "add", "storage-adapters", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    order = [
        line.split()[1]
        for line in out.splitlines()
        if line.strip().startswith("copy ")
    ]
    assert order.index("core-types") < order.index("vector-store-protocol")
    assert order.index("vector-store-protocol") < order.index("storage-adapters")
    assert (project / "src/demo_client/rag/core_types.py").is_file()


def test_no_dependency_file_is_ever_modified(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """FR-15's hard prohibition, on the files it names and one it implies."""
    project = _project(tmp_path)
    _add_dependencies(project)
    before = _dependency_hashes(project)

    for component in ("settings", "fixed-size-chunker"):
        code, _out, err = _run(
            "add", component, "--destination", str(project), "--force", capsys=capsys
        )
        assert code == EXIT_OK, err

    assert _dependency_hashes(project) == before, "ragkit add changed a dependency file"


def test_add_prints_the_packages_it_needs_and_the_command_to_get_them(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """FR-15: required and optional packages, with install commands."""
    project = _project(tmp_path)
    code, out, err = _run(
        "add", "settings", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    assert "pydantic>=2.13" in out
    assert "pydantic-settings>=2.14" in out
    assert 'uv add "pydantic>=2.13"' in out
    assert "Installed nothing, and changed no dependency file." in out

    code, out, err = _run(
        "add", "token-chunker", "--destination", str(project), "--force", capsys=capsys
    )
    assert code == EXIT_OK, err
    # Optional stays visibly separate from required -- a chunker offering a
    # cross-encoder must not read as demanding one.
    assert "tiktoken>=0.7.0" in out
    assert "add only the ones you use" in out


def test_add_prints_the_external_services_a_component_expects(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """FR-15: the API keys a component will ask for, before it is used."""
    project = _project(tmp_path)
    code, out, err = _run(
        "add", "remote-embeddings", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    assert "Embedding provider HTTP API" in out
    assert "RAG_EMBEDDINGS_API_KEY" in out
    assert "RAG_EMBEDDINGS_URL" in out


# ------------------------------------------------------------------ FR-18, AC-12


def test_a_second_add_conflicts_naming_the_file_the_owner_and_the_remedy(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """AC-12, and all three parts of FR-18."""
    project = _project(tmp_path)
    code, _out, err = _run(
        "add", "fixed-size-chunker", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    code, out, err = _run(
        "add", "fixed-size-chunker", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "fixed_size_chunker.py" in err  # which file
    assert "owned by: fixed-size-chunker" in err  # which component
    assert "--force" in err  # what to do
    assert "Traceback" not in err


def test_two_components_in_one_category_coexist(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The regression the per-component namespace exists to prevent.

    ``fixed-size-chunker`` and ``token-chunker`` both root at ``rag/chunking``
    and both ship a ``README.md`` and a ``manifest.json``. Under a flat layout
    the second ``add`` would have failed against the first; a Python module is
    named after itself and needs no help, but a README is not.
    """
    project = _project(tmp_path)
    for component in ("fixed-size-chunker", "token-chunker"):
        code, _out, err = _run(
            "add", component, "--destination", str(project), capsys=capsys
        )
        assert code == EXIT_OK, f"{component}: {err}"

    for component in ("fixed-size-chunker", "token-chunker"):
        assert (
            project / f"src/demo_client/rag/chunking/{component}/manifest.json"
        ).is_file(), component


def test_a_conflict_names_a_component_other_than_the_one_being_added(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """FR-18's case that ownership lookup exists for: not the component in flight.

    Staged by hand-placing one component's source where another's belongs,
    which is what a botched manual copy or a stale checkout looks like.
    """
    project = _project(tmp_path)
    core = _REGISTRY.get("core-types")
    imposter = core.directory / core.manifest.source

    target = project / "src/demo_client/rag/retrieval/dense_retrieval.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(imposter.read_bytes())

    code, _out, err = _run(
        "add", "dense-retrieval", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "dense_retrieval.py" in err
    assert "owned by: core-types" in err
    assert "--force" in err


def test_force_overwrites_inside_the_destination_root(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """AC-12: --force succeeds, and restores the component's real bytes."""
    project = _project(tmp_path)
    code, _out, err = _run(
        "add", "fixed-size-chunker", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_OK, err

    target = project / "src/demo_client/rag/chunking/fixed_size_chunker.py"
    target.write_text("# hand-edited\n", encoding="utf-8")

    code, _out, err = _run(
        "add",
        "fixed-size-chunker",
        "--destination",
        str(project),
        "--force",
        capsys=capsys,
    )
    assert code == EXIT_OK, err
    source = _REGISTRY.get("fixed-size-chunker").directory / "fixed_size_chunker.py"
    assert target.read_bytes() == source.read_bytes()


def test_force_still_refuses_a_path_outside_the_destination_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """AC-12's other half: force widens overwrite, never containment (NFR-3).

    The schema already refuses a ``..`` destination, so this reaches the write
    path by steering the planner there directly -- which is the honest way to
    prove the *write* refuses, rather than proving a validator does.
    """
    from rag_code_library.generator import planner

    project = _project(tmp_path)
    monkeypatch.setattr(
        planner,
        "resolve_destination",
        lambda _template, package: f"{package}/../../../escaped.py",
    )

    code, _out, err = _run(
        "add",
        "fixed-size-chunker",
        "--destination",
        str(project),
        "--force",
        capsys=capsys,
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "escape" in err or "outside" in err
    assert "Traceback" not in err
    assert not (tmp_path / "escaped.py").exists()
    assert not (tmp_path.parent / "escaped.py").exists()


# ----------------------------------------------------------------- FR-19, NFR-7


def test_an_unknown_component_exits_1_and_names_the_id(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    project = _project(tmp_path)
    code, out, err = _run(
        "add", "definitely-not-a-component", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert out == ""
    assert "definitely-not-a-component" in err
    assert "Traceback" not in err
    assert _written_files(project) == []


def test_a_missing_destination_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code, _out, err = _run(
        "add",
        "fixed-size-chunker",
        "--destination",
        str(tmp_path / "nowhere"),
        capsys=capsys,
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "does not exist" in err
    assert "Traceback" not in err
    assert not (tmp_path / "nowhere").exists()


def test_an_ambiguous_package_is_refused_until_one_is_given(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """package_detector returns None rather than guessing (the silent failure)."""
    project = tmp_path / "bare"
    project.mkdir()  # no pyproject.toml, no src/

    code, _out, err = _run(
        "add", "fixed-size-chunker", "--destination", str(project), capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "--package" in err
    assert _written_files(project) == []

    code, _out, err = _run(
        "add",
        "fixed-size-chunker",
        "--destination",
        str(project),
        "--package",
        "acme",
        capsys=capsys,
    )
    assert code == EXIT_OK, err
    assert (project / "src/acme/rag/chunking/fixed_size_chunker.py").is_file()


def test_a_partial_failure_reports_the_whole_plan_and_what_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A run that stops half-way must say how much of the plan is left undone.

    ``storage-adapters`` plans 12 files. Steering only its own destination out
    of the root lets the first 8 land and fails the 9th, which is the shape a
    real partial failure has -- the common case, not the first file.
    """
    from rag_code_library.generator import planner

    genuine = planner.resolve_destination

    def steer(template: str, package: str) -> str:
        if "storage_adapters.py" in template:
            return f"{package}/../../../escaped.py"
        return genuine(template, package)

    # Without --force, containment is caught during pre-flight and nothing is
    # written at all: the conflict detector resolves every target before the
    # first write, so the bad one is found while it can still cost nothing.
    clean = _project(tmp_path / "preflight")
    monkeypatch.setattr(planner, "resolve_destination", steer)
    code, _out, err = _run(
        "add", "storage-adapters", "--destination", str(clean), capsys=capsys
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "escape" in err
    assert not _written_files(clean), "a refused target still wrote files"

    # With --force the pre-flight is skipped, so the run gets as far as it can
    # and then has to account for everything it did not finish.
    project = _project(tmp_path / "partial")
    code, _out, err = _run(
        "add",
        "storage-adapters",
        "--destination",
        str(project),
        "--force",
        capsys=capsys,
    )
    assert code == EXIT_DOMAIN_ERROR
    # 12 planned, 8 written before the 9th refused -- not "9 of 10".
    assert "writing 8 of 12 file(s)" in err
    # The steered base gets the source's own declared name appended to it.
    assert "storage_adapters.py" in err
    assert "partial" in err
    assert "Traceback" not in err
    # The files that did land are real, which is what makes the report matter.
    assert (project / "src/demo_client/rag/core_types.py").is_file()
    assert not (tmp_path / "escaped.py").exists()


# ------------------------------------------------------- dependency resolution


class _StubEntry:
    """Enough of a ManifestEntry for resolve_order, and nothing more."""

    def __init__(self, component_id: str, dependencies: tuple[str, ...]) -> None:
        self.manifest = type(
            "Manifest", (), {"id": component_id, "dependencies": dependencies}
        )()


class _StubRegistry:
    """Duck-typed: resolve_order uses ``get``, ``ids`` and nothing else."""

    def __init__(self, graph: dict[str, tuple[str, ...]]) -> None:
        self._graph = graph

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._graph))

    def get(self, component_id: str) -> _StubEntry | None:
        if component_id not in self._graph:
            return None
        return _StubEntry(component_id, self._graph[component_id])


def test_a_dependency_cycle_names_the_path_that_closes_it() -> None:
    registry = _StubRegistry({"a": ("b",), "b": ("c",), "c": ("a",)})
    with pytest.raises(DependencyError) as excinfo:
        resolve_order("a", registry)
    assert "a -> b -> c -> a" in str(excinfo.value)


def test_a_missing_dependency_is_reported_with_the_ids_that_do_exist() -> None:
    registry = _StubRegistry({"a": ("ghost",)})
    with pytest.raises(UnknownComponentError) as excinfo:
        resolve_order("a", registry)
    assert "ghost" in str(excinfo.value)


def test_add_returns_dependencies_before_the_requested_component(tmp_path: Path) -> None:
    """The API contract the CLI renders, asserted directly."""
    project = _project(tmp_path / "api")
    result = add_component(
        "hybrid-retrieval",
        destination=project,
        registry=_REGISTRY,
        library_root=_LIBRARY_ROOT,
    )
    assert result.added[-1] == "hybrid-retrieval"
    assert "core-types" in result.added
    assert result.added.index("core-types") == 0


# --------------------------------------------------------- the real exit code


def test_the_console_entry_point_reports_a_conflict_with_exit_one() -> None:
    """Exit codes are properties of a process, so prove them in one."""
    import tempfile

    with tempfile.TemporaryDirectory() as holder:
        project = Path(holder) / "client"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[project]\nname = "client"\n', encoding="utf-8"
        )
        environment = {**os.environ, "PYTHONPATH": str(_SRC)}

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            # check=False on purpose: the exit code is the assertion, so the
            # call must return rather than raise on a non-zero result.
            return subprocess.run(
                [sys.executable, "-m", "rag_code_library.cli", *args],
                cwd=_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )

        first = run("add", "fixed-size-chunker", "--destination", str(project))
        assert first.returncode == EXIT_OK, first.stderr

        second = run("add", "fixed-size-chunker", "--destination", str(project))
        assert second.returncode == EXIT_DOMAIN_ERROR
        assert "fixed_size_chunker.py" in second.stderr
        assert "Traceback" not in second.stderr

#!/usr/bin/env python
"""Verify files written by `ragkit add` match their sources byte-for-byte (FR-25).

AC-10's proof, and two checks that follow from the same question
--------------------------------------------------------------
Copying is only half of "added a component": the copy has to be the file, and
the file has to work in a project that has never heard of this library. So
three things are asserted per written file:

1. **Byte-for-byte** against the component in the registry (FR-25, AC-10). Not
   "parses the same" or "imports the same" -- the bytes, because that is the
   claim, and anything weaker lets a rewritten header drift in unnoticed.
2. **Independence**: no copied Python file imports ``rag_code_library``. A
   destination that copied a chunker but had to import ``Chunk`` from the
   library has not been given a component, it has been given a dependency.
3. **No dependency file touched**: ``pyproject.toml`` is hashed before and
   after. FR-15 forbids ``ragkit add`` from modifying it, ``uv.lock`` or
   ``requirements*.txt``; asserting it costs one line and catches the single
   most tempting silent change.

The fixture project is a throwaway under the system temp directory, so nothing
here can reach a real one.
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_code_library.generator import add_component
from rag_code_library.registry import Registry

#: One component with a dependency (``core-types``), one directory-destination
#: component (``settings``), and one with optional packages (``token-chunker``)
#: -- the three shapes a copy has to get right.
COMPONENTS: tuple[str, ...] = ("dense-retrieval", "settings", "token-chunker")

_PYPROJECT = b'[project]\nname = "fixture-client"\nversion = "0.1.0"\n'


def imports_the_library(path: Path) -> tuple[str, ...]:
    """Fully-qualified module names this file imports from ``rag_code_library``.

    AST, not a substring search: a comment mentioning the package is not an
    import, and a fixture file whose docstring explains what it was copied from
    will mention it.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return (f"<could not parse: {exc}>",)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            found.append(node.module or "")
    return tuple(name for name in found if name.split(".")[0] == "rag_code_library")


def main() -> int:
    library_root = REPO / "components"
    registry = Registry.load(library_root)
    if not registry.ok:
        for issue in registry.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1

    problems: list[str] = []
    verified = 0

    with tempfile.TemporaryDirectory() as holder:
        project = Path(holder) / "fixture-client"
        project.mkdir()
        pyproject = project / "pyproject.toml"
        pyproject.write_bytes(_PYPROJECT)

        for component_id in COMPONENTS:
            try:
                result = add_component(
                    component_id,
                    destination=project,
                    registry=registry,
                    library_root=library_root,
                )
            except Exception as exc:  # noqa: BLE001 - this is the reporter
                problems.append(f"{component_id}: add failed: {exc}")
                continue

            for operation in result.operations:
                written = project / operation.target
                if not written.is_file():
                    problems.append(f"missing after add: {operation.target}")
                    continue
                if written.read_bytes() != operation.source.read_bytes():
                    problems.append(f"not byte-identical: {operation.target}")
                verified += 1

                if written.suffix == ".py":
                    for module in imports_the_library(written):
                        problems.append(
                            f"{operation.target} imports the library: {module}"
                        )

            for dependency in result.added:
                if dependency == component_id:
                    continue
                print(f"  {component_id} pulled in {dependency}")

        if pyproject.read_bytes() != _PYPROJECT:
            problems.append("pyproject.toml was modified by ragkit add")

    if problems:
        print(f"FAILED: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: {verified} file(s) across {len(COMPONENTS)} component(s) copied "
        "byte-for-byte; none import the library; no dependency file modified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

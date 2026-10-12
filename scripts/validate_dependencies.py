"""Prove every third-party import is declared somewhere (AC-14, FR-3).

A component that imports a package it never declared is a component that works
on the machine it was written on and fails on the machine it is copied to. AC-14
requires that no shipped file import anything that is neither a core dependency
of the project nor an optional package the component itself declares.

What "declared" means here
--------------------------
Four cases are accepted, in this order, because each is a different claim:

1. **Core.** The import is in ``pyproject.toml``'s ``[project].dependencies``.
   This is what the CLI and the registry genuinely need at all times.

2. **Development.** The import is in a dependency group and the file is a test.
   ``pytest`` is a real dependency of the suite and of nothing else; allowing
   it in a component's *implementation* would be wrong, so the exemption is
   scoped to test files rather than granted globally.

3. **Declared optional.** The owning component's manifest lists the package as
   optional or required **and** the import is guarded -- that is, it sits in a
   ``try`` whose handler catches ``ImportError``. An unguarded optional import
   is a component that cannot even be imported without its extras, which
   contradicts FR-3.

4. **Transitive.** A deliberately short, in-code list of imports that arrive
   with a declared optional package and cannot be avoided, each paired with the
   package that brings it and still required to be guarded. This list is
   visible in the output every run, so an entry cannot quietly accumulate.

5. **Renamed on PyPI.** Some packages install a module under a different name
   -- ``python-magic`` provides ``magic``, ``pillow`` provides ``PIL``. The
   import is translated back to its package name and then judged by rule 3, so
   the translation alone proves nothing: the owning component must still
   declare that package, and the import must still be guarded.

What is *not* a third-party import
----------------------------------
``import core_types`` inside another component resolves to a file in this
repository, not to a package. Components legitimately import one another --
``ragkit add`` copies the whole dependency graph -- and only component *tests*
are forbidden from reaching into the library package. So anything resolvable
inside the repository is local, and out of this check's scope.
"""

from __future__ import annotations

import ast
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Resolved once at import. Anything here needs no declaration anywhere.
stdlib_names = frozenset(sys.stdlib_module_names)

#: Imports that necessarily arrive with a declared optional package. Each entry
#: is paired with the package that provides it, and the pair is only accepted
#: when that component really does declare the provider -- so removing the
#: provider from the manifest re-breaks the check.
TRANSITIVE: dict[str, str] = {
    # `faiss` cannot be installed without numpy; components/storage/adapters
    # imports it lazily alongside faiss and raises a typed error if absent.
    "numpy": "faiss",
}

#: Import names whose module differs from the package that installs them.
#: Translation is all this buys: the resolved package must still be declared by
#: the owning component and the import must still be guarded, so an entry here
#: cannot launder an undeclared dependency.
IMPORT_PROVIDED_BY: dict[str, str] = {
    # PyPI name `python-magic`, module name `magic`.
    "magic": "python_magic",
}


@dataclass
class Violation:
    """An import that nothing declares."""

    module: str
    path: str
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  import {self.module!r}  {self.reason}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    transitive_used: list[tuple[str, str, str]] = field(default_factory=list)
    scanned: int = 0
    imports_seen: int = 0


def _normalise(name: str) -> str:
    """``sentence-transformers`` and ``sentence_transformers`` are one package."""
    return name.strip().lower().replace("-", "_")


def _requirement_name(specification: str) -> str:
    """Pull the package name out of ``"pydantic>=2.13"``."""
    for separator in ("[", ">", "=", "<", "!", "~", ";", " "):
        index = specification.find(separator)
        if index != -1:
            specification = specification[:index]
    return _normalise(specification)


def _shipped_python_files() -> list[Path]:
    files: list[Path] = []
    for pattern in ("components/**/*.py", "src/**/*.py", "scripts/*.py"):
        files.extend(_ROOT.glob(pattern))
    return sorted({p for p in files if "__pycache__" not in p.parts and p.is_file()})


def _local_modules() -> set[str]:
    """Top-level names that resolve to a file in this repository."""
    names = {"rag_code_library", "__future__"}
    for path in _shipped_python_files():
        names.add(_normalise(path.stem))
        # A directory of modules is importable as a namespace too.
        if path.parent.name not in {"components", "src", "scripts", "tests"}:
            names.add(_normalise(path.parent.name))
    return names


def _load_dependencies() -> tuple[set[str], set[str]]:
    """(core, development) package names from pyproject.toml."""
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    core = {
        _requirement_name(spec)
        for spec in project.get("dependencies", [])
    }
    dev: set[str] = set()
    for group in data.get("dependency-groups", {}).values():
        dev.update(_requirement_name(spec) for spec in group)
    return core, dev


def _load_manifest_packages() -> dict[str, tuple[set[str], str]]:
    """component id -> (declared package names, manifest path)."""
    declared: dict[str, tuple[set[str], str]] = {}
    for manifest_path in sorted((_ROOT / "components").rglob("manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        component_id = data.get("id")
        if not component_id:
            continue
        packages: set[str] = set()
        for key in ("optional_packages", "required_packages"):
            for package in data.get(key, []) or []:
                name = package.get("name") if isinstance(package, dict) else str(package)
                if name:
                    packages.add(_normalise(name))
        declared[component_id] = (packages, manifest_path.relative_to(_ROOT).as_posix())
    return declared


def _owning_manifest(path: Path) -> Path | None:
    """The manifest for the component this file belongs to, if any."""
    current = path.parent
    root = _ROOT.resolve()
    while current.resolve() != root:
        candidate = current / "manifest.json"
        if candidate.is_file():
            return candidate
        current = current.parent
    return None


def _is_test_file(path: Path) -> bool:
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return "tests" in path.parts


def _guarded_import_lines(tree: ast.AST) -> set[int]:
    """Line numbers of imports that sit in a ``try`` catching ``ImportError``.

    This is the shape FR-3 requires of an optional dependency::

        try:
            import faiss
        except ImportError as exc:
            raise MissingStorageDependency(...) from exc

    Anything else -- a bare module-level import of an optional package -- means
    the component cannot be imported at all without its extras.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches_import_error = False
        for handler in node.handlers:
            type_name = None
            if handler.type is not None:
                if isinstance(handler.type, ast.Name):
                    type_name = handler.type.id
                elif isinstance(handler.type, ast.Tuple):
                    type_name = next(
                        (e.id for e in handler.type.elts if isinstance(e, ast.Name)),
                        None,
                    )
            # A bare ``except ImportError`` and ``except X as exc`` both count;
            # an untyped ``except Exception`` does not, because it swallows
            # without saying what it was guarding against.
            if type_name == "ImportError":
                catches_import_error = True
        if not catches_import_error:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(inner.lineno)
    return guarded


def _imports(tree: ast.AST) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], node.lineno))
    return found


def check() -> Report:
    core, dev = _load_dependencies()
    manifest_packages = _load_manifest_packages()
    local = _local_modules()
    report = Report()

    for path in _shipped_python_files():
        report.scanned += 1
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            report.violations.append(
                Violation("<unreadable>", path.relative_to(_ROOT).as_posix(), 0,
                          f"could not be parsed: {exc}")
            )
            continue

        guarded = _guarded_import_lines(tree)
        relative = path.relative_to(_ROOT).as_posix()
        is_test = _is_test_file(path)
        manifest = _owning_manifest(path)
        component_packages: set[str] = set()
        if manifest is not None:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            component_id = data.get("id")
            if component_id and component_id in manifest_packages:
                component_packages = manifest_packages[component_id][0]

        for module, line in _imports(tree):
            if module in stdlib_names or module in local:
                continue
            report.imports_seen += 1
            name = _normalise(module)

            if name in core:
                continue
            if is_test and name in dev:
                continue

            package_name = IMPORT_PROVIDED_BY.get(name, name)

            if package_name in component_packages:
                if line in guarded:
                    continue
                report.violations.append(
                    Violation(
                        module, relative, line,
                        f"provided by the declared package {package_name!r}, "
                        "but the import is not guarded by try/except "
                        "ImportError",
                    )
                )
                continue

            provider = TRANSITIVE.get(name)
            if provider is not None:
                provider_normalised = _normalise(provider)
                if provider_normalised in component_packages and line in guarded:
                    report.transitive_used.append((module, provider, relative))
                    continue

            report.violations.append(
                Violation(
                    module, relative, line,
                    "not declared in pyproject.toml, not a component "
                    "dependency, and not a dev dependency of a test file",
                )
            )

    return report


def main() -> int:
    print("validate_dependencies: checking every shipped .py import is declared")

    core, dev = _load_dependencies()
    print(f"  core dependencies     : {', '.join(sorted(core)) or '(none)'}")
    print(f"  development group     : {', '.join(sorted(dev)) or '(none)'}")
    print(f"  transitive allowances : "
          f"{', '.join(f'{k} (via {v})' for k, v in TRANSITIVE.items()) or '(none)'}")
    print(f"  renamed on PyPI       : "
          f"{', '.join(f'{k} <- {v}' for k, v in IMPORT_PROVIDED_BY.items()) or '(none)'}")

    report = check()
    print(f"  files scanned         : {report.scanned}")
    print(f"  third-party imports   : {report.imports_seen}")

    if report.transitive_used:
        print(f"\n  accepted as transitive ({len(report.transitive_used)}):")
        for module, provider, path in sorted(set(report.transitive_used)):
            print(f"    {module:<18} via declared {provider:<14} in {path}")

    if report.violations:
        print(f"\n{len(report.violations)} undeclared import(s):")
        for violation in report.violations:
            print(f"  FAIL {violation}")
        print(
            "\nDeclare the package in pyproject.toml, or in the owning component's\n"
            "manifest as an optional dependency guarding the import with\n"
            "try/except ImportError. Do not add an entry to TRANSITIVE to make a\n"
            "real undeclared dependency pass."
        )
        return 1

    print(
        "\nOK: every third-party import is a core dependency, a development\n"
        "    dependency of a test, or a guarded import its component declares."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

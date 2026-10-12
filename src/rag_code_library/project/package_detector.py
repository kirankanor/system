"""Determine the importable package name of a destination project.

Why returning ``None`` matters
------------------------------
A destination template reads ``src/{project_package}/rag/chunking/...``, so
something has to supply the name. Guessing it is the failure mode worth
avoiding: writing into ``src/rac/`` when the project is called ``rag`` leaves a
copy that runs, imports cleanly, and is simply not the package the project
actually uses. That is a silently wrong result rather than an error, so where
the name cannot be *determined* this module says so and the CLI asks for it
explicitly (``--package``) instead of choosing.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

__all__ = ["detect_package", "normalize_package_name"]

#: Characters a module path may not contain; everything else collapses to one ``_``.
_ILLEGAL = re.compile(r"[^0-9A-Za-z_]+")


def normalize_package_name(name: str) -> str:
    """Turn a distribution name into a legal module name.

    ``rag-code-library`` -> ``rag_code_library``, which is what PEP 503
    normalization plus ``import`` legality give together. A leading digit is
    prefixed because ``2fast`` is a fine project name and an illegal identifier.
    """
    normalized = _ILLEGAL.sub("_", name.strip()).strip("_")
    if normalized[:1].isdigit():
        normalized = f"_{normalized}"
    return normalized


def detect_package(root: Path) -> str | None:
    """The destination's package name, or ``None`` when it cannot be determined.

    ``[project].name`` in ``pyproject.toml`` comes first: it is the name the
    project publishes under, and the one its own layout will match. Failing
    that, a single package directory under ``src/``. Anything else is ambiguous
    and returns ``None`` rather than picking between candidates.
    """
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        # An unreadable pyproject.toml is not this module's error to report:
        # nothing here needs it, and failing would turn "the package name is
        # unknown" into a TOML traceback.
        try:
            document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            document = {}
        project = document.get("project")
        name = project.get("name") if isinstance(project, dict) else None
        if isinstance(name, str) and name.strip():
            return normalize_package_name(name)

    src = root / "src"
    if src.is_dir():
        packages = [
            child.name
            for child in sorted(src.iterdir())
            if child.is_dir() and (child / "__init__.py").is_file()
        ]
        if len(packages) == 1:
            return normalize_package_name(packages[0])
    return None

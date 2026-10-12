#!/usr/bin/env python3
"""Validate every component manifest, path, id, dependency and destination.

This is the executable form of FR-24 and AC-5:

* FR-24 -- validate every manifest against the schema and fail with a non-zero
  exit listing each invalid component and reason; runnable standalone and from
  CI.
* AC-5  -- exit 0 over the full component set, and exit non-zero with a
  per-component reason when a manifest is deliberately corrupted.

Usage
-----
    python scripts/validate_components.py                 # the components/ tree
    python scripts/validate_components.py <dir-or-file>   # e.g. a corrupt fixture

Exit codes
----------
    0   every manifest valid
    1   at least one problem, each printed as ``<component> (<path>): <reason>``

Why it bootstraps ``sys.path``
------------------------------
``python`` on this host is the system interpreter, where the package is not
installed (the installed copy lives in the uv venv). Prepending ``src`` makes
the script run under any interpreter, from any working directory, with no
install step -- which is what "runnable standalone and from CI" requires, and
it guarantees it validates the working tree rather than a stale installed copy.
No task owns ``pyproject.toml``, so there is no supported alternative.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from rag_code_library.registry import Issue, Registry  # noqa: E402


def _display(issue: Issue, root: Path) -> str:
    """Render ``issue`` with its path relative to ``root`` where possible.

    Absolute paths make output machine- and cwd-dependent, which defeats diffing
    two CI runs; relative ones are stable. Paths outside ``root`` are left alone
    rather than mangled.
    """
    if not issue.path:
        return str(issue)
    try:
        relative = Path(issue.path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return str(issue)
    return str(replace(issue, path=relative.as_posix()))


def main(argv: list[str] | None = None) -> int:
    """Validate ``argv`` (or ``components/``) and return the process exit code."""
    args = sys.argv[1:] if argv is None else list(argv)
    targets = [Path(arg) for arg in args] or [_ROOT / "components"]

    total = 0
    issues: list[Issue] = []
    for target in targets:
        registry = Registry.load(target)
        total += registry.count
        issues += list(registry.issues)

    if issues:
        # Diagnostics on stderr, so automation can separate "what went wrong"
        # from the result line and still see both.
        for issue in issues:
            print(_display(issue, _ROOT), file=sys.stderr)
        print(
            f"FAILED: {len(issues)} problem(s) across {total} manifest(s)",
            file=sys.stderr,
        )
        return 1

    # Unreachable when a target is empty or missing: Registry.load already
    # reports both as issues, which is what stops an empty tree from passing
    # as "0 manifests, 0 problems".
    print(
        f"OK: {total} manifest(s) valid "
        "(schema, unique ids, declared files, dependencies, destinations)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

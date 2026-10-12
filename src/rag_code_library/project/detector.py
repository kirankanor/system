"""Resolve ``--destination`` to a real directory, or explain why it cannot be used.

A "project" is not an application architecture, so nothing here infers a
layout, a framework or a build system. Being wrong about those is what writes a
component somewhere the destination cannot import it from, which is worse than
refusing: a wrong path is a silently wrong answer, a refusal is a correct one
(FR-15 asks that writes stay inside a destination the human actually meant).

Containment is *not* decided here. That belongs to
``security/path_safety.py``, on the resolved real path, against a root cached
for the whole operation (NFR-3). Re-deriving it in this package would mean two
places could disagree, and the one that lost would be the one nobody was
watching.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ProjectRootError", "project_root"]


class ProjectRootError(RuntimeError):
    """The destination cannot be used as one.

    Raised rather than returned so ``main``'s domain-error handling renders it
    as one line instead of a traceback (FR-19, NFR-7).
    """


def project_root(destination: str | Path) -> Path:
    """Resolve ``destination`` to an existing directory, symlink-free.

    Resolved through the real filesystem so that a symlinked destination is
    measured by where it actually lands. ``SafeDestination`` resolves it again
    and caches it; the two must agree or containment would be measured against
    two different directories.
    """
    text = os.fspath(destination)
    if not text or not text.strip():
        raise ProjectRootError(
            "destination is empty; pass a directory with --destination"
        )
    root = Path(os.path.realpath(os.path.abspath(os.path.expanduser(text))))

    if not root.exists():
        raise ProjectRootError(
            f"destination {root} does not exist; create the project first, or "
            "point --destination at an existing one"
        )
    if not root.is_dir():
        raise ProjectRootError(
            f"destination {root} is a file, not a directory; pass the project "
            "root instead"
        )
    return root

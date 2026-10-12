"""Which component owns a file that is already in the destination (FR-18).

By content, not by a sidecar record
-----------------------------------
FR-25 and AC-10 require the copies ``ragkit add`` writes to be byte-for-byte
identical to their sources, so a provenance header cannot be added to the files
themselves without breaking the very check that proves the copy was faithful.

A separate state file -- ``.ragkit.json`` listing what was added -- was the
other option, and it was rejected for a duller reason: it drifts. It goes stale
the moment anyone edits a copied file, re-adds with ``--force``, copies a
component by hand, or checks out an older revision. Every one of those leaves
the file saying something untrue, and "which component owns this?" is exactly
the question asked in the middle of a confusing conflict, where a confidently
wrong answer costs the most time.

Comparing bytes against the registry needs no state, cannot drift, and answers
the question for *any* file in the destination -- whoever put it there, however
it arrived.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..registry import ManifestEntry, Registry

__all__ = ["owner_index", "owner_of"]


def _digest(path: Path) -> str | None:
    """SHA-256 of a file, or ``None`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _declared_files(entry: ManifestEntry) -> tuple[Path, ...]:
    """Every file this component would ship: source, docs, tests, manifest.

    The manifest counts as well as the three declared files, because ``add``
    copies it too (FR-15) and an existing ``manifest.json`` is a perfectly
    ordinary thing to collide with.
    """
    paths = [entry.directory / entry.manifest.source]
    for declared in (entry.manifest.documentation, entry.manifest.tests):
        if declared:
            paths.append(entry.directory / declared)
    paths.append(entry.path)
    return tuple(paths)


def owner_index(registry: Registry) -> dict[str, str]:
    """SHA-256 of every shipped file -> the component that ships it.

    ``setdefault`` on purpose: a digest already claimed keeps its first owner,
    so an empty file shared by two components reports whichever loads first
    rather than whichever the filesystem happened to enumerate last.
    """
    index: dict[str, str] = {}
    for entry in registry:
        for path in _declared_files(entry):
            digest = _digest(path)
            if digest is not None:
                index.setdefault(digest, entry.manifest.id)
    return index


def owner_of(path: Path, index: dict[str, str]) -> str | None:
    """The component whose shipped file ``path`` is a copy of, else ``None``.

    ``None`` is the honest answer for a file no component ships -- the user's
    own code, or a file edited past recognition. Claiming an owner for those
    would send someone editing the wrong manifest.
    """
    if not path.is_file():
        return None
    digest = _digest(path)
    return index.get(digest) if digest is not None else None

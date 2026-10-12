"""Report every file that already exists, before anything is written (FR-18).

All of them, not the first
--------------------------
``SafeDestination`` already refuses an existing target on its own, but it stops
at the first one. A user who ran ``add`` twice wants the whole list -- three
conflicts is one decision, three runs is three. So conflicts are collected up
front and reported together, and the write phase is left with nothing to
discover.

Ownership is provenance, not assumption
---------------------------------------
An existing file is most likely the same component's own previous copy, but
"likely" is not a thing to report to someone deciding whether to delete a file.
Ownership is looked up by comparing bytes against the registry (see
``provenance``), and an unrecognised file is reported as unowned rather than
attributed to whatever component happens to be in flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..registry import Registry
from ..security.path_safety import SafeDestination
from .planner import Operation
from .provenance import owner_index, owner_of

__all__ = ["Conflict", "detect"]


@dataclass(frozen=True, slots=True)
class Conflict:
    """One target that already exists, and what is at fault."""

    target: str
    """Path relative to the destination root -- the same string ``render`` uses."""

    owner: str | None
    """Component whose shipped file is already there, or ``None`` if unrecognised."""

    same_as_source: bool
    """The existing file is already byte-identical to what would be copied.

    True means the copy is a no-op the user could simply allow. Reported
    because "nothing would actually change" is the single most useful thing to
    know when deciding whether to pass ``--force``.
    """


def _identical(left: Path, right: Path) -> bool:
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


def detect(
    operations: tuple[Operation, ...], safe: SafeDestination, registry: Registry
) -> tuple[Conflict, ...]:
    """Every planned target that already exists, in plan order.

    Uses ``safe.resolve`` rather than ``safe.exists`` alone so that the check
    agrees with the write about what the target *is* -- including rejecting a
    target that escapes the root, which is then the caller's error to surface
    rather than something silently skipped here (NFR-3).
    """
    index = owner_index(registry)
    conflicts: list[Conflict] = []
    for operation in operations:
        if not safe.exists(operation.target):
            continue
        existing = safe.resolve(operation.target)
        same_as_source = _identical(existing, operation.source)
        conflicts.append(
            Conflict(
                target=operation.target,
                owner=operation.component_id
                if same_as_source
                else owner_of(existing, index),
                same_as_source=same_as_source,
            )
        )
    return tuple(conflicts)

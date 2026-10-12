"""Render a plan identically for both paths, then execute it (FR-31).

The whole of FR-31 lives in this module's shape: ``render`` takes the plan and
nothing else, so ``--dry-run`` and a real run cannot produce different text for
the same operations. The dry run stops between the two calls.
"""

from __future__ import annotations

from ..security.path_safety import SafeDestination, UnsafePathError
from .planner import Operation

__all__ = ["PartialFailure", "execute", "render"]


def render(operations: tuple[Operation, ...]) -> str:
    """The operations as text.

    Pure: no clock, no absolute paths, no filesystem. Byte-identical output for
    equal plans is a property of that, not a promise that two renderers were
    kept in step.
    """
    if not operations:
        return "  (nothing to copy)"
    width = max(len(operation.component_id) for operation in operations)
    return "\n".join(
        f"  copy  {operation.component_id:<{width}}  {operation.target}"
        for operation in operations
    )


class PartialFailure(RuntimeError):
    """The run stopped part-way, leaving the destination half-written.

    Carries what *was* written, because "some files were copied and one failed"
    without the list leaves the user with a destination they cannot reason
    about. The message is the whole report, so ``main`` can print one line
    (FR-19, NFR-7).
    """

    def __init__(
        self,
        written: tuple[Operation, ...],
        failed: Operation,
        cause: Exception,
        total: int,
    ) -> None:
        self.written = written
        self.failed = failed
        self.cause = cause
        # The whole plan's size, not written + 1: a run that fails at file 10
        # of 12 has left two more unattempted, and saying "9 of 10" understates
        # how much of the destination is still missing.
        super().__init__(
            f"stopped after writing {len(written)} of {total} file(s): "
            f"could not write {failed.target} ({cause}); the destination is now "
            "partial -- remove the copied files, or re-run with --force to "
            "overwrite what is already there"
        )


def execute(
    operations: tuple[Operation, ...], safe: SafeDestination, *, force: bool
) -> tuple[Operation, ...]:
    """Perform every operation through ``safe``, returning those that landed.

    Every write goes through ``SafeDestination`` and nowhere else (NFR-3), so
    containment, symlink and conflict checks are decided in exactly one place
    rather than re-derived here.

    ``force`` widens nothing: ``SafeDestination`` checks containment before it
    checks ``--force``, so force permits replacing a file that exists *inside*
    the root and cannot reach a path outside it.
    """
    written: list[Operation] = []
    total = len(operations)
    for operation in operations:
        try:
            safe.copy_file(operation.source, operation.target, force=force)
        except (UnsafePathError, OSError) as exc:
            raise PartialFailure(tuple(written), operation, exc, total) from exc
        written.append(operation)
    return tuple(written)

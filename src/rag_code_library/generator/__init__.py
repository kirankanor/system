"""``ragkit add``: resolve, plan, and copy a component into a project (FR-15).

Three questions, answered in order
----------------------------------
*Is this a directory I may write into* (``project``), *what does the component
need before it* (``dependency_manager``), and *where does each file land*
(``planner``). Only then does anything get written, and only through
``SafeDestination`` (NFR-3).

What this package deliberately does not do
------------------------------------------
It never installs a package and never edits ``pyproject.toml``, ``uv.lock``,
``requirements*.txt`` or any other dependency file -- not with ``--force``, not
on request. FR-15 forbids it, and the reason is worth stating: a copy that
quietly changed a project's dependency set would be the one part of the run
nobody could review afterwards. The needed commands are *printed* and left to
the human.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..project import detect_package, project_root
from ..registry import Registry
from ..security.path_safety import SafeDestination
from .conflict_detector import Conflict
from .conflict_detector import detect as detect_conflicts
from .copier import PartialFailure, execute, render
from .dependency_manager import (
    DependencyError,
    package_report,
    render_report,
    resolve_order,
)
from .planner import Operation, plan

__all__ = [
    "AddError",
    "AddResult",
    "ConflictError",
    "DependencyError",
    "PartialFailure",
    "add_component",
    "plan",
    "render",
    "render_report",
    "resolve_order",
]


class AddError(RuntimeError):
    """``ragkit add`` refused; the message is the whole report (FR-19, NFR-7)."""


class ConflictError(AddError):
    """Targets already exist and ``--force`` was not passed (FR-18).

    Carries the conflicts as data as well as a message, so a caller could
    render them its own way; the CLI does not, because one clear report is
    better than a formatter nobody asked for.
    """

    def __init__(self, conflicts: tuple[Conflict, ...], root: Path) -> None:
        self.conflicts = conflicts
        self.root = root
        lines = [f"{len(conflicts)} file(s) already exist in {root}:"]
        for conflict in conflicts:
            lines.append(f"  {conflict.target}")
            lines.append(
                f"    owned by: {conflict.owner or 'not a copy of any component'}"
            )
            if conflict.same_as_source:
                lines.append("    already identical to the component's file")
        lines.append(
            "Choose another --destination, delete those files, or re-run with "
            "--force to overwrite them."
        )
        super().__init__("\n".join(lines))


@dataclass(frozen=True, slots=True)
class AddResult:
    """Everything the CLI needs to report on, with nothing left to compute."""

    component_id: str
    added: tuple[str, ...]
    """Components written, dependencies before the one requested."""

    operations: tuple[Operation, ...]
    written: tuple[Operation, ...]
    """Empty on a dry run -- which is the point of one."""

    dry_run: bool
    report: str
    """The FR-15 package/service block, rendered."""


def add_component(
    component_id: str,
    *,
    destination: str | Path,
    registry: Registry,
    library_root: Path,
    package: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> AddResult:
    """Copy ``component_id`` and everything it depends on into ``destination``.

    Conflicts are collected *before* any write so that a run never stops on the
    first clash and leaves the user to discover the other two by re-running.
    ``--force`` skips that check entirely rather than widening it, and
    containment is still decided per write by ``SafeDestination`` -- force
    permits replacing a file inside the root, never reaching outside it.
    """
    root = project_root(destination)
    resolved_package = package or detect_package(root)
    if not resolved_package:
        raise AddError(
            f"cannot tell what package {root} uses: pyproject.toml has no "
            "[project].name and src/ holds no single package. Pass "
            "--package NAME to say where the component should land."
        )

    added = resolve_order(component_id, registry)
    operations = plan(component_id, registry, resolved_package, library_root)
    safe = SafeDestination(root)

    conflicts = () if force else detect_conflicts(operations, safe, registry)
    if conflicts:
        raise ConflictError(conflicts, root)

    written = () if dry_run else execute(operations, safe, force=force)

    return AddResult(
        component_id=component_id,
        added=added,
        operations=operations,
        written=written,
        dry_run=dry_run,
        report=render_report(
            package_report(registry.get(cid).manifest for cid in added)
        ),
    )

"""The single plan that both ``--dry-run`` and a real run follow (FR-31).

One planner, so the two cannot disagree
---------------------------------------
FR-31 requires the operations printed by ``--dry-run`` to be byte-identical to
the operations the real run performs. Two code paths that independently work out
what to write only satisfy that until the first person refactors one of them, so
here there is one: the plan is built once, ``copier.render`` draws it, and a
dry run simply stops before ``copier.execute``.

What is copied, and where it lands
----------------------------------
Four files per component (FR-15): source, docs, tests and the manifest. They
land in ``destination_base`` -- the directory the manifest's ``destination``
names, or its parent when the destination names a file. Component order comes
from ``dependency_manager.resolve_order``, so a dependency is always written
before the component that imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..project.path_resolver import destination_base, resolve_destination
from ..registry import ManifestEntry, Registry, UnknownComponentError
from .dependency_manager import resolve_order

__all__ = ["Operation", "plan"]


@dataclass(frozen=True, slots=True)
class Operation:
    """One file copy: where it comes from, where it lands, whose it is."""

    component_id: str
    source: Path
    """Absolute path in the library. The only thing read from during a run."""

    source_label: str
    """``source`` relative to the library root, for display only.

    Absolute paths would make the output depend on where the library happens to
    live, which is noise in a diff and unhelpful in a terminal.
    """

    target: str
    """Path relative to the destination root, posix. What gets validated."""


def _declared_files(
    entry: ManifestEntry,
) -> tuple[tuple[str, PurePosixPath, Path], ...]:
    """``(label, declared relative path, absolute source)`` for one entry.

    Ordered source, docs, tests, manifest -- the order a reader expects, and
    the order ``render`` shows.

    The *declared* path is kept, not just its basename: manifests write tests
    as ``tests/test_component.py``, and flattening that to ``test_component.py``
    would quietly drop the directory the component author chose. Whatever it
    contains still has to survive ``SafeDestination``, which rejects ``..`` and
    absolute forms before anything is opened (NFR-3).
    """
    pairs: list[tuple[str, PurePosixPath, Path]] = [
        ("source", PurePosixPath(entry.manifest.source), entry.directory / entry.manifest.source)
    ]
    for label, declared in (
        ("documentation", entry.manifest.documentation),
        ("tests", entry.manifest.tests),
    ):
        if declared:
            pairs.append(
                (label, PurePosixPath(declared), entry.directory / declared)
            )
    pairs.append(("manifest", PurePosixPath("manifest.json"), entry.path))
    return tuple(pairs)


def plan(
    component_id: str,
    registry: Registry,
    package: str,
    library_root: Path,
) -> tuple[Operation, ...]:
    """Every copy `ragkit add` would make, dependencies first.

    Two destinations, because there are two kinds of file
    -----------------------------------------------------
    The **source** lands on the manifest's own ``destination`` path, unchanged.
    That path is an import location: ``rag/storage/storage_adapters.py`` is
    what a user will ``import``, so it is not this module's to move.

    Everything else -- docs, tests, the manifest -- lands in a directory named
    for the component beside it. Without that, ``ragkit add storage-adapters``
    would fail on itself: it and ``vector-store-protocol`` both declare the
    destination directory ``rag/storage/`` and both ship a ``README.md`` and a
    ``manifest.json``, so the second copy would collide with the first within
    one run. A shared directory is fine for Python modules, which are named
    after themselves, and not for files that are not.

    ``library_root`` is only used to shorten ``source_label``; it is never
    written to, and containment of the *destination* is decided later by
    ``SafeDestination`` on the resolved real path (NFR-3).
    """
    operations: list[Operation] = []
    for added in resolve_order(component_id, registry):
        entry = registry.get(added)
        if entry is None:  # unreachable: resolve_order already raised
            raise UnknownComponentError(added, registry.ids) from None
        base = destination_base(
            resolve_destination(entry.manifest.destination, package)
        )
        namespace = base / added
        for label, declared, source in _declared_files(entry):
            target = base if label == "source" else namespace
            operations.append(
                Operation(
                    component_id=added,
                    source=source,
                    source_label=source.relative_to(library_root).as_posix(),
                    target=str(target / declared),
                )
            )
    return tuple(operations)

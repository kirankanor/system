"""Turn a loaded manifest into the concrete paths a caller copies (FR-11).

What resolution is
------------------
Dumb joining: the manifest declares ``source``, ``documentation`` and ``tests``
relative to the directory it was found in, and this module joins them. It does
not re-check existence -- ``validator.validate_entries`` already refused any
manifest whose declared files are missing, so anything that reaches here and
passed validation has them.

What resolution is not
----------------------
It is not the safety check. Rejecting a path that escapes its root belongs to
``src/rag_code_library/security/path_safety.py``, the single write choke point
(NFR-3). Nothing in this module writes anything.

``destination`` is deliberately *not* a field on :class:`ResolvedComponent`.
It is a template -- ``src/{project_package}/rag/...`` -- not a path: the
placeholder is substituted at ``ragkit add`` time against a project the
registry has never seen. Putting it beside real ``Path``s would imply it had
been resolved when it has not, so callers read ``manifest.destination``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .manifest import ComponentManifest

__all__ = ["ResolvedComponent", "UnknownComponentError", "resolve"]


class UnknownComponentError(ValueError):
    """Raised when a component id is not present in the loaded registry.

    A typed exception rather than an :class:`~.manifest.Issue`, because a bad
    id is a mistake by the caller of ``resolve``, not corrupt data in the
    registry: there is no file to point at and nothing to report in bulk.
    ``from None`` suppresses the KeyError that would otherwise show up in a
    traceback as a second, less useful error.
    """

    def __init__(self, component_id: str, known: Iterable[str]) -> None:
        self.component_id = component_id
        self.known = tuple(sorted(known))
        listing = ", ".join(self.known) if self.known else "<no components loaded>"
        super().__init__(f"unknown component {component_id!r}; known ids: {listing}")


@dataclass(frozen=True, slots=True)
class ResolvedComponent:
    """The files behind one component id, already checked to exist."""

    manifest: ComponentManifest
    directory: Path
    source: Path
    documentation: Path | None
    tests: Path | None


def resolve(directory: Path, manifest: ComponentManifest) -> ResolvedComponent:
    """Join ``manifest``'s declared relative paths onto ``directory``."""

    def relative(declared: str | None) -> Path | None:
        # ``""`` is legal for `tests`/`documentation` (the schema only says
        # "string"), and joining it would yield the directory itself -- which
        # would read as "the component's tests are this folder". Absent and
        # empty both mean "declares none".
        return directory / declared if declared else None

    return ResolvedComponent(
        manifest=manifest,
        directory=directory,
        source=directory / manifest.source,
        documentation=relative(manifest.documentation),
        tests=relative(manifest.tests),
    )

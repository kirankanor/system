"""Cross-manifest checks a single-file schema cannot express (FR-11, FR-24).

The schema answers "is this one file shaped correctly?". Everything here needs
the whole set in hand at once: an id is only duplicate relative to another
manifest, a path only missing relative to the directory it was declared in, a
dependency only unknown relative to what else is loaded.

Every check returns :class:`~.manifest.Issue` rather than raising, so one run
reports *all* the reasons rather than the first -- which is what makes the
output actionable instead of a whack-a-mole sequence of re-runs.

Destination safety goes beyond the schema
-----------------------------------------
``scripts/validate_components.py``'s own description commits it to checking
"destination safety", and the schema's ``destination`` *description* promises
absolute paths and backslashes are rejected -- but its *pattern* does not
deliver that for drive letters, interior backslashes, or backslash-style
traversal (``KNOWLEDGE-82bf789c`` has the measurements). So the two layers here
are:

1. :class:`~.manifest.ComponentManifest` enforces the schema pattern verbatim,
   which is what FR-24's "against the schema" means; and
2. this module adds the checks the schema prose claims but does not perform.

Rule 2 is stricter than the JSON Schema on purpose, and is recorded as such
rather than smuggled in. It cannot reject anything that would otherwise be
written: ``security/path_safety.py`` refuses those same targets at write time
(NFR-3), so this only moves the error from ``ragkit add`` to ``validate``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .manifest import Issue

if TYPE_CHECKING:
    # Imported for annotations only. Importing it for real would make
    # loader -> validator -> loader a cycle; with `from __future__ import
    # annotations` the annotation is never evaluated at runtime.
    from .loader import ManifestEntry

__all__ = ["validate_entries"]

# A path segment that could never be produced by the schema's own pattern but
# which Windows would still honour as a rooted location.
_DRIVE_QUALIFIED = re.compile(r"^[A-Za-z]:")


def validate_entries(entries: Sequence[ManifestEntry]) -> list[Issue]:
    """Run every cross-manifest check against a loaded set."""
    issues: list[Issue] = []
    issues += _duplicate_ids(entries)
    issues += _missing_files(entries)
    issues += _dependencies(entries)
    issues += _destinations(entries)
    return issues


def _duplicate_ids(entries: Sequence[ManifestEntry]) -> list[Issue]:
    """One id must name exactly one component (FR-11)."""
    by_id: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        by_id.setdefault(entry.manifest.id, []).append(entry)

    issues = []
    for component_id, group in sorted(by_id.items()):
        if len(group) < 2:
            continue
        declared_by = ", ".join(entry.path.as_posix() for entry in group)
        issues.append(
            Issue(
                component=component_id,
                path=group[0].path.as_posix(),
                reason=f"id declared by {len(group)} manifests: {declared_by}",
            )
        )
    return issues


def _missing_files(entries: Sequence[ManifestEntry]) -> list[Issue]:
    """source, documentation and tests must name files that exist.

    Existence only, deliberately. Unlike ``destination`` these three fields
    carry no containment rule -- the schema constrains them not at all, so
    forbidding ``..`` here would be inventing a rule the spec does not have,
    and would break a component that legitimately points at a file next door.
    ``destination`` does get containment because the schema's own description
    promises it and NFR-3 makes containment the project's rule; see the module
    docstring for where the boundary sits.
    """
    issues = []
    for entry in entries:
        manifest = entry.manifest
        declared = (
            ("source", manifest.source),
            ("documentation", manifest.documentation),
            ("tests", manifest.tests),
        )
        for label, value in declared:
            if not value:
                continue
            if not (entry.directory / value).is_file():
                issues.append(
                    Issue(
                        component=manifest.id,
                        path=entry.path.as_posix(),
                        reason=f"{label} file does not exist: {value}",
                    )
                )
    return issues


def _dependencies(entries: Sequence[ManifestEntry]) -> list[Issue]:
    """Every declared dependency must be a component that is actually loaded."""
    known = {entry.manifest.id for entry in entries}
    issues = []
    for entry in entries:
        manifest = entry.manifest
        for dependency in manifest.dependencies:
            if dependency == manifest.id:
                reason = f"{manifest.id} depends on itself"
            elif dependency not in known:
                reason = (
                    f"depends on unknown component {dependency!r}; "
                    f"loaded ids: {', '.join(sorted(known))}"
                )
            else:
                continue
            issues.append(
                Issue(
                    component=manifest.id,
                    path=entry.path.as_posix(),
                    reason=reason,
                )
            )
    return issues


def _destinations(entries: Sequence[ManifestEntry]) -> list[Issue]:
    """Reject targets the schema's pattern lets through (see module docstring)."""
    issues = []
    for entry in entries:
        destination = entry.manifest.destination
        if "\\" in destination:
            # The instance may not be traversing -- `src\rag\x.py` is not --
            # but the rule is about the class: the schema's `..` check tests
            # only for '/' separators, so `..\..\x` would slip past it while
            # Windows still resolves it as a traversal.
            reason = (
                "destination must use forward slashes only; the schema's '..' "
                f"check recognises '/' as the only separator: {destination!r}"
            )
        elif _DRIVE_QUALIFIED.match(destination):
            reason = (
                "destination must be relative to the destination root, "
                f"not drive-qualified: {destination!r}"
            )
        else:
            continue
        issues.append(
            Issue(
                component=entry.manifest.id,
                path=entry.path.as_posix(),
                reason=reason,
            )
        )
    return issues

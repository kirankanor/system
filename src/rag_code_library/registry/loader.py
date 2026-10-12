"""Discover and parse manifests, and expose the result as one Registry.

FR-11 asks for *a single* registry loader that discovers manifests, resolves
ids, and reports duplicates, missing files and schema violations as actionable
messages rather than tracebacks. This module is that loader:

* :func:`discover` finds the files,
* :func:`load_entries` parses them, turning every failure into an Issue,
* :class:`Registry` composes parsing with the cross-manifest checks and is what
  the CLI, the tests and the cookbook generator all read (FR-12).

Nothing in the parse path raises on bad data. A corrupt registry should be
*reportable* in one pass -- three broken manifests produce three issues, not one
exception -- because the alternative is fixing them one run at a time.

Empty means failure
-------------------
A target that does not exist, or that contains no ``*.json``, is reported as an
issue rather than loading as "zero components, zero problems". Silently
validating nothing is the vacuity failure this whole project keeps guarding
against: an exit code of 0 has to mean the components were actually checked.

Why ``*.json`` and not ``manifest.json``
----------------------------------------
One convention serves both trees this loader scans. The ``components/`` tree
contains only ``manifest.json`` files (verified), while the deliberately
corrupt fixtures under ``tests/fixtures/invalid_manifests/`` are named for the
corruption they demonstrate and live beside a ``README.md`` that must not be
parsed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .manifest import ComponentManifest, Issue, issue_from
from .resolver import ResolvedComponent, UnknownComponentError, resolve
from .validator import validate_entries

__all__ = ["ManifestEntry", "Registry", "discover", "load_entries"]


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """A parsed manifest plus the location it was declared relative to."""

    path: Path
    directory: Path
    manifest: ComponentManifest


def discover(target: Path) -> list[Path]:
    """Every ``*.json`` under ``target``, or ``target`` itself if it is a file.

    Sorted so validation output, test assertions and CI diffs are stable across
    runs and filesystems.
    """
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    return sorted(path for path in target.rglob("*.json") if path.is_file())


def load_entries(target: Path) -> tuple[list[ManifestEntry], list[Issue]]:
    """Parse every manifest under ``target``.

    Returns the entries that parsed and an issue for each that did not. Never
    raises for data problems; an :class:`OSError` while reading is reported too,
    since "permission denied" is exactly the kind of thing that should not
    escape as a traceback.
    """
    if not target.exists():
        return [], [Issue(path=target.as_posix(), reason="no such file or directory")]
    paths = discover(target)
    if not paths:
        return [], [
            Issue(path=target.as_posix(), reason="no manifest files found (expected *.json)")
        ]

    entries: list[ManifestEntry] = []
    issues: list[Issue] = []
    for path in paths:
        shown = path.as_posix()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            issues.append(Issue(path=shown, reason=f"unreadable manifest: {exc}"))
            continue
        except json.JSONDecodeError as exc:
            issues.append(Issue(path=shown, reason=f"not valid JSON: {exc}"))
            continue

        if not isinstance(raw, dict):
            issues.append(Issue(path=shown, reason="manifest must be a JSON object"))
            continue

        # Reported even when invalid: a malformed id is a common corruption, and
        # naming the component that failed is what makes the message actionable.
        declared = raw.get("id")
        component = declared if isinstance(declared, str) else None
        try:
            manifest = ComponentManifest.model_validate(raw)
        except ValidationError as exc:
            issues.append(issue_from(exc, path=shown, component=component))
            continue

        entries.append(ManifestEntry(path=path, directory=path.parent, manifest=manifest))
    return entries, issues


class Registry:
    """The loaded component set -- the single source the rest of the project reads.

    Build it with :meth:`load`, which does not raise. Check :attr:`ok` before
    trusting the result: when it is ``False``, :attr:`issues` says why, and ids
    may be duplicated (in which case :meth:`get` returns whichever entry loaded
    last -- the duplicate is reported, not silently resolved).
    """

    def __init__(
        self,
        target: Path,
        entries: list[ManifestEntry],
        issues: list[Issue],
    ) -> None:
        self.target = target
        self.entries = tuple(entries)
        self.issues = tuple(issues)
        self._by_id = {entry.manifest.id: entry for entry in entries}

    @classmethod
    def load(cls, target: Path | str) -> Registry:
        """Load ``target`` (a directory or a single manifest) and validate it."""
        path = Path(target)
        entries, issues = load_entries(path)
        issues += validate_entries(entries)
        return cls(path, entries, issues)

    @property
    def ok(self) -> bool:
        """True when every manifest parsed and every cross-check passed."""
        return not self.issues

    @property
    def manifests(self) -> tuple[ComponentManifest, ...]:
        """Parsed manifests, in discovery order."""
        return tuple(entry.manifest for entry in self.entries)

    @property
    def ids(self) -> tuple[str, ...]:
        """Declared component ids, sorted."""
        return tuple(sorted(self._by_id))

    @property
    def count(self) -> int:
        """How many manifests loaded, issues or not."""
        return len(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self.entries)

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._by_id

    def get(self, component_id: str) -> ManifestEntry | None:
        """The entry for ``component_id``, or ``None`` if there is none."""
        return self._by_id.get(component_id)

    def resolve(self, component_id: str) -> ResolvedComponent:
        """Concrete paths for ``component_id``.

        Raises :class:`~.resolver.UnknownComponentError` naming the ids that do
        exist, which is a caller mistake rather than a registry defect.
        """
        entry = self._by_id.get(component_id)
        if entry is None:
            raise UnknownComponentError(component_id, self._by_id) from None
        return resolve(entry.directory, entry.manifest)

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"{len(self.issues)} issue(s)"
        return f"<Registry {self.target.as_posix()}: {len(self.entries)} manifest(s), {state}>"

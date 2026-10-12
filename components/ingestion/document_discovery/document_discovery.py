"""Find the files worth parsing, and say precisely what happened to each one.

The whole point
---------------
Discovery is the stage where a pipeline learns what exists. It is also the
stage most often written as ``pathlib.Path.rglob("*")`` and then quietly
surprised by a symlink loop, an unreadable directory, or a 4 GB video dropped
next to the documents.

This component is deliberately *cheap*. It reports a path, a size and a
modification time, and it reports those because they are enough to answer the
question discovery actually owns: **what changed since the last run?** Content
hashing, format detection and parsing belong to later stages
(``document-identity-resolver``, ``file-type-validator``, the loaders).
Reading every candidate's bytes here would make an incremental scan as
expensive as a full one, which is the opposite of the point.

Four rules, and what each is for
--------------------------------
* **A bad entry never aborts the scan.** A file that cannot be stat'ed, a
  directory that cannot be opened, a link that escapes the root -- each becomes
  a record on ``DiscoveryReport.skipped`` with a reason. The other files are
  still found. An exception that stops the walk would turn one broken symlink
  into an empty corpus.
* **The scan never leaves the root it was given.** Symlinks and junctions are
  resolved and compared against the resolved source root *before* they are
  followed, so a link pointing at ``C:\\`` or at the user's home directory is
  reported and skipped rather than walked. Containment is decided on resolved
  paths, not on the text of the path.
* **Order is total and declared.** Records are sorted by a chosen key with
  ``relative_path`` as a permanent tie-break, so two runs over an unchanged
  tree emit byte-identical output. Determinism is not a nicety here: a
  non-deterministic scan makes "nothing changed" impossible to prove.
* **Identity is somebody else's job.** Two files with identical bytes get
  distinct records, and the same file reached twice is collapsed by *resolved
  path*, not by content. Hashing is what ``document-identity-resolver`` is for.

Snapshots, not state
--------------------
The component holds no memory between calls. A prior scan is passed back in as
a plain mapping of ``relative_path -> (size_bytes, modified_ns)``, which
serialises to JSON in one line. That keeps the component safe to run in a
fresh process each time and makes the comparison auditable: the caller can see
exactly what it handed over.

Known limitations
-----------------
Change detection compares ``(size_bytes, modified_ns)``. An edit that
preserves both -- a same-length rewrite with the timestamp restored, or a
filesystem with a coarse clock -- is reported as unchanged. That is a
deliberate trade, and the fix belongs downstream: give the changed candidates
to ``document-identity-resolver``, which hashes content. Discovery answers
"which files should we look at", never "are these the same bytes".
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "Discovery",
    "DiscoveryDelta",
    "DiscoveryError",
    "DiscoveryReport",
    "DiscoveredFile",
    "EntryState",
    "FileStamp",
    "InvalidSourceRoot",
    "InvalidScanOption",
    "SkipReason",
    "SkippedEntry",
    "SortKey",
    "classify",
]

#: Extensions this component will consider *by default*: the document formats
#: the library can actually open. ``None`` accepts everything; an empty
#: collection accepts nothing, which is reported rather than silently matching
#: every file in a tree.
DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json"}
)

#: A file's cheap identity for change detection: ``(size_bytes, modified_ns)``.
#: NOT a content identity -- see the module docstring.
FileStamp = tuple[int, int]

SortKey = Literal["relative_path", "modified_ns", "size_bytes"]

_SORT_FIELDS: dict[str, str] = {
    "relative_path": "relative_path",
    "modified_ns": "modified_ns",
    "size_bytes": "size_bytes",
}


class DiscoveryError(Exception):
    """Base for every error this component raises by contract, not by accident."""


class InvalidSourceRoot(DiscoveryError):
    """The configured source root does not exist or is not a directory."""


class InvalidScanOption(DiscoveryError):
    """A scan option was given a value that cannot produce a meaningful scan."""


class SkipReason(str, Enum):
    """Why an entry was not returned as a candidate file."""

    #: The entry's name matched an exclusion pattern.
    EXCLUDED = "excluded"
    #: The entry resolved outside the source root.
    SYMLINK_ESCAPE = "symlink-escape"
    #: Opening, stating or reading the entry raised an OS error.
    UNREADABLE = "unreadable"
    #: The entry exists but is not a regular file (socket, device, FIFO).
    NOT_A_FILE = "not-a-file"
    #: The entry is the same file as one already reported, reached another way.
    DUPLICATE = "duplicate"


class EntryState(str, Enum):
    """How a path relates a previous scan to a current one."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    #: Present in neither scan.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """One candidate file, with the cheap facts discovery is responsible for."""

    #: Absolute, resolved path. Safe to open -- discovery already checked it.
    path: Path
    #: Path relative to the source root, in POSIX form. This is the stable key
    #: used for snapshots and for reporting; it survives a tree being moved.
    relative_path: str
    size_bytes: int
    modified_ns: int

    def stamp(self) -> FileStamp:
        """Return the ``(size_bytes, modified_ns)`` pair used for comparison."""
        return (self.size_bytes, self.modified_ns)


@dataclass(frozen=True, slots=True)
class SkippedEntry:
    """An entry that was seen and deliberately not returned, with the reason."""

    relative_path: str
    reason: SkipReason
    #: Human-readable detail, e.g. the OS error text or the path a link
    #: escaped to. Contains no file content -- discovery never reads any.
    detail: str


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    """The complete outcome of one scan: found, skipped, and filtered counts."""

    source_root: Path
    files: tuple[DiscoveredFile, ...]
    skipped: tuple[SkippedEntry, ...]
    #: Entries dropped by the extension allow-list. Counted, not listed --
    #: a repository routinely contains thousands, and listing them would make
    #: the report unusable without telling anyone anything.
    filtered_by_extension: int
    #: Entries dropped by an exclusion pattern. Same reasoning.
    filtered_by_pattern: int
    recursive: bool

    def stamps(self) -> dict[str, FileStamp]:
        """Return the snapshot mapping to hand back as ``previous`` later.

        Keys are POSIX relative paths, so the mapping is JSON-serialisable as
       -is and can be persisted between processes.
        """
        return {record.relative_path: record.stamp() for record in self.files}

    def paths(self) -> tuple[str, ...]:
        """Return every discovered relative path, in scan order."""
        return tuple(record.relative_path for record in self.files)

    @property
    def total_entries_seen(self) -> int:
        """Everything the walk looked at, however it was then classified."""
        return (
            len(self.files)
            + len(self.skipped)
            + self.filtered_by_extension
            + self.filtered_by_pattern
        )


@dataclass(frozen=True, slots=True)
class DiscoveryDelta:
    """The classification of two scans: what was added, changed and removed."""

    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """True when nothing was added, modified or deleted."""
        return not (self.added or self.modified or self.deleted)

    def state(self, relative_path: str) -> EntryState:
        """Classify one path, including paths present in neither scan."""
        if relative_path in self.added:
            return EntryState.ADDED
        if relative_path in self.modified:
            return EntryState.MODIFIED
        if relative_path in self.deleted:
            return EntryState.DELETED
        if relative_path in self.unchanged:
            return EntryState.UNCHANGED
        return EntryState.UNKNOWN

    def changed(self) -> tuple[str, ...]:
        """Every path requiring work: added first, then modified, then deleted."""
        return (*self.added, *self.modified, *self.deleted)


def classify(
    previous: Mapping[str, FileStamp],
    current: Mapping[str, FileStamp],
) -> DiscoveryDelta:
    """Classify two snapshot mappings.

    Pure and module-level so it can be used on snapshots this process did not
    produce -- for example a JSON document read back from disk.

    A path whose stamp differs is ``MODIFIED``; one only in ``current`` is
    ``ADDED``; one only in ``previous`` is ``DELETED``. Both sides are sorted,
    so the delta is as deterministic as the scan that produced it.
    """
    added = tuple(sorted(set(current) - set(previous)))
    deleted = tuple(sorted(set(previous) - set(current)))
    modified = tuple(
        sorted(
            path
            for path in set(previous) & set(current)
            if previous[path] != current[path]
        )
    )
    unchanged = tuple(
        sorted(
            path
            for path in set(previous) & set(current)
            if previous[path] == current[path]
        )
    )
    return DiscoveryDelta(
        added=added,
        modified=modified,
        deleted=deleted,
        unchanged=unchanged,
    )


def _relative(root: Path, path: Path) -> str:
    """POSIX-form relative path, or ``"."`` for the root itself."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Not under the root -- should not happen for entries we walk, but a
        # resolved symlink can get here. Fall back to the absolute form rather
        # than raising and losing the rest of the scan.
        return path.as_posix()


def _is_junction(path: str) -> bool:
    """True for a Windows junction, which ``is_symlink`` does not report."""
    checker = getattr(os.path, "isjunction", None)
    return bool(checker(path)) if checker else False


class Discovery:
    """Scan a source root and report its candidate documents.

    Construction validates the options so that ``discover`` cannot fail for a
    reason the caller could have known about beforehand.

        Discovery(Path("docs"), allowed_extensions={".md"}).discover()
    """

    def __init__(
        self,
        source_root: Path | str,
        *,
        recursive: bool = True,
        allowed_extensions: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] = (),
        sort_by: SortKey = "relative_path",
    ) -> None:
        root = Path(source_root)
        if not root.is_dir():
            raise InvalidSourceRoot(
                f"source root is not a directory: {root}. "
                "Point Discovery at a directory that exists and is readable."
            )
        if not isinstance(recursive, bool):
            raise InvalidScanOption(
                f"recursive must be a bool, got {type(recursive).__name__}"
            )
        if sort_by not in _SORT_FIELDS:
            raise InvalidScanOption(
                f"sort_by must be one of {sorted(_SORT_FIELDS)}, got {sort_by!r}"
            )

        patterns = tuple(exclude_patterns)
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                raise InvalidScanOption(
                    "exclude_patterns must contain non-empty strings; "
                    f"got {pattern!r}"
                )

        if allowed_extensions is None:
            extensions = None
        else:
            extensions = frozenset(
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in allowed_extensions
            )

        #: Resolved once, here. Every containment check compares against this.
        self._root = root.resolve()
        self._recursive = recursive
        self._extensions = extensions
        self._patterns = patterns
        self._sort_by: str = _SORT_FIELDS[sort_by]

    @property
    def source_root(self) -> Path:
        """The resolved root every returned path is contained by."""
        return self._root

    @property
    def allowed_extensions(self) -> frozenset[str] | None:
        """The extension allow-list, or ``None`` when everything is accepted."""
        return self._extensions

    def _excluded(self, relative: str, name: str) -> bool:
        """True when an exclusion pattern matches the name or the relative path."""
        return any(
            fnmatch.fnmatchcase(name, pattern)
            or fnmatch.fnmatchcase(relative, pattern)
            for pattern in self._patterns
        )

    def _contains(self, candidate: Path) -> bool:
        """Containment on the *resolved* path, never on the path's text."""
        try:
            return candidate.resolve().is_relative_to(self._root)
        except OSError:
            return False

    def discover(self) -> DiscoveryReport:
        """Walk the root and return every candidate file, in declared order.

        Never raises for an individual entry: an unreadable file, an unreadable
        directory or an escaping link is recorded on ``skipped`` and the walk
        continues. It raises only if the root itself became unavailable.
        """
        files: list[DiscoveredFile] = []
        skipped: list[SkippedEntry] = []
        filtered_extension = 0
        filtered_pattern = 0
        #: Resolved paths already visited, so a link loop or a directory
        #: reachable two ways is walked exactly once.
        visited: set[str] = set()
        #: Resolved file path -> the relative path that first reported it. A
        #: file reachable two ways is reported once; the second name is kept as
        #: a DUPLICATE record rather than silently dropped.
        seen_files: dict[str, str] = {}
        pending: list[Path] = [self._root]

        while pending:
            current = pending.pop()
            key = str(current).casefold()
            if key in visited:
                continue
            visited.add(key)

            try:
                entries = sorted(os.scandir(current), key=lambda entry: entry.name)
            except OSError as exc:
                skipped.append(
                    SkippedEntry(
                        relative_path=_relative(self._root, current),
                        reason=SkipReason.UNREADABLE,
                        detail=f"could not open directory: {exc.strerror or exc}",
                    )
                )
                continue

            for entry in entries:
                path = Path(entry.path)
                relative = _relative(self._root, path)
                # Captured from the entry's own name, before any link is
                # followed: a filter the caller configured should be decided by
                # a name the caller can actually see.
                suffix = path.suffix.lower()

                if self._excluded(relative, entry.name):
                    filtered_pattern += 1
                    continue

                try:
                    is_link = entry.is_symlink() or _is_junction(entry.path)
                    if is_link:
                        if not self._contains(path):
                            skipped.append(
                                SkippedEntry(
                                    relative_path=relative,
                                    reason=SkipReason.SYMLINK_ESCAPE,
                                    detail=(
                                        f"resolves to {path.resolve()}, outside "
                                        f"the source root {self._root}"
                                    ),
                                )
                            )
                            continue
                        # Containment is already proven on the *resolved* path,
                        # so following the link here cannot leave the root.
                        opened = path.resolve()
                        if opened.is_dir():
                            if self._recursive:
                                pending.append(opened)
                            continue
                        if not opened.is_file():
                            skipped.append(
                                SkippedEntry(
                                    relative_path=relative,
                                    reason=SkipReason.NOT_A_FILE,
                                    detail="entry is not a regular file",
                                )
                            )
                            continue
                        stat = opened.stat()
                        # Report the link's own name, open the file it points
                        # at. Both facts matter and they are not the same fact.
                        path = opened
                    else:
                        if entry.is_dir(follow_symlinks=False):
                            if self._recursive:
                                pending.append(path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            skipped.append(
                                SkippedEntry(
                                    relative_path=relative,
                                    reason=SkipReason.NOT_A_FILE,
                                    detail="entry is not a regular file",
                                )
                            )
                            continue
                        stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    skipped.append(
                        SkippedEntry(
                            relative_path=relative,
                            reason=SkipReason.UNREADABLE,
                            detail=f"could not stat entry: {exc.strerror or exc}",
                        )
                    )
                    continue

                if self._extensions is not None and suffix not in self._extensions:
                    filtered_extension += 1
                    continue

                resolved = path.resolve()
                key = str(resolved).casefold()
                winner = seen_files.get(key)
                if winner is not None:
                    skipped.append(
                        SkippedEntry(
                            relative_path=relative,
                            reason=SkipReason.DUPLICATE,
                            detail=f"the same file is already reported as {winner}",
                        )
                    )
                    continue
                seen_files[key] = relative

                files.append(
                    DiscoveredFile(
                        path=resolved,
                        relative_path=relative,
                        size_bytes=stat.st_size,
                        modified_ns=stat.st_mtime_ns,
                    )
                )

        files.sort(key=self._sort_key)
        return DiscoveryReport(
            source_root=self._root,
            files=tuple(files),
            skipped=tuple(skipped),
            filtered_by_extension=filtered_extension,
            filtered_by_pattern=filtered_pattern,
            recursive=self._recursive,
        )

    def _sort_key(self, record: DiscoveredFile) -> tuple[object, str]:
        """Primary declared key with ``relative_path`` as a permanent tie-break.

        The tie-break is what makes the order *total*: two files with identical
        sizes and timestamps still have a fixed, reproducible order.
        """
        if self._sort_by == "relative_path":
            return (record.relative_path,)
        return (getattr(record, self._sort_by), record.relative_path)

    def delta(self, previous: Mapping[str, FileStamp]) -> DiscoveryDelta:
        """Scan now and classify against a snapshot this component returned.

        ``previous`` is normally ``DiscoveryReport.stamps()`` from an earlier
        run, but any ``{relative_path: (size, mtime_ns)}`` mapping works, so a
        snapshot read back from JSON can be passed straight in.
        """
        return classify(previous, self.discover().stamps())

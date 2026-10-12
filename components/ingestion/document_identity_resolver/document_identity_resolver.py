"""Give a document an identity its filename cannot provide, and say what changed.

Two identities, deliberately
----------------------------
The component separates two questions that are usually collapsed into one:

* **Content identity** -- which bytes are these? A SHA-256 of the whole file.
  Identical bytes give an identical identity wherever they are stored, which
  is what makes re-ingestion idempotent and cross-file duplicates findable.
* **File identity** -- which file is this? Its resolved path, size and mtime.
  This is what changes when a document is *edited*, and it is the thing a
  filename alone has never been able to answer, because two different files can
  share a name and one file can be renamed without changing at all.

Why the id is a truncation, and why this file repeats it
-------------------------------------------------------
``document_id`` is the first 32 hex characters of the content SHA-256. That is
the same rule as ``make_document_id`` in the ``settings`` component
(``config/models.py``) and the same rule ``core-types`` redefines for the
storage contracts. It is written out here for the third time because AC-13
forbids a component importing another component, and a copy whose equivalence
is documented and tested is better than a dependency that would make this
component uninstallable on its own -- which is the whole point of the library.
``test_the_document_id_matches_the_library_rule`` pins the exact value.

What a modified document becomes
--------------------------------
Because the id is derived from content, editing a file does **not** update its
identity -- it produces a new one, and the old identity is now a different
document that happens to be superseded. This component does not hide that. It
tracks each *path* separately, so ``Resolution`` reports the version number,
the digest the path held before, and therefore which identity is being retired.
A caller that wants "same document, new version" has to say so; nothing here
infers it from a filename.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DOCUMENT_ID_LENGTH",
    "ContentIdentity",
    "DocumentIdentityResolver",
    "FileIdentity",
    "IdentityError",
    "InvalidDigest",
    "MissingFile",
    "Resolution",
    "UnreadableFile",
    "make_document_id",
]

#: Matches ``make_document_id`` in config/models.py and storage/core_types.py.
DOCUMENT_ID_LENGTH = 32

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

#: Hashing streams through this buffer. A multi-gigabyte file must not become a
#: multi-gigabyte object just to be identified.
_CHUNK_SIZE = 1024 * 1024


class IdentityError(Exception):
    """Base for every failure this component reports."""


class MissingFile(IdentityError):
    """No such file."""


class UnreadableFile(IdentityError):
    """Exists, but could not be read to the end."""


class InvalidDigest(IdentityError):
    """A supplied digest is not a SHA-256 in lowercase hex."""


def make_document_id(content_sha256: str) -> str:
    """``sha256[:32]`` -- identical to the rule the rest of the library uses.

    Repeated rather than imported: a component may not depend on another
    component (AC-13), and this truncation is part of a shared data contract,
    so getting it wrong silently would desynchronise every stored vector.
    """
    if not _SHA256_RE.match(content_sha256):
        raise InvalidDigest(
            "content_sha256 must be 64 lowercase hexadecimal characters; "
            f"got {content_sha256!r}"
        )
    return content_sha256[:DOCUMENT_ID_LENGTH]


@dataclass(frozen=True)
class ContentIdentity:
    """Which bytes these are. Independent of where they are stored."""

    sha256: str
    size_bytes: int

    @property
    def document_id(self) -> str:
        return make_document_id(self.sha256)


@dataclass(frozen=True)
class FileIdentity:
    """Which file this is: a path, and the content currently at it."""

    path: str
    size_bytes: int
    mtime_ns: int
    content: ContentIdentity

    @property
    def document_id(self) -> str:
        return self.content.document_id


@dataclass(frozen=True)
class Resolution:
    """A file identity plus what the resolver knows about its history."""

    identity: FileIdentity
    #: 1 the first time this path is seen; incremented when its content changes.
    version: int
    #: The digest this path held on the previous resolution, if it changed.
    previous_sha256: str | None
    #: Other paths already known to hold byte-identical content. This is what a
    #: filename can never tell you, since the names differ by definition.
    identical_content_at: tuple[str, ...] = ()


class DocumentIdentityResolver:
    """Hash files, and remember enough to notice a duplicate or a change.

    The index is in memory and per-instance on purpose. A resolver that wrote
    to disk would be a second source of truth about identity, competing with
    the manifests; this one is disposable state that can always be rebuilt by
    re-resolving.
    """

    def __init__(self) -> None:
        self._by_content: dict[str, set[str]] = defaultdict(set)
        self._by_path: dict[str, str] = {}
        #: path -> digests held, in the order first seen. Drives ``version``.
        self._history: dict[str, list[str]] = {}

    # -- hashing -----------------------------------------------------------
    @staticmethod
    def hash_file(path: str | Path) -> ContentIdentity:
        """Content identity of a file, read in bounded chunks."""
        target = Path(path)
        try:
            size = target.stat().st_size
        except FileNotFoundError as exc:
            raise MissingFile(f"no such file: {target}") from exc
        except OSError as exc:
            raise UnreadableFile(f"could not stat {target}: {exc}") from exc

        digest = hashlib.sha256()
        try:
            with target.open("rb") as handle:
                while chunk := handle.read(_CHUNK_SIZE):
                    digest.update(chunk)
        except OSError as exc:
            raise UnreadableFile(f"could not read {target}: {exc}") from exc

        return ContentIdentity(sha256=digest.hexdigest(), size_bytes=size)

    def same_content(self, first: str | Path, second: str | Path) -> bool:
        """True when two paths hold byte-identical content."""
        return self.hash_file(first).sha256 == self.hash_file(second).sha256

    # -- resolving ---------------------------------------------------------
    def resolve(self, path: str | Path) -> Resolution:
        """Identify a file, and report duplicates and prior versions.

        Records the path in the index as a side effect, which is what makes the
        next call able to see a duplicate.
        """
        target = Path(path).resolve()
        key = str(target)
        content = self.hash_file(target)

        try:
            stat = target.stat()
        except OSError as exc:  # pragma: no cover - raced with the read
            raise UnreadableFile(f"could not stat {target}: {exc}") from exc

        identity = FileIdentity(
            path=key,
            size_bytes=content.size_bytes,
            mtime_ns=stat.st_mtime_ns,
            content=content,
        )

        previous = self._by_path.get(key)
        version = self._versions_for(key, content.sha256)

        known = self._by_content[content.sha256]
        identical = tuple(sorted(p for p in known if p != key))

        self._by_content[content.sha256].add(key)
        self._by_path[key] = content.sha256

        return Resolution(
            identity=identity,
            version=version,
            previous_sha256=previous,
            identical_content_at=identical,
        )

    def _versions_for(self, key: str, current: str) -> int:
        """How many distinct digests this path has held, counting this one.

        Re-resolving unchanged content does not bump the number, so ``version``
        means "distinct states this file has been in", not "times we looked".
        """
        seen = self._history.setdefault(key, [])
        if current not in seen:
            seen.append(current)
        return len(seen)

    # -- reporting ---------------------------------------------------------
    def paths_for_document(self, document_id: str) -> tuple[str, ...]:
        """Every path holding content with this document id."""
        matches = [
            paths
            for digest, paths in self._by_content.items()
            if make_document_id(digest) == document_id
        ]
        return tuple(sorted(path for group in matches for path in group))

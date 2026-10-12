"""Two answers to one question: *is this work already done?*

The whole point
---------------
An ingestion pipeline re-does work in two distinct ways, and both are
expensive in the same currency.

It re-embeds **content it has already embedded** -- because two chunks carry
identical text, because a document was re-ingested unchanged, because the same
paragraph appears in a template shared by a thousand pages. That is what the
deduplicator half answers.

It re-embeds **content whose stored vector came from a different configuration**
-- because someone swapped the embedding model, or changed the chunk size, and
the vectors already in the index no longer correspond to anything the current
code would produce. That is what the fingerprint half answers.

Both are silent when they go wrong. A pipeline that skips a chunk it should
have re-embedded does not error; it returns degraded answers, and nobody
notices for weeks. So both halves here default to **not reusing** when the
evidence is missing or unclear.

Exact, and only exact
---------------------
Duplicate detection hashes *normalized* text, where the normalization is
declared and deliberately narrow: Unicode NFC, then every run of whitespace
collapsed to a single space, then strip. Case is preserved on purpose.
Folding case would merge content that genuinely differs, which is similarity
matching by another name -- and near-duplicate matching is explicitly out of
scope. This component will happily tell you it found no duplicates in a corpus
full of near-misses, because claiming otherwise would be a lie about what it
computes.

The normalizer is injectable, so a caller who has already run
``text-normalizer`` can pass that same function here rather than normalizing
twice under two different rules.

Provenance is never thrown away
-------------------------------
Removing a duplicate does not mean pretending it was not there. Each removal
records the chunk it collapsed *into* and the document every appearance came
from, so an index that drops a duplicate can still say "this content also
appears in X". Without that, corpus-wide dedup would quietly erase the fact
that a shared template appeared in a thousand documents.

Retained chunks are returned as the **original objects**, not wrappers: what
goes in comes out unchanged, so the rest of the pipeline sees the same type it
always did.

Why a fingerprint, and what it is made of
-----------------------------------------
``IndexFingerprint`` is a parseable string -- ``model=...|dims=...|chunking=...``
-- not an opaque hash. An opaque hash would tell you *that* something differed;
this tells you **which field** differed, which is the difference between a
log line a human can act on and one they cannot. AC-24 asks for an error that
names the mismatch, so the value has to be nameable.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Appearance",
    "DedupError",
    "DedupOutcome",
    "Deduplicator",
    "DuplicateGroup",
    "EmbeddingReuseError",
    "EmbeddingReuseGate",
    "FingerprintMismatch",
    "IndexFingerprint",
    "MissingChunkText",
    "Scope",
    "default_normalize",
    "fingerprint_of",
]


class DedupError(Exception):
    """Base for every error this component raises by contract."""


class MissingChunkText(DedupError):
    """A chunk has no readable ``text``, so it has no content to compare."""


class EmbeddingReuseError(Exception):
    """Base for every error the fingerprint half raises by contract."""


class FingerprintMismatch(EmbeddingReuseError):
    """The stored vector came from a configuration that is no longer in force."""


class Scope(str, Enum):
    """What counts as a duplicate of what.

    ``DOCUMENT`` keeps cross-document duplicates (identical text in two
    documents is retained in both, because removing it would make one document
    silently incomplete). ``CORPUS`` collapses everything identical, wherever
    it came from. ``BOTH`` runs the document pass first and then the corpus
    pass, so a caller can see which level collapsed each duplicate.
    """

    DOCUMENT = "document"
    CORPUS = "corpus"
    BOTH = "both"


_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def default_normalize(text: str) -> str:
    """The declared normalization this component hashes.

    NFC, then whitespace runs collapsed to one space, then stripped. **Case is
    preserved** -- folding it would merge content that differs, which is
    similarity matching and out of scope (FR-43).
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _hash(text: str, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Appearance:
    """One place a piece of content was found. The unit of provenance."""

    document_id: str | None
    source_name: str | None
    chunk_id: str | None
    chunk_index: int | None

    @property
    def where(self) -> str:
        """A short human label for reports and run summaries."""
        document = self.document_id or self.source_name or "unknown-document"
        chunk = self.chunk_id or (f"#{self.chunk_index}" if self.chunk_index is not None else "?")
        return f"{document}/{chunk}"


def _appearance(chunk: object) -> Appearance:
    return Appearance(
        document_id=_text_attr(chunk, "document_id"),
        source_name=_text_attr(chunk, "source_name"),
        chunk_id=_text_attr(chunk, "chunk_id"),
        chunk_index=_int_attr(chunk, "chunk_index"),
    )


def _text_attr(chunk: object, name: str) -> str | None:
    value = getattr(chunk, name, None)
    return value if isinstance(value, str) and value.strip() else None


def _int_attr(chunk: object, name: str) -> int | None:
    value = getattr(chunk, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Content found more than once, and every place it was found.

    ``retained`` is what survived; ``collapsed`` is what was removed. Every
    entry carries its own document, so shared content keeps its attribution
    rather than being erased from the record.
    """

    content_hash: str
    #: ``"document"`` when every appearance shared a document, else ``"corpus"``.
    #: Computed from the data rather than assumed from the configured scope.
    level: str
    retained: Appearance
    collapsed: tuple[Appearance, ...]

    @property
    def documents(self) -> tuple[str, ...]:
        """Every document involved, sorted and de-duplicated."""
        names = [
            appearance.document_id or appearance.source_name
            for appearance in (self.retained, *self.collapsed)
        ]
        return tuple(sorted({name for name in names if name}))

    @property
    def removals(self) -> int:
        return len(self.collapsed)


@dataclass(frozen=True, slots=True)
class DedupOutcome:
    """The result of one deduplication pass."""

    #: The original chunk objects, unchanged, minus the duplicates.
    retained: tuple[object, ...]
    #: Only groups that actually collapsed something; clean content is absent.
    groups: tuple[DuplicateGroup, ...]
    scope: Scope

    @property
    def removed(self) -> int:
        return sum(group.removals for group in self.groups)

    @property
    def duplicate_groups(self) -> int:
        return len(self.groups)

    @property
    def is_clean(self) -> bool:
        """True when nothing was a duplicate."""
        return not self.groups

    def removals_by_document(self) -> dict[str, int]:
        """How many chunks each document lost, for a run summary."""
        counts: dict[str, int] = {}
        for group in self.groups:
            for appearance in group.collapsed:
                key = appearance.document_id or appearance.source_name or "unknown-document"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def collapsed_into(self, appearance: Appearance) -> str | None:
        """Which retained chunk absorbed this appearance, if any."""
        for group in self.groups:
            if appearance in group.collapsed:
                return group.retained.where
        return None


class Deduplicator:
    """Remove exact duplicates under a declared scope.

        Deduplicator(scope=Scope.CORPUS).dedupe(chunks)
    """

    def __init__(
        self,
        *,
        scope: Scope = Scope.CORPUS,
        normalize: Callable[[str], str] = default_normalize,
        algorithm: str = "sha256",
    ) -> None:
        if not isinstance(scope, Scope):
            raise DedupError(f"scope must be a Scope, got {scope!r}")
        if not callable(normalize):
            raise DedupError(f"normalize must be callable, got {type(normalize).__name__}")
        try:
            hashlib.new(algorithm, b"probe")
        except (ValueError, TypeError) as exc:
            raise DedupError(f"unknown hash algorithm {algorithm!r}: {exc}") from exc
        self._scope = scope
        self._normalize = normalize
        self._algorithm = algorithm

    @property
    def scope(self) -> Scope:
        return self._scope

    def content_hash(self, text: str) -> str:
        """Hash one piece of text under this deduplicator's rules."""
        if not isinstance(text, str):
            raise MissingChunkText(f"expected str to hash, got {type(text).__name__}")
        return _hash(self._normalize(text), self._algorithm)

    def dedupe(self, chunks: Sequence[object]) -> DedupOutcome:
        """Remove exact duplicates, returning the survivors and the record.

        Retained chunks are the **original objects**, not copies or wrappers.
        Order is preserved, and the first appearance of identical content wins,
        so the output is as deterministic as the input.
        """
        indexed: list[tuple[int, object, str]] = []
        for position, chunk in enumerate(chunks):
            if not hasattr(chunk, "text"):
                raise MissingChunkText(
                    f"chunk at index {position} has no 'text' attribute; got "
                    f"{type(chunk).__name__}. Deduplication compares content, "
                    "so there is nothing to compare without it."
                )
            text = getattr(chunk, "text", None)
            if not isinstance(text, str):
                raise MissingChunkText(
                    f"chunk at index {position} has text of type "
                    f"{type(text).__name__}, not str"
                )
            indexed.append((position, chunk, self.content_hash(text)))

        survivors = indexed
        groups: list[DuplicateGroup] = []

        if self._scope in (Scope.DOCUMENT, Scope.BOTH):
            survivors, first_pass = self._collapse(survivors, key_includes_document=True)
            groups.extend(first_pass)

        if self._scope in (Scope.CORPUS, Scope.BOTH):
            survivors, second_pass = self._collapse(survivors, key_includes_document=False)
            groups.extend(second_pass)

        # `groups` is already in input order: dict buckets preserve insertion,
        # and a bucket is created by the first appearance of its content.
        return DedupOutcome(
            retained=tuple(chunk for _, chunk, _ in survivors),
            groups=tuple(groups),
            scope=self._scope,
        )

    def _collapse(
        self,
        entries: list[tuple[int, object, str]],
        *,
        key_includes_document: bool,
    ) -> tuple[list[tuple[int, object, str]], list[DuplicateGroup]]:
        """Group identical content and keep the first of each group."""
        buckets: dict[object, list[tuple[int, object, str]]] = {}
        for entry in entries:
            _, chunk, digest = entry
            key: object = digest
            if key_includes_document:
                key = (_text_attr(chunk, "document_id") or _text_attr(chunk, "source_name"), digest)
            buckets.setdefault(key, []).append(entry)

        survivors: list[tuple[int, object, str]] = []
        groups: list[DuplicateGroup] = []
        for bucket in buckets.values():
            first = bucket[0]
            survivors.append(first)
            if len(bucket) == 1:
                continue

            kept_chunk = first[1]
            kept_appearance = _appearance(kept_chunk)
            collapsed = tuple(_appearance(chunk) for _, chunk, _ in bucket[1:])

            # Level describes the DATA, not the configured scope: was this
            # shared content confined to one document, or did it cross? An
            # unknown document counts as no attribution at all, so such a
            # group reports as corpus-level rather than claiming a
            # document-level sharing that was never established.
            shared = {
                kept_appearance.document_id or kept_appearance.source_name,
                *(
                    appearance.document_id or appearance.source_name
                    for appearance in collapsed
                ),
            }
            shared.discard(None)
            level = "document" if len(shared) == 1 else "corpus"

            groups.append(
                DuplicateGroup(
                    content_hash=first[2],
                    level=level,
                    retained=kept_appearance,
                    collapsed=collapsed,
                )
            )
        return survivors, groups


# ---------------------------------------------------------------------------
# Fingerprint-gated embedding reuse
# ---------------------------------------------------------------------------


def _canonical(value: object) -> str:
    """A stable serialisation of arbitrary chunking configuration."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - exotic input
        return repr(value)


#: (attribute, label) pairs for :class:`IndexFingerprint`. The label is what a
#: human sees in a mismatch message; the attribute is what Python reads. They
#: are listed once here so the two can never drift apart the way a hard-coded
#: tuple of names in each method can.
_FINGERPRINT_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "model"),
    ("dimensions", "dimensions"),
    ("chunking_hash", "chunking"),
)


@dataclass(frozen=True, slots=True)
class IndexFingerprint:
    """What an existing vector was actually produced from.

    Parseable rather than opaque on purpose: a bare hash says *that*
    something differed, while these three fields say **which** one did.
    AC-24 requires an error naming the mismatch, and a value that cannot be
    read back cannot be named.
    """

    model: str
    dimensions: int
    chunking_hash: str

    @property
    def value(self) -> str:
        """The canonical, storable string form."""
        return f"model={self.model}|dims={self.dimensions}|chunking={self.chunking_hash}"

    @classmethod
    def parse(cls, raw: str) -> IndexFingerprint:
        """Read a stored fingerprint back, or explain why it cannot be read."""
        if not isinstance(raw, str) or not raw.strip():
            raise FingerprintMismatch(
                "no stored index fingerprint, so there is nothing to reuse; "
                "the chunk must be embedded"
            )
        parts = raw.split("|")
        found: dict[str, str] = {}
        for part in parts:
            key, separator, value = part.partition("=")
            if not separator or key not in {"model", "dims", "chunking"}:
                raise FingerprintMismatch(
                    f"stored fingerprint {raw!r} is not in the expected "
                    "'model=...|dims=...|chunking=...' form, so it cannot be "
                    "compared; the chunk must be re-embedded rather than "
                    "guessed at"
                )
            found[key] = value
        missing = {"model", "dims", "chunking"} - set(found)
        if missing:
            raise FingerprintMismatch(
                f"stored fingerprint {raw!r} is missing {', '.join(sorted(missing))}; "
                "the chunk must be re-embedded"
            )
        try:
            dimensions = int(found["dims"])
        except ValueError as exc:
            raise FingerprintMismatch(
                f"stored fingerprint {raw!r} has a non-integer dimension "
                f"{found['dims']!r}"
            ) from exc
        return cls(model=found["model"], dimensions=dimensions, chunking_hash=found["chunking"])

    def differences(self, other: IndexFingerprint) -> tuple[str, ...]:
        """Names of the fields that disagree, sorted."""
        return tuple(
            sorted(
                label
                for attribute, label in _FINGERPRINT_FIELDS
                if getattr(self, attribute) != getattr(other, attribute)
            )
        )


def fingerprint_of(
    *,
    model: str,
    dimensions: int,
    chunking_config: object,
) -> IndexFingerprint:
    """Build a fingerprint from the three things that invalidate a vector.

    ``chunking_config`` may be any JSON-able value; it is canonically
    serialised, so the same configuration always hashes the same way
    regardless of key order.
    """
    if not isinstance(model, str) or not model.strip():
        raise EmbeddingReuseError(f"model must be a non-empty str, got {model!r}")
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise EmbeddingReuseError(f"dimensions must be a positive int, got {dimensions!r}")
    return IndexFingerprint(
        model=model,
        dimensions=dimensions,
        chunking_hash=_hash(_canonical(chunking_config))[:16],
    )


class EmbeddingReuseGate:
    """Decide whether a stored vector may be reused, and say why not.

        gate = EmbeddingReuseGate.configured(
            model="text-embedding-3-small", dimensions=1536,
            chunking_config={"size": 512, "overlap": 64},
        )
        gate.check(stored)          # raises FingerprintMismatch naming the field
    """

    def __init__(self, fingerprint: IndexFingerprint) -> None:
        if not isinstance(fingerprint, IndexFingerprint):
            raise EmbeddingReuseError(
                f"expected an IndexFingerprint, got {type(fingerprint).__name__}"
            )
        self._fingerprint = fingerprint

    @classmethod
    def configured(
        cls,
        *,
        model: str,
        dimensions: int,
        chunking_config: object,
    ) -> EmbeddingReuseGate:
        return cls(fingerprint_of(model=model, dimensions=dimensions, chunking_config=chunking_config))

    @property
    def fingerprint(self) -> IndexFingerprint:
        return self._fingerprint

    @property
    def value(self) -> str:
        """The value to store alongside every embedded chunk."""
        return self._fingerprint.value

    def may_reuse(self, stored: str | None) -> bool:
        """Non-throwing predicate. Nothing reused unless it is provably safe."""
        try:
            self.check(stored)
        except FingerprintMismatch:
            return False
        return True

    def check(self, stored: str | None) -> None:
        """Raise :class:`FingerprintMismatch` unless ``stored`` matches.

        The message names **which field** differs and shows both sides, so
        someone reading a log knows whether to re-embed or to investigate.
        """
        if stored is None:
            raise FingerprintMismatch(
                f"no stored index fingerprint for this chunk, so nothing can be "
                f"reused; the current configuration is {self.value}"
            )
        current = self._fingerprint
        recorded = IndexFingerprint.parse(stored)
        differing = current.differences(recorded)
        if not differing:
            return
        detail = ", ".join(
            f"{label}: stored {getattr(recorded, attribute)!r} != "
            f"current {getattr(current, attribute)!r}"
            for attribute, label in _FINGERPRINT_FIELDS
            if label in differing
        )
        raise FingerprintMismatch(
            f"stored embedding was produced by a configuration that is no "
            f"longer in force -- {detail}. Stored was {recorded.value}; "
            f"current is {current.value}. The stored vector must be discarded "
            f"and the chunk re-embedded, not reused."
        )

"""The write path: bounded batches, pre-write refusal, and records that prove
where their bytes came from.

Three failure modes this component exists to make loud
------------------------------------------------------
**Silent data loss on create.** Rebuilding an index to "fix" it is the most
natural impulse there is, and if it works by wiping the collection first, the
rows are gone before anyone notices the rebuild was unnecessary. So creation
and replacement are separated: :meth:`IndexLifecycle.create` refuses to sit on
top of rows that exist, and :meth:`IndexLifecycle.replace` is the only way
through -- it demands a reason and writes the loss down (FR-47).

**A payload the index cannot hold, discovered after the write.** A vector of
the wrong width, or a record missing a field the index is meant to be
queryable on, is only caught by the backend once rows are already landing.
:func:`validate_payload` checks every item *before* the first write and raises
:class:`PayloadMismatchError` naming the field or the width (FR-47).

**A big batch that half-fails.** One ``add()`` of ten thousand rows either
succeeds or throws, and a throw tells you nothing about which of the ten
thousand landed. So writes are split into bounded batches, and a batch that
fails is retried item by item -- which is how the report can say, per item,
what actually happened (FR-45, FR-46). Re-issuing an item is safe because
every write here is an upsert: writing an id twice replaces rather than
duplicates.

Why per-item retry is safe
--------------------------
The retry writes an id that may already exist. That is harmless precisely
because the write is an upsert -- the same content lands in the same place.
Without upsert semantics this recovery step would risk doubling every row that
the backend had already accepted before failing, which is why the two
requirements are specified together.

Records that survive the round trip (FR-48)
-------------------------------------------
A stored chunk is only useful if it can be traced back to the bytes it came
from, so :class:`StoredChunkRecord` carries the document id, chunk id, source
name, source offsets, and page or section reference alongside the vector and
the index fingerprint.

``core-types:Chunk`` has no dedicated offset or section fields, so those two
are read from the chunk's metadata under the conventional keys ``start_offset``,
``end_offset`` and ``section``. They are *always present on the record* --
``None`` when the chunker did not supply them -- because a field that is
sometimes absent cannot be asserted on, and AC-26 asks that every record
carry these. ``chunk_index`` and the page range are the citation location that
is always available regardless.

Self-contained on purpose
-------------------------
Depends on ``core-types`` and ``vector-store-protocol``, both declared in the
manifest so ``ragkit add`` copies them along.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any

from core_types import EmbeddedChunk, Metadata
from vector_store_protocol import validate_embedded_chunks

__all__ = [
    "PROVENANCE_FIELDS",
    "BatchItemResult",
    "BatchReport",
    "BatchWriteError",
    "DestructiveOperationRefused",
    "IndexDestruction",
    "IndexLifecycle",
    "IndexWriter",
    "IndexWriterError",
    "PAYLOAD_FIELD_KEYS",
    "PayloadMismatchError",
    "StoredChunkRecord",
    "bounded_batches",
    "record_from",
    "validate_payload",
]

#: Metadata keys this component reads back out as first-class record fields.
#: Declared once so the writer and its tests cannot drift on the spelling.
PAYLOAD_FIELD_KEYS: dict[str, str] = {
    "start_offset": "start_offset",
    "end_offset": "end_offset",
    "section": "section",
}

#: The provenance fields AC-26 asserts on every stored record.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "document_id",
    "chunk_id",
    "source_name",
    "chunk_index",
    "start_offset",
    "end_offset",
    "page_start",
    "page_end",
    "section",
)


class IndexWriterError(Exception):
    """Base for every error raised by contract here."""


class PayloadMismatchError(IndexWriterError):
    """The payload does not fit the target index -- detected *before* writing.

    Carries the field or the dimension that disagreed, plus the offending
    chunk's identity, so a failure names what to fix rather than only that
    something broke.
    """

    def __init__(self, message: str, *, chunk_id: str | None = None,
                 document_id: str | None = None) -> None:
        super().__init__(message)
        self.chunk_id = chunk_id
        self.document_id = document_id


class DestructiveOperationRefused(IndexWriterError):
    """An operation would have discarded stored rows, and was not authorised."""


class BatchWriteError(IndexWriterError):
    """Raised by :meth:`BatchReport.raise_if_failed` on demand."""

    def __init__(self, message: str, report: BatchReport) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- #
# FR-48 -- the stored record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StoredChunkRecord:
    """One chunk as it exists inside the index.

    Every provenance field is present on every record. Where the chunker
    supplied no value the field is ``None`` rather than absent -- a field that
    only exists sometimes cannot be asserted on, and AC-26 asks that every
    record carry these.
    """

    document_id: str | None
    chunk_id: str | None
    source_name: str | None
    chunk_index: int
    text: str
    start_offset: int | None
    end_offset: int | None
    page_start: int | None
    page_end: int | None
    section: str | None
    vector: tuple[float, ...]
    index_fingerprint: str
    metadata: Metadata = field(default_factory=dict)

    @property
    def provenance(self) -> dict[str, Any]:
        """The citation fields alone -- what AC-26 inspects."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_name": self.source_name,
            "chunk_index": self.chunk_index,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
        }

    @property
    def is_traceable(self) -> bool:
        """Whether enough survives to find this text in its source again.

        Needs the document, and either a page range or a section -- without at
        least one of those the record says *which* document but not *where in
        it*, which is not enough to cite.
        """
        if self.document_id is None:
            return False
        has_pages = self.page_start is not None and self.page_end is not None
        has_offsets = self.start_offset is not None and self.end_offset is not None
        return bool(has_pages or has_offsets or self.section)

    def field(self, name: str) -> Any:
        """Read a declared field by name, or say the name does not exist."""
        if not hasattr(self, name):
            raise PayloadMismatchError(
                f"index-writer: {name!r} is not a field of StoredChunkRecord; "
                f"the schema names fields such as "
                f"{', '.join(PROVENANCE_FIELDS[:4])}..."
            )
        return getattr(self, name)


def _metadata_int(metadata: Metadata, key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _metadata_str(metadata: Metadata, key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def record_from(item: EmbeddedChunk) -> StoredChunkRecord:
    """Build the stored record for one embedded chunk.

    Reads offsets and section from metadata under the conventional keys, so a
    chunker that records them keeps them, and one that does not produces
    ``None`` rather than a guess.

    Raises:
        PayloadMismatchError: ``item`` is not an ``EmbeddedChunk``, or its
            vector is not a sequence of finite numbers.
    """
    if not isinstance(item, EmbeddedChunk):
        raise PayloadMismatchError(
            f"index-writer: expected an EmbeddedChunk, got "
            f"{type(item).__name__}; a plain Chunk has no vector to store"
        )
    chunk = item.chunk
    metadata = dict(chunk.metadata or {})
    try:
        vector = tuple(float(value) for value in item.vector)
    except (TypeError, ValueError) as exc:
        raise PayloadMismatchError(
            f"index-writer: chunk {chunk.chunk_id!r} has a vector that is not "
            f"numeric: {exc}",
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
        ) from exc

    return StoredChunkRecord(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source_name=chunk.source_name,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        start_offset=_metadata_int(metadata, PAYLOAD_FIELD_KEYS["start_offset"]),
        end_offset=_metadata_int(metadata, PAYLOAD_FIELD_KEYS["end_offset"]),
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=_metadata_str(metadata, PAYLOAD_FIELD_KEYS["section"]),
        vector=vector,
        index_fingerprint=item.index_fingerprint,
        metadata=metadata,
    )


#: Every field a :class:`StoredChunkRecord` actually has. Read from the class
#: rather than listed by hand, so adding a field to the record cannot leave a
#: stale list behind claiming it is not addressable.
_RECORD_FIELDS: frozenset[str] = frozenset(
    spec.name for spec in dataclass_fields(StoredChunkRecord)
)


# --------------------------------------------------------------------------- #
# FR-45 / FR-46 -- bounded batches and per-item outcomes
# --------------------------------------------------------------------------- #


def bounded_batches(items: Sequence[Any], batch_size: int) -> list[list[Any]]:
    """Split ``items`` into consecutive batches of at most ``batch_size``.

    Raises:
        IndexWriterError: ``batch_size`` is not a positive integer.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise IndexWriterError(
            f"index-writer: batch_size must be an int, got "
            f"{type(batch_size).__name__} ({batch_size!r})"
        )
    if batch_size <= 0:
        raise IndexWriterError(
            f"index-writer: batch_size must be > 0, got {batch_size}; a "
            "bounded batch of zero or fewer would write nothing and report "
            "success"
        )
    return [list(items[start : start + batch_size])
            for start in range(0, len(items), batch_size)]


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """What happened to one item. The unit FR-46 reports in."""

    chunk_id: str | None
    document_id: str | None
    batch: int
    ok: bool
    error: str | None = None

    @property
    def where(self) -> str:
        document = self.document_id or "unknown-document"
        chunk = self.chunk_id or "?"
        return f"{document}/{chunk}"

    def describe(self) -> str:
        state = "written" if self.ok else f"FAILED: {self.error}"
        return f"[batch {self.batch}] {self.where} {state}"


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Every item's outcome, in input order.

    ``ok`` is ``False`` if *anything* failed -- partial success is still
    reported as not-ok, because a caller that only checks one boolean should
    be told the run was not clean.
    """

    results: tuple[BatchItemResult, ...]
    batch_size: int
    batches: int
    dimensions: int

    @property
    def succeeded(self) -> int:
        return sum(1 for result in self.results if result.ok)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.ok)

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def failures(self) -> tuple[BatchItemResult, ...]:
        return tuple(result for result in self.results if not result.ok)

    @property
    def total_items(self) -> int:
        return len(self.results)

    def raise_if_failed(self) -> None:
        """Raise :class:`BatchWriteError` carrying this report, if needed.

        Present so a caller can opt into an exception without the exception
        being the only way to learn what happened -- successes are already
        written by the time this runs, and the report in the error says which.
        """
        if self.ok:
            return
        raise BatchWriteError(
            f"index-writer: {self.failed} of {self.total_items} item(s) failed "
            f"to write; {self.succeeded} succeeded and remain written. First "
            f"failure: {self.failures[0].describe()}",
            self,
        )

    def summary(self) -> str:
        return (
            f"{self.succeeded}/{self.total_items} written in {self.batches} "
            f"batch(es) of <= {self.batch_size} at {self.dimensions} dimensions"
        )


# --------------------------------------------------------------------------- #
# FR-47 -- refuse before writing
# --------------------------------------------------------------------------- #


def _check_config(
    *,
    dimensions: int,
    required_fields: Sequence[str],
    batch_size: int,
) -> tuple[str, ...]:
    """Validate the writer's configuration, returning the required fields.

    Separate from :func:`validate_payload` so a writer can be constructed --
    and its misconfiguration caught -- without a payload to validate against.
    """
    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise IndexWriterError(
            f"index-writer: dimensions must be an int, got "
            f"{type(dimensions).__name__} ({dimensions!r})"
        )
    if dimensions <= 0:
        raise IndexWriterError(
            f"index-writer: dimensions must be > 0, got {dimensions}; an index "
            "of zero width cannot hold a vector"
        )
    bounded_batches([], batch_size)  # reuse its validation, discarding the split

    required = tuple(required_fields)
    unknown = [name for name in required if name not in _RECORD_FIELDS]
    if unknown:
        raise IndexWriterError(
            f"index-writer: required_fields names "
            f"{', '.join(repr(name) for name in unknown)}, which "
            f"{'is' if len(unknown) == 1 else 'are'} not "
            f"{'a field' if len(unknown) == 1 else 'fields'} of "
            "StoredChunkRecord; refusing to validate against a schema the "
            "record cannot satisfy"
        )
    return required


def validate_payload(
    items: Sequence[EmbeddedChunk],
    *,
    dimensions: int,
    required_fields: Sequence[str] = (),
    batch_size: int = 64,
) -> list[StoredChunkRecord]:
    """Check an entire payload against the target index, before writing it.

    Two checks, both raised as :class:`PayloadMismatchError` so the caller has
    one typed thing to catch:

    * **width** -- every vector must be ``dimensions`` wide;
    * **schema** -- every name in ``required_fields`` must be present and
      non-empty on every record.

    Configuration (``dimensions``, ``batch_size``, ``required_fields``) is
    checked first, so an unusable writer is caught before its payload.

    Raises:
        PayloadMismatchError: the first mismatch found, naming the field or
            width and the chunk it was found on.
        IndexWriterError: ``dimensions``, ``batch_size`` or
            ``required_fields`` is unusable.
        StorageInputError: from ``items`` itself -- it must be a non-empty
            sequence of ``EmbeddedChunk`` with a single vector width.
    """
    required = _check_config(
        dimensions=dimensions,
        required_fields=required_fields,
        batch_size=batch_size,
    )

    checked = validate_embedded_chunks(items)

    records: list[StoredChunkRecord] = []
    for item in checked:
        record = record_from(item)
        if len(record.vector) != dimensions:
            raise PayloadMismatchError(
                f"index-writer: chunk {record.chunk_id!r} has a "
                f"{len(record.vector)}-dimensional vector but the target index "
                f"holds {dimensions}-dimensional vectors. Re-embed with the "
                f"index's model, or point the writer at the index those "
                f"vectors were produced for -- writing it anyway would corrupt "
                f"the index rather than extend it.",
                chunk_id=record.chunk_id,
                document_id=record.document_id,
            )
        for name in required:
            value = record.field(name)
            missing = value is None or (isinstance(value, str) and not value.strip())
            if missing:
                raise PayloadMismatchError(
                    f"index-writer: chunk {record.chunk_id!r} is missing "
                    f"{name!r}, which this index declares required. The field "
                    f"is {value!r}; fill it upstream rather than storing a "
                    f"record the index cannot answer a query about.",
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                )
        records.append(record)
    return records


# --------------------------------------------------------------------------- #
# The writer
# --------------------------------------------------------------------------- #


class IndexWriter:
    """Writes embedded chunks in bounded, validated, upserting batches.

        writer = IndexWriter(store, dimensions=1536, batch_size=128)
        report = writer.upsert(items)
        report.raise_if_failed()

    Args:
        store: Anything with ``add(items) -> int``, i.e. a
            ``VectorStoreProtocol`` implementation. Injected rather than
            imported so this works against a real backend or a test double.
        dimensions: Width the target index holds. Every payload is checked
            against it before the first write.
        batch_size: Upper bound on rows per backend call.
        required_fields: Record fields the index declares it cannot answer
            queries without. Checked before writing.
    """

    def __init__(
        self,
        store: Any,
        *,
        dimensions: int,
        batch_size: int = 64,
        required_fields: Sequence[str] = (),
    ) -> None:
        if not hasattr(store, "add") or not callable(store.add):
            raise IndexWriterError(
                f"index-writer: store must provide add(items), got "
                f"{type(store).__name__}"
            )
        # Validate the configuration now, with no payload: a misconfigured
        # writer should fail at construction rather than on first write.
        required = _check_config(
            dimensions=dimensions,
            required_fields=required_fields,
            batch_size=batch_size,
        )
        self._store = store
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.required_fields = required
        self._written = 0
        self._reports: list[BatchReport] = []

    @property
    def written(self) -> int:
        """Items written successfully across every :meth:`upsert` so far."""
        return self._written

    @property
    def reports(self) -> tuple[BatchReport, ...]:
        """Every report this writer has produced, oldest first."""
        return tuple(self._reports)

    def upsert(self, items: Sequence[EmbeddedChunk]) -> BatchReport:
        """Write ``items``, returning what happened to each one.

        The whole payload is validated **before** the first backend call, so a
        mismatch costs no rows. Each batch that the backend rejects is then
        retried item by item -- that is the only way to learn which items in a
        failed batch were actually acceptable -- and upsert semantics make the
        retry safe even for items the backend had already accepted.

        Raises:
            PayloadMismatchError: the payload does not fit the index. Nothing
                was written.
            IndexWriterError: unusable ``batch_size`` or ``dimensions``.
        """
        records = validate_payload(
            items,
            dimensions=self.dimensions,
            required_fields=self.required_fields,
            batch_size=self.batch_size,
        )
        # validate_payload preserves input order, so records[i] belongs to
        # items[i]; pairing them here keeps the batch bookkeeping trivial.
        paired = list(zip(records, list(items)))
        batches = bounded_batches(paired, self.batch_size)
        results: list[BatchItemResult] = []

        for number, batch in enumerate(batches):
            try:
                self._store.add([item for _, item in batch])
            except Exception:  # noqa: BLE001 - recovered below, then reported
                results.extend(self._retry_individually(batch, number))
            else:
                results.extend(
                    BatchItemResult(
                        chunk_id=record.chunk_id,
                        document_id=record.document_id,
                        batch=number,
                        ok=True,
                    )
                    for record, _ in batch
                )

        report = BatchReport(
            results=tuple(results),
            batch_size=self.batch_size,
            batches=len(batches),
            dimensions=self.dimensions,
        )
        self._written += report.succeeded
        self._reports.append(report)
        return report

    def _retry_individually(
        self, batch: Sequence[tuple[StoredChunkRecord, EmbeddedChunk]], number: int
    ) -> list[BatchItemResult]:
        """Write each item of a rejected batch on its own, recording each.

        The only way to learn which items in a failed batch were acceptable.
        Safe because every write is an upsert: an item the backend had already
        accepted before failing simply lands in the same place again.
        """
        outcomes: list[BatchItemResult] = []
        for record, item in batch:
            try:
                self._store.add([item])
            except Exception as exc:  # noqa: BLE001 - reported per item
                outcomes.append(
                    _outcome(
                        record,
                        number,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                outcomes.append(_outcome(record, number, ok=True))
        return outcomes


def _outcome(
    record: StoredChunkRecord, number: int, *, ok: bool, error: str | None = None
) -> BatchItemResult:
    return BatchItemResult(
        chunk_id=record.chunk_id,
        document_id=record.document_id,
        batch=number,
        ok=ok,
        error=error,
    )


# --------------------------------------------------------------------------- #
# FR-47 -- creation and destruction are not the same operation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IndexDestruction:
    """A recorded loss of rows. There is no path that produces one silently."""

    index: str
    rows_removed: int
    reason: str

    def describe(self) -> str:
        return (
            f"destroyed {self.index!r}: {self.rows_removed} row(s) removed "
            f"because {self.reason}"
        )


class IndexLifecycle:
    """Guard for operations that could discard stored rows (FR-47).

    It does not own the index -- it owns the *record of what an operation
    would cost*, so a destructive one is always attributable afterwards.

        lifecycle = IndexLifecycle(count=store.count)
        lifecycle.create("docs")                  # refuses if rows exist
        lifecycle.replace("docs", reason="v2 schema",
                          clear=lambda name: store.delete(store.ids(name)))

    Args:
        count: Callable returning how many rows an index currently holds.
    """

    def __init__(self, *, count: Callable[[str], int]) -> None:
        if not callable(count):
            raise IndexWriterError(
                f"index-writer: count must be callable, got "
                f"{type(count).__name__}"
            )
        self._count = count
        self._destructions: list[IndexDestruction] = []

    @property
    def destructions(self) -> tuple[IndexDestruction, ...]:
        """Every destruction recorded so far, oldest first."""
        return tuple(self._destructions)

    @property
    def last_destruction(self) -> IndexDestruction | None:
        return self._destructions[-1] if self._destructions else None

    def rows(self, name: str) -> int:
        """How many rows ``name`` currently holds."""
        return int(self._count(name))

    def create(self, name: str, *, init: Callable[[str], None] | None = None) -> int:
        """Create an index, refusing to sit on top of stored rows.

        Returns the number of rows the index started with (always 0).

        Raises:
            DestructiveOperationRefused: rows exist and nothing has recorded
                why it is acceptable to lose them.
        """
        rows = self.rows(name)
        if rows:
            raise DestructiveOperationRefused(
                f"index-writer: creating {name!r} would discard {rows} "
                f"stored row(s). Nothing creates an index over data that "
                f"already exists. Call replace(name, reason=...) if losing "
                f"those rows is intended -- it will not proceed without a "
                f"reason, and it records the loss."
            )
        if init is not None:
            init(name)
        return rows

    def replace(
        self,
        name: str,
        *,
        reason: str,
        clear: Callable[[str], int],
    ) -> IndexDestruction:
        """The explicitly destructive path: clear, then record what was lost.

        This is the only operation here that discards rows, and it cannot be
        reached without a reason.

        Raises:
            IndexWriterError: ``reason`` is empty, or ``clear`` is not
                callable.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise IndexWriterError(
                f"index-writer: replacing {name!r} requires a non-empty "
                f"reason. A destructive operation with no stated reason "
                f"cannot be reviewed afterwards, which is exactly how data "
                f"gets lost without anyone knowing why."
            )
        if not callable(clear):
            raise IndexWriterError(
                f"index-writer: clear must be callable, got "
                f"{type(clear).__name__}"
            )
        rows_removed = int(clear(name))
        record = IndexDestruction(
            index=name, rows_removed=rows_removed, reason=reason.strip()
        )
        self._destructions.append(record)
        return record

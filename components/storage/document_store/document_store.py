"""Document-store contracts, and a backend that actually implements them.

``core-types`` declares ``DocumentStoreProtocol`` and ``SourceDocument``. This
component adds the error vocabulary and the input checks a backend
implementation needs, plus :class:`InMemoryDocumentStore` -- a real, working
implementation of ``put``/``get``/``delete``/``count``/``list``.

Why the backend lives here
--------------------------
Before this file declared a contract and nothing satisfied it. A protocol that
no code in the repository implements is documentation, not a store: nothing
ever called ``put()``, so the write path was unproven end to end. FR-45 asks
for at least one backend that works, and this is that backend.

It is deliberately in-memory. This library forbids modifying dependency files,
so a backend needing a server could not be exercised here at all -- and an
unexercised backend is precisely the claim FR-3 and NFR-8 exist to prevent. A
dict-backed store runs everywhere, proves the contract, and is the right
answer for the personal local use this project is scoped to.

Why the errors and filter check are repeated
--------------------------------------------
The library's contract components are copy-ready, and a file that imports its
own error hierarchy from elsewhere is only copy-ready together with that
elsewhere. So the names here deliberately mirror
``vector-store-protocol``'s -- same ``StorageInputError`` shape, same flat
filter rule -- while remaining importable on their own.

The conformance point the type cannot enforce
---------------------------------------------
``get`` returns ``None`` for an unknown id, and that is the single most
important behaviour in this contract. A missing document is an *expected*
condition in a RAG pipeline -- a chunk referencing a document that was never
ingested, or was deleted since -- so raising forces every caller to wrap a
plain lookup in a try/except. Returning ``None`` makes the caller handle it,
which is where that decision belongs.

Self-contained on purpose
-------------------------
Depends on ``core-types`` for ``SourceDocument`` and the protocol (declared in
the manifest, so ``ragkit add`` copies it along). ``core-types`` imports only
``pydantic``, which the runtime requires anyway. No optional packages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core_types import DocumentStoreProtocol, Metadata, SourceDocument

__all__ = [
    "DocumentStoreBackendError",
    "DocumentStoreError",
    "DocumentStoreInputError",
    "DocumentStoreProtocol",
    "InMemoryDocumentStore",
    "validate_filter",
    "validate_ids",
    "validate_source_document",
]


class DocumentStoreError(Exception):
    """Base class for every error raised by a document store (FR-4)."""


class DocumentStoreInputError(DocumentStoreError):
    """The *caller* passed something unusable -- a malformed document or
    filter."""


class DocumentStoreBackendError(DocumentStoreError):
    """The *backend* failed -- a rejected write, an unavailable collection."""


def validate_filter(filter: Metadata | None) -> dict[str, Any] | None:
    """Check a metadata filter, returning it as a plain dict (or ``None``).

    A filter is flat: field name to value. Nested mappings and sequences are
    rejected rather than dropped, because dropping one silently narrows the
    result set in a way the caller cannot observe.

    Raises:
        DocumentStoreInputError: the filter is not a mapping, or a value is
            nested.
    """
    if filter is None:
        return None
    if isinstance(filter, (str, bytes)) or not isinstance(filter, Mapping):
        raise DocumentStoreInputError(
            f"document-store: filter must be a mapping of field to value, "
            f"got {type(filter).__name__}"
        )
    flat: dict[str, Any] = {}
    for field, value in filter.items():
        if not isinstance(field, str) or not field:
            raise DocumentStoreInputError(
                f"document-store: filter field names must be non-empty "
                f"strings, got {field!r}"
            )
        if isinstance(value, Mapping):
            raise DocumentStoreInputError(
                f"document-store: filter value for {field!r} is a mapping. "
                "Filters are flat; a nested value is not portable across "
                "backends, and silently dropping it would narrow your result "
                "set without telling you"
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            raise DocumentStoreInputError(
                f"document-store: filter value for {field!r} is a "
                f"{type(value).__name__}. Filters are flat and take a single "
                "value; filter in a loop or post-filter instead"
            )
        flat[field] = value
    return flat


def validate_source_document(doc: Any) -> SourceDocument:
    """Check a document before it is written, returning it unchanged.

    The field-for-field rules are ``SourceDocument``'s own; what this adds is
    a message that names the component and the field, so a caller does not
    have to read a pydantic traceback to learn which value was wrong.

    Raises:
        DocumentStoreInputError: ``doc`` is not a ``SourceDocument``, or its
            own validation rejected it.
    """
    if not isinstance(doc, SourceDocument):
        raise DocumentStoreInputError(
            f"document-store: expected a SourceDocument, got "
            f"{type(doc).__name__}"
        )
    try:
        # Re-run the model's own validation so the rules live in one place.
        SourceDocument.model_validate(doc.model_dump())
    except Exception as exc:  # noqa: BLE001 - re-raised with a named message
        raise DocumentStoreInputError(
            f"document-store: document {getattr(doc, 'document_id', '<unset>')!r} "
            f"is not valid: {exc}"
        ) from exc
    return doc


def validate_ids(ids: Sequence[str]) -> list[str]:
    """Check a sequence of ids, returning it as a list.

    Raises:
        DocumentStoreInputError: ``ids`` is a bare string, or any id is empty
            or not a string.
    """
    if isinstance(ids, (str, bytes)):
        raise DocumentStoreInputError(
            f"document-store: ids must be a sequence of strings, got a "
            f"{type(ids).__name__} ({ids!r:.60}); a str is itself a sequence "
            "of characters. Wrap it: [ids]"
        )
    if isinstance(ids, (bytes, bytearray)) or not isinstance(ids, Sequence):
        raise DocumentStoreInputError(
            f"document-store: ids must be a sequence of strings, got "
            f"{type(ids).__name__}"
        )
    checked: list[str] = []
    for position, item in enumerate(ids):
        if not isinstance(item, str):
            raise DocumentStoreInputError(
                f"document-store: ids[{position}] must be a str, got "
                f"{type(item).__name__} ({item!r:.60})"
            )
        if not item:
            raise DocumentStoreInputError(
                f"document-store: ids[{position}] is empty; an id must "
                "identify something"
            )
        checked.append(item)
    return checked


# --------------------------------------------------------------------------- #
# A backend that actually implements the contract (FR-45)
# --------------------------------------------------------------------------- #

#: Fields addressed directly rather than through ``metadata`` when filtering.
#: A filter naming one of these matches the document's own field, so
#: ``list({"document_id": "d1"})`` works without the caller needing to know
#: that ``document_id`` is not metadata.
_TOP_LEVEL_FIELDS = ("document_id", "source_name", "page_count")


class InMemoryDocumentStore:
    """A working document-store backend, backed by a dict.

    Satisfies ``DocumentStoreProtocol`` (``put``, ``get``, ``list``) and adds
    the ``delete`` and ``count`` that FR-45 requires. Ids are the documents'
    own ``document_id``, so they are stable across restarts of the process and
    across calls -- there is no generated key for the caller to lose.

    Upsert, not append (FR-46)
    ---------------------------
    ``put`` keys on ``document_id``. Writing an id that already exists
    **replaces** the record in place: ``count()`` does not move, and no stale
    copy of the earlier version survives to be read back. Re-ingesting an
    unchanged document is therefore a no-op rather than a duplicate, which is
    what makes re-running ingestion safe.

    What this is not
    ----------------
    It holds everything in RAM and loses it when the process ends. It is a
    correct implementation of the contract, not a durable database; callers
    needing durability supply their own backend, and the protocol is what
    makes that a swap rather than a rewrite.
    """

    def __init__(self) -> None:
        self._documents: dict[str, SourceDocument] = {}
        self._put_calls = 0
        self._replacements = 0

    @property
    def put_calls(self) -> int:
        """How many times :meth:`put` was called, replacement or not."""
        return self._put_calls

    @property
    def replacements(self) -> int:
        """How many of those calls overwrote an existing record.

        ``put_calls == replacements`` means every write was a re-ingest;
        the gap between the two is how many genuinely new documents arrived.
        """
        return self._replacements

    def put(self, doc: SourceDocument) -> str:
        """Store or replace ``doc``, returning its ``document_id``.

        Raises:
            DocumentStoreInputError: ``doc`` is not a ``SourceDocument``, or
                its own validation rejected it.
        """
        checked = validate_source_document(doc)
        existed = checked.document_id in self._documents
        self._documents[checked.document_id] = checked
        self._put_calls += 1
        if existed:
            self._replacements += 1
        return checked.document_id

    def get(self, id: str) -> SourceDocument | None:
        """The document with this id, or ``None`` if there is none.

        ``None`` is the expected answer for an unknown id, not an error: a
        chunk referencing a document that was never ingested is normal.
        """
        if not isinstance(id, str):
            raise DocumentStoreInputError(
                f"document-store: id must be a str, got {type(id).__name__} "
                f"({id!r:.60}); a sequence of characters needs quoting"
            )
        return self._documents.get(id)

    def delete(self, ids: Sequence[str]) -> int:
        """Delete by id, returning how many were actually removed.

        Deleting an unknown id is a no-op rather than an error, which is what
        makes this safe to call after a failed re-ingest.
        """
        checked = validate_ids(ids)
        removed = 0
        for document_id in checked:
            if self._documents.pop(document_id, None) is not None:
                removed += 1
        return removed

    def count(self) -> int:
        """How many documents the store currently holds."""
        return len(self._documents)

    def list(self, filter: Metadata | None = None) -> list[SourceDocument]:
        """Every stored document matching ``filter``, insertion order.

        ``filter`` names top-level fields (``document_id``, ``source_name``,
        ``page_count``) directly and anything else is looked up in the
        document's ``metadata``.

        Note that a key no document uses simply matches nothing, so a typo
        returns an empty list rather than raising. The filter is validated for
        *shape* by :func:`validate_filter`, but its keys describe the caller's
        intent, and guessing at intent is not this store's job.
        """
        flat = validate_filter(filter)
        documents = list(self._documents.values())
        if flat is None:
            return documents
        matched: list[SourceDocument] = []
        for document in documents:
            if all(
                _matches(document, key, value) for key, value in flat.items()
            ):
                matched.append(document)
        return matched


def _matches(document: SourceDocument, key: str, value: Any) -> bool:
    """Whether one field of ``document`` equals ``value``."""
    if key in _TOP_LEVEL_FIELDS:
        return getattr(document, key) == value
    return document.metadata.get(key) == value

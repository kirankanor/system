"""Document-store contracts, independent of any particular backend.

``core-types`` declares ``DocumentStoreProtocol`` and ``SourceDocument``. This
component adds the error vocabulary and the input checks a backend
implementation needs, so a store can be written — and copied out — without
knowing the rest of the library.

Why the errors and filter check are repeated
--------------------------------------------
The library's contract components are copy-ready, and a file that imports its
own error hierarchy from elsewhere is only copy-ready together with that
elsewhere. So the names here deliberately mirror
``vector-store-protocol``'s — same ``StorageInputError`` shape, same flat
filter rule — while remaining importable on their own.

The conformance point the type cannot enforce
---------------------------------------------
``get`` returns ``None`` for an unknown id, and that is the single most
important behaviour in this contract. A missing document is an *expected*
condition in a RAG pipeline — a chunk referencing a document that was never
ingested, or was deleted since — so raising forces every caller to wrap a
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
    "validate_filter",
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

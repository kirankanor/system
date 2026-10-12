"""Vector-store operations and result contracts.

``core-types`` already declares ``VectorStoreProtocol`` and the result types.
This component re-declares the error vocabulary and the input checks that every
implementation needs, and re-exports the protocol, so a store implementation
can be copied out with one import and no knowledge of the rest of the library.

Why the errors are repeated rather than imported
-----------------------------------------------
The chunkers in this library each define their own ``ChunkingError``, and the
same reasoning applies here: a contract component has to be copy-ready on its
own. Importing an error hierarchy from somewhere else means copying *that*
somewhere else too, and a caller reading this file should be able to see every
failure it can raise without following a chain of imports. The names are
deliberately shared across this area's components so the duplication is
familiar rather than surprising.

What an implementation must get right
-------------------------------------
The two conformance points the *type* cannot enforce are called out here
because they are the ones that silently corrupt a result set:

1. **Results name their backend.** Every :class:`ScoredDocument` a query
   returns carries the backend's name in ``source_name``. Without it a caller
   merging results from a dense store and a sparse one cannot tell them
   apart, and the hybrid retriever upstream has nothing to merge on.
2. **Filters are flat.** ``filter`` is a mapping of field to value; a nested
   value is rejected rather than silently dropped, because dropping it would
   narrow the result set in a way the caller cannot see.

Self-contained on purpose
-------------------------
Depends on ``core-types`` for the shared types (declared in the manifest, so
``ragkit add`` copies it along). ``core-types`` itself imports only
``pydantic``, which the library's runtime requires anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from core_types import (
    Chunk,
    EmbeddedChunk,
    Metadata,
    ScoredDocument,
    VectorStoreProtocol,
)

__all__ = [
    "StorageBackendError",
    "StorageError",
    "StorageInputError",
    "MissingStorageDependency",
    "VectorStoreProtocol",
    "validate_embedded_chunks",
    "validate_filter",
    "validate_ids",
    "validate_k",
    "validate_vector",
]


class StorageError(Exception):
    """Base class for every error raised by a vector store (FR-4)."""


class StorageInputError(StorageError):
    """The *caller* passed something unusable -- a bad filter, id or vector."""


class StorageBackendError(StorageError):
    """The *backend* failed -- a rejected write, an unavailable index."""


class MissingStorageDependency(ImportError):
    """An optional backend dependency is not installed (FR-3).

    Deliberately an ``ImportError`` and deliberately not a
    :class:`StorageError`, so callers guarding optional backends with
    ``except ImportError`` catch it.
    """


def validate_filter(filter: Metadata | None) -> dict[str, Any] | None:
    """Check a metadata filter, and return it as a plain dict (or ``None``).

    A filter is flat: field name to value. Nested mappings and lists of
    values are rejected rather than dropped, because dropping one silently
    narrows the result set in a way the caller cannot observe -- they asked
    for a superset and got a subset with no warning.

    Args:
        filter: The candidate filter, or ``None`` for no filtering.

    Returns:
        The filter as a ``dict``, or ``None``.

    Raises:
        StorageInputError: the filter is not a mapping, or a value is
            nested.
    """
    if filter is None:
        return None
    if isinstance(filter, (str, bytes)) or not isinstance(filter, Mapping):
        raise StorageInputError(
            f"vector-store-protocol: filter must be a mapping of field to "
            f"value, got {type(filter).__name__}"
        )
    flat: dict[str, Any] = {}
    for field, value in filter.items():
        if not isinstance(field, str) or not field:
            raise StorageInputError(
                f"vector-store-protocol: filter field names must be non-empty "
                f"strings, got {field!r}"
            )
        if isinstance(value, Mapping):
            raise StorageInputError(
                f"vector-store-protocol: filter value for {field!r} is a "
                "mapping. Filters are flat -- nested values are not portable "
                "across backends, and silently dropping them would narrow "
                "your result set without telling you"
            )
        if isinstance(value, (list, tuple, set, frozenset)):
            raise StorageInputError(
                f"vector-store-protocol: filter value for {field!r} is a "
                f"{type(value).__name__}. Filters are flat and take a single "
                "value; filter in a loop or post-filter the results instead"
            )
        flat[field] = value
    return flat


def validate_ids(ids: Sequence[str]) -> list[str]:
    """Check a sequence of ids, returning it as a list.

    Raises:
        StorageInputError: ``ids`` is a bare string (which *is* a sequence of
            characters), or any id is empty or not a string.
    """
    if isinstance(ids, (str, bytes)):
        raise StorageInputError(
            f"vector-store-protocol: ids must be a sequence of strings, got "
            f"a {type(ids).__name__} ({ids!r:.60}); a str is itself a "
            "sequence of characters. Wrap it: [ids]"
        )
    if isinstance(ids, (bytes, bytearray)) or not isinstance(ids, Sequence):
        raise StorageInputError(
            f"vector-store-protocol: ids must be a sequence of strings, got "
            f"{type(ids).__name__}"
        )
    checked: list[str] = []
    for position, item in enumerate(ids):
        if not isinstance(item, str):
            raise StorageInputError(
                f"vector-store-protocol: ids[{position}] must be a str, got "
                f"{type(item).__name__} ({item!r:.60})"
            )
        if not item:
            raise StorageInputError(
                f"vector-store-protocol: ids[{position}] is empty; an id must "
                "identify something"
            )
        checked.append(item)
    return checked


def validate_vector(vector: Sequence[float]) -> list[float]:
    """Check one embedding vector, returning it as a list of floats.

    Rejects an empty vector (there is no such thing as a zero-width point in
    a vector space) and any non-finite value, because ``NaN`` and ``inf``
    compare against nothing and poison a nearest-neighbour search silently.

    Raises:
        StorageInputError: the vector is empty, not a sequence, or contains
            a non-numeric or non-finite value.
    """
    if isinstance(vector, (str, bytes, bytearray)):
        raise StorageInputError(
            f"vector-store-protocol: a vector must be a sequence of floats, "
            f"got a {type(vector).__name__}"
        )
    if isinstance(vector, (bytes, bytearray)) or not isinstance(vector, Sequence):
        raise StorageInputError(
            f"vector-store-protocol: a vector must be a sequence of floats, "
            f"got {type(vector).__name__}"
        )
    if not vector:
        raise StorageInputError(
            "vector-store-protocol: a vector must have at least one value; "
            "an empty vector has no position in a vector space"
        )
    out: list[float] = []
    for position, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StorageInputError(
                f"vector-store-protocol: vector[{position}] must be a number, "
                f"got {type(value).__name__} ({value!r:.60})"
            )
        number = float(value)
        if not isfinite(number):
            raise StorageInputError(
                f"vector-store-protocol: vector[{position}] is {number!r}. "
                "NaN and infinity compare against nothing and poison a "
                "nearest-neighbour search without failing visibly"
            )
        out.append(number)
    return out


def validate_k(k: Any) -> int:
    """Check a result-count argument, returning it as an int.

    Raises:
        StorageInputError: ``k`` is not a positive int.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise StorageInputError(
            f"vector-store-protocol: k must be an int, got "
            f"{type(k).__name__} ({k!r})"
        )
    if k <= 0:
        raise StorageInputError(
            f"vector-store-protocol: k must be > 0, got {k}; asking for no "
            "results should be an empty list at the caller, not a query"
        )
    return k


def validate_embedded_chunks(items: Sequence[EmbeddedChunk]) -> list[EmbeddedChunk]:
    """Check items before they are written, returning them as a list.

    Every chunk must carry a vector, and every vector in one call must have
    the same width: a store that accepted mixed widths would answer queries
    against an index whose geometry is not one geometry.

    Raises:
        StorageInputError: ``items`` is not a sequence, is empty, or any
            chunk has a vector of a differing width.
    """
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise StorageInputError(
            f"vector-store-protocol: items must be a sequence of "
            f"EmbeddedChunk, got {type(items).__name__}"
        )
    if not items:
        raise StorageInputError(
            "vector-store-protocol: items is empty; there is nothing to add"
        )
    checked: list[EmbeddedChunk] = []
    width: int | None = None
    for position, item in enumerate(items):
        if not isinstance(item, EmbeddedChunk):
            raise StorageInputError(
                f"vector-store-protocol: items[{position}] must be an "
                f"EmbeddedChunk, got {type(item).__name__}"
            )
        validate_vector(item.vector)
        if width is None:
            width = len(item.vector)
        elif len(item.vector) != width:
            raise StorageInputError(
                f"vector-store-protocol: items[{position}] has a "
                f"{len(item.vector)}-wide vector but item 0 has {width}. "
                "One call must add one vector space, or the index has no "
                "single geometry to search against"
            )
        checked.append(item)
    return checked

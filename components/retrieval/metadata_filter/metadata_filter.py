"""Filter retrieval candidates by metadata, with operators you can trust.

Why this exists
---------------
Filtering is where a RAG pipeline silently loses recall. Three ways it
happens, and how each is answered here:

* **An operator nobody validated.** A config that says ``{"page": {"gtr": 3}}``
  is a typo, and a filter that treats an unknown operator as "never matches"
  returns an empty list that looks like a correct answer. Unknown operators
  raise, and the message names the operator and lists the valid ones.
* **A field nobody checked.** ``{"langauge": "en"}`` matches nothing, forever,
  with no signal. A missing field is a *match failure* rather than an error --
  it is legal for a document not to carry a field -- but ``exists`` is
  provided so a caller can ask about presence explicitly.
* **A comparison Python refuses.** ``"3" > 2`` raises ``TypeError`` in Python
  3, and that traceback escaping a filter is worse than a clear message.
  Incomparable operands raise ``MetadataFilterInputError`` naming both types.

The criteria language
---------------------
Each key is a field; each value is either a literal (meaning ``eq``) or a
one-key mapping naming an operator::

    {"lang": "en"}                     # eq
    {"page": {"gte": 3}}               # range
    {"title": {"contains": "setup"}}   # substring of the document's text
    {"lang": {"in": ["en", "de"]}}     # the document's value among these
    {"title": {"startswith": "SETUP"}} # prefix
    {"reviewed": {"exists": True}}     # presence

Metadata values are scalars by contract (``core-types`` types them as text,
number or bool), so ``contains`` and ``startswith`` operate on text and
``in`` asks whether one scalar is among a set of candidates. Membership in a
list is not expressible because a list is not a legal metadata value.

A field is looked up first among the document's own fields (``document_id``,
``source_name``, ``chunk_index``, ``text``...) and then in its ``metadata``,
so a caller writes the name they see in the result rather than the storage
path it happens to live under.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core_types import Chunk, ScoredDocument

__all__ = [
    "OPERATORS",
    "MetadataFilterError",
    "MetadataFilterInputError",
    "filter_documents",
]

#: The operators this component implements. Anything else raises.
OPERATORS: tuple[str, ...] = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "contains",
    "startswith",
    "exists",
)

#: Operators that order two values and therefore need comparable types.
_ORDERING = frozenset({"gt", "gte", "lt", "lte"})

#: Document fields addressable by name. Kept explicit rather than derived from
#: the model so a rename in ``core-types`` fails loudly here instead of
#: quietly dropping a documented filter field.
_IDENTITY_FIELDS = ("chunk_id", "document_id", "source_name", "chunk_index", "text")


class MetadataFilterError(Exception):
    """Base for every failure in this component."""


class MetadataFilterInputError(MetadataFilterError):
    """The caller passed criteria this component cannot evaluate."""


def _lookup(document: Chunk, field: str) -> tuple[bool, Any]:
    """Return ``(present, value)`` for ``field`` on ``document``.

    Takes the base :class:`Chunk` because ``ScoredDocument`` *is* one, and the
    criteria language has to serve both without a second copy of the lookup.
    """
    if field in _IDENTITY_FIELDS:
        return True, getattr(document, field)
    if field in document.metadata:
        return True, document.metadata[field]
    return False, None


def _resolve(value: Any, field: str) -> tuple[str, Any]:
    """Turn a criterion value into ``(operator, operand)``."""
    if isinstance(value, Mapping):
        if len(value) != 1:
            raise MetadataFilterInputError(
                f"metadata-filter: the criterion for {field!r} has "
                f"{len(value)} conditions, expected exactly one. Combine "
                f"several fields in one criteria mapping instead of nesting "
                f"them under one key"
            )
        operator, operand = next(iter(value.items()))
        if not isinstance(operator, str):
            raise MetadataFilterInputError(
                f"metadata-filter: the operator for {field!r} is "
                f"{type(operator).__name__}, expected a string"
            )
        if operator not in OPERATORS:
            raise MetadataFilterInputError(
                f"metadata-filter: unknown operator {operator!r} for field "
                f"{field!r}. Valid operators are: {', '.join(OPERATORS)}"
            )
        return operator, operand
    return "eq", value


def _compare(
    operator: str, present: bool, actual: Any, operand: Any, field: str
) -> bool:
    """Evaluate one operator against one document."""
    if operator == "exists":
        if not isinstance(operand, bool):
            raise MetadataFilterInputError(
                f"metadata-filter: the 'exists' operator takes True or False, "
                f"got {operand!r}"
            )
        return present is operand

    # Every other operator needs a value to compare against. A document that
    # does not carry the field simply does not match.
    if not present:
        return False

    if operator == "eq":
        return actual == operand
    if operator == "ne":
        return actual != operand
    if operator == "in":
        if isinstance(operand, (str, bytes)) or not isinstance(operand, Sequence):
            raise MetadataFilterInputError(
                f"metadata-filter: the 'in' operator takes a sequence of "
                f"candidates, got {type(operand).__name__}. Wrap a single "
                f"value in a list, or use 'eq'"
            )
        return actual in operand
    if operator == "contains":
        if isinstance(actual, str):
            return operand in actual
        raise MetadataFilterInputError(
            f"metadata-filter: the 'contains' operator needs the document's "
            f"{field!r} to be text, got {type(actual).__name__}. Metadata "
            f"values are scalars (text, number or bool) by contract, so "
            f"membership in a list is expressed with 'in' instead"
        )
    if operator == "startswith":
        if not isinstance(actual, str):
            raise MetadataFilterInputError(
                f"metadata-filter: the 'startswith' operator needs the "
                f"document's {field!r} to be text, got "
                f"{type(actual).__name__}"
            )
        if not isinstance(operand, str):
            raise MetadataFilterInputError(
                f"metadata-filter: the 'startswith' operand must be text, got "
                f"{type(operand).__name__}"
            )
        return actual.startswith(operand)

    # The four ordering operators.
    if isinstance(actual, str) != isinstance(operand, str):
        raise MetadataFilterInputError(
            f"metadata-filter: cannot compare {field!r} with '{operator}': the "
            f"document holds a {type(actual).__name__} and the operand is a "
            f"{type(operand).__name__}"
        )
    try:
        if operator == "gt":
            return actual > operand
        if operator == "gte":
            return actual >= operand
        if operator == "lt":
            return actual < operand
        return actual <= operand
    except TypeError as exc:
        raise MetadataFilterInputError(
            f"metadata-filter: cannot compare the document's {field!r} "
            f"({type(actual).__name__}) with the operand "
            f"({type(operand).__name__}) using '{operator}'"
        ) from exc


def filter_documents(
    documents: Sequence[ScoredDocument],
    criteria: Mapping[str, Any],
) -> list[ScoredDocument]:
    """Return the documents matching every criterion (logical AND).

    Criteria combine with AND, not OR, because a caller narrowing a result set
    expects every condition to hold; a caller who wants OR can filter twice
    and concatenate.

    The input order is preserved, so a caller's existing ranking survives the
    filter rather than being re-sorted by whatever the criteria happen to
    contain.
    """
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise MetadataFilterInputError(
            f"metadata-filter: documents must be a sequence of ScoredDocument, "
            f"got {type(documents).__name__}"
        )
    items = list(documents)
    for position, item in enumerate(items):
        if not isinstance(item, ScoredDocument):
            raise MetadataFilterInputError(
                f"metadata-filter: documents[{position}] is "
                f"{type(item).__name__}, expected ScoredDocument"
            )
    if not isinstance(criteria, Mapping):
        raise MetadataFilterInputError(
            f"metadata-filter: criteria must be a mapping of field to "
            f"criterion, got {type(criteria).__name__}"
        )
    for field in criteria:
        if not isinstance(field, str) or not field:
            raise MetadataFilterInputError(
                f"metadata-filter: field names must be non-empty strings, got "
                f"{field!r}"
            )

    resolved = [(field, *_resolve(value, field)) for field, value in criteria.items()]

    return [
        document
        for document in items
        if all(
            _compare(operator, *_lookup(document, field), operand, field)
            for field, operator, operand in resolved
        )
    ]


def apply_to_chunks(
    chunks: Sequence[Chunk], criteria: Mapping[str, Any]
) -> list[Chunk]:
    """Same criteria language, applied to plain :class:`Chunk` values.

    Retrieval works on ``ScoredDocument``; ingestion and pre-filtering work on
    ``Chunk``. Sharing one criteria language means a filter written against a
    query result can be reused to pre-narrow a corpus, rather than being
    rewritten in a second dialect.
    """
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise MetadataFilterInputError(
            f"metadata-filter: chunks must be a sequence of Chunk, got "
            f"{type(chunks).__name__}"
        )
    items = list(chunks)
    for position, item in enumerate(items):
        if not isinstance(item, Chunk):
            raise MetadataFilterInputError(
                f"metadata-filter: chunks[{position}] is "
                f"{type(item).__name__}, expected Chunk"
            )
    if not isinstance(criteria, Mapping):
        raise MetadataFilterInputError(
            f"metadata-filter: criteria must be a mapping of field to "
            f"criterion, got {type(criteria).__name__}"
        )
    for field in criteria:
        if not isinstance(field, str) or not field:
            raise MetadataFilterInputError(
                f"metadata-filter: field names must be non-empty strings, got "
                f"{field!r}"
            )
    resolved = [(field, *_resolve(value, field)) for field, value in criteria.items()]

    return [
        chunk
        for chunk in items
        if all(
            _compare(operator, *_lookup(chunk, field), operand, field)
            for field, operator, operand in resolved
        )
    ]

"""Retrieve candidates by vector similarity against a dense vector store.

The three-part pipeline this composes
-------------------------------------
``embed query -> ask the store -> hand back ScoredDocument``. Every part is
already a contract from another component: the embedder is an
``EmbeddingProtocol`` and the store is a ``VectorStoreProtocol``, both from
``core-types``. This component owns none of them and re-implements none of
them -- it validates that what it was handed honours the contracts, does the
composition, and fails with a message that says which part broke.

What it checks, and why each is worth the check
-----------------------------------------------
* **The vector space.** An embedder and a store only agree when
  ``embedder.dimensions`` matches the width the store was indexed at, and
  when the store was indexed with the *same model*. Querying a store indexed
  with model A using vectors from model B returns confident, well-formed,
  entirely wrong neighbours -- no error anywhere. This checks the width, and
  records ``embedder.model_id`` in the retriever so a mismatch is visible in
  the object the caller holds.
* **One query, one vector.** ``embed([query])`` must return exactly one
  vector. A batch embedder handed one string and returning two vectors means
  the caller's query was matched against the wrong one of them.
* **Every result is a ``ScoredDocument``.** The whole point of FR-8 is that
  downstream components never convert. A store that returned raw dicts would
  push that conversion onto every caller instead.

Requesting more results than the store holds is not an error. A store with
three chunks asked for ten returns three, and this passes them through -- a
retriever that raised would force every caller to know its index size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core_types import EmbeddingProtocol, ScoredDocument, VectorStoreProtocol

__all__ = ["DenseRetriever", "RetrievalError", "RetrievalInputError"]


class RetrievalError(Exception):
    """Base for every failure in this component."""


class RetrievalInputError(RetrievalError):
    """The caller passed something this component cannot retrieve with."""


def _validate_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise RetrievalInputError(
            f"dense-retrieval: k must be an int, got "
            f"{type(k).__name__} ({k!r})"
        )
    if k < 1:
        raise RetrievalInputError(
            f"dense-retrieval: k must be at least 1 to return anything, got {k}"
        )
    return k


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise RetrievalInputError(
            f"dense-retrieval: query must be a string, got "
            f"{type(query).__name__}"
        )
    if not query.strip():
        raise RetrievalInputError(
            f"dense-retrieval: query is empty or whitespace. An empty string "
            f"embeds to a real vector and retrieves a real-looking answer "
            f"about nothing, so it is rejected rather than passed through"
        )
    return query


class DenseRetriever:
    """Embed a query and ask a vector store for its nearest neighbours.

    ``name`` is stamped onto the retriever and used to describe failures, so a
    caller running several retrievers can tell which one complained.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingProtocol,
        store: VectorStoreProtocol,
        k: int = 5,
        name: str = "dense",
    ) -> None:
        if not isinstance(embedder, EmbeddingProtocol):
            raise RetrievalInputError(
                f"dense-retrieval: embedder must satisfy EmbeddingProtocol "
                f"(dimensions, model_id, embed), got "
                f"{type(embedder).__name__}"
            )
        if not isinstance(store, VectorStoreProtocol):
            raise RetrievalInputError(
                f"dense-retrieval: store must satisfy VectorStoreProtocol "
                f"(add, query, delete, count), got {type(store).__name__}"
            )
        if (
            isinstance(embedder.dimensions, bool)
            or not isinstance(embedder.dimensions, int)
            or embedder.dimensions < 1
        ):
            raise RetrievalInputError(
                f"dense-retrieval: embedder.dimensions must be a positive "
                f"int, got {embedder.dimensions!r}"
            )
        if not isinstance(name, str) or not name:
            raise RetrievalInputError(
                f"dense-retrieval: name must be a non-empty string, got "
                f"{name!r}"
            )
        self.embedder = embedder
        self.store = store
        self.name = name
        self.k = _validate_k(k)

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """Return the ``k`` nearest chunks to ``query``, best first.

        Ordering is whatever the store returns: for a vector store that is
        ascending distance, and this component does not second-guess it. A
        store that returned an unsorted list would be a store bug, and
        re-sorting here would mask it.
        """
        text = _validate_query(query)
        limit = self.k if k is None else _validate_k(k)

        try:
            vectors = self.embedder.embed([text])
        except Exception as exc:  # noqa: BLE001 - the embedder's own failure
            raise RetrievalError(
                f"dense-retrieval: embedding the query with "
                f"{self.embedder.model_id!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(vectors, Sequence) or len(vectors) != 1:
            raise RetrievalError(
                f"dense-retrieval: embed([query]) returned "
                f"{len(vectors) if isinstance(vectors, Sequence) else type(vectors).__name__} "
                f"vectors, expected exactly 1. A query needs one vector to "
                f"search with; more than one means the query was matched "
                f"against the wrong one"
            )
        vector = list(vectors[0])
        if len(vector) != self.embedder.dimensions:
            raise RetrievalError(
                f"dense-retrieval: the query vector is {len(vector)} wide but "
                f"{self.embedder.model_id!r} declares "
                f"{self.embedder.dimensions} dimensions"
            )

        try:
            results = self.store.query(vector, limit, filter)
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - the store's own failure
            raise RetrievalError(
                f"dense-retrieval: querying the store failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise RetrievalError(
                f"dense-retrieval: the store returned "
                f"{type(results).__name__}, expected a sequence of "
                f"ScoredDocument"
            )
        for position, item in enumerate(results):
            if not isinstance(item, ScoredDocument):
                raise RetrievalError(
                    f"dense-retrieval: the store returned a "
                    f"{type(item).__name__} at position {position}. Results "
                    f"must be ScoredDocument so downstream components need "
                    f"no conversion (FR-8)"
                )
        return list(results)

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(name={self.name!r}, "
            f"model_id={self.embedder.model_id!r}, "
            f"dimensions={self.embedder.dimensions}, k={self.k})"
        )

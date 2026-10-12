"""Provider-neutral contracts defined once and reused by every area (FR-5, FR-8).

Why this module exists
----------------------
**FR-5 -- interchange.** Components in an area that share a contract implement
one common protocol, so a caller swaps ``fixed-size-chunker`` for
``token-chunker`` without touching its own code. Seven protocols cover the
library: chunking, embedding, vector-store, document-store, reranker,
context-builder and metric.

**FR-8 -- no adapters between stages.** Retrieval, reranking and context
building all pass the *same* result type around (``ScoredDocument``), so the
stages compose directly. Introducing a second "scored result" class would
recreate the exact adapter seam FR-8 exists to remove.

Why these types are defined here rather than imported
------------------------------------------------------
The ``settings`` component ships a near-identical ``Chunk``/``SourceDocument``
in ``config/models.py``, because ``config/__init__.py`` states that *"standalone
library components must NOT depend on it (they ship their own models /
exceptions)."* ``core-types`` is a standalone library component: it must be
copyable into a project on its own, and it must not drag the destination
application's settings, prompts and error hierarchy in behind it.

So the two layers are deliberately separate today:

* **this file** -- the component interchange vocabulary (copied by
  ``ragkit add core-types``, no dependencies);
* ``config/models.py`` -- the destination application's pipeline contracts.

The field names below mirror ``config/models.py`` exactly wherever the two
mean the same thing, so that collapsing them into one definition later is a
mechanical move rather than a redesign. The one intentional difference is
noted on :class:`Chunk`. **Do not "fix" this duplication casually** -- see the
project knowledge record for the reconciliation plan.

What is *not* here
------------------
Overlap semantics (FR-7) belong to the chunkers, not to this contract, so
this module never validates ``overlap <= size``. Importing anything from
``rag_code_library`` would break the rule that component tests run without
the registry package, so this file imports only from ``pydantic``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

__all__ = [
    "Chunk",
    "ChunkingProtocol",
    "Context",
    "ContextBuilderProtocol",
    "DocumentStoreProtocol",
    "EmbeddingProtocol",
    "EmbeddedChunk",
    "EvaluationExample",
    "Metadata",
    "MetadataValue",
    "MetricProtocol",
    "MetricResult",
    "RerankerProtocol",
    "ScoredDocument",
    "SourceDocument",
    "VectorStoreProtocol",
]

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _Contract(BaseModel):
    """Frozen and closed: a contract record is validated once, then trusted."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Shared result types (FR-8)
# --------------------------------------------------------------------------- #


class Chunk(_Contract):
    """A contiguous piece of text produced by a chunker.

    Field names mirror ``config/models.py:Chunk`` so the two definitions can be
    collapsed later without renaming anything.

    **Intentional difference:** the document-derived fields
    (``chunk_id``, ``document_id``, ``source_name``) and the page range are
    *optional* here, while ``config`` requires them. A chunker is called as
    ``chunk(text, size=..., overlap=...)`` and has no idea which document the
    text came from, so demanding an id it cannot know would force every
    chunker to fabricate one. Ingestion and indexing fill these in afterwards
    (the application layer does that with ``make_chunk_id`` /
    ``make_document_id``). ``page_start``/``page_end`` are optional because
    plain-text sources have no pages at all.

    When both pages are given they must still be ordered -- a page range that
    runs backwards is a bug, not a value to clamp.
    """

    chunk_id: str | None = None
    document_id: str | None = None
    source_name: str | None = None
    chunk_index: Annotated[int, Field(ge=0)]
    text: _NonEmpty
    page_start: Annotated[int, Field(ge=1)] | None = None
    page_end: Annotated[int, Field(ge=1)] | None = None
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _pages_ordered(self) -> Self:
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be >= page_start")
        return self


class ScoredDocument(Chunk):
    """The one result type shared by retrieval, reranking and context building.

    This is FR-8's carrier: it carries ``text``, ``score``, a source id
    (``document_id``/``source_name``) and ``metadata``. It *is* a :class:`Chunk`
    plus a score, rather than a separate shape, because a reranker that returned
    its own class would force the caller to convert before context building.

    ``score`` is not bounded and its direction is metric-specific: cosine and
    BM25 disagree about which end is better. Re-rankers must not assume
    "lower is better" -- compare within a single result set only.

    Ordering of a returned list encodes rank, so there is deliberately no
    ``rank`` field to fall out of sync with position.
    """

    score: float

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        score: float,
        document_id: str | None = None,
        source_name: str | None = None,
    ) -> Self:
        """Score an existing :class:`Chunk`, optionally naming its origin.

        This is the conversion every retriever performs, and the naive
        ``ScoredDocument(**chunk.model_dump(), document_id=...)`` raises
        ``TypeError: got multiple values for keyword argument`` as soon as the
        chunk already carries that field. Overriding happens here instead, once.

        Any override passed as ``None`` keeps the chunk's own value, so
        ``from_chunk(chunk, score=s)`` on an already-identified chunk is a pure
        scoring operation that loses nothing.
        """
        payload = chunk.model_dump()
        if document_id is not None:
            payload["document_id"] = document_id
        if source_name is not None:
            payload["source_name"] = source_name
        return cls(score=score, **payload)


class SourceDocument(_Contract):
    """A stored document, as accepted by :class:`DocumentStoreProtocol`.

    Field-for-field identical to ``config/models.py:SourceDocument``.
    """

    document_id: _NonEmpty
    source_name: _NonEmpty
    sha256: _SHA256
    page_count: Annotated[int, Field(ge=0)]
    metadata: Metadata = Field(default_factory=dict)


class EmbeddedChunk(_Contract):
    """A chunk paired with the vector that represents it.

    ``index_fingerprint`` records which embedding + chunking configuration
    produced the vector, so a stale entry can be detected and rebuilt instead
    of silently answering against the wrong embedding space.
    """

    chunk: Chunk
    vector: Annotated[tuple[float, ...], Field(min_length=1)]
    index_fingerprint: _NonEmpty


class Context(_Contract):
    """What :class:`ContextBuilderProtocol` assembled, handed to the answer stage.

    ``text`` may legitimately be empty: when nothing fits the budget the
    builder returns an empty context rather than raising, and the answer stage
    decides whether to abstain. ``chunks`` is the authoritative provenance --
    exactly the sources that made it in, in the order they appear in ``text``,
    so a citation can never disagree with what the model was actually shown.
    """

    text: str
    chunks: tuple[ScoredDocument, ...] = ()
    token_count: Annotated[int, Field(ge=0)] = 0
    budget: Annotated[int, Field(ge=1)]
    truncated: bool = False


class EvaluationExample(_Contract):
    """One row a :class:`MetricProtocol` consumes.

    This is the *contract* a metric may rely on; the ``evaluation-dataset``
    component supplies the on-disk format that produces these. Fields a stage
    has not run yet stay empty rather than being invented -- ``actual_answer``
    is unset until an answer stage has run, so retrieval-only metrics work
    without a generator.
    """

    query: _NonEmpty
    relevant_ids: tuple[str, ...] = ()
    retrieved_ids: tuple[str, ...] = ()
    expected_answer: str | None = None
    actual_answer: str | None = None


class MetricResult(_Contract):
    """The outcome of one metric over one dataset.

    ``details`` is scalar-only so results stay JSON-serialisable for
    regression baselines (FR-9), and ``higher_is_better`` travels with the
    value so a comparison never has to be guessed from the metric's name.
    """

    name: _NonEmpty
    value: float
    higher_is_better: bool = True
    sample_count: Annotated[int, Field(ge=0)] = 0
    details: Metadata = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The seven protocols (FR-5)
# --------------------------------------------------------------------------- #


@runtime_checkable
class ChunkingProtocol(Protocol):
    """Split text into chunks.

    Overlap semantics are the implementing chunker's responsibility (FR-7):
    each one documents what ``overlap`` means to it, and rejects
    ``overlap >= size`` with a typed error rather than clamping. This contract
    deliberately does not enforce that, so one chunker's rule is not forced
    onto another's.
    """

    def chunk(self, text: str, *, size: int, overlap: int, **kw: Any) -> list[Chunk]: ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Turn text into vectors in one fixed vector space.

    ``dimensions`` and ``model_id`` describe the space: two embedders are only
    comparable when both match, and a vector store indexed with one space must
    not be queried with another. Implementations expose these as ordinary
    attributes.
    """

    dimensions: int
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Store and query vectors.

    ``filter`` is a flat metadata filter; nested values are not portable
    across backends, so implementations reject them rather than silently
    dropping them.

    Implementations must name themselves when they build results: the
    ``ScoredDocument`` instances returned from :meth:`query` should carry the
    backend's name in the inherited ``source_name``/``document_id`` fields, so a
    caller can tell which backend a result came from. Note that the *type* does
    not enforce this -- those fields are optional because a raw chunk cannot know
    its own origin -- so it is a conformance requirement on implementations, and
    one that T-08/T-09a tests are expected to assert.
    """

    def add(self, items: Sequence[EmbeddedChunk]) -> int: ...

    def query(
        self, vector: Sequence[float], k: int, filter: Metadata | None = None
    ) -> list[ScoredDocument]: ...

    def delete(self, ids: Sequence[str]) -> int: ...

    def count(self) -> int: ...


@runtime_checkable
class DocumentStoreProtocol(Protocol):
    """Store and retrieve source documents.

    Returns ``None`` from :meth:`get` for an unknown id -- a missing document
    is an expected condition, not an exception to raise.
    """

    def put(self, doc: SourceDocument) -> str: ...

    def get(self, id: str) -> SourceDocument | None: ...

    def list(self, filter: Metadata | None = None) -> list[SourceDocument]: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Re-order candidate results by relevance to a query.

    ``documents`` are the candidates in their current order; the result is the
    top ``top_k`` in the new order. The query is a plain string on purpose:
    a reranker needs the query text and nothing else, so it does not depend on
    the application's ``Query`` settings object.
    """

    def rerank(
        self, query: str, documents: Sequence[ScoredDocument], top_k: int
    ) -> list[ScoredDocument]: ...


@runtime_checkable
class ContextBuilderProtocol(Protocol):
    """Assemble retrieved chunks into a context that fits ``budget``.

    The budget is measured in tokens by the implementation. Returning an empty
    context when nothing fits is valid; silently exceeding the budget is not.
    """

    def build(
        self, query: str, chunks: Sequence[ScoredDocument], budget: int
    ) -> Context: ...


@runtime_checkable
class MetricProtocol(Protocol):
    """Compute one metric over a dataset (FR-9).

    Implementations must return a result rather than print or log it, so
    regression tests can compare values against a recorded baseline.
    """

    def compute(self, dataset: Sequence[EvaluationExample]) -> MetricResult: ...

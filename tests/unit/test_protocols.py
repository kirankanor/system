"""The seven component protocols and the shared result type (FR-5, FR-8).

FR-5 says components that share a contract are interchangeable, so a caller
written against a protocol must accept *any* conforming implementation without
being touched. FR-8 says retrieval, reranking and context building pass one
result type between them, so no stage needs an adapter before it can start.

Both claims are about behaviour across boundaries, so every test here drives a
caller written once and runs it against more than one implementation, or pushes
one value through three stages and asserts nothing was converted in between.

This file exercises the component itself, not the registry package, so it
imports nothing from ``rag_code_library`` (AC-13).
"""

from __future__ import annotations

import inspect
import sys
import typing
from pathlib import Path

import pytest

# The gate command is literally `python -m pytest`, and `python` on this host is
# the system interpreter (3.14.5). Only one root is needed: the component's own
# directory, so `core_types` resolves to the copy under test. `core_types.py`
# imports nothing but pydantic, so no sibling component roots go on the path --
# adding them would leak module names into the rest of the session for no
# benefit. Both the component root and the path insertion are derived from this
# file, never from the current working directory, so the gate runs from anywhere.
_COMPONENT_DIR = Path(__file__).resolve().parents[2] / "components" / "storage" / "core_types"
if str(_COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPONENT_DIR))

from core_types import (  # noqa: E402
    Chunk,
    ChunkingProtocol,
    Context,
    ContextBuilderProtocol,
    DocumentStoreProtocol,
    EmbeddingProtocol,
    EvaluationExample,
    MetricProtocol,
    MetricResult,
    RerankerProtocol,
    ScoredDocument,
    VectorStoreProtocol,
)

PROTOCOLS = (
    ChunkingProtocol,
    EmbeddingProtocol,
    VectorStoreProtocol,
    DocumentStoreProtocol,
    RerankerProtocol,
    ContextBuilderProtocol,
    MetricProtocol,
)


# --------------------------------------------------------------------------- #
# FR-5: a caller written once serves every conforming implementation
# --------------------------------------------------------------------------- #


def chunk_with(impl: ChunkingProtocol, text: str) -> list[Chunk]:
    """A caller expressed purely against the protocol. Never changes (FR-5)."""
    return impl.chunk(text, size=6, overlap=2)


class FixedChunker:
    def chunk(self, text: str, *, size: int, overlap: int, **kw: object) -> list[Chunk]:
        return [Chunk(text=text[i : i + size], chunk_index=n)
                for n, i in enumerate(range(0, len(text), size))]


class SlidingChunker:
    """Different algorithm, different internal state -- same contract."""

    def chunk(self, text: str, *, size: int, overlap: int, **kw: object) -> list[Chunk]:
        step = size - overlap
        out: list[Chunk] = []
        start = 0
        while start < len(text):
            out.append(Chunk(text=text[start : start + size], chunk_index=len(out)))
            start += step
        return out


@pytest.mark.parametrize("impl", [FixedChunker(), SlidingChunker()])
def test_one_caller_drives_every_chunker(impl):
    """FR-5: the caller is written against the protocol, not the class."""
    chunks = chunk_with(impl, "abcdefghijklmno")
    assert chunks, "a conforming chunker must produce something"
    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_the_two_chunkers_actually_differ():
    """Otherwise the interchange test above would prove nothing."""
    a = chunk_with(FixedChunker(), "abcdefghijklmno")
    b = chunk_with(SlidingChunker(), "abcdefghijklmno")
    assert [c.text for c in a] != [c.text for c in b]


class FixedBuilder:
    def build(self, query, chunks, budget):
        return Context(text="", chunks=(), budget=budget)


class ProportionalBuilder:
    def build(self, query, chunks, budget):
        kept: list[ScoredDocument] = []
        tokens = 0
        for candidate in sorted(chunks, key=lambda c: -c.score):
            cost = len(candidate.text.split())
            if tokens + cost > budget:
                break
            kept.append(candidate)
            tokens += cost
        return Context(
            text=" ".join(c.text for c in kept),
            chunks=tuple(kept),
            token_count=tokens,
            budget=budget,
            truncated=len(kept) < len(chunks),
        )


@pytest.mark.parametrize("builder", [FixedBuilder(), ProportionalBuilder()])
def test_one_caller_drives_every_context_builder(builder):
    ctx: ContextBuilderProtocol = builder
    result = ctx.build("q", [], budget=10)
    assert isinstance(result, Context)
    assert result.budget == 10


# --------------------------------------------------------------------------- #
# FR-8: one result type flows retrieval -> reranking -> context unchanged
# --------------------------------------------------------------------------- #


def retrieve(candidates: list[Chunk]) -> list[ScoredDocument]:
    return [
        ScoredDocument.from_chunk(
            c,
            score=1.0 / (position + 1),
            document_id="doc-1",
            source_name="source.txt",
        )
        for position, c in enumerate(candidates)
    ]


class ScoreOnlyReranker:
    """Re-ranks by its own idea of relevance and returns the *same* class."""

    def rerank(self, query, documents, top_k):
        return list(reversed(documents))[:top_k]


class BudgetBuilder:
    def build(self, query, chunks, budget):
        kept = tuple(chunks[: max(budget, 0)])
        return Context(
            text=" ".join(c.text for c in kept),
            chunks=kept,
            token_count=sum(len(c.text.split()) for c in kept),
            budget=budget,
            truncated=len(kept) < len(chunks),
        )


def test_scored_document_survives_three_stages_without_conversion():
    """FR-8: no adapter exists anywhere in this pipeline -- by construction."""
    chunks = [Chunk(text=f"passage {i}", chunk_index=i) for i in range(4)]

    retrieved = retrieve(chunks)
    assert all(isinstance(c, ScoredDocument) for c in retrieved)

    reranked = ScoreOnlyReranker().rerank("anything", retrieved, top_k=3)
    # The reranker handed back the very objects it was given, re-ordered --
    # there is no intermediate class to map through.
    assert all(isinstance(c, ScoredDocument) for c in reranked)
    assert set(id(c) for c in reranked) <= set(id(c) for c in retrieved)

    ctx = BudgetBuilder().build("anything", reranked, budget=2)
    assert all(isinstance(c, ScoredDocument) for c in ctx.chunks)


def test_scored_document_carries_exactly_fr8_fields():
    """text, score, source id, metadata -- the four FR-8 names."""
    doc = ScoredDocument(
        text="the passage",
        chunk_index=2,
        score=0.42,
        document_id="doc-7",
        source_name="report.pdf",
        metadata={"page": 3},
    )
    assert doc.text == "the passage"
    assert doc.score == 0.42
    assert doc.document_id == "doc-7"
    assert doc.metadata == {"page": 3}


def test_scored_document_is_a_chunk_so_no_widening_is_needed():
    assert issubclass(ScoredDocument, Chunk)


def test_from_chunk_scores_without_losing_identity():
    chunk = Chunk(text="p", chunk_index=0, document_id="d9", source_name="a.txt")
    doc = ScoredDocument.from_chunk(chunk, score=0.5)
    assert doc.score == 0.5
    assert doc.document_id == "d9" and doc.source_name == "a.txt"


def test_from_chunk_can_name_an_unidentified_chunk():
    """A chunker never knows its document; the retriever supplies it."""
    chunk = Chunk(text="p", chunk_index=0)
    doc = ScoredDocument.from_chunk(
        chunk, score=1.0, document_id="d1", source_name="notes.md"
    )
    assert doc.document_id == "d1"
    assert doc.source_name == "notes.md"


def test_naive_model_dump_spread_collides_but_from_chunk_does_not():
    """Pin the failure that motivated `from_chunk`, so it is not re-discovered.

    Spreading a chunk's dump while also naming an id it already carries raises
    TypeError; `from_chunk` is the supported route and must not.
    """
    chunk = Chunk(text="p", chunk_index=0, document_id="d1")
    with pytest.raises(TypeError):
        ScoredDocument(**chunk.model_dump(), document_id="override", score=1.0)
    assert (
        ScoredDocument.from_chunk(chunk, score=1.0, document_id="override").document_id
        == "override"
    )


def test_score_direction_is_not_encoded_in_the_type():
    """Cosine and BM25 disagree about which end is better; neither is clamped."""
    high = ScoredDocument(text="x", chunk_index=0, score=1e6)
    low = ScoredDocument(text="x", chunk_index=0, score=-1e6)
    assert high.score > low.score


# --------------------------------------------------------------------------- #
# The contract itself: signatures match the specification's protocol table
# --------------------------------------------------------------------------- #


def _params(protocol: typing.Any, method: str) -> list[inspect.Parameter]:
    """Parameters of a protocol method, minus the implicit `self`."""
    params = list(inspect.signature(getattr(protocol, method)).parameters.values())
    return params[1:] if params and params[0].name == "self" else params


def test_all_seven_protocols_exist():
    assert len(PROTOCOLS) == 7
    assert len(set(PROTOCOLS)) == 7


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_protocols_are_runtime_checkable(protocol):
    """isinstance() must work -- callers use it as a capability check.

    Without @runtime_checkable, isinstance against a Protocol raises TypeError
    rather than returning a bool, so the isinstance call below is the proof;
    merely asserting the object is a class would pass either way.
    """
    assert isinstance(object(), protocol) is False


def test_chunking_signature_matches_spec():
    names = [p.name for p in _params(ChunkingProtocol, "chunk")]
    kinds = {p.name: p.kind for p in _params(ChunkingProtocol, "chunk")}
    assert names[0] == "text"
    assert kinds["size"] is inspect.Parameter.KEYWORD_ONLY
    assert kinds["overlap"] is inspect.Parameter.KEYWORD_ONLY
    assert kinds["kw"] is inspect.Parameter.VAR_KEYWORD


def test_embedding_signature_matches_spec():
    assert [p.name for p in _params(EmbeddingProtocol, "embed")] == ["texts"]
    hints = typing.get_type_hints(EmbeddingProtocol.embed)
    assert hints["return"] == list[list[float]]
    # dimensions/model_id are declared members, not optional extras.
    members = set(getattr(EmbeddingProtocol, "__annotations__", {}))
    assert {"dimensions", "model_id"} <= members


def test_vector_store_signature_matches_spec():
    assert [p.name for p in _params(VectorStoreProtocol, "add")] == ["items"]
    assert [p.name for p in _params(VectorStoreProtocol, "query")] == [
        "vector",
        "k",
        "filter",
    ]
    assert [p.name for p in _params(VectorStoreProtocol, "delete")] == ["ids"]
    assert [p.name for p in _params(VectorStoreProtocol, "count")] == []


def test_document_store_signature_matches_spec():
    assert [p.name for p in _params(DocumentStoreProtocol, "put")] == ["doc"]
    assert [p.name for p in _params(DocumentStoreProtocol, "get")] == ["id"]
    assert [p.name for p in _params(DocumentStoreProtocol, "list")] == ["filter"]


def test_reranker_signature_returns_scored_document():
    assert [p.name for p in _params(RerankerProtocol, "rerank")] == [
        "query",
        "documents",
        "top_k",
    ]
    hints = typing.get_type_hints(RerankerProtocol.rerank)
    assert hints["return"] == list[ScoredDocument]


def test_context_builder_signature_returns_context():
    assert [p.name for p in _params(ContextBuilderProtocol, "build")] == [
        "query",
        "chunks",
        "budget",
    ]
    assert typing.get_type_hints(ContextBuilderProtocol.build)["return"] is Context


def test_metric_signature_returns_metric_result():
    assert [p.name for p in _params(MetricProtocol, "compute")] == ["dataset"]
    assert (
        typing.get_type_hints(MetricProtocol.compute)["return"] is MetricResult
    )


# --------------------------------------------------------------------------- #
# The protocols actually discriminate
# --------------------------------------------------------------------------- #


def test_a_bare_object_conforms_to_nothing():
    for protocol in PROTOCOLS:
        assert not isinstance(object(), protocol)


def test_embedder_declares_dimensions_and_model_id_as_real_members():
    """An embedder without them is not an embedder (FR-5).

    Unlike the other six members these are annotations rather than methods,
    so this is what proves `typing` collected them as protocol members at all.
    Without it, isinstance would quietly accept anything with an `embed`
    method and the vector-space contract would be decorative.
    """

    class EmbedOnly:
        def embed(self, texts):
            return [[] for _ in texts]

    assert not isinstance(EmbedOnly(), EmbeddingProtocol)

    class Complete(EmbedOnly):
        dimensions = 8
        model_id = "fake"

    assert isinstance(Complete(), EmbeddingProtocol)


def test_runtime_checkable_checks_presence_only_not_signature_or_type():
    """Limitation of typing.runtime_checkable -- recorded, not worked around.

    isinstance() against these protocols confirms the members *exist* and
    nothing more: it cannot see parameters, return types or field types. An
    object that passes may still be wrong, which is why the signature tests
    above assert the real contract separately rather than relying on this.
    """

    class WrongOnPurpose:
        dimensions = "not-an-int"
        model_id = None

        def embed(self, texts):
            return "not a list of vectors"

        def chunk(self):
            return "not a list of Chunk"

    assert isinstance(WrongOnPurpose(), EmbeddingProtocol)
    assert isinstance(WrongOnPurpose(), ChunkingProtocol)


# --------------------------------------------------------------------------- #
# The contracts hold their own shape (failure paths)
# --------------------------------------------------------------------------- #


def test_context_rejects_a_zero_budget():
    """A zero-token budget means the caller should not be building context."""
    with pytest.raises(ValueError):
        Context(text="", budget=0)


def test_metric_result_requires_a_name():
    with pytest.raises(ValueError):
        MetricResult(name="  ", value=1.0)


def test_evaluation_example_rejects_an_empty_query():
    with pytest.raises(ValueError):
        EvaluationExample(query="   ")


def test_scored_document_rejects_a_negative_index():
    with pytest.raises(ValueError):
        ScoredDocument(text="x", chunk_index=-1, score=1.0)

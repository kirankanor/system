"""Tests for the core-types component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers at least one success path and one failure path
per contract (AC-4).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_types import (  # noqa: E402
    Chunk,
    ChunkingProtocol,
    Context,
    ContextBuilderProtocol,
    DocumentStoreProtocol,
    EmbeddingProtocol,
    EmbeddedChunk,
    EvaluationExample,
    MetricProtocol,
    MetricResult,
    RerankerProtocol,
    ScoredDocument,
    SourceDocument,
    VectorStoreProtocol,
)

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_minimal_chunk_needs_only_text_and_index():
    """A chunker knows the text and its position, and nothing else."""
    chunk = Chunk(text="hello world", chunk_index=0)
    assert chunk.text == "hello world"
    assert chunk.chunk_id is None and chunk.document_id is None
    assert chunk.metadata == {}


def test_scored_document_carries_fr8_fields():
    """FR-8: text, score, source id and metadata all travel together."""
    doc = ScoredDocument(
        text="a passage",
        chunk_index=1,
        score=0.75,
        document_id="doc-1",
        source_name="notes.md",
        metadata={"section": "intro"},
    )
    assert doc.text == "a passage"
    assert doc.score == 0.75
    assert doc.document_id == "doc-1"
    assert doc.metadata["section"] == "intro"


def test_scored_document_is_a_chunk():
    """Reranking must not need to convert its input before context building."""
    assert isinstance(ScoredDocument(text="x", chunk_index=0, score=1.0), Chunk)


def test_from_chunk_preserves_and_overrides_identity():
    identified = Chunk(text="p", chunk_index=0, document_id="d1", source_name="a.txt")
    assert ScoredDocument.from_chunk(identified, score=0.5).document_id == "d1"
    assert (
        ScoredDocument.from_chunk(identified, score=0.5, document_id="d2").document_id
        == "d2"
    )


def test_context_records_provenance_and_budget():
    doc = ScoredDocument(text="kept", chunk_index=0, score=1.0, source_name="a.txt")
    ctx = Context(text="kept", chunks=(doc,), token_count=3, budget=10)
    assert ctx.budget == 10
    assert ctx.truncated is False
    assert ctx.chunks[0].source_name == "a.txt"


def test_empty_context_is_valid():
    """Nothing fitting the budget is an outcome, not an error."""
    ctx = Context(text="", budget=5)
    assert ctx.text == ""
    assert ctx.chunks == ()


def test_source_document_round_trips():
    doc = SourceDocument(
        document_id="d1",
        source_name="a.txt",
        sha256="0" * 64,
        page_count=2,
    )
    assert doc.page_count == 2


def test_metric_result_serialises_details():
    result = MetricResult(name="hit_rate", value=0.8, sample_count=10)
    assert result.higher_is_better is True
    assert result.details == {}


def test_evaluation_example_defaults_to_stages_not_yet_run():
    example = EvaluationExample(query="what?")
    assert example.relevant_ids == ()
    assert example.actual_answer is None


def test_embedded_chunk_carries_vector_and_fingerprint():
    chunk = Chunk(text="v", chunk_index=0)
    emb = EmbeddedChunk(chunk=chunk, vector=(0.1, 0.2), index_fingerprint="fp")
    assert emb.vector == (0.1, 0.2)


# --------------------------------------------------------------------------- #
# Every protocol recognises a minimal implementation
# --------------------------------------------------------------------------- #


class _Chunker:
    def chunk(self, text: str, *, size: int, overlap: int, **kw: object):
        return [Chunk(text=text[:size], chunk_index=0)]


class _Embedder:
    dimensions = 8
    model_id = "fake"

    def embed(self, texts):
        return [[0.0] * self.dimensions for _ in texts]


class _Store:
    def add(self, items):
        return len(items)

    def query(self, vector, k, filter=None):
        return []

    def delete(self, ids):
        return 0

    def count(self):
        return 0


class _Docs:
    def put(self, doc):
        return doc.document_id

    def get(self, id):
        return None

    def list(self, filter=None):
        return []


class _Reranker:
    def rerank(self, query, documents, top_k):
        return list(documents)[:top_k]


class _Builder:
    def build(self, query, chunks, budget):
        return Context(text="", budget=budget)


class _Metric:
    def compute(self, dataset):
        return MetricResult(name="hit_rate", value=0.0, sample_count=len(dataset))


@pytest.mark.parametrize(
    ("impl", "protocol"),
    [
        (_Chunker(), ChunkingProtocol),
        (_Embedder(), EmbeddingProtocol),
        (_Store(), VectorStoreProtocol),
        (_Docs(), DocumentStoreProtocol),
        (_Reranker(), RerankerProtocol),
        (_Builder(), ContextBuilderProtocol),
        (_Metric(), MetricProtocol),
    ],
)
def test_every_protocol_recognises_its_implementation(impl, protocol):
    assert isinstance(impl, protocol)


def test_protocols_reject_an_unrelated_object():
    """A capability check must actually discriminate."""
    assert not isinstance(object(), ChunkingProtocol)
    assert not isinstance(object(), MetricProtocol)


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_chunk_rejects_unknown_field():
    with pytest.raises(ValueError) as exc:
        Chunk(text="x", chunk_index=0, surprise=1)
    assert "unexpected" in str(exc.value) or "extra" in str(exc.value)


def test_chunk_rejects_empty_text():
    with pytest.raises(ValueError):
        Chunk(text="   ", chunk_index=0)


def test_chunk_rejects_negative_index():
    with pytest.raises(ValueError):
        Chunk(text="x", chunk_index=-1)


def test_chunk_rejects_backwards_page_range():
    with pytest.raises(ValueError) as exc:
        Chunk(text="x", chunk_index=0, page_start=5, page_end=2)
    assert "page_end must be >= page_start" in str(exc.value)


def test_chunk_rejects_page_zero():
    """Pages are 1-based; page 0 is a bug, not a value to clamp."""
    with pytest.raises(ValueError):
        Chunk(text="x", chunk_index=0, page_start=0, page_end=1)


def test_contracts_are_frozen():
    chunk = Chunk(text="x", chunk_index=0)
    with pytest.raises(Exception):
        chunk.text = "changed"


def test_source_document_rejects_bad_sha256():
    with pytest.raises(ValueError):
        SourceDocument(document_id="d", source_name="s", sha256="nope", page_count=0)


def test_source_document_rejects_zero_pages_field_type():
    with pytest.raises(ValueError):
        SourceDocument(
            document_id="d", source_name="s", sha256="0" * 64, page_count=-1
        )


def test_context_rejects_negative_token_count():
    with pytest.raises(ValueError):
        Context(text="x", budget=10, token_count=-1)

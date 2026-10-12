"""Tests for the dense-retrieval component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4).

Both dependencies are supplied as fakes honouring the core-types protocols, so
these tests assert the composition rather than any particular backend. The
retriever is constructed in-process; nothing here talks to a service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types, which lives in its own component
# directory under components/storage. The manifest declares it a dependency,
# so `ragkit add` copies it beside this file; here we reach across to the
# sibling so the test runs from the component tree.
sys.path.insert(0, str(COMPONENT.parent.parent / "storage" / "core_types"))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
from dense_retrieval import (  # noqa: E402
    DenseRetriever,
    RetrievalError,
    RetrievalInputError,
)

import pytest  # noqa: E402


class FakeEmbedder:
    """Satisfies EmbeddingProtocol; every text maps to the same vector."""

    dimensions = 2
    model_id = "fake-v1"

    def __init__(self, seen: list[str] | None = None):
        self.seen = seen if seen is not None else []

    def embed(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    """Satisfies VectorStoreProtocol; answers deterministically."""

    def __init__(self, count: int = 2):
        self._count = count
        self.last_query: tuple[Any, int, Any] | None = None

    def add(self, items):
        return len(items)

    def delete(self, ids):
        return len(ids)

    def count(self):
        return self._count

    def query(self, vector, k, filter=None):
        self.last_query = (vector, k, filter)
        return [
            core.ScoredDocument.from_chunk(
                core.Chunk(chunk_index=i, text=f"hit {i}", chunk_id=f"c{i}"),
                score=float(i),
            )
            for i in range(min(k, self._count))
        ]


@pytest.fixture
def retriever():
    return DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(count=2), k=2)


# --------------------------------------------------------------------------- #
# Success paths -- the composition
# --------------------------------------------------------------------------- #


def test_the_dependencies_satisfy_their_protocols(retriever):
    assert isinstance(retriever.embedder, core.EmbeddingProtocol)
    assert isinstance(retriever.store, core.VectorStoreProtocol)


def test_retrieve_returns_the_stores_results(retriever):
    assert [h.chunk_id for h in retriever.retrieve("q")] == ["c0", "c1"]


def test_results_are_scored_documents(retriever):
    for hit in retriever.retrieve("q"):
        assert isinstance(hit, core.ScoredDocument)


def test_the_query_is_embedded_exactly_once():
    seen: list[str] = []
    retriever = DenseRetriever(embedder=FakeEmbedder(seen), store=FakeStore())
    retriever.retrieve("what is a vector store")
    assert seen == ["what is a vector store"]


def test_the_query_vector_is_what_reaches_the_store():
    store = FakeStore(1)
    DenseRetriever(embedder=FakeEmbedder(), store=store).retrieve("q")
    vector, _k, _filter = store.last_query
    assert vector == [1.0, 0.0]


def test_k_defaults_to_the_configured_value(retriever):
    retriever.retrieve("q")
    assert retriever.store.last_query[1] == 2


def test_a_per_call_k_overrides_the_default(retriever):
    retriever.retrieve("q", k=1)
    assert retriever.store.last_query[1] == 1


def test_the_filter_is_passed_through_untouched(retriever):
    retriever.retrieve("q", filter={"lang": "en"})
    assert retriever.store.last_query[2] == {"lang": "en"}


def test_a_blank_filter_passes_through_as_none(retriever):
    retriever.retrieve("q")
    assert retriever.store.last_query[2] is None


def test_the_retriever_exposes_its_wiring(retriever):
    assert retriever.name == "dense"
    assert retriever.embedder.model_id == "fake-v1"
    assert retriever.k == 2


def test_the_repr_names_the_model(retriever):
    assert "fake-v1" in repr(retriever)


# --------------------------------------------------------------------------- #
# Requesting more results than exist
# --------------------------------------------------------------------------- #


def test_asking_for_more_than_the_store_holds_returns_what_exists():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(count=2))
    assert len(retriever.retrieve("q", k=500)) == 2


def test_an_empty_store_returns_an_empty_list():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(count=0))
    assert retriever.retrieve("q") == []


def test_an_over_large_k_is_still_valid():
    """A store may legitimately hold fewer than k; returning what exists is correct."""
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(count=1), k=100)
    assert len(retriever.retrieve("q")) == 1


# --------------------------------------------------------------------------- #
# Failure paths -- the caller
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [0, -1, -10])
def test_a_non_positive_k_is_rejected(k):
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), k=k)
    assert "at least 1" in str(exc.value)


@pytest.mark.parametrize("k", [True, 1.5, "3", None, object()])
def test_a_non_int_k_is_rejected(k):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), k=k)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_a_bad_per_call_k_is_rejected(k):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve("q", k=k)


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "\n"])
def test_a_blank_query_is_rejected(query):
    """An empty string embeds to a real vector and retrieves about nothing."""
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve(query)
    assert "empty" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"], b"q"])
def test_a_non_string_query_is_rejected(query):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve(query)


@pytest.mark.parametrize(
    "embedder,store,label",
    [(object(), FakeStore(), "embedder"), (FakeEmbedder(), object(), "store")],
)
def test_a_dependency_that_is_not_its_protocol_is_rejected(embedder, store, label):
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=embedder, store=store)
    assert label in str(exc.value)
    assert "Protocol" in str(exc.value)


def test_an_embedder_with_an_unusable_dimension_count_is_rejected():
    class NoDimensions:
        model_id = "x"

        def embed(self, texts):
            return []

    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=NoDimensions(), store=FakeStore())
    assert "dimensions" in str(exc.value)


@pytest.mark.parametrize("dimensions", [0, -1, True, "2", 2.0])
def test_a_non_positive_dimension_count_is_rejected(dimensions):
    class Bad:
        model_id = "x"

        def __init__(self):
            self.dimensions = dimensions

        def embed(self, texts):
            return []

    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=Bad(), store=FakeStore())


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), name=name)


# --------------------------------------------------------------------------- #
# Failure paths -- the dependencies
# --------------------------------------------------------------------------- #


def test_an_embedder_returning_the_wrong_number_of_vectors_is_caught():
    class TwoVectors:
        dimensions = 2
        model_id = "bad-v1"

        def embed(self, texts):
            return [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=TwoVectors(), store=FakeStore()).retrieve("q")
    assert "exactly 1" in str(exc.value)


def test_an_embedder_returning_no_vectors_is_caught():
    class NoVectors:
        dimensions = 2
        model_id = "bad-v1"

        def embed(self, texts):
            return []

    with pytest.raises(RetrievalError):
        DenseRetriever(embedder=NoVectors(), store=FakeStore()).retrieve("q")


def test_an_embedder_returning_a_vector_of_the_wrong_width_is_caught():
    class Wide:
        dimensions = 2
        model_id = "wide-v1"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=Wide(), store=FakeStore()).retrieve("q")
    assert "3 wide" in str(exc.value)


def test_a_broken_embedder_is_wrapped_with_its_model_id():
    class Broken:
        dimensions = 2
        model_id = "unreachable-v1"

        def embed(self, texts):
            raise RuntimeError("service is down")

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=Broken(), store=FakeStore()).retrieve("q")
    assert "unreachable-v1" in str(exc.value)
    assert "service is down" in str(exc.value)


def test_a_broken_embedder_keeps_its_cause():
    class Broken:
        dimensions = 2
        model_id = "broken"

        def embed(self, texts):
            raise RuntimeError("nope")

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=Broken(), store=FakeStore()).retrieve("q")
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_broken_store_is_wrapped():
    class Broken(FakeStore):
        def query(self, vector, k, filter=None):
            raise RuntimeError("index corrupt")

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=Broken()).retrieve("q")
    assert "index corrupt" in str(exc.value)


def test_a_store_returning_non_documents_is_rejected():
    class BadStore(FakeStore):
        def query(self, vector, k, filter=None):
            return [{"chunk_id": "c0"}]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=BadStore()).retrieve("q")
    assert "ScoredDocument" in str(exc.value)


def test_a_store_returning_a_non_sequence_is_rejected():
    class BadStore(FakeStore):
        def query(self, vector, k, filter=None):
            return "not a list"

    with pytest.raises(RetrievalError):
        DenseRetriever(embedder=FakeEmbedder(), store=BadStore()).retrieve("q")


def test_a_failure_the_retriever_raises_itself_is_not_double_wrapped():
    class BadStore(FakeStore):
        def query(self, vector, k, filter=None):
            raise RetrievalInputError("vector-store-protocol: k must be >= 1")

    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=BadStore()).retrieve("q")
    assert "vector-store-protocol" in str(exc.value)


# --------------------------------------------------------------------------- #
# Message provenance and the manifest
# --------------------------------------------------------------------------- #


def test_messages_name_the_component():
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve("")
    assert "dense-retrieval" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(RetrievalInputError, RetrievalError)
    assert not issubclass(RetrievalError, ValueError)
    assert not issubclass(RetrievalError, TypeError)


def test_every_failure_is_catchable_under_the_base_class():
    for call in (
        lambda: DenseRetriever(embedder=object(), store=FakeStore()),
        lambda: DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve(""),
        lambda: DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), k=0),
    ):
        with pytest.raises(RetrievalError):
            call()


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []
    assert manifest["category"] == "retrieval"

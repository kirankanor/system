"""Tests for the batch-embeddings component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs -- the provider here is always a fake,
so nothing is networked, flaky or charged for.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports the embedding-protocol contract, which lives in its
# own component directory. The manifest declares it a dependency, so `ragkit
# add` copies it beside this file; here we reach across to the sibling.
sys.path.insert(0, str(COMPONENT.parent / "embedding_protocol"))
sys.path.insert(0, str(COMPONENT))

import batch_embeddings  # noqa: E402
from batch_embeddings import DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE, BatchEmbedder  # noqa: E402
from embedding_protocol import (  # noqa: E402
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProviderError,
)

import pytest  # noqa: E402


class FakeEmbedder:
    """A conforming embedder: three-wide vectors, text length in position 0."""

    dimensions = 3
    model_id = "fake"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 1.0] for text in texts]


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_calls_are_split_and_order_is_preserved():
    provider = FakeEmbedder()
    wrapped = BatchEmbedder(provider, batch_size=2)
    out = wrapped.embed(["aa", "bbb", "cccc", "d"])
    assert provider.calls == [["aa", "bbb"], ["cccc", "d"]]
    assert [vector[0] for vector in out] == [2.0, 3.0, 4.0, 1.0]


def test_a_final_short_batch_is_allowed():
    provider = FakeEmbedder()
    wrapped = BatchEmbedder(provider, batch_size=3)
    out = wrapped.embed(["a", "b", "c", "d", "e"])
    assert provider.calls == [["a", "b", "c"], ["d", "e"]]
    assert len(out) == 5


def test_one_text_per_batch_when_batch_size_is_one():
    provider = FakeEmbedder()
    wrapped = BatchEmbedder(provider, batch_size=1)
    wrapped.embed(["a", "b"])
    assert provider.calls == [["a"], ["b"]]


def test_the_wrapper_delegates_the_declared_space():
    wrapped = BatchEmbedder(FakeEmbedder())
    assert wrapped.dimensions == 3
    assert wrapped.model_id == "fake"


def test_the_wrapper_satisfies_the_embedding_protocol():
    from embedding_protocol import EmbeddingProtocol

    assert isinstance(BatchEmbedder(FakeEmbedder()), EmbeddingProtocol)


def test_the_wrapper_composes_with_another_wrapper():
    """A wrapper around a wrapper is still a conforming embedder."""
    inner = BatchEmbedder(FakeEmbedder(), batch_size=1)
    outer = BatchEmbedder(inner, batch_size=2)
    out = outer.embed(["a", "bb"])
    assert [vector[0] for vector in out] == [1.0, 2.0]


def test_an_empty_list_never_calls_the_provider():
    provider = FakeEmbedder()
    wrapped = BatchEmbedder(provider)
    assert wrapped.embed([]) == []
    assert provider.calls == []


def test_embed_one():
    assert BatchEmbedder(FakeEmbedder()).embed_one("hello") == [5.0, 0.0, 1.0]


def test_the_default_batch_size_is_declared():
    assert DEFAULT_BATCH_SIZE == 32
    assert BatchEmbedder(FakeEmbedder()).batch_size == DEFAULT_BATCH_SIZE


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_a_provider_returning_too_few_vectors_is_rejected():
    """The failure this component exists to catch: silent misalignment."""

    class TooFew(FakeEmbedder):
        def embed(self, texts):
            return [[1.0, 0.0, 1.0]]

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(TooFew()).embed(["a", "b"])
    assert "1 vector(s) for 2 input(s)" in str(exc.value)


def test_a_provider_returning_too_many_vectors_is_rejected():
    class TooMany(FakeEmbedder):
        def embed(self, texts):
            return [[1.0, 0.0, 1.0]] * (len(texts) + 1)

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(TooMany()).embed(["a"])
    assert "2 vector(s) for 1 input(s)" in str(exc.value)


def test_vectors_of_differing_widths_are_rejected():
    class Ragged(FakeEmbedder):
        def embed(self, texts):
            return [[1.0], [1.0, 2.0]]

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(Ragged()).embed(["a", "b"])
    assert "wide" in str(exc.value)


def test_a_width_disagreeing_with_the_declared_one_is_rejected():
    class WrongWidth(FakeEmbedder):  # declares 3
        def embed(self, texts):
            return [[1.0, 2.0] for _ in texts]

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(WrongWidth()).embed(["a"])
    assert "declares 3" in str(exc.value)


def test_a_failing_batch_is_wrapped_with_its_identity():
    class Down(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("quota exhausted")

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(Down()).embed(["a", "b", "c", "d", "e"])
    assert "batch 0" in str(exc.value) and "quota exhausted" in str(exc.value)


def test_the_original_exception_is_chained_not_swallowed():
    class Down(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("quota exhausted")

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(Down()).embed(["a"])
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "quota exhausted" in str(exc.value.__cause__)


def test_a_non_sequence_return_is_rejected():
    class NotAList(FakeEmbedder):
        def embed(self, texts):
            return "not vectors"

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(NotAList()).embed(["a"])
    assert "expected a list" in str(exc.value)


def test_a_non_numeric_vector_value_is_rejected():
    class Letters(FakeEmbedder):
        def embed(self, texts):
            return [["a", "b"] for _ in texts]

    with pytest.raises(EmbeddingProviderError) as exc:
        BatchEmbedder(Letters()).embed(["a"])
    assert "non-numeric" in str(exc.value)


def test_a_vector_that_is_not_a_sequence_is_rejected():
    class Scalar(FakeEmbedder):
        def embed(self, texts):
            return [1.0 for _ in texts]

    with pytest.raises(EmbeddingProviderError):
        BatchEmbedder(Scalar()).embed(["a"])


@pytest.mark.parametrize("batch_size", [0, -1, 2048, MAX_BATCH_SIZE + 1])
def test_an_out_of_range_batch_size_is_rejected(batch_size):
    with pytest.raises(EmbeddingInputError):
        BatchEmbedder(FakeEmbedder(), batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [True, 1.5, "4", None])
def test_a_non_int_batch_size_is_rejected(batch_size):
    with pytest.raises(EmbeddingInputError):
        BatchEmbedder(FakeEmbedder(), batch_size=batch_size)


def test_the_boundary_batch_sizes_are_accepted():
    assert BatchEmbedder(FakeEmbedder(), batch_size=1).batch_size == 1
    assert BatchEmbedder(FakeEmbedder(), batch_size=MAX_BATCH_SIZE).batch_size == (
        MAX_BATCH_SIZE
    )


def test_an_object_without_the_protocol_is_rejected_by_name():
    with pytest.raises(EmbeddingInputError) as exc:
        BatchEmbedder(object())
    assert "dimensions" in str(exc.value) and "model_id" in str(exc.value)


def test_an_object_missing_only_embed_is_rejected():
    class Almost:
        dimensions = 3
        model_id = "x"

    with pytest.raises(EmbeddingInputError) as exc:
        BatchEmbedder(Almost())
    assert "embed" in str(exc.value)


def test_failures_are_catchable_under_the_area_base_class():
    with pytest.raises(EmbeddingError):
        BatchEmbedder(object())


def test_messages_name_the_component():
    with pytest.raises(EmbeddingInputError) as exc:
        BatchEmbedder(FakeEmbedder(), batch_size=0)
    assert "batch-embeddings" in str(exc.value)


def test_the_manifest_declares_the_protocol_dependency():
    """The import below is real, so the manifest must say so."""
    import json

    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert "embedding-protocol" in manifest["dependencies"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []

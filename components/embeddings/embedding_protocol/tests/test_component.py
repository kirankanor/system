"""Tests for the embedding-protocol component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_protocol import (  # noqa: E402
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProtocol,
    EmbeddingProviderError,
    MissingEmbeddingDependency,
    Vector,
    validate_texts,
)

import pytest  # noqa: E402


class TinyEmbedder:
    """A conforming embedder that inherits from nothing."""

    dimensions = 3
    model_id = "tiny"

    def embed(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_a_plain_class_satisfies_the_protocol():
    """Structural: no base class, no registration step."""
    assert isinstance(TinyEmbedder(), EmbeddingProtocol)


def test_a_class_missing_an_attribute_does_not_satisfy_the_protocol():
    class Incomplete:
        dimensions = 3
        model_id = "x"

    assert not isinstance(Incomplete(), EmbeddingProtocol)


def test_an_empty_list_is_accepted():
    """Nothing to embed is an outcome, not an error."""
    assert validate_texts([]) == []


def test_a_tuple_in_comes_out_as_a_list():
    """Callers can index and slice the result without checking its type."""
    out = validate_texts(("a", "b"))
    assert out == ["a", "b"]
    assert isinstance(out, list)


def test_valid_text_passes_through_unchanged():
    texts = ["alpha", "beta", "gamma"]
    assert validate_texts(texts) == texts


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad",
    [None, 3, 1.5, {"a": 1}, b"raw bytes", bytearray(b"raw")],
    ids=["none", "int", "float", "dict", "bytes", "bytearray"],
)
def test_non_sequences_are_rejected(bad):
    with pytest.raises(EmbeddingInputError):
        validate_texts(bad)


def test_a_bare_string_is_rejected_because_it_is_a_sequence_of_characters():
    """The subtle one: 'abc' *is* a Sequence[str], and would embed per letter."""
    with pytest.raises(EmbeddingInputError) as exc:
        validate_texts("hello")
    assert "character" in str(exc.value)
    assert "Wrap it" in str(exc.value), "the message should show the fix"


def test_a_bare_bytes_is_rejected_for_the_same_reason():
    with pytest.raises(EmbeddingInputError):
        validate_texts(b"hello")


def test_an_empty_element_is_rejected_by_position():
    """An empty chunk means the pipeline upstream produced nothing."""
    with pytest.raises(EmbeddingInputError) as exc:
        validate_texts(["ok", "", "later"])
    assert "texts[1]" in str(exc.value)


def test_a_list_containing_an_empty_element_is_not_the_same_as_an_empty_list():
    """The distinction worth pinning: [] is fine, [""] is not."""
    assert validate_texts([]) == []
    with pytest.raises(EmbeddingInputError):
        validate_texts([""])


@pytest.mark.parametrize("bad", [5, 1.5, None, b"b"], ids=["int", "float", "none", "bytes"])
def test_a_non_string_element_is_rejected_by_position(bad):
    with pytest.raises(EmbeddingInputError) as exc:
        validate_texts(["ok", bad])
    assert "texts[1]" in str(exc.value)


def test_the_message_names_the_component():
    """NFR-7: a caller can tell where the complaint came from."""
    with pytest.raises(EmbeddingInputError) as exc:
        validate_texts(None)
    assert "embedding-protocol" in str(exc.value)


def test_every_failure_is_catchable_under_the_one_base_class():
    assert issubclass(EmbeddingInputError, EmbeddingError)
    assert issubclass(EmbeddingProviderError, EmbeddingError)
    with pytest.raises(EmbeddingError):
        validate_texts(None)


def test_the_base_is_distinguishable_from_a_programming_error():
    assert not issubclass(EmbeddingError, ValueError)
    assert not issubclass(EmbeddingError, TypeError)


def test_the_missing_dependency_error_is_an_import_error_not_a_domain_error():
    """FR-3: callers guard optional components with `except ImportError`."""
    assert issubclass(MissingEmbeddingDependency, ImportError)
    assert not issubclass(MissingEmbeddingDependency, EmbeddingError)


def test_vector_is_a_list_of_floats():
    """The alias stays ordinary and assignable -- not a NewType wrapper."""
    value: Vector = [0.1, 0.2]
    assert isinstance(value, list)

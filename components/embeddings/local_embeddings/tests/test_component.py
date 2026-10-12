"""Tests for the local-embeddings component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

`sentence-transformers` is not installed here, and installing it would modify
dependency files, which this library forbids (FR-2). Every encode test
therefore runs against an injected fake model, and the FR-3 guard is proven by
blocking the import so the assertion is about the guard rather than about the
machine running the tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports the embedding-protocol contract, which lives in its
# own component directory. The manifest declares it a dependency, so `ragkit
# add` copies it beside this file; here we reach across to the sibling.
sys.path.insert(0, str(COMPONENT.parent / "embedding_protocol"))
sys.path.insert(0, str(COMPONENT))

import local_embeddings  # noqa: E402
from embedding_protocol import (  # noqa: E402
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProtocol,
    EmbeddingProviderError,
)
from local_embeddings import (  # noqa: E402
    MISSING_DEPENDENCY_MESSAGE,
    MODEL_ENV_VAR,
    LocalEmbedder,
)

import pytest  # noqa: E402


class FakeModel:
    """Stands in for a sentence-transformers model."""

    def __init__(self, dimension=3, out=None):
        self._dimension = dimension
        self._out = out
        self.encoded: list[list[str]] = []

    def get_sentence_embedding_dimension(self):
        return self._dimension

    def encode(self, texts):
        self.encoded.append(list(texts))
        if self._out is not None:
            return self._out
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def _with_model(model, **kwargs):
    embedder = LocalEmbedder(**kwargs)
    embedder._model = model
    return embedder


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_a_model_id_passed_directly_is_used():
    assert LocalEmbedder("explicit").model_id == "explicit"


def test_a_model_id_comes_from_the_environment_when_not_passed():
    embedder = LocalEmbedder(env={MODEL_ENV_VAR: "from-env"})
    assert embedder.model_id == "from-env"


def test_an_explicit_model_id_beats_the_environment():
    embedder = LocalEmbedder("explicit", env={MODEL_ENV_VAR: "from-env"})
    assert embedder.model_id == "explicit"


def test_the_embedder_satisfies_the_embedding_protocol():
    assert isinstance(LocalEmbedder("m"), EmbeddingProtocol)


def test_embed_returns_one_vector_per_text_in_order():
    embedder = _with_model(FakeModel(), model_name="m")
    assert embedder.embed(["a", "bb", "ccc"]) == [
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
    ]


def test_embed_one():
    assert _with_model(FakeModel(), model_name="m").embed_one("hello") == [
        5.0, 0.0, 1.0
    ]


def test_an_empty_input_encodes_nothing():
    model = FakeModel(out=AssertionError("must not be called"))
    assert _with_model(model, model_name="m").embed([]) == []
    assert model.encoded == []


def test_dimensions_come_from_the_model():
    assert _with_model(FakeModel(dimension=7), model_name="m").dimensions == 7


def test_the_model_is_built_once_and_reused():
    embedder = LocalEmbedder("m")
    embedder._model = FakeModel()
    first = embedder.model
    assert embedder.model is first


def test_a_tuple_of_texts_is_accepted():
    assert _with_model(FakeModel(), model_name="m").embed(("a", "bb")) == [
        [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
    ]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_no_model_id_from_either_source_is_rejected():
    """The manifest is explicit that there is no default; this keeps it so."""
    with pytest.raises(EmbeddingInputError) as exc:
        LocalEmbedder(env={})
    assert MODEL_ENV_VAR in str(exc.value)


def test_an_empty_model_id_is_rejected():
    with pytest.raises(EmbeddingInputError):
        LocalEmbedder("", env={})


def test_a_non_string_model_id_is_rejected():
    with pytest.raises(EmbeddingInputError):
        LocalEmbedder(5, env={})


def test_a_model_returning_the_wrong_count_is_rejected():
    model = FakeModel(out=[[1.0, 0.0, 1.0]])
    with pytest.raises(EmbeddingProviderError) as exc:
        _with_model(model, model_name="m").embed(["a", "b"])
    assert "1 vector(s) for 2 text(s)" in str(exc.value)


def test_a_model_returning_a_non_list_is_rejected():
    model = FakeModel(out="not vectors")
    with pytest.raises(EmbeddingProviderError):
        _with_model(model, model_name="m").embed(["a"])


def test_a_non_numeric_vector_value_is_rejected():
    model = FakeModel(out=[["a", "b"]])
    with pytest.raises(EmbeddingProviderError) as exc:
        _with_model(model, model_name="m").embed(["a"])
    assert "non-numeric" in str(exc.value)


def test_a_model_raising_is_wrapped_and_chained():
    class Down(FakeModel):
        def encode(self, texts):
            raise RuntimeError("cuda out of memory")

    with pytest.raises(EmbeddingProviderError) as exc:
        _with_model(Down(), model_name="m").embed(["a"])
    assert "cuda out of memory" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_model_reporting_no_usable_width_is_rejected():
    with pytest.raises(EmbeddingProviderError):
        _with_model(FakeModel(dimension=0), model_name="m").dimensions


def test_a_model_reporting_a_non_int_width_is_rejected():
    with pytest.raises(EmbeddingProviderError):
        _with_model(FakeModel(dimension="3"), model_name="m").dimensions


def test_bad_input_is_rejected_before_the_model_is_touched():
    model = FakeModel(out=AssertionError("must not be called"))
    with pytest.raises(EmbeddingInputError):
        _with_model(model, model_name="m").embed(["ok", ""])


def test_a_bare_string_is_rejected_before_the_model_is_touched():
    model = FakeModel(out=AssertionError("must not be called"))
    with pytest.raises(EmbeddingInputError):
        _with_model(model, model_name="m").embed("hello")


def test_failures_are_catchable_under_the_area_base_class():
    with pytest.raises(EmbeddingError):
        LocalEmbedder(env={})


def test_messages_name_the_component():
    with pytest.raises(EmbeddingInputError) as exc:
        LocalEmbedder(env={})
    assert "local-embeddings" in str(exc.value)


# --------------------------------------------------------------------------- #
# FR-3: the optional dependency is guarded, with the manifest's message
# --------------------------------------------------------------------------- #


def test_the_optional_dependency_is_guarded_with_the_manifest_message(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError.

    Blocked through sys.modules so the assertion holds whether or not this
    machine has sentence-transformers installed.
    """
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = LocalEmbedder("m")
    with pytest.raises(ImportError) as exc:
        embedder.embed(["hello"])
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == MISSING_DEPENDENCY_MESSAGE


def test_the_missing_dependency_error_is_an_import_error():
    from embedding_protocol import MissingEmbeddingDependency

    assert issubclass(MissingEmbeddingDependency, ImportError)
    assert not issubclass(MissingEmbeddingDependency, EmbeddingError)


def test_importing_the_module_does_not_require_the_dependency():
    """The module must import on a machine that never installed it."""
    assert hasattr(local_embeddings, "LocalEmbedder")
    assert local_embeddings.MISSING_DEPENDENCY_MESSAGE == MISSING_DEPENDENCY_MESSAGE


def test_the_manifest_declares_the_dependency_optional():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_packages"] == []
    assert [entry["name"] for entry in manifest["optional_packages"]] == [
        "sentence-transformers"
    ]
    assert "embedding-protocol" in manifest["dependencies"]

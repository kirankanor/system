"""Tests for the cross-encoder-reranker component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

`sentence-transformers` is NOT installed in this repository, and installing it
would modify dependency files -- which this library forbids. So the model is
exercised through an injected stand-in that speaks the slice of
`CrossEncoder` the component calls, and the missing-wheel path is exercised
for real. The real model is unexecuted; the component README says so plainly.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types, which lives in its own component
# directory under components/storage. The manifest declares it a dependency,
# so `ragkit add` copies it beside this file; here we reach across to the
# sibling so the test runs from the component tree.
sys.path.insert(0, str(COMPONENT.parent.parent / "storage" / "core_types"))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
from cross_encoder_reranker import (  # noqa: E402
    MANIFEST_MESSAGE,
    CrossEncoderReranker,
    MissingDependency,
    RerankingError,
    RerankingInputError,
)

import pytest  # noqa: E402


def scored(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    return core.ScoredDocument(chunk_id=cid, text=text, score=0.0, **kwargs)


@pytest.fixture
def candidates():
    return [
        scored("c0", "Python error handling with exceptions", chunk_index=0),
        scored("c1", "Cooking pasta with salt", chunk_index=1),
        scored("c2", "Java exceptions and Python compared", chunk_index=2),
    ]


class FakeCrossEncoder:
    """Speaks the slice of sentence_transformers.CrossEncoder we call."""

    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append([list(pair) for pair in pairs])
        return self._scores


def reranker(scores, **kwargs) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        "cross-encoder/ms-marco", model=FakeCrossEncoder(scores), **kwargs
    )


# --- success paths ------------------------------------------------------- #


def test_the_model_orders_the_candidates(candidates):
    out = reranker([0.9, 0.1, 0.5]).rerank("q", candidates, 3)
    assert [d.chunk_id for d in out] == ["c0", "c2", "c1"]
    assert [d.score for d in out] == [0.9, 0.5, 0.1]


def test_every_candidate_is_scored_then_truncated(candidates):
    fake = FakeCrossEncoder([0.1, 0.9, 0.5])
    CrossEncoderReranker("m", model=fake).rerank("q", candidates, 1)
    assert len(fake.calls[0]) == 3


def test_the_query_is_paired_with_each_text_in_order(candidates):
    fake = FakeCrossEncoder([0.1, 0.2, 0.3])
    CrossEncoderReranker("m", model=fake).rerank("how do I", candidates, 3)
    pairs = fake.calls[0]
    assert [p[0] for p in pairs] == ["how do I"] * 3
    assert [p[1] for p in pairs] == [c.text for c in candidates]


def test_the_document_is_kept_and_only_the_score_replaced(candidates):
    out = reranker([0.9, 0.1, 0.5]).rerank("q", candidates, 3)
    by_id = {d.chunk_id: d for d in out}
    assert by_id["c0"].text == "Python error handling with exceptions"
    assert by_id["c0"].score == 0.9


def test_a_tied_score_breaks_on_chunk_index(candidates):
    out = reranker([0.5, 0.5, 0.5]).rerank("q", candidates, 3)
    assert [d.chunk_id for d in out] == ["c0", "c1", "c2"]


def test_a_tie_does_not_break_on_input_order(candidates):
    forward = [d.chunk_id for d in reranker([0.5] * 3).rerank("q", candidates, 3)]
    backward = [
        d.chunk_id for d in reranker([0.5] * 3).rerank("q", list(reversed(candidates)), 3)
    ]
    assert forward == backward


def test_more_than_exists_returns_what_exists(candidates):
    assert len(reranker([0.1, 0.2, 0.3]).rerank("q", candidates, 100)) == 3


def test_an_empty_candidate_list_is_not_an_error():
    assert reranker([]).rerank("anything", [], 3) == []


def test_an_injected_model_is_not_reloaded(candidates):
    fake = FakeCrossEncoder([0.1, 0.2, 0.3])
    model = CrossEncoderReranker("m", model=fake)
    assert model.is_loaded() is True
    model.rerank("q", candidates, 1)
    model.rerank("q", candidates, 1)
    assert len(fake.calls) == 2  # used, never replaced


def test_the_default_name_carries_the_model_name():
    assert CrossEncoderReranker("ms-marco-MiniLM", model=FakeCrossEncoder([])).name == (
        "cross-encoder:ms-marco-MiniLM"
    )


def test_the_component_satisfies_reranker_protocol(candidates):
    assert isinstance(reranker([0.1, 0.2, 0.3]), core.RerankerProtocol)


# --- the guarded dependency (FR-3) --------------------------------------- #


def test_the_wheel_is_never_a_top_level_import():
    """DECISION-bf2566ea: heavy libraries are never imported at module level."""
    tree = ast.parse(Path(sys.modules[CrossEncoderReranker.__module__].__file__).read_text("utf-8"))
    top_level = set()
    for node in tree.body:  # direct children of the module only
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "sentence_transformers" not in top_level


def test_the_module_imports_with_the_wheel_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert hasattr(sys.modules[CrossEncoderReranker.__module__], "CrossEncoderReranker")


def test_the_reranker_constructs_with_the_wheel_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert CrossEncoderReranker("cross-encoder/ms-marco").is_loaded() is False


def test_a_missing_wheel_raises_import_error(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError):
        CrossEncoderReranker("m").rerank("q", candidates, 2)


def test_the_error_names_the_package(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert isinstance(exc.value, ImportError)
    assert "sentence-transformers" in str(exc.value)


def test_the_message_is_the_manifest_verbatim(candidates, monkeypatch):
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert str(exc.value) == MANIFEST_MESSAGE == manifest["missing_dependency_message"]


def test_the_original_import_error_is_kept_as_the_cause(candidates, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(MissingDependency) as exc:
        CrossEncoderReranker("m").rerank("q", candidates, 2)
    assert isinstance(exc.value.__cause__, ImportError)


def test_the_manifest_declares_it_optional_not_required():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_packages"] == [
        {"name": "sentence-transformers", "specifier": ">=2.2.0"}
    ]
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == ["core-types"]


# --- failure paths ------------------------------------------------------- #


@pytest.mark.parametrize("model_name", ["", "   ", None, 5])
def test_an_unusable_model_name_is_rejected(model_name):
    """No silent default: picking a model is the caller's decision."""
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker(model_name)


@pytest.mark.parametrize("device", [3, ["cpu"], {}])
def test_an_unusable_device_is_rejected(device):
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker("m", device=device)


@pytest.mark.parametrize("name", ["", 5])
def test_an_explicit_unusable_name_is_rejected(name):
    with pytest.raises(RerankingInputError):
        CrossEncoderReranker("m", model=FakeCrossEncoder([]), name=name)


def test_name_none_derives_the_default():
    assert CrossEncoderReranker("m", model=FakeCrossEncoder([]), name=None).name == (
        "cross-encoder:m"
    )


@pytest.mark.parametrize("query", ["", "  ", None, 3])
def test_an_unusable_query_is_rejected(candidates, query):
    with pytest.raises(RerankingInputError):
        reranker([0.1, 0.2, 0.3]).rerank(query, candidates, 1)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "3"])
def test_an_unusable_top_k_is_rejected(candidates, top_k):
    with pytest.raises(RerankingInputError):
        reranker([0.1, 0.2, 0.3]).rerank("q", candidates, top_k)


def test_a_wrong_score_count_is_rejected(candidates):
    """Otherwise candidates would drop out of the ranking silently."""
    with pytest.raises(RerankingError) as exc:
        reranker([0.5]).rerank("q", candidates, 3)
    assert "1 scores for 3 document(s)" in str(exc.value)


def test_a_model_returning_nothing_is_rejected(candidates):
    with pytest.raises(RerankingError):
        reranker([]).rerank("q", candidates, 3)


def test_a_non_numeric_score_is_wrapped(candidates):
    class Bad:
        def predict(self, pairs):
            return ["not", "scores", "here"]

    with pytest.raises(RerankingError):
        CrossEncoderReranker("m", model=Bad()).rerank("q", candidates, 3)


def test_a_broken_model_is_wrapped_and_chained(candidates):
    class Broken:
        def predict(self, pairs):
            raise RuntimeError("cuda out of memory")

    with pytest.raises(RerankingError) as exc:
        CrossEncoderReranker("ms-marco-MiniLM", model=Broken()).rerank("q", candidates, 3)
    assert "ms-marco-MiniLM" in str(exc.value)
    assert "cuda out of memory" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_the_errors_form_a_hierarchy():
    assert issubclass(RerankingInputError, RerankingError)
    assert not issubclass(RerankingError, ValueError)
    assert not issubclass(MissingDependency, RerankingInputError)

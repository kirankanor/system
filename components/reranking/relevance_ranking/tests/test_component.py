"""Tests for the relevance-ranking component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

No optional package is involved -- the scoring is arithmetic in the component,
so these tests run anywhere Python does.
"""

from __future__ import annotations

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
from relevance_ranking import (  # noqa: E402
    DEFAULT_STOPWORDS,
    RankingError,
    RankingInputError,
    RelevanceRanker,
    score_documents,
    tokenize,
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


@pytest.fixture
def ranker():
    return RelevanceRanker()


# --- success paths ------------------------------------------------------- #


def test_the_relevant_candidate_ranks_first(ranker, candidates):
    out = ranker.rerank("python exception", candidates, 3)
    assert out[0].chunk_id in ("c0", "c2")
    assert out[-1].chunk_id == "c1"


def test_scores_stay_in_the_unit_interval(ranker, candidates):
    for document in ranker.score("python exception handling salt", candidates):
        assert 0.0 <= document.score <= 1.0


def test_a_full_match_scores_exactly_one(ranker):
    out = ranker.score(
        "python exception", [scored("c0", "a python exception and another python")]
    )
    assert out[0].score == pytest.approx(1.0)


def test_a_term_shared_by_every_candidate_cannot_separate_them(ranker):
    """'shared' is in all three, so it cannot rank them -- but it still counts."""
    docs = [
        scored("c0", "python shared", chunk_index=0),
        scored("c1", "python shared", chunk_index=1),
        scored("c2", "gamma shared", chunk_index=2),
    ]
    out = {d.chunk_id: d.score for d in ranker.score("python shared", docs)}
    # c0 and c1 cover the whole query; c2 covers only the shared term.
    assert out["c0"] == pytest.approx(1.0)
    assert out["c1"] == pytest.approx(1.0)
    # c2 is above zero -- a shared term raises every floor -- but below the
    # candidates that also cover the discriminating term.
    assert 0.0 < out["c2"] < out["c0"]


def test_score_preserves_input_order_while_rerank_orders(ranker, candidates):
    assert [d.chunk_id for d in ranker.score("python", candidates)] == ["c0", "c1", "c2"]
    assert [d.chunk_id for d in ranker.rerank("python", candidates, 3)] == ["c0", "c2", "c1"]


def test_a_tie_breaks_on_chunk_index(ranker):
    tied = [scored("z", "identical", chunk_index=1), scored("a", "identical", chunk_index=0)]
    assert [d.chunk_id for d in ranker.rerank("identical", tied, 2)] == ["a", "z"]


def test_a_tie_breaks_on_chunk_id_when_indices_tie(ranker):
    tied = [scored("b", "same", chunk_index=0), scored("a", "same", chunk_index=0)]
    assert [d.chunk_id for d in ranker.rerank("same", tied, 2)] == ["a", "b"]


def test_permuting_the_input_does_not_permute_a_tie(ranker, candidates):
    forward = [d.chunk_id for d in ranker.rerank("python exception", candidates, 3)]
    backward = [d.chunk_id for d in ranker.rerank("python exception", list(reversed(candidates)), 3)]
    assert forward == backward


def test_scoring_returns_new_documents(ranker, candidates):
    before = [d.score for d in candidates]
    ranker.score("python", candidates)
    assert [d.score for d in candidates] == before


def test_asking_for_more_than_exists_returns_what_exists(ranker, candidates):
    assert len(ranker.rerank("python", candidates, 100)) == 3


def test_an_empty_candidate_list_is_not_an_error(ranker):
    assert ranker.rerank("anything", [], 3) == []


def test_stopwords_are_dropped_and_overridable():
    assert "the" not in tokenize("the python and the snake")
    assert "the" in DEFAULT_STOPWORDS
    assert tokenize("the python", stopwords=frozenset()) == frozenset({"the", "python"})


def test_the_component_satisfies_reranker_protocol(ranker):
    assert isinstance(ranker, core.RerankerProtocol)


def test_the_function_form_matches_the_method(ranker, candidates):
    assert [d.score for d in score_documents("python", candidates)] == [
        d.score for d in ranker.score("python", candidates)
    ]


# --- failure paths ------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(ranker, candidates, query):
    with pytest.raises(RankingInputError) as exc:
        ranker.rerank(query, candidates, 1)
    assert "relevance-ranking" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(ranker, candidates, query):
    with pytest.raises(RankingInputError):
        ranker.rerank(query, candidates, 1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_rejected(ranker, candidates, top_k):
    with pytest.raises(RankingInputError) as exc:
        ranker.rerank("q", candidates, top_k)
    assert "at least 1" in str(exc.value)


@pytest.mark.parametrize("top_k", [True, 1.5, "3", None])
def test_a_non_int_top_k_is_rejected(ranker, candidates, top_k):
    with pytest.raises(RankingInputError):
        ranker.rerank("q", candidates, top_k)


def test_a_non_document_is_rejected_with_its_position(ranker, candidates):
    with pytest.raises(RankingInputError) as exc:
        ranker.rerank("q", candidates + ["nope"], 3)
    assert "[3]" in str(exc.value)


@pytest.mark.parametrize("bad", ["list", 3, None, {"a": 1}])
def test_a_non_sequence_of_documents_is_rejected(ranker, bad):
    with pytest.raises(RankingInputError):
        ranker.rerank("q", bad, 1)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(RankingInputError):
        RelevanceRanker(name=name)


def test_tokenize_rejects_a_non_string():
    with pytest.raises(RankingInputError):
        tokenize(3)


def test_no_stemming_is_pinned_not_hidden(ranker):
    """'exception' never matches 'exceptions'. Documented, so not a bug."""
    docs = [scored("c0", "handling exceptions gracefully")]
    assert ranker.score("exception", docs)[0].score == 0.0


def test_a_match_is_a_whole_word_not_a_substring(ranker):
    assert ranker.score("python", [scored("c0", "pythonic code")])[0].score == 0.0


def test_the_errors_form_a_hierarchy():
    assert issubclass(RankingInputError, RankingError)
    assert not issubclass(RankingError, ValueError)


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_no_packages():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["category"] == "reranking"

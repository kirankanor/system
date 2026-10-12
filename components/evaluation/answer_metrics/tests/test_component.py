"""Component tests for `answer-metrics`.

Runs standalone: ``python -m pytest components/evaluation/answer_metrics/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the two the component is built around: groundedness is
computed **only** from the evidence you supply, and a term that evidence has
never heard of takes the MAXIMUM weight rather than zero, so fabrication
scores low instead of scoring a perfect 1.0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from answer_metrics import (
    DEFAULT_STOPWORDS,
    AnswerMetrics,
    AnswerMetricsError,
    AnswerMetricsInputError,
    AnswerScore,
    tokenize,
)

#: With exactly one passage of evidence the two IDF weights are closed form:
#: a present term gets ``idf(1,1) = log(4/3)`` and an absent one gets the
#: maximum ``idf(1,0) = log(4)``. "One of two terms matched" is therefore
#: exactly this, not a number we accepted from the component.
ONE_OF_TWO = math.log(4 / 3) / (math.log(4 / 3) + math.log(4))


@pytest.fixture
def am() -> AnswerMetrics:
    return AnswerMetrics()


EVIDENCE = ["alpha beta"]
QUERY = "alpha beta"


# --------------------------------------------------------------------------- #
# Normal cases -- both endpoints, exactly
# --------------------------------------------------------------------------- #


def test_an_answer_that_matches_the_evidence_is_fully_grounded(am):
    assert am.groundedness("alpha beta", EVIDENCE) == 1.0
    assert am.groundedness("beta alpha", EVIDENCE) == 1.0
    assert am.groundedness("alpha", EVIDENCE) == 1.0


def test_an_answer_sharing_nothing_with_the_evidence_is_zero(am):
    assert am.groundedness("gamma delta", EVIDENCE) == 0.0


def test_an_answer_that_addresses_the_question_fully_is_relevant(am):
    assert am.relevance("alpha beta", QUERY) == 1.0
    assert am.relevance("alpha beta gamma", QUERY) == 1.0


def test_an_answer_that_ignores_the_question_is_irrelevant(am):
    assert am.relevance("gamma delta", QUERY) == 0.0


def test_the_partial_case_is_the_closed_form_value(am):
    # One of two terms present, in both directions.
    assert am.groundedness("alpha gamma", ["alpha beta"]) == pytest.approx(ONE_OF_TWO, abs=1e-15)
    assert am.relevance("alpha gamma", "alpha beta") == pytest.approx(ONE_OF_TWO, abs=1e-15)
    assert ONE_OF_TWO == pytest.approx(0.17185550924272538, abs=1e-15)


def test_score_returns_both_metrics_plus_the_term_count(am):
    score = am.score(answer="alpha beta", query="alpha", evidence=EVIDENCE)
    assert isinstance(score, AnswerScore)
    assert score.groundedness == 1.0
    assert score.relevance == 1.0
    assert score.answer_terms == 2
    # Deliberately NOT a single averaged number: averaging would let a
    # faithful answer that ignores the question look like an irrelevant one
    # copied from the evidence -- exactly where a regression likes to hide.
    assert not hasattr(score, "mean")


def test_as_metrics_returns_both_separately(am):
    metrics = am.score(answer="alpha", query="alpha beta", evidence=EVIDENCE).as_metrics()
    assert sorted(metrics) == ["groundedness", "relevance"]
    assert metrics["groundedness"] == 1.0
    # The query asks about alpha AND beta; the answer only covers alpha, so
    # "beta" takes the maximum weight -- the same closed-form partial.
    assert metrics["relevance"] == pytest.approx(ONE_OF_TWO, abs=1e-15)


# --------------------------------------------------------------------------- #
# 1. Groundedness is computed ONLY from the supplied evidence
# --------------------------------------------------------------------------- #


def test_the_same_answer_scores_differently_against_different_evidence(am):
    answer = "alpha gamma"
    assert am.groundedness(answer, ["alpha beta"]) == pytest.approx(ONE_OF_TWO, abs=1e-15)
    assert am.groundedness(answer, ["gamma delta"]) == pytest.approx(ONE_OF_TWO, abs=1e-15)
    assert am.groundedness(answer, ["alpha gamma"]) == 1.0
    assert am.groundedness(answer, ["beta delta"]) == 0.0


def test_the_score_is_deterministic(am):
    answer = "alpha gamma delta"
    evidence = ["alpha beta", "delta omega"]
    runs = {am.groundedness(answer, evidence) for _ in range(50)}
    assert len(runs) == 1


def test_ids_used_as_evidence_cannot_score_well(am):
    # The README says to pass the passages the model saw, not document ids.
    # Nothing here can detect that mistake -- ids are perfectly good strings
    # -- so assert the honest consequence: every answer term is absent from
    # an id-shaped corpus, so the score is 0 and never 1.0.
    assert am.groundedness("alpha", ["doc_0001", "doc_0002"]) == 0.0
    assert am.groundedness("alpha", ["doc_0001"]) == 0.0


# --------------------------------------------------------------------------- #
# 2. The unseen-term rule -- the direction the error fails in
# --------------------------------------------------------------------------- #


def test_a_fabricated_answer_scores_near_zero_not_near_one(am):
    fabricated = am.groundedness(
        "The quantum tunnelling manifold repolarises the substrate.",
        ["The json module serialises objects into JSON text."],
    )
    assert fabricated == 0.0  # every distinctive term is absent


def test_the_unseen_term_takes_the_maximum_weight_not_zero(am):
    # If absent terms weighted 0, this would read 1.0 -- the confabulated
    # answer would win. Assert the direction that makes it fail safe.
    assert am.groundedness("gamma", ["alpha"]) == 0.0
    # A mix keeps a real number between the endpoints, never 1.0.
    mixed = am.groundedness("alpha gamma", ["alpha beta"])
    assert 0.0 < mixed < 1.0
    assert mixed == pytest.approx(ONE_OF_TWO, abs=1e-15)


def test_a_higher_stakes_fabrication_beats_an_honest_paraphrase(am):
    evidence = ["The json module serialises objects into JSON text."]
    honest = am.groundedness("json serialises objects.", evidence)
    invented = am.groundedness("Rust compiles its borrow checker at runtime.", evidence)
    assert honest > invented
    assert honest == 1.0
    assert invented == 0.0


def test_an_unseen_query_term_keeps_relevance_below_one(am):
    # "how" is a stopword so it does not count, but "serialise" vs the
    # answer's "serialised" does: no stemming, so it is genuinely absent.
    score = am.relevance("Objects are serialised by json.", "json serialise objects")
    assert 0.0 < score < 1.0


# --------------------------------------------------------------------------- #
# 3. No stemming, no lemmatising
# --------------------------------------------------------------------------- #


def test_exception_does_not_match_exceptions(am):
    assert am.groundedness("exceptions", ["the exception was caught"]) == 0.0
    assert am.groundedness("exception", ["the exception was caught"]) == 1.0


def test_run_does_not_match_running(am):
    assert am.relevance("running fast", "run fast") == pytest.approx(ONE_OF_TWO, abs=1e-15)


def test_case_folding_is_the_only_folding(am):
    # Lowercased by the tokenizer -- documented, and the one transform applied.
    assert am.groundedness("ALPHA BETA", EVIDENCE) == 1.0


# --------------------------------------------------------------------------- #
# Degenerate answers
# --------------------------------------------------------------------------- #


def test_an_empty_answer_scores_zero_rather_than_raising(am):
    # An evaluation has to be able to score a model that produced nothing;
    # crashing the run on one empty output would be worse than the score.
    assert am.groundedness("", EVIDENCE) == 0.0
    assert am.relevance("", QUERY) == 0.0
    assert am.score(answer="", query=QUERY, evidence=EVIDENCE).answer_terms == 0


def test_a_stopword_only_answer_scores_zero_not_one(am):
    # Vacuously grounded is not grounded. 1.0 here would let an empty string
    # top a leaderboard.
    assert am.groundedness("the and of", EVIDENCE) == 0.0
    assert am.groundedness("the and of", EVIDENCE) != 1.0


def test_scores_stay_inside_the_unit_interval(am):
    for answer in ("", "alpha", "alpha beta gamma", "x y z w", "the a of"):
        for evidence in (["alpha"], ["alpha beta gamma"], ["x y z w"]):
            score = am.groundedness(answer, evidence)
            assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


def test_tokenize_returns_a_distinct_lowercase_set():
    assert tokenize("Alpha alpha ALPHA") == frozenset({"alpha"})
    assert tokenize("hello, world!") == frozenset({"hello", "world"})
    assert tokenize("the and of") == frozenset()


def test_stopwords_are_removed_by_default():
    assert "the" in DEFAULT_STOPWORDS
    # Question words are stopwords: they are what the question is about
    # only in form.
    assert {"what", "when", "where", "which", "who", "how", "why"} <= DEFAULT_STOPWORDS


def test_a_custom_stopword_list_is_accepted():
    assert tokenize("alpha beta", stopwords=frozenset({"alpha"})) == frozenset({"beta"})


def test_repeating_a_word_does_not_weight_it_more(am):
    # The tokenizer returns a set: "alpha alpha alpha" is still one term.
    # Repeating a word three times does not make an answer three times more
    # grounded -- it makes it three times longer.
    assert am.groundedness("alpha alpha alpha", ["alpha"]) == 1.0
    assert am.score(answer="alpha alpha alpha", query="alpha", evidence=["alpha"]).answer_terms == 1


# --------------------------------------------------------------------------- #
# Invalid inputs -- failure paths
# --------------------------------------------------------------------------- #


def test_an_empty_evidence_list_is_refused(am):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        am.groundedness("alpha", [])
    assert "no evidence" in str(excinfo.value)


def test_evidence_with_no_distinctive_terms_is_refused(am):
    # Every answer would score 0 against it, which is a property of the
    # evidence, not of the answer.
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        am.groundedness("alpha", ["the and of"])
    assert "no distinctive terms" in str(excinfo.value)


def test_an_empty_query_is_refused(am):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        am.relevance("alpha", "")
    assert "no distinctive terms" in str(excinfo.value)
    assert "no question" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["text", 42, None])
def test_evidence_that_is_not_a_sequence_is_refused(am, bad):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        am.groundedness("alpha", bad)
    assert "sequence of passages" in str(excinfo.value)
    if isinstance(bad, str):
        assert "single characters" in str(excinfo.value)


@pytest.mark.parametrize("bad", [1, None, ["nested"]])
def test_evidence_containing_non_strings_is_refused(am, bad):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        am.groundedness("alpha", [bad])
    assert "only strings" in str(excinfo.value)


@pytest.mark.parametrize("bad", [["the"], set(), None, 5, {}])
def test_a_stopword_collection_that_is_not_a_frozenset_is_refused(bad):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        AnswerMetrics(stopwords=bad)
    assert "frozenset" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_an_unusable_name_is_refused(bad):
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        AnswerMetrics(name=bad)
    assert "non-empty" in str(excinfo.value)


def test_tokenize_rejects_a_non_string():
    with pytest.raises(AnswerMetricsInputError) as excinfo:
        tokenize(None)
    assert "must be a string" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Documented guarantees
# --------------------------------------------------------------------------- #


def test_the_errors_are_the_components_own_hierarchy():
    assert issubclass(AnswerMetricsInputError, AnswerMetricsError)
    assert not issubclass(AnswerMetricsError, ValueError)


def test_the_component_is_reusable_without_reconfiguration(am):
    assert am.groundedness("alpha", ["alpha"]) == 1.0
    assert am.groundedness("beta", ["alpha"]) == 0.0
    assert am.stopwords is DEFAULT_STOPWORDS


def test_the_repr_is_informative(am):
    assert repr(am) == f"AnswerMetrics(name='answer-metrics', stopwords={len(DEFAULT_STOPWORDS)})"


def test_no_network_or_optional_dependency_is_used():
    import answer_metrics as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for banned in (
        "import openai",
        "import anthropic",
        "import httpx",
        "import socket",
        "import requests",
        "import rag_code_library",
    ):
        assert banned not in source

"""Component tests for `retrieval-metrics`.

Runs standalone: ``python -m pytest components/evaluation/retrieval_metrics/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the one the component is built around: all three metrics are
computed against values worked out on paper, and a degenerate ``k`` is
refused or clamped rather than silently scoring zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from retrieval_metrics import (
    DEFAULT_K,
    QueryScore,
    RetrievalMetrics,
    RetrievalMetricsError,
    RetrievalMetricsInputError,
    RetrievalScore,
    unique_in_order,
)


@pytest.fixture
def m3() -> RetrievalMetrics:
    return RetrievalMetrics(k=3)


# --------------------------------------------------------------------------- #
# Normal cases -- every value hand-computed first
# --------------------------------------------------------------------------- #

# Worked out on paper:
#   A  retrieved c1,c2,c3 / relevant c2  -> rank 2, hit 1, rr 1/2, recall 1/1
#   B  retrieved c5,c6,c7 / relevant c2  -> no hit,  hit 0, rr 0,   recall 0
#   C  retrieved c3,c2     / relevant c1,c2 -> rank 2, hit 1, rr 1/2, recall 1/2
WORKED = [
    (["c1", "c2", "c3"], ["c2"]),
    (["c5", "c6", "c7"], ["c2"]),
    (["c3", "c2"], ["c1", "c2"]),
]


def test_a_relevant_chunk_at_rank_two_scores_half(m3):
    score = m3.score(retrieved=["c1", "c2", "c3"], relevant=["c2"])
    assert score.hit == 1.0
    assert score.reciprocal_rank == 0.5
    assert score.rank == 2
    assert score.recall == 1.0
    assert score.missed is False


def test_a_complete_miss_scores_zero_everywhere(m3):
    score = m3.score(retrieved=["c5", "c6", "c7"], relevant=["c2"])
    assert score.hit == 0.0
    assert score.reciprocal_rank == 0.0
    assert score.rank is None
    assert score.recall == 0.0
    assert score.missed is True


def test_recall_counts_only_the_labels_that_were_found(m3):
    score = m3.score(retrieved=["c3", "c2"], relevant=["c1", "c2"])
    assert score.hit == 1.0
    assert score.reciprocal_rank == 0.5
    assert score.recall == 0.5


def test_rank_is_one_based(m3):
    assert m3.score(retrieved=["target"], relevant=["target"]).reciprocal_rank == 1.0
    assert m3.score(retrieved=["x", "target"], relevant=["target"]).reciprocal_rank == 0.5
    assert m3.score(retrieved=["x", "y", "target"], relevant=["target"]).reciprocal_rank == 1 / 3


def test_the_first_relevant_chunk_decides_the_rank(m3):
    # c2 is relevant at rank 2 and c9 is also relevant at rank 3; MRR takes
    # the FIRST one, not the best-scoring one.
    score = m3.score(retrieved=["c1", "c2", "c9"], relevant=["c2", "c9"])
    assert score.rank == 2
    assert score.reciprocal_rank == 0.5


def test_the_corpus_means_are_the_mean_of_the_parts(m3):
    corpus = m3.score_many(WORKED)
    assert corpus.hit_rate == pytest.approx(2 / 3)
    assert corpus.mrr == pytest.approx(1 / 3)
    assert corpus.recall == pytest.approx(0.5)
    assert corpus.examples == 3
    assert corpus.k == 3
    assert len(corpus.per_query) == 3


def test_the_default_cutoff_is_conventional_not_prescriptive():
    assert DEFAULT_K == 10
    assert RetrievalMetrics().k == 10


def test_missed_queries_is_the_diagnosis(m3):
    corpus = m3.score_many(WORKED)
    assert len(corpus.missed_queries) == 1
    # Only query B missed; the two that hit at rank 2 are not in there.
    assert corpus.missed_queries[0].rank is None
    assert all(query.missed for query in corpus.missed_queries)


def test_as_metrics_is_exactly_the_three_named_means(m3):
    metrics = m3.score_many(WORKED).as_metrics()
    assert sorted(metrics) == ["hit_rate", "mrr", "recall"]
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    # These names are the contract with evaluation-dataset: a rename would
    # silently stop the regression check from seeing a metric at all.
    assert set(metrics) == {"hit_rate", "mrr", "recall"}


def test_a_query_score_is_frozen(m3):
    score = m3.score(retrieved=["c1"], relevant=["c1"])
    with pytest.raises(Exception):
        score.hit = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Degenerate k -- the scenario names this as observable
# --------------------------------------------------------------------------- #


def test_k_past_the_end_of_the_list_is_clamped_not_an_error():
    # recall@100 over a 2-result list is a real question with a real answer.
    score = RetrievalMetrics(k=100).score(retrieved=["only", "two"], relevant=["two"])
    assert score.hit == 1.0
    assert score.rank == 2
    assert score.recall == 1.0


def test_k_narrower_than_the_relevant_set_lowers_recall():
    # Two valid answers exist, but only one fits in the top-1 window: at k=1
    # recall is 1/2, at k=2 it is 2/2. This is why an evaluation must name
    # its cutoff rather than inherit one.
    assert RetrievalMetrics(k=1).score(retrieved=["c1", "c2"], relevant=["c1", "c2"]).recall == 0.5
    assert RetrievalMetrics(k=2).score(retrieved=["c1", "c2"], relevant=["c1", "c2"]).recall == 1.0
    # Neither result fits the top-1 window, so nothing was found at all.
    narrow = RetrievalMetrics(k=1).score(retrieved=["x", "c1"], relevant=["c1", "c2"])
    assert narrow.recall == 0.0
    assert narrow.hit == 0.0


def test_an_empty_retrieval_with_a_label_is_a_miss(m3):
    # A real outcome, not a mistake: the retriever returned nothing.
    score = m3.score(retrieved=[], relevant=["c1"])
    assert score.hit == 0.0
    assert score.reciprocal_rank == 0.0
    assert score.recall == 0.0
    assert score.missed is True


@pytest.mark.parametrize("k", [0, -1, -100])
def test_a_cutoff_below_one_is_refused(k):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=k)
    assert "at least 1" in str(excinfo.value)


def test_the_cutoff_error_explains_why_zero_is_meaningless():
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=0)
    message = str(excinfo.value)
    assert "inspect no results" in message
    assert "say nothing about the retriever" in message


@pytest.mark.parametrize(("k", "kind"), [(1.5, "float"), (True, "bool"), ("10", "str"), (None, "NoneType")])
def test_a_cutoff_that_is_not_an_int_is_refused(k, kind):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=k)
    assert "must be an int" in str(excinfo.value)
    assert kind in str(excinfo.value)


def test_a_bool_is_not_a_cutoff_even_though_it_is_an_int_subclass():
    # True == 1, so without this check k=True would silently mean "top 1".
    with pytest.raises(RetrievalMetricsInputError):
        RetrievalMetrics(k=True)


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


def test_a_duplicate_id_occupies_one_slot(m3):
    # Deduplication happens BEFORE the cutoff, so padding cannot buy extra
    # chances either.
    padded = m3.score(retrieved=["c1", "c1", "c1", "c2"], relevant=["c2"])
    assert padded.reciprocal_rank == 0.5  # c2 sits at position 2 after dedup
    assert padded.rank == 2


def test_a_retriever_cannot_inflate_recall_by_repeating_one_chunk(m3):
    honest = m3.score(retrieved=["c1", "c2"], relevant=["c1"])
    repeated = m3.score(retrieved=["c1", "c1", "c1", "c1"], relevant=["c1"])
    assert honest.recall == 1.0
    assert repeated.recall == 1.0  # not 4.0, and not > 1.0


def test_a_repeated_label_is_still_one_target(m3):
    assert m3.score(retrieved=["c1", "c2"], relevant=["c1", "c1"]).recall == 1.0


def test_unique_in_order_is_the_exported_rule():
    assert unique_in_order(["b", "a", "b", "c", "a"]) == ("b", "a", "c")
    assert unique_in_order([]) == ()
    assert unique_in_order(["x"]) == ("x",)


# --------------------------------------------------------------------------- #
# score_many input shapes
# --------------------------------------------------------------------------- #


def test_a_mapping_example_is_accepted(m3):
    # The shape a JSON dataset round-trips to, so loading and scoring needs
    # no adapter.
    corpus = m3.score_many([{"retrieved": ["c1"], "relevant": ["c1"]}])
    assert corpus.hit_rate == 1.0


def test_meaning_does_not_depend_on_example_order(m3):
    forward = m3.score_many(WORKED)
    backward = m3.score_many(list(reversed(WORKED)))
    assert (backward.hit_rate, backward.mrr, backward.recall) == (
        forward.hit_rate,
        forward.mrr,
        forward.recall,
    )


def test_the_corpora_are_reported_identically(m3):
    assert repr(m3) == "RetrievalMetrics(k=3, name='retrieval-metrics')"


# --------------------------------------------------------------------------- #
# Invalid inputs -- failure paths
# --------------------------------------------------------------------------- #


def test_an_empty_label_is_refused(m3):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score(retrieved=["a"], relevant=[])
    assert "relevant was empty" in str(excinfo.value)
    # It explains the failure mode rather than just refusing.
    assert "unlabelled query" in str(excinfo.value)


@pytest.mark.parametrize(("label", "value"), [("retrieved", "abc"), ("relevant", "abc")])
def test_a_bare_string_is_refused_because_it_is_a_list_of_characters(m3, label, value):
    kwargs = {"retrieved": ["a"], "relevant": ["a"]}
    kwargs[label] = value
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score(**kwargs)
    assert "single characters" in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 42, b"bytes"])
def test_a_retrieved_that_is_not_a_sequence_of_ids_is_refused(m3, bad):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score(retrieved=bad, relevant=["a"])
    assert "sequence of ids" in str(excinfo.value)


@pytest.mark.parametrize("bad", [1, None, ["nested"], 1.5])
def test_an_id_that_is_not_a_string_is_refused(m3, bad):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score(retrieved=[bad], relevant=["a"])
    assert "only id strings" in str(excinfo.value)


def test_an_empty_corpus_is_refused(m3):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score_many([])
    assert "no examples to score" in str(excinfo.value)


def test_a_mapping_missing_a_key_names_the_key(m3):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score_many([{"retrieved": ["a"]}])
    assert "'relevant'" in str(excinfo.value)


def test_a_pair_of_the_wrong_length_is_refused(m3):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        m3.score_many([("only-one",)])
    assert "must be a" in str(excinfo.value)


def test_an_empty_name_is_refused():
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(name="")
    assert "non-empty" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Documented guarantees
# --------------------------------------------------------------------------- #


def test_the_errors_are_the_components_own_hierarchy():
    assert issubclass(RetrievalMetricsInputError, RetrievalMetricsError)
    assert not issubclass(RetrievalMetricsError, ValueError)
    assert not issubclass(RetrievalMetricsError, KeyError)


def test_no_network_or_optional_dependency_is_used():
    import retrieval_metrics as module

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


def test_the_types_are_the_documented_ones(m3):
    corpus = m3.score_many(WORKED)
    assert isinstance(corpus, RetrievalScore)
    assert isinstance(corpus.per_query[0], QueryScore)

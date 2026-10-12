"""Tests for the sparse-retrieval component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4).

BM25 is implemented in this component, so these tests need no optional
package. One test blocks the `rank_bm25` import to prove the results do not
depend on it, and one asserts the manifest declares no optional packages.
"""

from __future__ import annotations

import json
import math
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
from sparse_retrieval import (  # noqa: E402
    DEFAULT_STOPWORDS,
    SparseRetriever,
    RetrievalError,
    RetrievalInputError,
    tokenize,
)

import pytest  # noqa: E402


@pytest.fixture
def corpus():
    return [
        core.Chunk(chunk_index=0, text="Python error handling with exceptions",
                   chunk_id="c0", source_name="a.md"),
        core.Chunk(chunk_index=1, text="Java error handling compared to Python",
                   chunk_id="c1", source_name="b.md"),
        core.Chunk(chunk_index=2, text="Cooking pasta with salt",
                   chunk_id="c2", source_name="c.md"),
    ]


@pytest.fixture
def retriever(corpus):
    return SparseRetriever(corpus, k=5)


# --------------------------------------------------------------------------- #
# Success paths -- ranking
# --------------------------------------------------------------------------- #


def test_the_document_matching_more_query_terms_ranks_first(retriever):
    assert [h.chunk_id for h in retriever.retrieve("python error")] == ["c0", "c1"]


def test_a_single_term_query_ranks_its_documents(retriever):
    assert [h.chunk_id for h in retriever.retrieve("python")] == ["c0", "c1"]


def test_an_unrelated_document_is_excluded(retriever):
    assert [h.chunk_id for h in retriever.retrieve("pasta")] == ["c2"]


def test_results_are_scored_documents(retriever):
    for hit in retriever.retrieve("python"):
        assert isinstance(hit, core.ScoredDocument)


def test_provenance_survives(retriever):
    """The chunk already knows its origin, so it is preserved, not overwritten."""
    assert [h.source_name for h in retriever.retrieve("pasta")] == ["c.md"]


def test_text_survives(retriever):
    assert retriever.retrieve("pasta")[0].text == "Cooking pasta with salt"


def test_scores_are_positive_and_finite(retriever):
    for hit in retriever.retrieve("python error"):
        assert hit.score > 0.0
        assert math.isfinite(hit.score)


def test_the_k_limit_is_honoured(corpus):
    retriever = SparseRetriever(corpus, k=1)
    assert len(retriever.retrieve("python error")) == 1


def test_a_per_call_k_overrides_the_default(retriever):
    assert len(retriever.retrieve("python", k=1)) == 1


def test_len_reports_the_corpus_size(retriever):
    assert len(retriever) == 3


def test_average_document_length_is_exposed(retriever):
    assert retriever.average_document_length == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Determinism -- the property the task names
# --------------------------------------------------------------------------- #


def test_the_same_query_returns_the_same_order_every_time(retriever):
    first = [h.chunk_id for h in retriever.retrieve("error handling")]
    for _ in range(10):
        assert [h.chunk_id for h in retriever.retrieve("error handling")] == first


def test_permuting_the_corpus_does_not_change_the_ranking(corpus):
    """Ties break on the document's own index, not on input position."""
    forward = [h.chunk_id for h in SparseRetriever(corpus).retrieve("python")]
    backward = [h.chunk_id for h in SparseRetriever(list(reversed(corpus))).retrieve("python")]
    assert forward == backward


def test_ties_break_on_chunk_index_then_id():
    tied = [
        core.Chunk(chunk_index=1, text="same words here", chunk_id="b"),
        core.Chunk(chunk_index=0, text="same words here", chunk_id="a"),
    ]
    assert [h.chunk_id for h in SparseRetriever(tied).retrieve("same words")] == ["a", "b"]


def test_identical_documents_rank_by_id():
    tied = [
        core.Chunk(chunk_index=0, text="identical text", chunk_id="z"),
        core.Chunk(chunk_index=0, text="identical text", chunk_id="a"),
    ]
    assert [h.chunk_id for h in SparseRetriever(tied).retrieve("identical")] == ["a", "z"]


# --------------------------------------------------------------------------- #
# Requesting more results than exist
# --------------------------------------------------------------------------- #


def test_asking_for_more_than_matches_returns_what_exists(retriever):
    assert len(retriever.retrieve("pasta", k=100)) == 1


def test_a_query_no_document_claims_returns_empty(retriever):
    assert retriever.retrieve("xylophone") == []


def test_an_empty_corpus_returns_empty():
    assert SparseRetriever([]).retrieve("anything") == []


def test_an_empty_corpus_reports_zero_size():
    assert len(SparseRetriever([])) == 0


def test_an_empty_corpus_has_no_average_length():
    assert SparseRetriever([]).average_document_length == 0.0


def test_a_document_that_tokenizes_to_nothing_is_tolerated():
    """Punctuation-only text is constructible and yields a zero-length document."""
    retriever = SparseRetriever(
        [
            core.Chunk(chunk_index=0, text="---", chunk_id="punct"),
            core.Chunk(chunk_index=1, text="python", chunk_id="real"),
        ]
    )
    assert [h.chunk_id for h in retriever.retrieve("python")] == ["real"]


def test_a_zero_length_document_does_not_divide_by_zero():
    """avgdl can be small; the length normalisation must stay finite."""
    retriever = SparseRetriever(
        [
            core.Chunk(chunk_index=0, text="...", chunk_id="punct"),
            core.Chunk(chunk_index=1, text="python", chunk_id="real"),
        ]
    )
    for hit in retriever.retrieve("python"):
        assert math.isfinite(hit.score)


# --------------------------------------------------------------------------- #
# Tokenization and IDF
# --------------------------------------------------------------------------- #


def test_tokenize_lowercases():
    assert tokenize("Python PYTHON python") == ["python", "python", "python"]


def test_tokenize_keeps_numbers():
    assert "42" in tokenize("error code 42")


def test_tokenize_drops_english_stopwords():
    assert "the" not in tokenize("the and of a")


def test_tokenize_keeps_content_words():
    assert tokenize("the python and the snake") == ["python", "snake"]


def test_stopwords_are_a_documented_constant():
    assert "the" in DEFAULT_STOPWORDS


def test_stopwords_can_be_overridden():
    assert tokenize("the python", stopwords=frozenset()) == ["the", "python"]


def test_tokenize_rejects_a_non_string():
    with pytest.raises(RetrievalInputError):
        tokenize(3)


def test_a_rare_term_has_a_higher_idf_than_a_common_one(retriever):
    assert retriever.idf("pasta") > retriever.idf("error")


def test_a_term_no_document_contains_has_the_unseen_idf(retriever):
    assert retriever.idf("xylophone") > 0.0


def test_idf_is_positive_even_for_a_term_in_every_document():
    """The +1 smoothing; without it this would be negative."""
    every = [
        core.Chunk(chunk_index=0, text="python", chunk_id="a"),
        core.Chunk(chunk_index=1, text="python", chunk_id="b"),
        core.Chunk(chunk_index=2, text="python", chunk_id="c"),
    ]
    assert SparseRetriever(every).idf("python") > 0.0


def test_idf_rejects_a_non_string():
    with pytest.raises(RetrievalInputError):
        SparseRetriever([]).idf(3)


# --------------------------------------------------------------------------- #
# The length parameter, demonstrated
# --------------------------------------------------------------------------- #


def _short_and_long():
    return [
        core.Chunk(chunk_index=0, text="python", chunk_id="short"),
        core.Chunk(chunk_index=1,
                   text=("filler words here and there " * 2) + "python python",
                   chunk_id="long"),
    ]


def test_length_normalisation_prefers_the_short_document():
    assert [h.chunk_id for h in SparseRetriever(_short_and_long()).retrieve("python")][0] == "short"


def test_turning_normalisation_off_lets_the_long_document_win():
    """Proof that b is doing the work, not the assertion being vacuous."""
    assert [h.chunk_id for h in SparseRetriever(_short_and_long(), b=0.0).retrieve("python")][0] == "long"


def test_the_short_document_is_not_simply_preferred_for_being_first():
    """Same corpus, reversed order: the ranking must not follow the input."""
    reversed_corpus = list(reversed(_short_and_long()))
    assert [h.chunk_id for h in SparseRetriever(reversed_corpus).retrieve("python")][0] == "short"


def test_full_normalisation_at_b_one_also_prefers_the_short_document():
    assert [h.chunk_id for h in SparseRetriever(_short_and_long(), b=1.0).retrieve("python")][0] == "short"


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("b", [-0.1, 1.1, 2.0])
def test_an_out_of_range_b_is_rejected(b):
    with pytest.raises(RetrievalInputError) as exc:
        SparseRetriever([], b=b)
    assert "between 0" in str(exc.value)


@pytest.mark.parametrize("k1", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_an_unusable_k1_is_rejected(k1):
    with pytest.raises(RetrievalInputError):
        SparseRetriever([], k1=k1)


@pytest.mark.parametrize("b", [True, "0.5", None])
def test_a_non_numeric_b_is_rejected(b):
    with pytest.raises(RetrievalInputError):
        SparseRetriever([], b=b)


def test_a_non_chunk_in_the_corpus_is_rejected_by_position():
    with pytest.raises(RetrievalInputError) as exc:
        SparseRetriever(
            [core.Chunk(chunk_index=0, text="x", chunk_id="c0"), "not a chunk"]
        )
    assert "[1]" in str(exc.value)


@pytest.mark.parametrize("documents", ["corpus", 3, None, {"a": 1}])
def test_a_non_sequence_corpus_is_rejected(documents):
    with pytest.raises(RetrievalInputError):
        SparseRetriever(documents)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(retriever, query):
    with pytest.raises(RetrievalInputError):
        retriever.retrieve(query)


@pytest.mark.parametrize("query", [None, 3, ["q"], b"q"])
def test_a_non_string_query_is_rejected(retriever, query):
    with pytest.raises(RetrievalInputError):
        retriever.retrieve(query)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_an_unusable_per_call_k_is_rejected(retriever, k):
    with pytest.raises(RetrievalInputError):
        retriever.retrieve("python", k=k)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_an_unusable_constructor_k_is_rejected(k):
    with pytest.raises(RetrievalInputError):
        SparseRetriever([], k=k)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(RetrievalInputError):
        SparseRetriever([], name=name)


# --------------------------------------------------------------------------- #
# No optional dependency, and FR-3
# --------------------------------------------------------------------------- #


def test_the_manifest_declares_no_optional_packages():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == ["core-types"]


def test_ranking_is_unaffected_by_the_optional_library_being_blocked(monkeypatch):
    """FR-3: nothing here imports rank_bm25, so blocking it changes nothing."""
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    retriever = SparseRetriever(
        [
            core.Chunk(chunk_index=0, text="python error handling", chunk_id="c0"),
            core.Chunk(chunk_index=1, text="python typing", chunk_id="c1"),
        ]
    )
    assert [h.chunk_id for h in retriever.retrieve("python error")] == ["c0", "c1"]


def test_the_module_never_imports_the_optional_library():
    """Checked on the AST, so a docstring that merely discusses it passes."""
    import ast

    path = Path(sys.modules[SparseRetriever.__module__].__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "rank_bm25" not in imported


# --------------------------------------------------------------------------- #
# Message provenance and the hierarchy
# --------------------------------------------------------------------------- #


def test_messages_name_the_component():
    with pytest.raises(RetrievalInputError) as exc:
        SparseRetriever([], k1=-1)
    assert "sparse-retrieval" in str(exc.value)


def test_the_errors_are_their_own_hierarchy():
    """Independent of dense-retrieval's, so this component copies out alone."""
    assert issubclass(RetrievalInputError, RetrievalError)
    assert not issubclass(RetrievalError, ValueError)


def test_every_failure_is_catchable_under_the_base_class():
    for call in (
        lambda: SparseRetriever([]).retrieve(""),
        lambda: SparseRetriever([], b=9),
        lambda: SparseRetriever(["x"]),
    ):
        with pytest.raises(RetrievalError):
            call()

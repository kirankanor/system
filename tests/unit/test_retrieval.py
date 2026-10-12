"""Gate tests for the retrieval area (T-09a).

Five components: dense, sparse, hybrid, metadata filter and result
normalizer. They compose rather than nest -- each works on the ``ScoredDocument``
type from ``core-types`` and none imports another retrieval component, so any
one of them can be copied out on its own.

The four properties the task names as observable are asserted here by name:

1. ranked results are deterministic,
2. requesting more results than exist is handled,
3. mixed-source scores are normalized,
4. an invalid filter operator raises a typed error.

The sparse component implements BM25 itself rather than importing
``rank-bm25``; its manifest declares no optional packages, and that is
asserted so the claim cannot drift.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL = ROOT / "components" / "retrieval"
CORE_TYPES = ROOT / "components" / "storage" / "core_types"

for directory in (
    CORE_TYPES,
    RETRIEVAL / "result_normalizer",
    RETRIEVAL / "metadata_filter",
    RETRIEVAL / "dense",
    RETRIEVAL / "sparse",
    RETRIEVAL / "hybrid",
):
    sys.path.insert(0, str(directory))

import core_types as core  # noqa: E402
from dense_retrieval import (  # noqa: E402
    DenseRetriever,
    RetrievalError,
    RetrievalInputError,
)
from hybrid_retrieval import (  # noqa: E402
    HybridError,
    HybridInputError,
    SOURCE_KEY,
    fuse,
)
from metadata_filter import (  # noqa: E402
    OPERATORS,
    MetadataFilterError,
    MetadataFilterInputError,
    apply_to_chunks,
    filter_documents,
)
from retrieval_result_normalizer import (  # noqa: E402
    METHODS,
    NormalizationError,
    NormalizationInputError,
    normalize_scores,
)
from sparse_retrieval import (  # noqa: E402
    SparseRetriever,
    tokenize,
)
# dense and sparse each carry their own copy of the retrieval error
# hierarchy, so a component stays copy-ready. The names collide on purpose,
# and the tests below assert against the hierarchy the component under test
# actually raises.
from sparse_retrieval import (  # noqa: E402
    RetrievalError as SparseRetrievalError,
    RetrievalInputError as SparseRetrievalInputError,
)

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures and test doubles
# --------------------------------------------------------------------------- #


def scored(cid: str, score: float, **kwargs: Any) -> core.ScoredDocument:
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("text", cid)
    kwargs.setdefault("chunk_index", 0)
    return core.ScoredDocument(chunk_id=cid, score=score, **kwargs)


@pytest.fixture
def documents():
    return [
        scored("c0", 1.0, source_name="a.md", chunk_index=0),
        scored("c1", 3.0, source_name="b.md", chunk_index=1),
        scored("c2", 5.0, source_name="c.md", chunk_index=2),
    ]


@pytest.fixture
def tagged():
    """Documents carrying the metadata the filter tests exercise."""
    return [
        scored(
            "c0", 1.0, source_name="a.md", chunk_index=0,
            metadata={"lang": "en", "page": 1, "tags": "api rest"},
        ),
        scored(
            "c1", 2.0, source_name="b.md", chunk_index=1,
            metadata={"lang": "fr", "page": 9, "tags": "api"},
        ),
        scored(
            "c2", 3.0, source_name="c.md", chunk_index=2,
            metadata={"lang": "en", "page": 5, "tags": ""},
        ),
    ]


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

    def add(self, items):
        return len(items)

    def delete(self, ids):
        return len(ids)

    def count(self):
        return self._count

    def query(self, vector, k, filter=None):
        # Behaves like a real store: returns at most what it holds.
        return [
            core.ScoredDocument.from_chunk(
                core.Chunk(chunk_index=i, text=f"hit {i}", chunk_id=f"c{i}"),
                score=float(i),
            )
            for i in range(min(k, self._count))
        ]


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


# --------------------------------------------------------------------------- #
# Observable 3: mixed-source scores are normalized
# --------------------------------------------------------------------------- #


def test_min_max_maps_onto_zero_to_one_with_the_best_at_one(documents):
    assert [round(d.score, 4) for d in normalize_scores(documents)] == [0.0, 0.5, 1.0]


def test_distance_like_scores_are_inverted_so_the_best_still_wins(documents):
    """Cosine distance is lower-is-better; the normalizer corrects direction."""
    assert [round(d.score, 4) for d in normalize_scores(documents, higher_is_better=False)] == [1.0, 0.5, 0.0]


def test_a_constant_list_does_not_become_nan():
    """The min-max divide-by-zero, which a naive implementation ships."""
    out = normalize_scores([scored("c0", 2.0), scored("c1", 2.0)])
    assert [d.score for d in out] == [0.5, 0.5]
    assert all(math.isfinite(d.score) for d in out)


def test_a_constant_list_under_z_score_is_zero():
    out = normalize_scores(
        [scored("c0", 2.0), scored("c1", 2.0), scored("c2", 2.0)], method="z_score"
    )
    assert [d.score for d in out] == [0.0, 0.0, 0.0]


def test_equal_scores_share_an_average_rank():
    """Two tied documents must not come out differently ranked."""
    out = normalize_scores([scored("c0", 1.0), scored("c1", 1.0)], method="rank")
    assert out[0].score == out[1].score == 0.5


def test_a_single_document_is_the_best_and_only():
    """Each method reports 'no distinction' differently; all are finite."""
    single = [scored("c0", 9.0)]
    # min_max has no range to map across, so it reports the neutral midpoint.
    assert normalize_scores(single)[0].score == 0.5
    # rank has no competition, so the only document is the best.
    assert normalize_scores(single, method="rank")[0].score == 1.0
    # z-score has no deviation to measure, so it reports none.
    assert normalize_scores(single, method="z_score")[0].score == 0.0


def test_rank_uses_position_not_magnitude():
    """One far-away outlier must not flatten everything else."""
    outlier = [scored("c0", 0.0), scored("c1", 0.1), scored("c2", 1000.0)]
    ranked = [round(d.score, 4) for d in normalize_scores(outlier, method="rank")]
    assert ranked == [0.0, 0.5, 1.0]
    minmax = [round(d.score, 4) for d in normalize_scores(outlier, method="min_max")]
    assert minmax[1] < 0.001  # the outlier's effect, exactly


def test_z_score_is_unbounded_and_centred(documents):
    out = normalize_scores(documents, method="z_score")
    assert round(sum(d.score for d in out), 9) == 0.0
    assert any(d.score < 0 for d in out)


def test_normalizing_preserves_order_and_does_not_mutate(documents):
    original = [d.score for d in documents]
    out = normalize_scores(documents)
    assert [d.chunk_id for d in out] == [d.chunk_id for d in documents]
    assert [d.score for d in documents] == original  # ScoredDocument is frozen


def test_an_empty_result_set_normalizes_to_itself():
    assert normalize_scores([]) == []


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_survives_a_single_and_a_tie(method):
    assert normalize_scores([scored("c0", 3.0)], method=method)
    assert normalize_scores([scored("c0", 3.0), scored("c1", 3.0)], method=method)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_rejected_before_it_poisons_the_others(value):
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0), scored("c1", value)])
    assert "result-normalizer" in str(exc.value)


def test_an_unknown_method_lists_the_valid_ones():
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0)], method="softmax")
    assert "softmax" in str(exc.value)
    for method in METHODS:
        assert method in str(exc.value)


@pytest.mark.parametrize("bad", ["docs", 3, None, {1: 2}])
def test_a_non_sequence_of_documents_is_rejected(bad):
    with pytest.raises(NormalizationInputError):
        normalize_scores(bad)


def test_a_non_document_in_the_list_is_rejected_by_position():
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0), core.Chunk(chunk_index=0, text="x")])
    assert "[1]" in str(exc.value)


def test_the_normalizer_errors_form_a_hierarchy():
    assert issubclass(NormalizationInputError, NormalizationError)
    assert not issubclass(NormalizationError, ValueError)


# --------------------------------------------------------------------------- #
# Observable 4: an invalid filter operator raises a typed error
# --------------------------------------------------------------------------- #


def test_an_unknown_operator_raises_a_typed_error_naming_the_operator(tagged):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(tagged, {"page": {"gtr": 3}})
    assert isinstance(exc.value, MetadataFilterError)
    assert "'gtr'" in str(exc.value)


def test_the_unknown_operator_message_lists_every_valid_operator(tagged):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(tagged, {"page": {"nope": 3}})
    for operator in OPERATORS:
        assert operator in str(exc.value)


def test_an_unknown_operator_is_not_silently_treated_as_a_mismatch(tagged):
    """The failure this guards: a typo returning an empty, 'correct' list."""
    with pytest.raises(MetadataFilterInputError):
        filter_documents(tagged, {"page": {"gtr": 3}})
    # A genuine no-match is an empty list, so a typo and a real miss must
    # never look the same to the caller.
    assert filter_documents(tagged, {"page": {"gt": 100}}) == []


def test_a_non_string_operator_is_rejected(tagged):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(tagged, {"page": {3: 3}})


def test_more_than_one_condition_under_one_key_is_rejected(tagged):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(tagged, {"page": {"gt": 1, "lt": 9}})
    assert "exactly one" in str(exc.value)


def test_comparing_incompatible_types_is_a_typed_error_not_a_traceback(tagged):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(tagged, {"page": {"gt": "three"}})
    assert "metadata-filter" in str(exc.value)
    assert "int" in str(exc.value) and "str" in str(exc.value)


def test_an_in_operator_needs_a_sequence_not_a_bare_string(tagged):
    """Otherwise 'in' on a string would mean per-character membership."""
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(tagged, {"lang": {"in": "en"}})
    assert "sequence" in str(exc.value)


def test_exists_takes_a_boolean(tagged):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(tagged, {"page": {"exists": "yes"}})


def test_contains_on_non_text_is_a_typed_error(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"chunk_index": {"contains": 1}})
    assert "text" in str(exc.value)


def test_startswith_on_non_text_is_a_typed_error(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"chunk_index": {"startswith": "1"}})


@pytest.mark.parametrize("bad", ["x", 3, None])
def test_a_non_mapping_criteria_is_rejected(bad):
    with pytest.raises(MetadataFilterInputError):
        filter_documents([], bad)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_an_unusable_field_name_is_rejected(bad):
    with pytest.raises(MetadataFilterInputError):
        filter_documents([], {bad: 1})


# --------------------------------------------------------------------------- #
# The filter's success paths
# --------------------------------------------------------------------------- #


def test_a_literal_criterion_means_equality(tagged):
    assert [d.chunk_id for d in filter_documents(tagged, {"lang": "en"})] == ["c0", "c2"]


def test_ordering_operators_work(tagged):
    assert [d.chunk_id for d in filter_documents(tagged, {"page": {"gte": 5}})] == ["c1", "c2"]
    assert [d.chunk_id for d in filter_documents(tagged, {"page": {"lt": 5}})] == ["c0"]
    assert [d.chunk_id for d in filter_documents(tagged, {"page": {"lte": 1}})] == ["c0"]
    assert [d.chunk_id for d in filter_documents(tagged, {"page": {"ne": 9}})] == ["c0", "c2"]


def test_in_and_contains_and_startswith(tagged):
    assert [d.chunk_id for d in filter_documents(tagged, {"lang": {"in": ["fr", "de"]}})] == ["c1"]
    assert [d.chunk_id for d in filter_documents(tagged, {"tags": {"contains": "rest"}})] == ["c0"]
    assert [d.chunk_id for d in filter_documents(tagged, {"source_name": {"startswith": "b."}})] == ["c1"]


def test_exists_reports_presence(tagged):
    assert [d.chunk_id for d in filter_documents(tagged, {"absent": {"exists": False}})] == ["c0", "c1", "c2"]
    assert filter_documents(tagged, {"absent": {"exists": True}}) == []


def test_an_identity_field_is_addressable_by_its_result_name(tagged):
    """The caller sees chunk_index, not some storage path."""
    assert [d.chunk_id for d in filter_documents(tagged, {"chunk_index": {"lt": 2}})] == ["c0", "c1"]


def test_a_document_missing_the_field_does_not_match(tagged):
    assert filter_documents(tagged, {"absent": "x"}) == []


def test_criteria_combine_with_and(tagged):
    out = filter_documents(tagged, {"lang": "en", "page": {"gte": 5}})
    assert [d.chunk_id for d in out] == ["c2"]


def test_the_filter_preserves_the_callers_ranking(tagged):
    assert [d.chunk_id for d in filter_documents(tagged, {"lang": "en"})] == ["c0", "c2"]


def test_the_same_criteria_language_serves_plain_chunks(corpus):
    out = apply_to_chunks(corpus, {"source_name": {"startswith": "b."}})
    assert [c.chunk_id for c in out] == ["c1"]


def test_applying_to_non_chunks_is_rejected(corpus):
    with pytest.raises(MetadataFilterInputError):
        apply_to_chunks(["not a chunk"], {"x": 1})


# --------------------------------------------------------------------------- #
# Observable 1: ranked results are deterministic
# --------------------------------------------------------------------------- #


def test_bm25_ranks_the_two_term_document_first(corpus):
    retriever = SparseRetriever(corpus, k=5)
    assert [h.chunk_id for h in retriever.retrieve("python error")] == ["c0", "c1"]


def test_bm25_returns_the_same_order_every_time(corpus):
    """Observable 1, for the sparse retriever."""
    retriever = SparseRetriever(corpus, k=5)
    first = [h.chunk_id for h in retriever.retrieve("error handling")]
    for _ in range(5):
        assert [h.chunk_id for h in retriever.retrieve("error handling")] == first


def test_ties_break_on_chunk_index_then_id():
    tied = [
        core.Chunk(chunk_index=1, text="same words here", chunk_id="b"),
        core.Chunk(chunk_index=0, text="same words here", chunk_id="a"),
    ]
    retriever = SparseRetriever(tied)
    assert [h.chunk_id for h in retriever.retrieve("same words")] == ["a", "b"]


def test_bm25_excludes_documents_sharing_no_term(corpus):
    assert SparseRetriever(corpus).retrieve("xylophone") == []


def test_bm25_scores_are_positive_and_finite(corpus):
    for hit in SparseRetriever(corpus).retrieve("python"):
        assert hit.score > 0.0
        assert math.isfinite(hit.score)


def test_bm25_results_are_scored_documents_with_provenance(corpus):
    hits = SparseRetriever(corpus).retrieve("pasta")
    assert hits and all(isinstance(h, core.ScoredDocument) for h in hits)
    # The chunk already knows its origin, so it is preserved rather than
    # overwritten with the retriever's name.
    assert [h.source_name for h in hits] == ["c.md"]


def test_hybrid_fusion_is_deterministic():
    dense = [scored("c0", 0.2), scored("c1", 0.9), scored("c3", 0.4)]
    sparse = [scored("c1", 8.0), scored("c2", 5.0), scored("c0", 1.0)]
    first = [d.chunk_id for d in fuse(dense, sparse)]
    for _ in range(5):
        assert [d.chunk_id for d in fuse(dense, sparse)] == first


def test_rrf_puts_a_document_found_by_both_sources_first():
    dense = [scored("c0", 0.2), scored("c1", 0.9), scored("c3", 0.4)]
    sparse = [scored("c1", 8.0), scored("c2", 5.0), scored("c0", 1.0)]
    assert [d.chunk_id for d in fuse(dense, sparse)][0] == "c1"


def test_rrf_arithmetic_matches_the_formula():
    """fused = SUM weight / (rrf_k + rank), rank counting from 1."""
    dense = [scored("c0", 0.2), scored("c1", 0.9)]
    sparse = [scored("c1", 8.0), scored("c2", 5.0)]
    out = {d.chunk_id: d.score for d in fuse(dense, sparse, rrf_k=60)}
    assert out["c1"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["c0"] == pytest.approx(1 / 61)
    assert out["c2"] == pytest.approx(1 / 62)


# --------------------------------------------------------------------------- #
# Observable 2: requesting more results than exist is handled
# --------------------------------------------------------------------------- #


def test_bm25_returning_more_than_matches_returns_what_exists(corpus):
    assert len(SparseRetriever(corpus).retrieve("pasta", k=100)) == 1


def test_dense_returning_more_than_the_store_holds_returns_what_exists():
    store = FakeStore(count=2)
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=store, k=50)
    assert len(retriever.retrieve("anything")) == 2


def test_hybrid_fusing_two_partial_lists_returns_the_union():
    assert len(fuse([scored("c0", 1.0)], [scored("c0", 1.0), scored("c1", 0.5)])) == 2


def test_an_empty_corpus_is_not_an_error():
    assert SparseRetriever([]).retrieve("anything") == []
    assert len(SparseRetriever([])) == 0


def test_fusing_two_empty_lists_returns_empty():
    assert fuse([], []) == []


# --------------------------------------------------------------------------- #
# Dense retrieval
# --------------------------------------------------------------------------- #


def test_dense_satisfies_its_composed_protocols():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore())
    assert isinstance(retriever.embedder, core.EmbeddingProtocol)
    assert isinstance(retriever.store, core.VectorStoreProtocol)


def test_dense_returns_the_stores_results():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(count=2))
    assert [h.chunk_id for h in retriever.retrieve("q")] == ["c0", "c1"]


def test_dense_embeds_exactly_the_caller_query():
    seen: list[str] = []
    retriever = DenseRetriever(embedder=FakeEmbedder(seen), store=FakeStore())
    retriever.retrieve("what is BM25")
    assert seen == ["what is BM25"]


def test_dense_passes_the_filter_through_to_the_store():
    class Recording(FakeStore):
        def __init__(self):
            super().__init__(1)
            self.seen = None

        def query(self, vector, k, filter=None):
            self.seen = filter
            return super().query(vector, k, filter)

    store = Recording()
    DenseRetriever(embedder=FakeEmbedder(), store=store).retrieve("q", filter={"lang": "en"})
    assert store.seen == {"lang": "en"}


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_an_unusable_k_is_rejected(k):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), k=k)
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve("q", k=k)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(query):
    """An empty string embeds and retrieves a real-looking answer about nothing."""
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve(query)
    assert "dense-retrieval" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(query):
    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=FakeEmbedder(), store=FakeStore()).retrieve(query)


def test_a_store_returning_the_wrong_number_of_vectors_is_caught():
    class TwoVectors:
        dimensions = 2
        model_id = "bad-v1"

        def embed(self, texts):
            return [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=TwoVectors(), store=FakeStore()).retrieve("q")
    assert "exactly 1" in str(exc.value)


def test_a_vector_width_the_store_never_indexed_is_caught():
    """The embedder disagrees with its own declared width, so the store would."""

    class Wide:
        dimensions = 2
        model_id = "wide-v1"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=Wide(), store=FakeStore()).retrieve("q")
    assert "3 wide" in str(exc.value)


def test_a_broken_embedder_is_wrapped_with_the_model_id():
    class Broken:
        dimensions = 2
        model_id = "broken-v1"

        def embed(self, texts):
            raise RuntimeError("service down")

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=Broken(), store=FakeStore()).retrieve("q")
    assert "broken-v1" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_store_returning_non_documents_is_rejected():
    class BadStore(FakeStore):
        def query(self, vector, k, filter=None):
            return [{"chunk_id": "c0"}]

    with pytest.raises(RetrievalError) as exc:
        DenseRetriever(embedder=FakeEmbedder(), store=BadStore()).retrieve("q")
    assert "ScoredDocument" in str(exc.value)


@pytest.mark.parametrize(
    "embedder,store",
    [(object(), FakeStore()), (FakeEmbedder(), object())],
    ids=["embedder", "store"],
)
def test_a_contract_violating_dependency_is_rejected_at_construction(embedder, store):
    with pytest.raises(RetrievalInputError) as exc:
        DenseRetriever(embedder=embedder, store=store)
    assert "Protocol" in str(exc.value)


def test_an_embedder_with_an_unusable_dimension_count_is_rejected():
    class NoDimensions:
        model_id = "x"

        def embed(self, texts):
            return []

    with pytest.raises(RetrievalInputError):
        DenseRetriever(embedder=NoDimensions(), store=FakeStore())


def test_dense_is_reprable_and_names_itself():
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=FakeStore(), name="corp-a")
    assert "corp-a" in repr(retriever)
    assert "fake-v1" in repr(retriever)


# --------------------------------------------------------------------------- #
# Sparse retrieval
# --------------------------------------------------------------------------- #


def test_tokenizer_lowercases_keeps_numbers_and_drops_stopwords():
    assert tokenize("The ERROR_Code 42 in Python") == ["error", "code", "42", "python"]


def test_a_common_term_has_lower_idf_than_a_rare_one(corpus):
    retriever = SparseRetriever(corpus)
    assert retriever.idf("error") < retriever.idf("pasta")


def test_bm25_length_normalisation_is_visible():
    retriever = SparseRetriever(
        [
            core.Chunk(chunk_index=0, text="python", chunk_id="short"),
            core.Chunk(chunk_index=1, text="python " * 40, chunk_id="long"),
        ]
    )
    assert retriever.average_document_length > 1.0


def test_a_longer_document_does_not_win_on_word_count_alone():
    """What the b parameter is for, proven by turning it off."""
    corpus = [
        core.Chunk(chunk_index=0, text="python", chunk_id="short"),
        core.Chunk(
            chunk_index=1,
            text=("filler words here and there " * 2) + "python python",
            chunk_id="long",
        ),
    ]
    normalized = SparseRetriever(corpus).retrieve("python")
    assert [h.chunk_id for h in normalized][0] == "short"

    # Without length normalisation the longer document wins on repetition
    # alone, which is the behaviour b exists to prevent.
    unnormalised = SparseRetriever(corpus, b=0.0).retrieve("python")
    assert [h.chunk_id for h in unnormalised][0] == "long"


@pytest.mark.parametrize("b", [-0.1, 1.1, 2.0])
def test_an_out_of_range_b_is_rejected(b):
    with pytest.raises(SparseRetrievalInputError) as exc:
        SparseRetriever([], b=b)
    assert "between 0" in str(exc.value)


@pytest.mark.parametrize("k1", [-1.0, float("nan"), float("inf")])
def test_an_unusable_k1_is_rejected(k1):
    with pytest.raises(SparseRetrievalInputError):
        SparseRetriever([], k1=k1)


def test_a_non_chunk_in_the_corpus_is_rejected_by_position():
    with pytest.raises(SparseRetrievalInputError) as exc:
        SparseRetriever([core.Chunk(chunk_index=0, text="x", chunk_id="c0"), "nope"])
    assert "[1]" in str(exc.value)


@pytest.mark.parametrize("query", ["", "  ", None, 3])
def test_an_unusable_query_is_rejected(corpus, query):
    with pytest.raises(SparseRetrievalInputError):
        SparseRetriever(corpus).retrieve(query)


def test_idf_rejects_a_non_string_term():
    with pytest.raises(SparseRetrievalInputError):
        SparseRetriever([]).idf(3)


def test_the_sparse_errors_are_their_own_hierarchy_not_dense_s():
    """Each component carries its own, so copying one out needs only itself."""
    assert issubclass(SparseRetrievalInputError, SparseRetrievalError)
    assert not issubclass(SparseRetrievalError, RetrievalError)


def test_the_manifest_declares_no_optional_dependencies():
    """BM25 is implemented here, so no wheel is required to run it."""
    manifest = json.loads(
        (RETRIEVAL / "sparse" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == ["core-types"]


def test_bm25_runs_with_the_optional_library_blocked(monkeypatch):
    """FR-3 guard: results must not depend on a package that may be absent."""
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    retriever = SparseRetriever(
        [
            core.Chunk(chunk_index=0, text="python error handling", chunk_id="c0"),
            core.Chunk(chunk_index=1, text="python typing", chunk_id="c1"),
        ]
    )
    # c0 matches both terms, c1 only one; the ranking holds with no wheel.
    assert [h.chunk_id for h in retriever.retrieve("python error")] == ["c0", "c1"]


# --------------------------------------------------------------------------- #
# Hybrid fusion
# --------------------------------------------------------------------------- #


def test_either_source_may_be_empty():
    assert [d.chunk_id for d in fuse([scored("c0", 1.0), scored("c1", 0.5)], [])] == ["c0", "c1"]
    assert [d.chunk_id for d in fuse([], [scored("c0", 1.0)])] == ["c0"]


def test_the_winning_source_is_recorded():
    out = fuse([scored("c0", 0.2), scored("c1", 0.9)], [scored("c1", 8.0), scored("c2", 5.0)])
    by_id = {d.chunk_id: d.metadata[SOURCE_KEY] for d in out}
    assert by_id == {"c1": "dense+sparse", "c2": "sparse", "c0": "dense"}


def test_fusion_preserves_text_and_provenance():
    dense = [scored("c0", 0.2, text="the passage", source_name="a.md")]
    out = fuse(dense, [])
    assert out[0].text == "the passage"
    assert out[0].source_name == "a.md"


def test_weighted_fusion_refuses_scores_that_are_not_normalized():
    """The silent failure this component exists to remove."""
    with pytest.raises(HybridInputError) as exc:
        fuse([scored("c0", 8.0)], [scored("c1", 0.5)], method="weighted")
    assert "result-normalizer" in str(exc.value)


def test_weighted_fusion_adds_normalized_scores():
    out = fuse(
        [scored("c0", 0.5), scored("c1", 1.0)],
        [scored("c1", 1.0), scored("c2", 0.0)],
        method="weighted",
    )
    by_id = {d.chunk_id: round(d.score, 6) for d in out}
    assert by_id == {"c1": 2.0, "c0": 0.5, "c2": 0.0}


def test_weights_shift_the_result_toward_the_heavier_source():
    dense = [scored("c0", 0.2), scored("c1", 0.9)]
    sparse = [scored("c1", 8.0), scored("c2", 5.0)]
    default = {d.chunk_id: d.score for d in fuse(dense, sparse)}
    boosted = {d.chunk_id: d.score for d in fuse(dense, sparse, sparse_weight=3.0)}
    assert boosted["c1"] > default["c1"]
    assert boosted["c2"] > default["c2"]
    assert boosted["c0"] == pytest.approx(default["c0"])  # dense-only, unchanged


def test_a_larger_rrf_k_flattens_the_list():
    dense = [scored("c0", 0.9), scored("c1", 0.1)]
    flat = {d.chunk_id: d.score for d in fuse(dense, [], rrf_k=1000)}
    sharp = {d.chunk_id: d.score for d in fuse(dense, [], rrf_k=1)}
    assert (sharp["c0"] - sharp["c1"]) > (flat["c0"] - flat["c1"])


@pytest.mark.parametrize("method", ["softmax", "combsum", ""])
def test_an_unknown_fusion_method_lists_the_valid_ones(method):
    with pytest.raises(HybridInputError) as exc:
        fuse([scored("c0", 1.0)], [], method=method)
    assert repr(method) in str(exc.value)
    assert "rrf" in str(exc.value) and "weighted" in str(exc.value)


def test_a_duplicate_id_in_one_list_is_rejected():
    """It would be fused twice and outrank genuinely better documents."""
    with pytest.raises(HybridInputError) as exc:
        fuse([], [scored("c0", 1.0), scored("c0", 0.5)])
    assert "'c0'" in str(exc.value)


@pytest.mark.parametrize("weight", [-1.0, 0.0, "1", None])
def test_an_unusable_weight_is_rejected(weight):
    with pytest.raises(HybridInputError):
        fuse([scored("c0", 1.0)], [], sparse_weight=weight)


@pytest.mark.parametrize("rrf_k", [0, -1, True, 1.5, "60"])
def test_an_unusable_rrf_k_is_rejected(rrf_k):
    with pytest.raises(HybridInputError):
        fuse([scored("c0", 1.0)], [], rrf_k=rrf_k)


@pytest.mark.parametrize("bad", ["list", 3, None])
def test_a_non_sequence_source_is_rejected(bad):
    with pytest.raises(HybridInputError):
        fuse(bad, [])


def test_a_non_document_in_a_source_is_rejected_by_position():
    with pytest.raises(HybridInputError) as exc:
        fuse([scored("c0", 1.0), "nope"], [])
    assert "[1]" in str(exc.value)


def test_the_hybrid_errors_form_a_hierarchy():
    assert issubclass(HybridInputError, HybridError)
    assert not issubclass(HybridError, ValueError)


# --------------------------------------------------------------------------- #
# Message provenance and the composed pipeline
# --------------------------------------------------------------------------- #


def test_every_component_names_itself_in_its_messages():
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([], method="nope")
    assert "result-normalizer" in str(exc.value)

    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents([], {"x": {"nope": 1}})
    assert "metadata-filter" in str(exc.value)

    with pytest.raises(SparseRetrievalInputError) as exc:
        SparseRetriever([], k1=-1)
    assert "sparse-retrieval" in str(exc.value)

    with pytest.raises(HybridInputError) as exc:
        fuse([], [], method="nope")
    assert "hybrid-retrieval" in str(exc.value)


def test_the_five_components_compose_end_to_end(corpus):
    """dense + sparse -> normalize -> fuse -> filter, as the pipeline runs."""
    sparse_hits = SparseRetriever(corpus, k=3).retrieve("python error")
    dense_hits = [scored("c0", 0.9), scored("c3", 0.4)]

    fused = fuse(
        normalize_scores(dense_hits),
        normalize_scores(sparse_hits),
        method="weighted",
    )
    assert fused and fused[0].chunk_id == "c0"

    narrowed = filter_documents(fused, {"chunk_index": {"gte": 0}})
    assert [d.chunk_id for d in narrowed] == [d.chunk_id for d in fused]


def test_no_retrieval_component_imports_another():
    """Each is copy-ready on its own; none reaches into a sibling."""
    import ast

    module_names = (
        "dense_retrieval",
        "sparse_retrieval",
        "hybrid_retrieval",
        "metadata_filter",
        "retrieval_result_normalizer",
    )
    for name in module_names:
        module = sys.modules[name]
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for other in module_names:
            assert other not in imported, f"{name} imports {other}"


def test_every_manifest_declares_the_core_types_dependency():
    for directory in ("dense", "sparse", "hybrid", "metadata_filter", "result_normalizer"):
        manifest = json.loads(
            (RETRIEVAL / directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["dependencies"] == ["core-types"], directory
        assert manifest["category"] == "retrieval", directory
        assert manifest["required_packages"] == [], directory

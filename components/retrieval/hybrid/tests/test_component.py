"""Tests for the hybrid-retrieval component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

The property that matters most is asserted by name: weighted fusion refuses
scores that are not on a shared scale, because adding raw cosine distance to
raw BM25 is the standard silent failure this component exists to remove.
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
from hybrid_retrieval import (  # noqa: E402
    DEFAULT_RRF_K,
    METHODS,
    SOURCE_KEY,
    HybridError,
    HybridInputError,
    fuse,
)

import pytest  # noqa: E402


def scored(cid: str, score: float, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("text", cid)
    kwargs.setdefault("chunk_index", 0)
    return core.ScoredDocument(chunk_id=cid, score=score, **kwargs)


@pytest.fixture
def dense():
    return [scored("c0", 0.2), scored("c1", 0.9), scored("c3", 0.4)]


@pytest.fixture
def sparse():
    return [scored("c1", 8.0), scored("c2", 5.0), scored("c0", 1.0)]


@pytest.fixture
def normalized(dense, sparse):
    """Both lists on one [0, 1] scale, as weighted fusion requires."""
    def minmax(items):
        lo, hi = min(d.score for d in items), max(d.score for d in items)
        span = hi - lo
        return [d.model_copy(update={"score": (d.score - lo) / span}) for d in items]

    return minmax(dense), minmax(sparse)


# --------------------------------------------------------------------------- #
# Success paths -- reciprocal rank fusion
# --------------------------------------------------------------------------- #


def test_rrf_puts_a_document_found_by_both_sources_first(dense, sparse):
    assert fuse(dense, sparse)[0].chunk_id == "c1"


def test_rrf_arithmetic_matches_the_formula(dense, sparse):
    """fused = SUM weight / (rrf_k + rank), rank counting from 1."""
    out = {d.chunk_id: d.score for d in fuse(dense, sparse, rrf_k=60)}
    assert out["c1"] == pytest.approx(1 / 62 + 1 / 61)  # dense rank 2, sparse rank 1
    assert out["c0"] == pytest.approx(1 / 61 + 1 / 63)  # dense rank 1, sparse rank 3
    assert out["c2"] == pytest.approx(1 / 62)           # sparse rank 2 only
    assert out["c3"] == pytest.approx(1 / 63)           # dense rank 3 only


def test_rrf_ignores_the_actual_score_magnitudes():
    """The whole point: rank, not magnitude, so scale cannot break it."""
    low = [scored("a", 0.1), scored("b", 0.09)]
    high = [scored("a", 1e6), scored("b", 9e5)]
    assert [d.chunk_id for d in fuse(low, [])] == [d.chunk_id for d in fuse(high, [])]


def test_the_default_rrf_k_is_the_documented_value():
    assert DEFAULT_RRF_K == 60


def test_the_winning_source_is_recorded(dense, sparse):
    by_id = {d.chunk_id: d.metadata[SOURCE_KEY] for d in fuse(dense, sparse)}
    assert by_id == {"c1": "dense+sparse", "c0": "dense+sparse",
                     "c2": "sparse", "c3": "dense"}


def test_fusion_preserves_text_and_provenance():
    dense = [scored("c0", 0.2, text="the passage", source_name="a.md")]
    out = fuse(dense, [])
    assert out[0].text == "the passage"
    assert out[0].source_name == "a.md"


def test_fusion_preserves_metadata_and_adds_only_the_source_key():
    dense = [scored("c0", 0.2, metadata={"lang": "en"})]
    out = fuse(dense, [scored("c0", 1.0)])
    assert out[0].metadata["lang"] == "en"
    assert out[0].metadata[SOURCE_KEY] == "dense+sparse"


def test_results_are_scored_documents(dense, sparse):
    for item in fuse(dense, sparse):
        assert isinstance(item, core.ScoredDocument)


def test_a_larger_rrf_k_flattens_the_list(dense, sparse):
    flat = {d.chunk_id: d.score for d in fuse(dense, sparse, rrf_k=1000)}
    sharp = {d.chunk_id: d.score for d in fuse(dense, sparse, rrf_k=1)}
    assert (sharp["c1"] - sharp["c2"]) > (flat["c1"] - flat["c2"])


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_same_inputs_return_the_same_order_every_time(dense, sparse):
    first = [d.chunk_id for d in fuse(dense, sparse)]
    for _ in range(10):
        assert [d.chunk_id for d in fuse(dense, sparse)] == first


def test_ties_break_on_chunk_index_then_id():
    """RRF ties constantly, so this ordering must be total and stable."""
    a = [scored("c1", 1.0, chunk_index=1), scored("c0", 1.0, chunk_index=0)]
    b = [scored("c0", 9.0, chunk_index=0), scored("c1", 9.0, chunk_index=1)]
    assert [d.chunk_id for d in fuse(a, b)] == ["c0", "c1"]


def test_rrf_reads_rank_from_position_not_from_score():
    """Documented contract: each list must arrive ordered best first.

    RRF cannot sort internally -- it does not know which end of a score is
    better, because dense and sparse disagree and ScoredDocument carries no
    direction. So position *is* the rank, and a permuted input is a permuted
    ranking.
    """
    ordered = [scored("c1", 0.9), scored("c0", 0.2)]
    assert [d.chunk_id for d in fuse(ordered, [])] == ["c1", "c0"]
    # Same documents, same scores, reversed positions -> reversed ranking.
    assert [d.chunk_id for d in fuse(list(reversed(ordered)), [])] == ["c0", "c1"]


def test_rrf_fuses_on_position_even_when_scores_contradict_it():
    """A store that returned an unsorted list is a store bug, surfaced here."""
    contradictory = [scored("worse", 0.1), scored("better", 0.9)]
    out = {d.chunk_id: d.score for d in fuse(contradictory, [])}
    assert out["worse"] == pytest.approx(1 / 61)  # rank 1 by position
    assert out["better"] == pytest.approx(1 / 62)  # rank 2 by position


def test_weighted_fusion_is_independent_of_input_order(normalized):
    """It reads the scores, not the positions, so permuting changes nothing."""
    dense, sparse = normalized
    assert [d.chunk_id for d in fuse(dense, sparse, method="weighted")] == [
        d.chunk_id for d in fuse(list(reversed(dense)), list(reversed(sparse)), method="weighted")
    ]


# --------------------------------------------------------------------------- #
# Weighted fusion, and the guard that defines it
# --------------------------------------------------------------------------- #


def test_weighted_fusion_refuses_raw_scores(dense, sparse):
    """The silent failure this component exists to remove."""
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, sparse, method="weighted")
    assert "result-normalizer" in str(exc.value)


def test_the_refusal_names_the_offending_list(dense, sparse):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, sparse, method="weighted")
    assert "sparse" in str(exc.value)


def test_weighted_fusion_works_over_normalized_scores(normalized):
    dense, sparse = normalized
    out = {d.chunk_id: d.score for d in fuse(dense, sparse, method="weighted")}
    assert out["c1"] == pytest.approx(2.0)  # best of both, 1.0 + 1.0
    assert out["c2"] == pytest.approx(4 / 7)  # sparse only
    assert out["c3"] == pytest.approx(2 / 7)  # dense only
    assert out["c0"] == pytest.approx(0.0)  # last of both
    assert out["c1"] > out["c2"] > out["c3"] > out["c0"]


def test_weighted_fusion_rejects_a_single_out_of_range_score():
    with pytest.raises(HybridInputError):
        fuse([scored("c0", 1.5)], [], method="weighted")


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf")])
def test_the_out_of_range_check_is_inclusive_of_the_bounds(value):
    with pytest.raises(HybridInputError):
        fuse([scored("c0", value)], [], method="weighted")


def test_the_out_of_range_check_covers_both_sources():
    with pytest.raises(HybridInputError):
        fuse([scored("c0", 0.5)], [scored("c1", 2.0)], method="weighted")


def test_weights_shift_the_result_toward_the_heavier_source(dense, sparse):
    default = {d.chunk_id: d.score for d in fuse(dense, sparse)}
    boosted = {d.chunk_id: d.score for d in fuse(dense, sparse, sparse_weight=3.0)}
    assert boosted["c1"] > default["c1"]  # found by both, sparse weighted up
    assert boosted["c2"] > default["c2"]  # sparse-only, so it gains
    # c3 is dense-only, so a sparse weight cannot touch it.
    assert boosted["c3"] == pytest.approx(default["c3"])


def test_a_heavy_dense_weight_keeps_its_own_first_hit_first():
    dense = [scored("c0", 0.9), scored("c1", 0.1)]
    out = fuse(dense, [], dense_weight=10.0)
    assert out[0].chunk_id == "c0"


# --------------------------------------------------------------------------- #
# Empty sources
# --------------------------------------------------------------------------- #


def test_an_empty_sparse_list_passes_the_dense_through(dense):
    assert [d.chunk_id for d in fuse(dense, [])] == ["c0", "c1", "c3"]


def test_an_empty_dense_list_passes_the_sparse_through(sparse):
    assert [d.chunk_id for d in fuse([], sparse)] == ["c1", "c2", "c0"]


def test_two_empty_lists_return_empty():
    assert fuse([], []) == []


def test_the_source_key_says_which_retriever_actually_found_it(dense):
    by_id = {d.chunk_id: d.metadata[SOURCE_KEY] for d in fuse(dense, [])}
    assert set(by_id.values()) == {"dense"}


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["softmax", "combsum", ""])
def test_an_unknown_method_lists_the_valid_ones(dense, sparse, method):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, sparse, method=method)
    assert repr(method) in str(exc.value)
    for name in METHODS:
        assert name in str(exc.value)


def test_a_duplicate_id_in_one_list_is_rejected():
    """It would be fused twice and outrank genuinely better documents."""
    with pytest.raises(HybridInputError) as exc:
        fuse([], [scored("c0", 1.0), scored("c0", 0.5)])
    assert "'c0'" in str(exc.value)


def test_the_duplicate_message_says_why_it_matters():
    with pytest.raises(HybridInputError) as exc:
        fuse([scored("c0", 1.0), scored("c0", 0.9)], [])
    assert "twice" in str(exc.value)


def test_a_duplicate_id_in_either_list_is_rejected(dense, sparse):
    with pytest.raises(HybridInputError):
        fuse(dense + [dense[0]], sparse)
    with pytest.raises(HybridInputError):
        fuse(dense, sparse + [sparse[0]])


def test_the_model_rejects_an_empty_chunk_id_upstream():
    """So fusion's duplicate check never has to handle id-less documents.

    If an empty chunk_id were possible, every such document would collapse
    into one key during fusion. It is not possible: core-types requires a
    non-empty id, which is why `fuse` can treat "" as a real identity.
    """
    with pytest.raises(Exception):
        scored("", 1.0)


@pytest.mark.parametrize("weight", [-1.0, 0.0])
def test_a_negative_or_zero_weight_is_rejected(dense, weight):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, [], sparse_weight=weight)
    assert "sparse_weight" in str(exc.value)


def test_a_zero_weight_message_explains_the_preferred_alternative(dense):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, [], sparse_weight=0.0)
    assert "Omit the source" in str(exc.value)


@pytest.mark.parametrize("weight", ["1", None, True, object()])
def test_a_non_numeric_weight_is_rejected(dense, weight):
    with pytest.raises(HybridInputError):
        fuse(dense, [], sparse_weight=weight)


@pytest.mark.parametrize("rrf_k", [0, -1, True, 1.5, "60", None])
def test_an_unusable_rrf_k_is_rejected(dense, sparse, rrf_k):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, sparse, rrf_k=rrf_k)
    assert "rrf_k" in str(exc.value)


@pytest.mark.parametrize("bad", ["list", 3, None, {"a": 1}])
def test_a_non_sequence_source_is_rejected(bad):
    with pytest.raises(HybridInputError):
        fuse(bad, [])
    with pytest.raises(HybridInputError):
        fuse([], bad)


def test_a_non_document_in_a_source_is_rejected_by_position(dense):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, ["not a document"])
    assert "[0]" in str(exc.value)


# --------------------------------------------------------------------------- #
# Message provenance, hierarchy and the manifest
# --------------------------------------------------------------------------- #


def test_messages_name_the_component(dense, sparse):
    with pytest.raises(HybridInputError) as exc:
        fuse(dense, sparse, method="nope")
    assert "hybrid-retrieval" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(HybridInputError, HybridError)
    assert not issubclass(HybridError, ValueError)
    assert not issubclass(HybridError, TypeError)


def test_every_failure_is_catchable_under_the_base_class(dense, sparse):
    for call in (
        lambda: fuse(dense, sparse, method="nope"),
        lambda: fuse(dense, sparse, method="weighted"),
        lambda: fuse([], [scored("c0", 1.0), scored("c0", 0.5)]),
        lambda: fuse(dense, [], sparse_weight=-1.0),
    ):
        with pytest.raises(HybridError):
            call()


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []
    assert manifest["category"] == "retrieval"


def test_the_component_does_not_import_the_normalizer():
    """`weighted` asks the caller to normalize; it never does it itself."""
    import ast

    tree = ast.parse((COMPONENT / "hybrid_retrieval.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "retrieval_result_normalizer" not in imported

"""Tests for the retrieval-result-normalizer component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

The same behaviour is asserted end to end in `tests/unit/test_retrieval.py`;
this file covers the component's own API on its own terms.
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
from retrieval_result_normalizer import (  # noqa: E402
    CONSTANT_SCORE,
    METHODS,
    NormalizationError,
    NormalizationInputError,
    normalize_scores,
)

import pytest  # noqa: E402


def scored(cid: str, score: float, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("text", cid)
    kwargs.setdefault("chunk_index", 0)
    return core.ScoredDocument(chunk_id=cid, score=score, **kwargs)


@pytest.fixture
def spread():
    return [scored("c0", 1.0), scored("c1", 3.0), scored("c2", 5.0)]


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_min_max_maps_onto_zero_to_one(spread):
    assert [round(d.score, 6) for d in normalize_scores(spread)] == [0.0, 0.5, 1.0]


def test_rank_puts_the_best_at_one_and_the_worst_at_zero(spread):
    out = normalize_scores(spread, method="rank")
    assert [round(d.score, 6) for d in out] == [0.0, 0.5, 1.0]


def test_inverting_direction_flips_the_result(spread):
    forward = normalize_scores(spread)
    backward = normalize_scores(spread, higher_is_better=False)
    assert [round(d.score, 6) for d in backward] == [1.0, 0.5, 0.0]
    for a, b in zip(forward, backward):
        assert a.score + b.score == pytest.approx(1.0)


def test_z_score_is_centred_on_zero(spread):
    out = normalize_scores(spread, method="z_score")
    assert round(sum(d.score for d in out), 9) == 0.0


def test_order_is_preserved_and_scores_change(spread):
    out = normalize_scores(spread)
    assert [d.chunk_id for d in out] == ["c0", "c1", "c2"]


def test_the_originals_are_not_mutated(spread):
    before = [d.score for d in spread]
    normalize_scores(spread)
    assert [d.score for d in spread] == before


def test_the_results_are_scored_documents(spread):
    for item in normalize_scores(spread):
        assert isinstance(item, core.ScoredDocument)


def test_an_empty_list_returns_an_empty_list():
    assert normalize_scores([]) == []


def test_every_method_is_reachable():
    for method in METHODS:
        assert normalize_scores(
            [scored("c0", 1.0), scored("c1", 2.0)], method=method
        )


def test_metadata_survives_normalization():
    out = normalize_scores([scored("c0", 1.0, metadata={"lang": "en"})])
    assert out[0].metadata == {"lang": "en"}


# --------------------------------------------------------------------------- #
# The cases a naive implementation gets wrong
# --------------------------------------------------------------------------- #


def test_a_constant_list_is_finite_not_nan():
    """min-max divides by max - min, which is zero here."""
    out = normalize_scores([scored("c0", 2.0), scored("c1", 2.0)])
    assert [d.score for d in out] == [CONSTANT_SCORE, CONSTANT_SCORE]
    assert all(math.isfinite(d.score) for d in out)


def test_the_constant_is_documented_as_the_neutral_value():
    assert CONSTANT_SCORE == 0.5


def test_a_constant_under_rank_is_not_zero():
    out = normalize_scores(
        [scored("c0", 7.0), scored("c1", 7.0), scored("c2", 7.0)], method="rank"
    )
    assert all(d.score == 0.5 for d in out)


def test_a_constant_under_z_score_is_zero():
    out = normalize_scores(
        [scored("c0", 7.0), scored("c1", 7.0)], method="z_score"
    )
    assert all(d.score == 0.0 for d in out)


def test_ties_share_an_average_rank():
    """Two tied documents must not be told apart arbitrarily."""
    out = normalize_scores(
        [scored("c0", 1.0), scored("c1", 1.0), scored("c2", 4.0)], method="rank"
    )
    assert out[0].score == out[1].score
    assert out[2].score == 1.0


def test_rank_survives_an_outlier_that_min_max_flattens():
    outlier = [scored("c0", 0.01), scored("c1", 0.02), scored("c2", 900.0)]
    ranked = [round(d.score, 6) for d in normalize_scores(outlier, method="rank")]
    assert ranked == [0.0, 0.5, 1.0]
    flattened = normalize_scores(outlier)
    assert flattened[1].score < 0.001


def test_a_single_document_reports_no_distinction():
    assert normalize_scores([scored("c0", 9.0)])[0].score == CONSTANT_SCORE
    assert normalize_scores([scored("c0", 9.0)], method="rank")[0].score == 1.0
    assert normalize_scores([scored("c0", 9.0)], method="z_score")[0].score == 0.0


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_rejected(value):
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0), scored("c1", value)])
    assert "result-normalizer" in str(exc.value)


def test_an_unknown_method_names_it_and_lists_the_valid_ones():
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0)], method="softmax")
    assert "softmax" in str(exc.value)
    for method in METHODS:
        assert method in str(exc.value)


@pytest.mark.parametrize("bad", ["docs", b"docs", 3, None, {"a": 1}])
def test_a_non_sequence_of_documents_is_rejected(bad):
    with pytest.raises(NormalizationInputError):
        normalize_scores(bad)


@pytest.mark.parametrize("bad", ["text", 3, object()], ids=["str", "int", "obj"])
def test_a_non_document_in_the_list_is_rejected(bad):
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores([scored("c0", 1.0), bad])
    assert "[1]" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(NormalizationInputError, NormalizationError)
    assert not issubclass(NormalizationError, ValueError)
    assert not issubclass(NormalizationError, TypeError)


# --------------------------------------------------------------------------- #
# Message provenance and the manifest
# --------------------------------------------------------------------------- #


def test_messages_name_the_component():
    with pytest.raises(NormalizationInputError) as exc:
        normalize_scores("nope")
    assert "result-normalizer" in str(exc.value)


def test_every_failure_is_catchable_under_the_base_class():
    for call in (
        lambda: normalize_scores("nope"),
        lambda: normalize_scores([scored("c0", float("nan"))]),
        lambda: normalize_scores([scored("c0", 1.0)], method="bogus"),
    ):
        with pytest.raises(NormalizationError):
            call()


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []
    assert manifest["category"] == "retrieval"

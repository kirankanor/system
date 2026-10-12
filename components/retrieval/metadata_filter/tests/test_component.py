"""Tests for the metadata-filter component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

The behaviour that most matters is asserted by name: an invalid filter
operator raises a typed error rather than matching nothing.
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
from metadata_filter import (  # noqa: E402
    OPERATORS,
    MetadataFilterError,
    MetadataFilterInputError,
    apply_to_chunks,
    filter_documents,
)

import pytest  # noqa: E402


def scored(cid: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("text", cid)
    kwargs.setdefault("chunk_index", 0)
    return core.ScoredDocument(chunk_id=cid, score=1.0, **kwargs)


@pytest.fixture
def documents():
    return [
        scored("c0", source_name="a.md", chunk_index=0,
               metadata={"lang": "en", "page": 1, "tags": "api rest"}),
        scored("c1", source_name="b.md", chunk_index=1,
               metadata={"lang": "fr", "page": 9, "tags": "api"}),
        scored("c2", source_name="c.md", chunk_index=2,
               metadata={"lang": "en", "page": 5, "tags": ""}),
    ]


@pytest.fixture
def corpus():
    return [
        core.Chunk(chunk_index=0, text="a", chunk_id="k0", metadata={"lang": "en"}),
        core.Chunk(chunk_index=1, text="b", chunk_id="k1", metadata={"lang": "fr"}),
    ]


# --------------------------------------------------------------------------- #
# Observable requirement: an invalid filter operator raises a typed error
# --------------------------------------------------------------------------- #


def test_an_unknown_operator_raises(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"page": {"gtr": 3}})


def test_the_error_is_typed_as_a_filter_error(documents):
    with pytest.raises(MetadataFilterError):
        filter_documents(documents, {"page": {"gtr": 3}})


def test_the_message_names_the_operator(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"gtr": 3}})
    assert "gtr" in str(exc.value)


def test_the_message_names_the_field(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"gtr": 3}})
    assert "page" in str(exc.value)


def test_the_message_lists_every_valid_operator(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"nope": 3}})
    for operator in OPERATORS:
        assert operator in str(exc.value)


def test_the_component_rejects_the_operator_rather_than_matching_nothing(documents):
    """A typo and a genuine miss must not look the same."""
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"page": {"gtr": 3}})
    assert filter_documents(documents, {"page": {"gt": 100}}) == []


@pytest.mark.parametrize("operator", ["GTR", "Equals", "like", "regex", ""])
def test_near_miss_operators_are_rejected_too(documents, operator):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"page": {operator: 3}})


# --------------------------------------------------------------------------- #
# Success paths -- every operator
# --------------------------------------------------------------------------- #


def test_a_literal_means_equality(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": "en"})] == ["c0", "c2"]


def test_ne(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": {"ne": "en"}})] == ["c1"]


@pytest.mark.parametrize(
    "operator,expected",
    [("gt", ["c1"]), ("gte", ["c1", "c2"]), ("lt", ["c0"]), ("lte", ["c0", "c2"])],
)
def test_ordering_operators(documents, operator, expected):
    assert [d.chunk_id for d in filter_documents(documents, {"page": {operator: 5}})] == expected


def test_in(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": {"in": ["fr", "de"]}})] == ["c1"]


def test_in_accepts_a_single_candidate_wrapped_in_a_list(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": {"in": ["en"]}})] == ["c0", "c2"]


def test_contains(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"tags": {"contains": "rest"}})] == ["c0"]


def test_contains_matches_a_substring_of_itself(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"tags": {"contains": "api"}})] == ["c0", "c1"]


def test_startswith(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"source_name": {"startswith": "b."}})] == ["c1"]


def test_exists_true(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"absent": {"exists": True}})] == []


def test_exists_false(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"absent": {"exists": False}})] == ["c0", "c1", "c2"]


def test_a_document_field_is_addressable_by_its_result_name(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"chunk_index": {"lt": 2}})] == ["c0", "c1"]


def test_text_is_addressable(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"text": "c1"})] == ["c1"]


def test_a_metadata_key_is_addressable_directly(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": "fr"})] == ["c1"]


# --------------------------------------------------------------------------- #
# Semantics the tests pin down
# --------------------------------------------------------------------------- #


def test_a_document_missing_the_field_does_not_match(documents):
    assert filter_documents(documents, {"absent": "x"}) == []


def test_ne_on_a_missing_field_does_not_match(documents):
    """Absence is not inequality; `exists` is how you ask about absence."""
    assert filter_documents(documents, {"absent": {"ne": "x"}}) == []


def test_criteria_combine_with_and(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": "en", "page": {"gte": 5}})] == ["c2"]


def test_and_that_cannot_both_hold_returns_empty(documents):
    assert filter_documents(documents, {"lang": "en", "page": {"gt": 100}}) == []


def test_the_callers_ranking_survives(documents):
    assert [d.chunk_id for d in filter_documents(documents, {"lang": "en"})] == ["c0", "c2"]


def test_an_empty_criteria_matches_everything(documents):
    assert [d.chunk_id for d in filter_documents(documents, {})] == ["c0", "c1", "c2"]


def test_an_empty_list_filters_to_empty():
    assert filter_documents([], {"lang": "en"}) == []


def test_results_are_still_scored_documents(documents):
    for item in filter_documents(documents, {"lang": "en"}):
        assert isinstance(item, core.ScoredDocument)


# --------------------------------------------------------------------------- #
# Failure paths beyond the operator
# --------------------------------------------------------------------------- #


def test_a_type_mismatch_is_a_typed_error_not_a_traceback(documents):
    """Python raises TypeError on int > str; that must not escape."""
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"gt": "three"}})
    assert "int" in str(exc.value) and "str" in str(exc.value)


def test_a_type_mismatch_names_the_field(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"lt": "three"}})
    assert "page" in str(exc.value)


def test_comparing_text_with_a_number_is_rejected(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"lang": {"gt": 3}})


def test_in_needs_a_sequence_not_a_string(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"lang": {"in": "en"}})
    assert "sequence" in str(exc.value)


@pytest.mark.parametrize("operand", [3, None, {"a": 1}])
def test_in_rejects_a_non_sequence_operand(documents, operand):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"lang": {"in": operand}})


def test_more_than_one_condition_under_one_key_is_rejected(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"page": {"gt": 1, "lt": 9}})
    assert "exactly one" in str(exc.value)


def test_a_non_string_operator_is_rejected(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"page": {3: 3}})


def test_exists_requires_a_boolean(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"page": {"exists": "yes"}})


def test_contains_on_non_text_is_rejected(documents):
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(documents, {"chunk_index": {"contains": 1}})
    assert "text" in str(exc.value)


def test_startswith_on_non_text_is_rejected(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"chunk_index": {"startswith": "1"}})


def test_startswith_needs_a_text_operand(documents):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(documents, {"source_name": {"startswith": 3}})


@pytest.mark.parametrize("bad", ["x", 3, None, [1]])
def test_a_non_mapping_criteria_is_rejected(bad):
    with pytest.raises(MetadataFilterInputError):
        filter_documents([], bad)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_an_unusable_field_name_is_rejected(bad):
    with pytest.raises(MetadataFilterInputError):
        filter_documents([], {bad: 1})


@pytest.mark.parametrize("bad", ["docs", 3, None, {"a": 1}])
def test_a_non_sequence_of_documents_is_rejected(bad):
    with pytest.raises(MetadataFilterInputError):
        filter_documents(bad, {})


def test_a_non_document_in_the_list_is_rejected_by_position():
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents(["nope"], {})
    assert "[0]" in str(exc.value)


# --------------------------------------------------------------------------- #
# apply_to_chunks -- the same language over plain chunks
# --------------------------------------------------------------------------- #


def test_apply_to_chunks_uses_the_same_operators(corpus):
    assert [c.chunk_id for c in apply_to_chunks(corpus, {"lang": {"in": ["en"]}})] == ["k0"]


def test_apply_to_chunks_respects_identity_fields(corpus):
    assert [c.chunk_id for c in apply_to_chunks(corpus, {"chunk_index": 1})] == ["k1"]


def test_apply_to_chunks_preserves_the_chunks(corpus):
    out = apply_to_chunks(corpus, {"lang": "en"})
    assert out and isinstance(out[0], core.Chunk)


def test_apply_to_chunks_rejects_non_chunks():
    with pytest.raises(MetadataFilterInputError):
        apply_to_chunks(["nope"], {})


def test_apply_to_chunks_rejects_a_bad_criteria():
    with pytest.raises(MetadataFilterInputError):
        apply_to_chunks([], "not a mapping")


def test_apply_to_chunks_shares_the_operator_validation(corpus):
    with pytest.raises(MetadataFilterInputError) as exc:
        apply_to_chunks(corpus, {"lang": {"nope": 1}})
    assert "nope" in str(exc.value)


# --------------------------------------------------------------------------- #
# Message provenance and the manifest
# --------------------------------------------------------------------------- #


def test_messages_name_the_component():
    with pytest.raises(MetadataFilterInputError) as exc:
        filter_documents([], {"x": {"bad": 1}})
    assert "metadata-filter" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(MetadataFilterInputError, MetadataFilterError)
    assert not issubclass(MetadataFilterError, ValueError)
    assert not issubclass(MetadataFilterError, TypeError)


def test_every_failure_is_catchable_under_the_base_class():
    for call in (
        lambda: filter_documents([], "x"),
        lambda: filter_documents([], {"x": {"bad": 1}}),
        lambda: filter_documents([], {None: 1}),
    ):
        with pytest.raises(MetadataFilterError):
            call()


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []
    assert manifest["category"] == "retrieval"

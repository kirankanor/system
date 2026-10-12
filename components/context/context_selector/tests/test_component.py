"""Tests for the context-selector component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

No optional package is involved. The token counter is injected, so this runs
anywhere Python does -- and so the budget guarantee is measured rather than
assumed.
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
from context_selector import (  # noqa: E402
    ContextError,
    ContextInputError,
    ContextSelector,
)

import pytest  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def scored(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("source_name", f"{cid}.md")
    return core.ScoredDocument(chunk_id=cid, text=text, score=1.0, **kwargs)


@pytest.fixture
def selector():
    return ContextSelector(counter=words)


@pytest.fixture
def candidates():
    return [
        scored("c0", "alpha beta", chunk_index=0),
        scored("c1", "gamma delta", chunk_index=1),
        scored("c2", "epsilon zeta", chunk_index=2),
    ]


# --- success paths ------------------------------------------------------- #


def test_the_budget_is_never_exceeded(selector, candidates):
    for limit in range(1, 12):
        assert selector.select("q", candidates, limit).token_count <= limit


def test_the_token_count_is_the_real_cost_of_the_returned_text(selector, candidates):
    built = selector.select("q", candidates, 100)
    assert built.token_count == words(built.text)


def test_passages_are_joined_in_the_order_given(selector, candidates):
    built = selector.select("q", candidates, 100)
    assert built.text == "alpha beta\n\ngamma delta\n\nepsilon zeta"
    assert [c.chunk_id for c in built.chunks] == ["c0", "c1", "c2"]


def test_the_budget_and_truncation_are_recorded(selector, candidates):
    built = selector.select("q", candidates, 5)
    assert built.budget == 5
    assert built.truncated is True


def test_a_generous_budget_is_not_truncated(selector, candidates):
    assert selector.select("q", candidates, 100).truncated is False


# --- deduplication ------------------------------------------------------- #


def test_a_duplicate_passage_is_kept_once(selector):
    dup = [scored("c0", "same words", chunk_index=0), scored("c1", "same words", chunk_index=1)]
    assert len(selector.select("q", dup, 100).chunks) == 1


def test_the_better_ranked_copy_of_a_duplicate_survives(selector):
    dup = [scored("dense-7", "identical", chunk_index=0), scored("sparse-3", "identical", chunk_index=1)]
    assert [c.chunk_id for c in selector.select("q", dup, 100).chunks] == ["dense-7"]


def test_layout_differences_do_not_defeat_deduplication(selector):
    dup = [scored("c0", "alpha   beta\n\ngamma", chunk_index=0), scored("c1", "alpha beta gamma", chunk_index=1)]
    assert len(selector.select("q", dup, 100).chunks) == 1


def test_a_repeated_chunk_id_is_deduplicated(selector):
    dup = [scored("dup", "first", chunk_index=0), scored("dup", "second", chunk_index=1)]
    assert [c.chunk_id for c in selector.select("q", dup, 100).chunks] == ["dup"]


def test_chunks_without_ids_are_still_deduplicated_by_text(selector):
    untagged = [
        core.ScoredDocument(chunk_id=None, text="same", score=1.0, chunk_index=0),
        core.ScoredDocument(chunk_id=None, text="same", score=1.0, chunk_index=1),
    ]
    assert len(selector.select("q", untagged, 100).chunks) == 1


def test_deduplication_happens_before_the_budget_is_spent(selector):
    dup = [scored("c0", "alpha beta", chunk_index=0), scored("c1", "alpha beta", chunk_index=1),
           scored("c2", "gamma delta", chunk_index=2)]
    assert [c.chunk_id for c in selector.select("q", dup, 4).chunks] == ["c0", "c2"]


def test_distinct_passages_are_all_kept(selector, candidates):
    assert len(selector.select("q", candidates, 100).chunks) == 3


def test_deduplicate_can_be_called_on_its_own(selector):
    dup = [scored("c0", "same", chunk_index=0), scored("c1", "same", chunk_index=1)]
    assert [c.chunk_id for c in selector.deduplicate(dup)] == ["c0"]


# --- an all-duplicate input still returns context ------------------------- #


def test_an_all_duplicate_input_still_returns_context(selector):
    dup = [scored(f"c{i}", "repeated evidence", chunk_index=i) for i in range(6)]
    built = selector.select("q", dup, 100)
    assert built.text == "repeated evidence"
    assert len(built.chunks) == 1
    assert built.token_count == 2


def test_an_all_duplicate_input_is_not_empty(selector):
    dup = [scored(f"c{i}", "same words", chunk_index=i) for i in range(4)]
    assert selector.select("q", dup, 100).text != ""


def test_an_all_duplicate_input_still_honours_the_budget(selector):
    dup = [scored(f"c{i}", "same words", chunk_index=i) for i in range(4)]
    built = selector.select("q", dup, 1)
    assert built.token_count <= built.budget
    assert built.chunks == ()


def test_an_empty_input_is_an_empty_context_not_an_error(selector):
    built = selector.select("q", [], 10)
    assert (built.text, built.chunks, built.token_count, built.truncated) == ("", (), 0, False)
    assert built.budget == 10


# --- selection policy ---------------------------------------------------- #


def test_nothing_is_crammed_in_when_the_first_passage_is_too_big(selector, candidates):
    """A half-passage is worse than none; truncate first if you want it in."""
    built = selector.select("q", candidates, 1)
    assert built.chunks == ()
    assert built.truncated is True


def test_a_passage_is_never_truncated_by_the_selector(selector, candidates):
    """Truncation is token-budget's job, not a selector's."""
    for chunk in selector.select("q", candidates, 100).chunks:
        assert chunk.text in ("alpha beta", "gamma delta", "epsilon zeta")


def test_a_custom_separator_is_used(selector, candidates):
    built = ContextSelector(counter=words, separator=" | ").select("q", candidates, 100)
    assert built.text == "alpha beta | gamma delta | epsilon zeta"


def test_a_character_counter_measures_in_characters(selector, candidates):
    """The unit is whatever the caller's counter says -- and Context says so."""
    built = ContextSelector(counter=len).select("q", candidates, 100)
    assert built.token_count == len(built.text)


def test_the_selector_satisfies_context_builder_protocol(selector):
    assert isinstance(selector, core.ContextBuilderProtocol)


def test_build_is_an_alias_for_select(selector, candidates):
    assert selector.build("q", candidates, 100) == selector.select("q", candidates, 100)


def test_the_results_are_scored_documents(selector, candidates):
    for chunk in selector.select("q", candidates, 100).chunks:
        assert isinstance(chunk, core.ScoredDocument)


# --- failure paths ------------------------------------------------------- #


def test_an_absent_counter_is_rejected_with_the_reason():
    """No silent default: a guessed token count overshoots real budgets."""
    with pytest.raises(ContextInputError) as exc:
        ContextSelector()
    assert "counter is required" in str(exc.value)
    assert "overshoots" in str(exc.value)


def test_a_non_callable_counter_is_rejected():
    with pytest.raises(ContextInputError):
        ContextSelector(counter=12)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(selector, candidates, query):
    with pytest.raises(ContextInputError) as exc:
        selector.select(query, candidates, 10)
    assert "context-selector" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(selector, candidates, query):
    with pytest.raises(ContextInputError):
        selector.select(query, candidates, 10)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10", None])
def test_an_unusable_budget_is_rejected(selector, candidates, limit):
    with pytest.raises(ContextInputError):
        selector.select("q", candidates, limit)


def test_a_non_document_is_rejected_with_its_position(selector, candidates):
    with pytest.raises(ContextInputError) as exc:
        selector.select("q", candidates + ["nope"], 10)
    assert "[3]" in str(exc.value)


@pytest.mark.parametrize("bad", ["a string", 3, None, {"a": 1}])
def test_a_non_sequence_of_chunks_is_rejected(selector, bad):
    with pytest.raises(ContextInputError):
        selector.select("q", bad, 10)


@pytest.mark.parametrize("separator", [3, None, ["x"]])
def test_an_unusable_separator_is_rejected(separator):
    with pytest.raises(ContextInputError):
        ContextSelector(counter=words, separator=separator)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(ContextInputError):
        ContextSelector(counter=words, name=name)


def test_a_broken_counter_is_reported(candidates):
    broken = ContextSelector(counter=lambda text: "nope")
    with pytest.raises(ContextError) as exc:
        broken.select("q", candidates, 10)
    assert "non-negative int" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(ContextInputError, ContextError)
    assert not issubclass(ContextError, ValueError)


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["category"] == "context"

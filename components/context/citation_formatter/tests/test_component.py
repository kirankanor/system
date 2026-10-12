"""Tests for the citation-formatter component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

No optional package is involved.
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
from citation_formatter import (  # noqa: E402
    CitationFormatter,
    CitationInputError,
    CitationOrphanSpanError,
)

import pytest  # noqa: E402


def chunk(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("source_name", f"{cid}.md")
    return core.ScoredDocument(chunk_id=cid, text=text, score=1.0, **kwargs)


def ctx(text: str, chunks=(), **kwargs) -> core.Context:
    return core.Context(
        text=text, chunks=tuple(chunks), budget=kwargs.pop("budget", 100),
        token_count=kwargs.pop("token_count", 0),
        truncated=kwargs.pop("truncated", False),
    )


@pytest.fixture
def formatter():
    return CitationFormatter()


@pytest.fixture
def cited():
    return ctx(
        "Alpha says [1]. Beta adds [2]. See also [1].",
        [chunk("c0", "alpha"), chunk("c1", "beta")],
    )


# --- success paths ------------------------------------------------------- #


def test_valid_spans_do_not_raise(formatter, cited):
    formatter.validate(cited)


def test_text_with_no_spans_is_valid(formatter):
    """A model is allowed to answer without citing."""
    assert formatter.format(ctx("a plain answer", [])) == "a plain answer"


def test_format_returns_the_text_unchanged(formatter, cited):
    """Appending a bibliography would blow the budget upstream enforced."""
    assert formatter.format(cited) == cited.text


def test_spans_report_their_character_offsets(formatter):
    assert formatter.spans("ab [12] cd [3]") == [(12, 3), (3, 11)]


def test_references_lists_cited_numbers_in_order(formatter, cited):
    assert formatter.references(cited) == [1, 2, 1]


def test_bracketed_non_numbers_are_ordinary_text(formatter):
    assert formatter.spans("see [note], [1.5] and [2]") == [(2, 22)]


def test_the_bibliography_lists_distinct_cited_sources_in_order(formatter, cited):
    assert formatter.bibliography(cited) == ["c0.md", "c1.md"]


def test_the_bibliography_omits_uncited_passages(formatter):
    assert formatter.bibliography(ctx("[2]", [chunk("c0", "a"), chunk("c1", "b")])) == ["c1.md"]


def test_the_bibliography_omits_unreachable_spans(formatter):
    assert formatter.bibliography(ctx("[9]", [chunk("c0", "a")])) == []


def test_a_label_falls_back_to_document_id(formatter):
    bare = core.ScoredDocument(chunk_id="c9", document_id="doc-1", text="x", score=1.0, chunk_index=0)
    assert formatter.bibliography(ctx("[1]", [bare])) == ["doc-1"]


def test_a_label_falls_back_to_chunk_id(formatter):
    bare = core.ScoredDocument(chunk_id="c9", text="x", score=1.0, chunk_index=0)
    assert formatter.bibliography(ctx("[1]", [bare])) == ["c9"]


def test_a_label_falls_back_to_a_positional_name(formatter):
    bare = core.ScoredDocument(chunk_id=None, text="x", score=1.0, chunk_index=0)
    assert formatter.bibliography(ctx("[1]", [bare])) == ["passage 1"]


def test_a_custom_pattern_is_supported():
    parenthesised = CitationFormatter(pattern=r"\((\d+)\)")
    assert parenthesised.spans("see (1) here") == [(1, 4)]


def test_zero_based_numbering_is_supported():
    zero = CitationFormatter(one_based=False)
    assert zero.spans("[0]") == [(0, 0)]
    zero.validate(ctx("[0]", [chunk("c0", "a")]))  # resolves


def test_the_repr_shows_the_configuration(formatter):
    shown = repr(formatter)
    assert shown.startswith("CitationFormatter(")
    assert "one_based=True" in shown
    assert "citation-formatter" in shown


# --- failure paths ------------------------------------------------------- #


def test_an_orphan_span_raises_a_typed_error(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(ctx("a [1] and ghost [3]", [chunk("c0", "a")]))


def test_the_error_names_the_offending_span(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.validate(ctx("a [1] ghost [3]", [chunk("c0", "a")]))
    assert "[3]" in str(exc.value)
    assert "[1] through [1]" in str(exc.value)


def test_the_error_carries_every_offending_span(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.validate(ctx("[1] [9] [2] [8]", [chunk("c0", "a"), chunk("c1", "b")]))
    assert [n for n, _ in exc.value.spans] == [9, 8]


def test_zero_is_an_orphan_under_one_based_numbering(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(ctx("zero [0]", [chunk("c0", "a")]))


def test_a_span_with_no_chunks_at_all_is_an_orphan(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.format(ctx("claims [1]", []))
    assert "none at all" in str(exc.value)


def test_a_span_beyond_the_last_chunk_is_an_orphan(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(ctx("[1] [2] [3]", [chunk("c0", "a"), chunk("c1", "b")]))


def test_format_raises_rather_than_rendering_bad_text(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.format(ctx("[7]", [chunk("c0", "a")]))


def test_an_unreachable_span_does_not_reach_the_bibliography(formatter):
    """bibliography() is safe on orphaned text; only format()/validate() raise."""
    assert formatter.bibliography(ctx("[7]", [chunk("c0", "a")])) == []


def test_a_pattern_that_is_not_a_valid_regex_is_rejected():
    with pytest.raises(CitationInputError):
        CitationFormatter(pattern="[")


def test_a_pattern_with_too_many_groups_is_rejected():
    with pytest.raises(CitationInputError) as exc:
        CitationFormatter(pattern=r"\[(\d+)\]-(\d+)")
    assert "exactly one capture group" in str(exc.value)


def test_a_pattern_with_no_group_is_rejected():
    with pytest.raises(CitationInputError):
        CitationFormatter(pattern=r"\[\d+\]")


@pytest.mark.parametrize("one_based", ["yes", 1, None])
def test_an_unusable_one_based_is_rejected(one_based):
    with pytest.raises(CitationInputError):
        CitationFormatter(one_based=one_based)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(CitationInputError):
        CitationFormatter(name=name)


def test_a_non_context_is_rejected(formatter):
    with pytest.raises(CitationInputError) as exc:
        formatter.validate("not a context")
    assert "must be a Context" in str(exc.value)


@pytest.mark.parametrize("text", [None, 3, ["a"], {}])
def test_spans_reject_a_non_string(formatter, text):
    with pytest.raises(CitationInputError):
        formatter.spans(text)


def test_the_orphan_error_is_an_exception_with_span_details(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.validate(ctx("[5]", [chunk("c0", "a")]))
    assert isinstance(exc.value.spans, tuple)
    assert exc.value.spans == ((5, 0),)


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["category"] == "context"

"""Tests for the recursive-chunker component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs -- every expected value below is either
stated in the component's contract or reproducible by hand from the algorithm.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recursive_chunker import (  # noqa: E402
    SEPARATORS,
    Chunk,
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_short_text_fits_in_one_chunk():
    """Text that fits is returned whole, indexed from 0."""
    chunks = chunk("hello", size=10)
    assert [c.text for c in chunks] == ["hello"]
    assert [c.chunk_index for c in chunks] == [0]


def test_empty_text_yields_no_chunks():
    """Nothing to chunk is an outcome, not an error."""
    assert chunk("", size=10) == []


def test_overlap_zero_rejoins_input_exactly():
    """FR invariant: pieces are contiguous slices -- separators are never lost.

    The text deliberately mixes paragraphs, a line break and single spaces so
    every separator tier participates; the join must be byte-for-byte exact.
    """
    text = "Para one.\n\nPara two here.\n\nLast para\nwith a line break and words."
    chunks = chunk(text, size=10)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text


def test_every_chunk_respects_the_size_cap():
    """The size cap holds across every separator tier."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk(text, size=13)
    assert all(len(c.text) <= 13 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_chunk_indices_are_sequential_from_zero():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk(text, size=13)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_separator_priority_prefers_paragraphs_over_hard_cuts():
    """Cuts land on the \\n\\n boundary, never inside it.

    A blind size=5 window would cut "AAAA\\n" / "\\nBBB" / "B"; walking the
    priority yields whole parts with the boundary as its own piece.
    """
    chunks = chunk("AAAA\n\nBBBB", size=5)
    assert [c.text for c in chunks] == ["AAAA", "\n\n", "BBBB"]


def test_absent_separator_is_skipped_not_an_error():
    """\\n\\n does not appear in this text; the walk continues to " "."""
    chunks = chunk("one two three four", size=7, separators=("\n\n", " ", ""))
    assert [c.text for c in chunks] == ["one ", "two ", "three ", "four"]


def test_hard_cut_fallback_when_no_separator_matches():
    """No separator at all: the empty-string sentinel hard-cuts by character."""
    chunks = chunk("abcdefghij", size=4)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_custom_separators_replace_the_default_priority():
    chunks = chunk("a|b|c|d", size=5, separators=("|", ""))
    assert [c.text for c in chunks] == ["a|b|", "c|d"]
    assert "".join(c.text for c in chunks) == "a|b|c|d"


def test_default_separators_constant():
    assert SEPARATORS == ("\n\n", "\n", " ", "")


def test_full_overlap_example_is_deterministic():
    """The contract's worked example: 2 chunks, second starts with the first's tail.

    Pieces are ["alpha ", "beta ", "gamma ", "delta"]; the first three merge to
    "alpha beta gamma " (17 <= 20), "delta" closes it, and the next chunk opens
    with its final 5 characters.
    """
    chunks = chunk("alpha beta gamma delta", size=20, overlap=5)
    assert len(chunks) == 2
    first, second = chunks
    assert first.text == "alpha beta gamma "
    assert second.text == "amma delta"
    assert second.text.startswith(first.text[-5:])
    assert all(len(c.text) <= 20 for c in chunks)


def test_overlap_is_truncated_to_room_so_the_cap_holds():
    """FR-7 semantics: overlap gives way before the size contract does.

    Pieces are 11/12/10/10/10 chars. When "epsilon zeta " (13) fills a chunk
    of size 13 entirely, the following chunk gets ZERO overlap rather than
    exceeding size.
    """
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk(text, size=13, overlap=3)
    assert all(len(c.text) <= 13 for c in chunks)
    assert chunks[0].text == "alpha beta "
    assert chunks[2].text == "epsilon zeta "
    assert chunks[3].text == "ta eta theta "  # overlap re-attached where room exists


def test_chunk_fields_default_to_unset_identity():
    """A chunker cannot know its document; the application fills ids in."""
    c = chunk("hello", size=10)[0]
    assert c.chunk_id is None and c.document_id is None and c.source_name is None
    assert c.page_start is None and c.page_end is None
    assert c.metadata == {}


def test_chunks_are_frozen_and_do_not_share_metadata():
    c = chunk("hello", size=10)[0]
    with pytest.raises(Exception):
        c.text = "changed"
    # default_factory, not a shared class-level dict
    a, b = Chunk(chunk_index=0, text="a"), Chunk(chunk_index=1, text="b")
    assert a.metadata == b.metadata == {}
    assert a.metadata is not b.metadata


def test_chunk_construction_guards():
    """The returned type defends its own invariants (empty text, negative index)."""
    with pytest.raises(ChunkingError):
        Chunk(chunk_index=-1, text="x")
    with pytest.raises(ChunkingError):
        Chunk(chunk_index=0, text="")


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("size", "overlap"), [(10, 10), (10, 15), (1, 2)])
def test_overlap_at_or_above_size_is_rejected_not_clamped(size, overlap):
    """FR-7: never clamped. At equality the step would be zero -- a hang."""
    with pytest.raises(OverlapTooLarge) as exc:
        chunk("some text here", size=size, overlap=overlap)
    assert str(size) in str(exc.value) and str(overlap) in str(exc.value)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),  # zero-width windows are not a thing
        (-1, 0),  # negative size
        (10, -1),  # negative overlap
        (True, 0),  # bool is an int, but size=True would silently chunk to width 1
        (10, 1.5),  # float is not an int
    ],
)
def test_invalid_size_or_overlap_types_and_signs(size, overlap):
    with pytest.raises(InvalidChunkSize):
        chunk("text", size=size, overlap=overlap)


@pytest.mark.parametrize(
    "separators",
    [
        " ",  # a bare str is a sequence of one-char strings -- rejected, not split
        ("\n", 5),  # non-str element
        [1, 2],  # non-str elements
        42,  # not a sequence at all
    ],
)
def test_separators_must_be_a_sequence_of_strings(separators):
    with pytest.raises(InvalidChunkSize):
        chunk("text", size=10, separators=separators)


@pytest.mark.parametrize("text", [None, b"bytes", 3])
def test_non_str_text_raises_chunking_error(text):
    with pytest.raises(ChunkingError):
        chunk(text, size=10)


def test_unknown_option_is_rejected_by_name():
    """A silently dropped option reads as if it worked; reject it instead."""
    with pytest.raises(ChunkingError) as exc:
        chunk("text", size=10, step=2)
    assert "step" in str(exc.value)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches every failure, typed subclasses narrow it."""
    assert issubclass(InvalidChunkSize, ChunkingError)
    assert issubclass(OverlapTooLarge, ChunkingError)
    with pytest.raises(ChunkingError):
        chunk("text", size=0)

"""Tests for the fixed-size-chunker component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs -- every expected value below is either
stated in the component's contract or reproducible by hand from `step =
size - overlap`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixed_size_chunker import (  # noqa: E402
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
    """Text narrower than the window is returned whole, indexed from 0."""
    chunks = chunk("hello", size=10)
    assert [c.text for c in chunks] == ["hello"]
    assert [c.chunk_index for c in chunks] == [0]


def test_empty_text_yields_no_chunks():
    """Nothing to chunk is an outcome, not an error."""
    assert chunk("", size=10) == []


def test_windows_advance_by_size_minus_overlap():
    """The defining arithmetic: size=5, overlap=2 -> step 3 -> starts 0, 3, 6."""
    chunks = chunk("0123456789", size=5, overlap=2)
    assert [c.text for c in chunks] == ["01234", "34567", "6789"]


def test_windows_stop_at_the_end_instead_of_emitting_a_duplicate_tail():
    """Regression: the step must not keep walking after the text is covered.

    With overlap>0 the step is smaller than the window, so `range()` alone
    reaches start=9 and re-emits a single character that the window starting at
    6 already carries. The caller would pay to embed a duplicate and gain
    nothing back. The window at 6 ends at 11 >= 10, so the loop stops there.
    """
    chunks = chunk("0123456789", size=5, overlap=2)
    assert chunks[-1].text == "6789"
    assert len(chunks) == 3
    # The final chunk always runs to the end of the input.
    assert "0123456789".endswith(chunks[-1].text)


def test_no_duplicate_tail_across_a_range_of_configurations():
    """The same redundancy, checked without hand-picking the numbers."""
    text = "abcdefghij"
    for size in (3, 4, 6, 9):
        for overlap in range(size):
            chunks = chunk(text, size=size, overlap=overlap)
            if len(chunks) > 1:
                # The last chunk must reach the end; if it did not, some tail
                # of the input was never covered at all.
                assert text.endswith(chunks[-1].text), (size, overlap)
            # No two windows are byte-identical unless the input itself repeats
            # that substring -- and `text` has no repeated runs, so an exact
            # duplicate consecutive pair always means a redundant tail.
            for previous, current in zip(chunks, chunks[1:]):
                assert current.text != previous.text, (size, overlap, previous.text)


def test_overlap_zero_rejoins_input_exactly():
    """Windows tile the input: nothing lost, nothing duplicated."""
    text = "Para one.\n\nPara two here.\n\nLast para\nwith a line break and words."
    chunks = chunk(text, size=11)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text


def test_every_chunk_respects_the_size_cap():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    chunks = chunk(text, size=13)
    assert chunks
    assert max(len(c.text) for c in chunks) <= 13


def test_chunk_indices_are_sequential_from_zero():
    chunks = chunk("alpha beta gamma delta epsilon", size=7, overlap=3)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_exactly_the_overlap():
    """The heart of FR-7: previous tail == next head, character for character."""
    text = "0123456789abcdef"  # no repeated runs, so sharing is unambiguous
    chunks = chunk(text, size=6, overlap=2)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.text[:2] == previous.text[-2:]


def test_overlap_is_truncated_rather_than_breaking_the_cap():
    """Overlap gives way before the size contract does (FR-7 semantics).

    A window starting at 8 is only two characters from the end, so it cannot
    also carry the full overlap -- it is truncated, and the cap still holds.
    """
    text = "abcdefghijklmnop"  # 16 chars
    chunks = chunk(text, size=8, overlap=5)  # step 3 -> starts 0, 3, 6, 9
    assert [c.text for c in chunks] == ["abcdefgh", "defghijk", "ghijklmn", "jklmnop"]
    assert all(len(c.text) <= 8 for c in chunks)
    # The last window runs to the end rather than being padded to size.
    assert chunks[-1].text == "jklmnop"
    assert len(chunks[-1].text) < 8


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
        (10, False),  # overlap=False is 0 in disguise
    ],
)
def test_invalid_size_or_overlap_types_and_signs(size, overlap):
    with pytest.raises(InvalidChunkSize):
        chunk("text", size=size, overlap=overlap)


@pytest.mark.parametrize("text", [None, b"bytes", 3])
def test_non_str_text_raises_chunking_error(text):
    with pytest.raises(ChunkingError):
        chunk(text, size=10)


def test_unknown_option_is_rejected_by_name():
    """A silently dropped option reads as if it worked; reject it instead."""
    with pytest.raises(ChunkingError) as exc:
        chunk("text", size=10, separator="|")
    assert "separator" in str(exc.value)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches every failure, typed subclasses narrow it."""
    assert issubclass(InvalidChunkSize, ChunkingError)
    assert issubclass(OverlapTooLarge, ChunkingError)
    assert not issubclass(ChunkingError, ValueError)  # distinguishable from bugs
    with pytest.raises(ChunkingError):
        chunk("text", size=0)


def test_failure_messages_name_the_component():
    """NFR-7: a caller can tell which chunker complained without a traceback."""
    with pytest.raises(InvalidChunkSize) as exc:
        chunk("text", size=0)
    assert "fixed-size-chunker" in str(exc.value)

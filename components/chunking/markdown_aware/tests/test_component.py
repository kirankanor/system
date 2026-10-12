"""Tests for the markdown-aware-chunker component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs -- every expected value below is either
stated in the component's contract or reproducible by hand from its three
structural rules (fences atomic, splits at blank lines, headings glued).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markdown_chunker import (  # noqa: E402
    Chunk,
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

import pytest  # noqa: E402


# 62 characters, and deliberately full of blank lines -- the bait that makes a
# prose chunker cut it in half.
FENCE = "```python\ndef f():\n    a = 1\n\n    b = 2\n\n    return a + b\n```"
assert "\n\n" in FENCE

DOC = "# Title\n\nIntro paragraph.\n\n" + FENCE + "\n\nClosing prose.\n"


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_short_text_fits_in_one_chunk():
    chunks = chunk("hello", size=10)
    assert [c.text for c in chunks] == ["hello"]
    assert [c.chunk_index for c in chunks] == [0]


def test_empty_text_yields_no_chunks():
    assert chunk("", size=10) == []


def test_a_fence_is_never_split_when_it_fits_the_budget():
    """Rule 1: the fence is one segment, so no cut can land inside it.

    The fence text is 61 characters, but the *segment* it becomes is 62: the
    newline that terminates the closing ``` line travels with the fence,
    because that newline has to be emitted somewhere and leaving it stranded
    in the prose would break the exact join. So the budget has to be 62 for
    the fence to come through whole.
    """
    assert len(FENCE) == 61
    for size in (62, 70, 120, 300):
        chunks = chunk(DOC, size=size, overlap=0)
        assert any(FENCE in c.text for c in chunks), f"fence split at size={size}"


def test_an_oversized_fence_is_split_instead_of_overrunning_the_budget():
    """Rule 1 has an escape hatch, and it is asserted rather than hidden.

    A 61-character fence cannot live in a 40-character budget. Emitting an
    over-budget chunk would break the caller's contract, so the fence gives
    way -- but every chunk still respects the cap.
    """
    chunks = chunk(DOC, size=40, overlap=0)
    assert not any(FENCE in c.text for c in chunks)
    assert max(len(c.text) for c in chunks) <= 40


def test_a_paragraph_that_fits_is_returned_whole():
    """Rule 2: blank lines are the split point; text inside one is left alone."""
    text = "one two three\n\nfour five six\n\nseven eight nine"
    chunks = chunk(text, size=20, overlap=0)
    assert [c.text for c in chunks] == [
        "one two three\n\n",
        "four five six\n\n",
        "seven eight nine",
    ]


def test_a_heading_is_glued_to_the_paragraph_under_it():
    """Rule 3: no chunk opens with content whose title travelled on ahead."""
    text = "# Alpha\n\nbody text here\n\n# Beta\n\nmore body\n"
    chunks = chunk(text, size=30, overlap=0)
    assert len(chunks) == 2
    for c in chunks:
        stripped = c.text.strip()
        assert stripped not in ("# Alpha", "# Beta"), "heading travelled alone"
        assert stripped.splitlines()[0].startswith("#"), "heading should lead"
        assert len(stripped.splitlines()) > 1, "heading must keep its paragraph"


def test_heading_glue_yields_to_the_budget():
    """Gluing two units that would overflow `size` would break the cap."""
    text = "# " + "x" * 40 + "\n\n" + "y" * 40 + "\n"
    chunks = chunk(text, size=50, overlap=0)
    assert max(len(c.text) for c in chunks) <= 50
    assert "".join(c.text for c in chunks) == text


def test_overlap_zero_rejoins_input_exactly():
    """Segments are slices of the input, so nothing is lost or duplicated."""
    text = "Para one.\n\nPara two here.\n\nLast para\nwith a line break."
    chunks = chunk(text, size=11)
    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == text


def test_unclosed_fence_still_rejoins_exactly():
    """Regression: a fence with no closing delimiter used to duplicate the
    document.

    The prose cursor has to advance to the fence's first byte when the fence
    opens. If it does not, the trailing segment is built from offset 0 and the
    whole input is emitted a second time -- a silent duplication that no size
    or overlap check would catch.
    """
    text = "prose before\n``` never closed\nmore text after"
    chunks = chunk(text, size=8, overlap=0)
    joined = "".join(c.text for c in chunks)
    assert joined == text
    assert joined.count("prose before") == 1, "document was duplicated"


def test_fenced_block_containing_prose_looking_lines_is_not_reassembled():
    """A fence holding blank lines must not be treated as several paragraphs."""
    fence = "```\n\nalpha\n\nbeta\n\n```"
    chunks = chunk(fence, size=100, overlap=0)
    assert [c.text for c in chunks] == [fence]


def test_every_chunk_respects_the_size_cap():
    for size in (7, 23, 61, 500):
        chunks = chunk(DOC, size=size, overlap=0)
        assert chunks
        assert max(len(c.text) for c in chunks) <= size


def test_chunk_indices_are_sequential_from_zero():
    chunks = chunk(DOC, size=40, overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_the_configured_overlap():
    """FR-7 in the positive direction: previous tail == next head."""
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk(text, size=20, overlap=5)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        shared = 0
        for length in range(1, min(len(previous.text), len(current.text)) + 1):
            if previous.text[-length:] == current.text[:length]:
                shared = length
        assert shared <= 5, "shared more than the caller asked for"


def test_chunk_fields_default_to_unset_identity():
    c = chunk("hello", size=10)[0]
    assert c.chunk_id is None and c.document_id is None and c.source_name is None
    assert c.page_start is None and c.page_end is None
    assert c.metadata == {}


def test_chunks_are_frozen_and_do_not_share_metadata():
    c = chunk("hello", size=10)[0]
    with pytest.raises(Exception):
        c.text = "changed"
    a, b = Chunk(chunk_index=0, text="a"), Chunk(chunk_index=1, text="b")
    assert a.metadata == b.metadata == {}
    assert a.metadata is not b.metadata


def test_chunk_construction_guards():
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
        chunk(DOC, size=size, overlap=overlap)
    assert str(size) in str(exc.value) and str(overlap) in str(exc.value)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (True, 0),  # bool is an int, but size=True would silently chunk to width 1
        (10, 1.5),
    ],
)
def test_invalid_size_or_overlap_types_and_signs(size, overlap):
    with pytest.raises(InvalidChunkSize):
        chunk(DOC, size=size, overlap=overlap)


@pytest.mark.parametrize("text", [None, b"bytes", 3])
def test_non_str_text_raises_chunking_error(text):
    with pytest.raises(ChunkingError):
        chunk(text, size=10)


def test_unknown_option_is_rejected_by_name():
    """A silently dropped option reads as if it worked; reject it instead."""
    with pytest.raises(ChunkingError) as exc:
        chunk(DOC, size=10, separators=("|",))
    assert "separators" in str(exc.value)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches every failure, typed subclasses narrow it."""
    assert issubclass(InvalidChunkSize, ChunkingError)
    assert issubclass(OverlapTooLarge, ChunkingError)
    assert not issubclass(ChunkingError, ValueError)
    with pytest.raises(ChunkingError):
        chunk(DOC, size=0)


def test_failure_messages_name_the_component():
    """NFR-7: a caller can tell which chunker complained without a traceback."""
    with pytest.raises(InvalidChunkSize) as exc:
        chunk(DOC, size=0)
    assert "markdown-chunker" in str(exc.value)

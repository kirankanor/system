"""Tests for the structure-aware-chunker component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs -- every expected value below is either
stated in the component's contract or reproducible by hand from its rules
(lines atomic, tables packed by row, headings glued to what follows).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structure_aware_chunker import (  # noqa: E402
    Chunk,
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

import pytest  # noqa: E402


# Four rows of sixteen characters each: 64 in total, every line beginning and
# ending with a pipe so each one reads as a row.
ROW1 = "| name  | qty |\n"
ROW2 = "|-------|-----|\n"
ROW3 = "| bolt  |  12 |\n"
ROW4 = "| nut   |  30 |\n"
TABLE = ROW1 + ROW2 + ROW3 + ROW4
assert len(TABLE) == 64 and len(ROW1) == 16

# The geometry every heading test below shares:
#   PREAMBLE is too wide to pack with anything else at size=22,
#   but each heading does fit with HEAD_BODY, so gluing is observable
#   as "the heading moved forward" rather than "everything fit anyway".
PREAMBLE = "aaaaaaaa\n"  # 8
HEAD_BODY = "bbbbbbbb\n"  # 9
SIZE = 22


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_short_text_fits_in_one_chunk():
    chunks = chunk("hello", size=10)
    assert [c.text for c in chunks] == ["hello"]
    assert [c.chunk_index for c in chunks] == [0]


def test_empty_text_yields_no_chunks():
    assert chunk("", size=10) == []


def test_a_table_that_fits_is_a_single_chunk():
    """Rule 1: a run of rows is one unit, so no cut can land inside it."""
    chunks = chunk(TABLE, size=200, overlap=0)
    assert [c.text for c in chunks] == [TABLE]


def test_an_oversized_table_breaks_at_row_boundaries():
    """Rule 1's escape hatch: rows repack, cells are never cut.

    At size=40 two of the sixteen-character rows fit together (32), a third
    would make 48 -- so the table comes out as two chunks of two whole rows.
    """
    chunks = chunk(TABLE, size=40, overlap=0)
    assert [c.text for c in chunks] == [ROW1 + ROW2, ROW3 + ROW4]
    assert all(len(c.text) <= 40 for c in chunks)


def test_a_table_too_narrow_for_two_rows_emits_one_row_per_chunk():
    """Even when nothing can be packed, each emitted piece is a whole row."""
    chunks = chunk(TABLE, size=30, overlap=0)
    assert [c.text for c in chunks] == [ROW1, ROW2, ROW3, ROW4]
    assert all(len(c.text) <= 30 for c in chunks)


def test_a_numbered_heading_glues_to_the_line_under_it():
    """Rule 2: the heading moves forward rather than closing the last chunk."""
    text = PREAMBLE + "1. Scope\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [
        PREAMBLE,
        "1. Scope\n" + HEAD_BODY,
    ]


def test_a_multi_level_numbered_heading_glues_too():
    """"2.3 Details" carries no trailing bracket, and is still a section."""
    text = PREAMBLE + "2.3 Details\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [
        PREAMBLE,
        "2.3 Details\n" + HEAD_BODY,
    ]


def test_an_all_caps_heading_glues_to_the_line_under_it():
    text = PREAMBLE + "OVERVIEW\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [PREAMBLE, "OVERVIEW\n" + HEAD_BODY]


def test_an_atx_heading_glues_to_the_line_under_it():
    text = PREAMBLE + "## Title\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [PREAMBLE, "## Title\n" + HEAD_BODY]


def test_a_plain_line_is_not_treated_as_a_heading():
    """The negative case, in the exact geometry where gluing would show.

    "Section One" is a title in every language a reader would recognise, but
    it carries no marker this component is willing to trust -- so it packs
    backwards with the preamble like any other line, instead of pulling the
    body forward with it.
    """
    text = PREAMBLE + "Section One\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [
        PREAMBLE + "Section One\n",
        HEAD_BODY,
    ]


def test_a_numbered_list_item_is_not_treated_as_a_heading():
    """`3 items here` starts with a digit and is still not a section.

    Without this the numbered rule would swallow every enumerated list item.
    """
    text = PREAMBLE + "3 items here\n" + HEAD_BODY
    chunks = chunk(text, size=SIZE, overlap=0)
    assert [c.text for c in chunks] == [
        PREAMBLE + "3 items here\n",
        HEAD_BODY,
    ]


def test_a_line_shorter_than_the_budget_is_left_whole():
    """Line atomicity: a short line is never cut to tidy up its neighbours."""
    text = "x" * 50 + "\nshort\n" + "y" * 50 + "\n"
    chunks = chunk(text, size=30, overlap=0)
    for c in chunks:
        if "short" in c.text:
            assert "short\n" in c.text, "the short line was cut in half"


def test_a_line_longer_than_the_budget_breaks_at_word_boundaries():
    """Words first; a character cut only fires for a single over-long word."""
    text = "alpha beta gamma delta\n"
    chunks = chunk(text, size=10, overlap=0)
    assert [c.text for c in chunks] == ["alpha ", "beta ", "gamma ", "delta\n"]


def test_overlap_zero_rejoins_input_exactly():
    """Units are contiguous slices of the input, newlines included."""
    text = TABLE + "\n2.3 Details\nSome trailing prose.\n"
    for size in (7, 23, 64, 500):
        chunks = chunk(text, size=size, overlap=0)
        assert chunks
        assert "".join(c.text for c in chunks) == text, f"lossless failed at {size}"


def test_every_chunk_respects_the_size_cap():
    text = TABLE + "OVERVIEW\n" + "prose " * 30 + "\n"
    for size in (1, 7, 23, 61, 500):
        chunks = chunk(text, size=size, overlap=0)
        assert chunks
        assert max(len(c.text) for c in chunks) <= size


def test_chunk_indices_are_sequential_from_zero():
    chunks = chunk(TABLE, size=30, overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_the_configured_overlap():
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
        chunk(TABLE, size=size, overlap=overlap)
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
        chunk(TABLE, size=size, overlap=overlap)


@pytest.mark.parametrize("text", [None, b"bytes", 3])
def test_non_str_text_raises_chunking_error(text):
    with pytest.raises(ChunkingError):
        chunk(text, size=10)


def test_unknown_option_is_rejected_by_name():
    """A silently dropped option reads as if it worked; reject it instead."""
    with pytest.raises(ChunkingError) as exc:
        chunk(TABLE, size=10, separators=("|",))
    assert "separators" in str(exc.value)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches every failure, typed subclasses narrow it."""
    assert issubclass(InvalidChunkSize, ChunkingError)
    assert issubclass(OverlapTooLarge, ChunkingError)
    assert not issubclass(ChunkingError, ValueError)
    with pytest.raises(ChunkingError):
        chunk(TABLE, size=0)


def test_failure_messages_name_the_component():
    """NFR-7: a caller can tell which chunker complained without a traceback."""
    with pytest.raises(InvalidChunkSize) as exc:
        chunk(TABLE, size=0)
    assert "structure-aware-chunker" in str(exc.value)

"""Tests for the token-chunker component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

`tiktoken` is an *optional* dependency (FR-3) and is not installed here --
installing it would modify dependency files, which this library forbids. Every
windowing test therefore runs against a character-level codec, one character
per token, which makes expected boundaries exactly predictable while still
exercising the real window/overlap/decode path. The FR-3 guard itself is
tested separately against the genuine import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# Import the component the way a copy of it would be imported: the file that
# sits next to this tests/ directory, nothing else.
sys.path.insert(0, str(COMPONENT))

import token_chunker  # noqa: E402
from token_chunker import (  # noqa: E402
    DEFAULT_ENCODING,
    MISSING_DEPENDENCY_MESSAGE,
    Chunk,
    ChunkingError,
    InvalidChunkSize,
    InvalidEncoding,
    MissingTokenizer,
    OverlapTooLarge,
    chunk,
)

import pytest  # noqa: E402


class CharacterCodec:
    """One character per token, so token windows equal character windows."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)


@pytest.fixture(autouse=True)
def _character_tokenizer(monkeypatch):
    """Give the chunker a tokenizer so its windowing runs for real.

    Without this the chunker could only ever raise, since tiktoken is absent.
    The FR-3 guard is tested separately against the real import.
    """
    monkeypatch.setattr(
        token_chunker, "_load_tokenizer", lambda encoding: CharacterCodec()
    )


# Captured at import time, before any fixture has patched it: the FR-3 test
# below restores this to prove the *real* import guard fires.
_REAL_LOAD_TOKENIZER = token_chunker._load_tokenizer


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_short_text_fits_in_one_chunk():
    chunks = chunk("hello", size=10)
    assert [c.text for c in chunks] == ["hello"]
    assert [c.chunk_index for c in chunks] == [0]


def test_empty_text_yields_no_chunks():
    """Nothing to chunk is an outcome, not an error."""
    assert chunk("", size=10) == []


def test_empty_text_never_touches_the_optional_dependency(monkeypatch):
    """An empty string returns [] without loading a tokenizer at all.

    Restored to the real loader here, so the assertion holds on a machine that
    has tiktoken *and* on one that does not -- the point is that the code path
    never reaches the import either way.
    """
    monkeypatch.setattr(token_chunker, "_load_tokenizer", _REAL_LOAD_TOKENIZER)

    def _must_not_be_called(encoding):  # pragma: no cover - the assertion
        raise AssertionError("the tokenizer was loaded for empty text")

    monkeypatch.setattr(token_chunker, "_load_tokenizer", _must_not_be_called)
    assert chunk("", size=10) == []


def test_windows_advance_by_size_minus_overlap():
    """The defining arithmetic: size=4, overlap=1 -> step 3 -> starts 0, 3, 6."""
    chunks = chunk("abcdefghij", size=4, overlap=1)
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]


def test_windows_stop_at_the_end_instead_of_emitting_a_duplicate_tail():
    """Regression: the step must not keep walking after the tokens are covered.

    With overlap>0 the step is smaller than the window, so `range()` alone
    reaches start=9 and re-emits one token the window at 6 already carries.
    The caller would pay to embed a duplicate and gain nothing back.
    """
    chunks = chunk("abcdefghij", size=4, overlap=1)
    assert len(chunks) == 3
    assert chunks[-1].text == "ghij"
    assert "abcdefghij".endswith(chunks[-1].text)


def test_windows_without_overlap_rejoin_exactly():
    """Windows tile the input: nothing lost, nothing duplicated."""
    chunks = chunk("abcdefghij", size=4, overlap=0)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]
    assert "".join(c.text for c in chunks) == "abcdefghij"


def test_every_chunk_respects_the_size_cap_in_tokens():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    for size in (3, 7, 19):
        chunks = chunk(text, size=size, overlap=0)
        assert chunks
        assert max(len(c.text) for c in chunks) <= size


def test_chunk_indices_are_sequential_from_zero():
    chunks = chunk("abcdefghijklmnop", size=5, overlap=2)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_share_the_configured_overlap():
    """FR-7 in the positive direction: previous tail == next head."""
    text = "abcdefghijklmnopqrstuvwxyz0123"  # no repeated runs
    chunks = chunk(text, size=8, overlap=3)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.text[-3:] == current.text[:3]


def test_windows_decoding_to_nothing_are_dropped_and_indices_stay_contiguous():
    """A tokenizer may map a window to nothing; indices are assigned after."""

    class DroppingCodec:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(value) for value in ids if chr(value) != "!")

    token_chunker._load_tokenizer = lambda encoding: DroppingCodec()
    try:
        # windows "ab", "!!", "cd" -> the middle decodes to "" and is dropped.
        chunks = chunk("ab!!cd", size=2, overlap=0)
    finally:
        token_chunker._load_tokenizer = lambda encoding: CharacterCodec()
    assert [c.text for c in chunks] == ["ab", "cd"]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_the_manifest_declares_the_dependency_optional_and_nothing_required():
    """The manifest and the module must not disagree about what is needed."""
    manifest = json.loads(
        (COMPONENT / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["required_packages"] == [], "token-chunker must need nothing"
    assert [entry["name"] for entry in manifest["optional_packages"]] == ["tiktoken"]
    assert manifest["optional_packages"][0]["specifier"] == ">=0.7.0"
    assert DEFAULT_ENCODING == "cl100k_base"


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
        chunk("some text here", size=size, overlap=overlap)
    assert str(size) in str(exc.value) and str(overlap) in str(exc.value)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [
        (0, 0),  # zero-width token windows are not a thing
        (-1, 0),  # negative size
        (10, -1),  # negative overlap
        (True, 0),  # bool is an int, but size=True would silently budget one token
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


@pytest.mark.parametrize("encoding", ["", 3, None])
def test_unusable_encoding_is_rejected_before_any_import(encoding):
    """Checked before the tokenizer loads, so no tiktoken is needed to prove it."""
    with pytest.raises(InvalidEncoding):
        chunk("text", size=10, encoding=encoding)


def test_unknown_option_is_rejected_by_name():
    """A silently dropped option reads as if it worked; reject it instead."""
    with pytest.raises(ChunkingError) as exc:
        chunk("text", size=10, separators=("|",))
    assert "separators" in str(exc.value)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches every failure, typed subclasses narrow it."""
    assert issubclass(InvalidChunkSize, ChunkingError)
    assert issubclass(OverlapTooLarge, ChunkingError)
    assert issubclass(InvalidEncoding, ChunkingError)
    assert not issubclass(ChunkingError, ValueError)
    with pytest.raises(ChunkingError):
        chunk("text", size=0)


def test_missing_tokenizer_is_an_import_error_not_a_chunking_error():
    """FR-3: callers guard optional components with `except ImportError`."""
    assert issubclass(MissingTokenizer, ImportError)
    assert not issubclass(MissingTokenizer, ChunkingError)


def test_optional_dependency_guard_carries_the_manifest_message(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError.

    Restores the genuine loader so the real `import tiktoken` runs rather than
    the fixture's codec, then confirms its absence surfaces the manifest's
    message byte for byte -- not a paraphrase that could drift from it.

    Preconditions: tiktoken is absent. Installing it would modify dependency
    files, which this library forbids, so the precondition holds by policy.
    """
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(token_chunker, "_load_tokenizer", _REAL_LOAD_TOKENIZER)
    with pytest.raises(ImportError) as exc:
        chunk("hello world", size=4, overlap=1)
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == MISSING_DEPENDENCY_MESSAGE


def test_failure_messages_name_the_component():
    """NFR-7: a caller can tell which chunker complained without a traceback."""
    with pytest.raises(InvalidChunkSize) as exc:
        chunk("text", size=0)
    assert "token-chunker" in str(exc.value)

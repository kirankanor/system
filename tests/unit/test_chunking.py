"""Cross-component contract for the five chunking components (T-06's gate).

    python -m pytest tests/unit/test_chunking.py -q

Each component also ships its own ``tests/test_component.py`` covering its
internals; this file covers the boundary the five *share*, which is what FR-5
interchangeability means in practice -- a caller can swap one chunker for
another without changing a line.

It is also where the specification's chunking requirements are pinned down:

- **FR-4** -- real implementations with typed errors: every failure below
  raises a component-defined exception class, never a bare ``ValueError``.
- **FR-7** -- overlap is documented *and* enforced: ``overlap >= size`` is
  rejected with a typed error rather than silently clamped.
- **AC-4** -- each component gets a success path and a failure path here as
  well as in its own test file.

Components are imported the way a copy of them would be imported (AC-13):
their own file, with its directory on ``sys.path`` -- never
``rag_code_library``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHUNKING = ROOT / "components" / "chunking"

# Import each component exactly as a standalone copy would be: its own
# directory on the path, its own module name. Module names are unique, so the
# five can sit on the path together.
for _name in (
    "fixed_size",
    "recursive",
    "markdown_aware",
    "structure_aware",
    "token_based",
):
    sys.path.insert(0, str(CHUNKING / _name))

import fixed_size_chunker as fixed  # noqa: E402
import markdown_chunker as markdown  # noqa: E402
import recursive_chunker as recursive  # noqa: E402
import structure_aware_chunker as structure  # noqa: E402
import token_chunker as token  # noqa: E402

#: label -> module. All five are exercised through every shared-contract test.
CHUNKERS = {
    "fixed-size": fixed,
    "recursive": recursive,
    "markdown": markdown,
    "structure-aware": structure,
    "token": token,
}

#: A single short sentence whose chunk boundaries are identical under all five
#: chunkers at size=20/overlap=5, which makes exact-overlap assertions possible
#: without reaching into any implementation's internals.
SENTENCE = "alpha beta gamma delta"

#: A longer, deliberately non-periodic paragraph for lossless/cap assertions.
#: Non-periodic matters: with repeating text, two adjacent chunks really can be
#: identical, and "shared characters" would then mean something else.
PARAGRAPH = (
    "Retrieval augmented generation grounds an answer in retrieved passages. "
    "The chunker decides where those passages begin and end, and a boundary "
    "that falls inside a code block or a table changes what the model sees."
)

SIZES = (17, 40, 96, 400)


@pytest.fixture(autouse=True)
def _character_tokenizer(monkeypatch):
    """Give token-chunker a tokenizer so its windowing runs for real.

    ``tiktoken`` is an *optional* dependency (FR-3) and is not installed here,
    so without this the token chunker could only ever raise. A one-character-
    per-token codec keeps expected boundaries trivially predictable while still
    exercising the genuine window/overlap/decode path -- the logic under test.
    The FR-3 guard itself is tested separately against the real import.
    """

    class CharacterCodec:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(value) for value in ids)

    monkeypatch.setattr(token, "_load_tokenizer", lambda encoding: CharacterCodec())


# Captured at import time, while no fixture has patched it yet: the FR-3 test
# below restores this to prove the *real* import guard fires.
_REAL_LOAD_TOKENIZER = token._load_tokenizer


def _shared(prev: str, nxt: str) -> int:
    """Length of the longest tail of ``prev`` equal to a prefix of ``nxt``.

    Checked from the longest candidate down: a one-character check would pass or
    fail on the *alignment* rather than the overlap, because a real overlap of
    ``"g.\\n\\n"`` ends the previous chunk with ``"\\n"`` while starting the next
    with ``"g"``.
    """
    for length in range(min(len(prev), len(nxt)), 0, -1):
        if prev[-length:] == nxt[:length]:
            return length
    return 0


# --------------------------------------------------------------------------- #
# FR-5: the five are interchangeable at their boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_signature_is_the_same_shape(mod):
    """One call site, five implementations: keyword arguments in, chunks out."""
    chunks = mod.chunk(PARAGRAPH, size=40, overlap=5)
    assert isinstance(chunks, list)
    assert chunks and all(type(c).__name__ == "Chunk" for c in chunks)
    # The fields every stage downstream relies on, identically named everywhere.
    for field in ("chunk_index", "text", "metadata"):
        assert hasattr(chunks[0], field)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_chunk_index_is_contiguous_from_zero(mod):
    chunks = mod.chunk(PARAGRAPH, size=40, overlap=5)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_chunk_is_frozen(mod):
    """A chunk handed to the next stage must not be mutable underneath it."""
    chunk = mod.chunk(SENTENCE, size=20)[0]
    with pytest.raises(Exception):
        chunk.text = "rewritten"


# --------------------------------------------------------------------------- #
# FR-7: overlap is enforced, never clamped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_overlap_equal_to_size(mod):
    """``overlap == size`` gives a step of zero: chunking would never advance."""
    with pytest.raises(mod.OverlapTooLarge) as exc:
        mod.chunk(SENTENCE, size=10, overlap=10)
    message = str(exc.value)
    assert "10" in message  # names the setting at fault
    assert "size" in message


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_overlap_larger_than_size(mod):
    with pytest.raises(mod.OverlapTooLarge):
        mod.chunk(SENTENCE, size=10, overlap=11)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_negative_overlap(mod):
    with pytest.raises(mod.InvalidChunkSize) as exc:
        mod.chunk(SENTENCE, size=10, overlap=-1)
    assert "overlap" in str(exc.value)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_zero_size(mod):
    with pytest.raises(mod.InvalidChunkSize) as exc:
        mod.chunk(SENTENCE, size=0, overlap=0)
    assert "size" in str(exc.value)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_bool_size(mod):
    """``True`` is an ``int`` -- accepting it would silently chunk to width 1."""
    with pytest.raises(mod.InvalidChunkSize):
        mod.chunk(SENTENCE, size=True)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
@pytest.mark.parametrize("kwargs", [{"size": "20"}, {"size": 20, "overlap": "5"}])
def test_rejects_non_int_configuration(mod, kwargs):
    with pytest.raises(mod.InvalidChunkSize):
        mod.chunk(SENTENCE, **kwargs)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_exact_overlap_is_shared_between_consecutive_chunks(mod):
    """The heart of FR-7: the previous chunk's tail is the next chunk's head."""
    chunks = mod.chunk(SENTENCE, size=20, overlap=5)
    assert len(chunks) == 2
    assert chunks[1].text[:5] == chunks[0].text[-5:]
    assert _shared(chunks[0].text, chunks[1].text) == 5


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
@pytest.mark.parametrize("overlap", [0, 1, 7, 16])
def test_overlap_never_exceeds_the_configured_amount(mod, overlap):
    """Enforcement, not just presence: sharing more than asked is also a bug."""
    chunks = mod.chunk(PARAGRAPH, size=40, overlap=overlap)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert _shared(previous.text, current.text) <= overlap


# --------------------------------------------------------------------------- #
# FR-4: input validation and typed errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_errors_are_typed_not_bare_valueerror(mod):
    """A caller must be able to catch this component's failures selectively."""
    assert issubclass(mod.ChunkingError, Exception)
    assert not issubclass(mod.ChunkingError, ValueError)
    assert issubclass(mod.InvalidChunkSize, mod.ChunkingError)
    assert issubclass(mod.OverlapTooLarge, mod.ChunkingError)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_non_str_text(mod):
    with pytest.raises(mod.ChunkingError) as exc:
        mod.chunk(42, size=10)
    assert "str" in str(exc.value)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_rejects_unknown_option_by_name(mod):
    """Rejected by name rather than ignored: a dropped option must be audible."""
    with pytest.raises(mod.ChunkingError) as exc:
        mod.chunk(SENTENCE, size=10, separator="|")
    assert "separator" in str(exc.value)


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
def test_empty_text_yields_no_chunks_not_an_error(mod):
    assert mod.chunk("", size=10, overlap=2) == []


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
@pytest.mark.parametrize("size", SIZES)
def test_no_chunk_exceeds_size(mod, size):
    chunks = mod.chunk(PARAGRAPH, size=size, overlap=0)
    assert chunks
    assert max(len(c.text) for c in chunks) <= size


@pytest.mark.parametrize("mod", list(CHUNKERS.values()), ids=list(CHUNKERS))
@pytest.mark.parametrize("size", SIZES)
def test_chunks_rejoin_to_the_input_without_overlap(mod, size):
    """Without overlap the pieces tile the input exactly -- nothing dropped."""
    chunks = mod.chunk(PARAGRAPH, size=size, overlap=0)
    assert "".join(c.text for c in chunks) == PARAGRAPH


# --------------------------------------------------------------------------- #
# Per-component behaviour: what makes each one worth having
# --------------------------------------------------------------------------- #


def test_fixed_size_advances_by_size_minus_overlap():
    chunks = fixed.chunk("0123456789", size=5, overlap=2)
    # step = 3: windows start at 0, 3, 6
    assert [c.text for c in chunks] == ["01234", "34567", "6789"]


def test_fixed_size_without_overlap_tiles_exactly():
    chunks = fixed.chunk("0123456789", size=4, overlap=0)
    assert [c.text for c in chunks] == ["0123", "4567", "89"]


def test_recursive_prefers_paragraph_breaks_over_hard_cuts():
    text = "first para\n\nsecond para\n\nthird para"
    chunks = recursive.chunk(text, size=20, overlap=0)
    # A hard cut would land mid-word; a paragraph break never does.
    assert all(not c.text.rstrip().endswith("par") for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_recursive_accepts_a_custom_separator_priority():
    text = "aaa|bbb|ccc|ddd"
    chunks = recursive.chunk(text, size=8, overlap=0, separators=("|", ""))
    assert [c.text for c in chunks] == ["aaa|bbb|", "ccc|ddd"]


def test_recursive_falls_back_to_a_hard_cut_when_no_separator_matches():
    chunks = recursive.chunk("abcdefghij", size=4, overlap=0, separators=("|", ""))
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_recursive_rejects_a_bare_string_separator():
    with pytest.raises(recursive.InvalidChunkSize) as exc:
        recursive.chunk("a b", size=4, separators="\n")
    assert "sequence" in str(exc.value)


def test_recursive_rejects_a_non_string_separator_element():
    with pytest.raises(recursive.InvalidChunkSize):
        recursive.chunk("a b", size=4, separators=[1])


FENCE = "```python\ndef f():\n    a = 1\n\n    b = 2\n\n    return a + b\n```"
DOC = "# Title\n\nIntro paragraph.\n\n" + FENCE + "\n\nClosing prose.\n"


def test_markdown_never_splits_a_fence_that_fits():
    for size in (62, 80, 140):
        chunks = markdown.chunk(DOC, size=size, overlap=0)
        assert any(FENCE in c.text for c in chunks), f"split at size={size}"


def test_markdown_splits_a_fence_that_cannot_fit():
    """Exceeding the budget is the worse failure, so the fence gives way."""
    assert len(FENCE) > 40
    chunks = markdown.chunk(DOC, size=40, overlap=0)
    assert not any(FENCE in c.text for c in chunks)
    assert max(len(c.text) for c in chunks) <= 40


def test_markdown_keeps_a_short_paragraph_whole():
    text = "one two three\n\nfour five six"
    chunks = markdown.chunk(text, size=30, overlap=0)
    assert "one two three\n\n" in "".join(c.text for c in chunks)


def test_markdown_glues_a_heading_to_its_paragraph():
    text = "# Alpha\n\nbody text here\n\n# Beta\n\nmore body\n"
    chunks = markdown.chunk(text, size=40, overlap=0)
    for chunk in chunks:
        stripped = chunk.text.strip()
        if stripped.startswith("#"):
            # The heading must not have travelled alone into its own chunk.
            assert stripped != "# Alpha" and stripped != "# Beta"


def test_markdown_stays_lossless_with_an_unclosed_fence():
    """Regression: an unclosed fence used to re-emit the whole document."""
    text = "prose before\n``` never closed\nmore text after"
    chunks = markdown.chunk(text, size=8, overlap=0)
    assert "".join(c.text for c in chunks) == text


TABLE = "| name  | qty |\n|-------|-----|\n| bolt  |  12 |\n| nut   |  30 |\n"


def test_structure_keeps_a_table_that_fits_whole():
    chunks = structure.chunk(TABLE + "\ntrailing prose\n", size=200, overlap=0)
    assert any(TABLE in c.text for c in chunks)


def test_structure_breaks_an_oversized_table_at_rows_not_through_a_cell():
    chunks = structure.chunk(TABLE, size=30, overlap=0)
    assert max(len(c.text) for c in chunks) <= 30
    for chunk in chunks:
        # A row is either entirely present or entirely absent -- never half.
        if "30" in chunk.text:
            assert "|  30 |" in chunk.text


def test_structure_recognises_a_numbered_section_heading():
    text = "1. Scope\nThe scope line follows immediately.\n"
    chunks = structure.chunk(text, size=60, overlap=0)
    assert any(c.text.startswith("1. Scope") for c in chunks)


def test_structure_does_not_treat_a_sentence_as_a_heading():
    """The ALL-CAPS heuristic must not fire on ordinary prose."""
    text = "this is an ordinary sentence\nthe next line follows it\n"
    units = structure._units(text)
    assert not any(structure._is_heading(line) for line in units if line.strip())


def test_token_chunker_guards_its_optional_dependency(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError."""
    manifest = json.loads(
        (CHUNKING / "token_based" / "manifest.json").read_text(encoding="utf-8")
    )
    # Restore the real loader so the actual import guard runs rather than the
    # fixture's codec, then confirm tiktoken's absence surfaces the manifest's
    # message byte for byte -- not a paraphrase that could drift from it.
    monkeypatch.setattr(token, "_load_tokenizer", _REAL_LOAD_TOKENIZER)
    with pytest.raises(ImportError) as exc:
        token.chunk("hello world", size=4, overlap=1)
    assert str(exc.value) == manifest["missing_dependency_message"]


def test_token_windows_are_measured_in_tokens():
    """With a character codec, one token == one character, so this is exact.

    Only three windows: the one starting at token 6 reaches the end, so the
    step never advances to token 9 -- a fourth window there would be a
    duplicate of characters the third already carries.
    """
    chunks = token.chunk("abcdefghij", size=4, overlap=1)
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]


def test_token_without_overlap_rejoins_exactly():
    chunks = token.chunk("abcdefghij", size=4, overlap=0)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_token_rejects_an_empty_encoding():
    with pytest.raises(token.InvalidEncoding) as exc:
        token.chunk("x", size=4, encoding="")
    assert "encoding" in str(exc.value)


def test_token_rejects_a_non_string_encoding():
    with pytest.raises(token.InvalidEncoding):
        token.chunk("x", size=4, encoding=7)

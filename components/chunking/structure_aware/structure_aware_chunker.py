"""Split text on document *structure*: sections, tables, list items, lines.

The other four chunkers all assume prose. This one assumes a document with a
shape -- a report, an outline, a spec, a log -- where the structure carries the
meaning and cutting across it destroys it.

What it recognises
------------------
- **Section headings**: ATX (``# Title``), numbered (``1. Intro``, ``2.3
  Details``), or a line in ALL CAPS. Each heading is glued to what follows it,
  so a chunk never opens with content whose title went to the previous chunk.
- **Tables**: consecutive ``| row |`` lines form one unit that is kept whole.
  A table split down the middle is a table with no column meanings.
- **Lines**: a line -- a list item, a key/value pair, a log record -- is never
  split across chunks. Structure-aware means the units stay intact.

Only a unit that genuinely cannot fit is broken down further, at word
boundaries and then characters, because exceeding the budget is worse than
losing a seam.

Overlap semantics (FR-7)
------------------------
Lines and table runs are contiguous slices of the input (``"".join(pieces) ==
text``), then merged greedily into chunks of at most ``size``. Consecutive
chunks share text: when a chunk closes, its final ``overlap`` characters become
the head of the next one. The tail is truncated to the room the incoming unit
leaves, so overlap yields before the size contract does -- a chunk is never
larger than ``size``. ``overlap >= size`` is rejected outright and never
clamped: at equality the step is zero and chunking never advances, which is
exactly the silent failure FR-7 exists to rule out.

Self-contained on purpose
-------------------------
The manifest declares no dependencies, so this file imports only the standard
library and defines the types it returns. :class:`Chunk` mirrors the field
names of the canonical ``core_types.Chunk``, and ``chunk()`` matches the shared
``ChunkingProtocol.chunk(text, *, size, overlap, **kw)`` shape structurally --
what makes FR-5's "swap the component, keep the caller" true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chunk",
    "ChunkingError",
    "InvalidChunkSize",
    "OverlapTooLarge",
    "chunk",
]

#: ATX heading: 1-6 hashes then a space.
_ATX = re.compile(r"^#{1,6}\s")
#: Numbered section: "1. Intro", "2.3 Details", "12) Note". Two alternatives
#: rather than one, because the trailing ``[.)]`` is what makes "1. Intro"
#: distinct from an ordinary sentence -- but a multi-level heading such as
#: "2.3 Details" carries no trailing bracket at all. With ``[.)]`` made optional
#: instead, "3 items here" (a list item, not a section) would match too, so the
#: multi-level form is spelled out on its own.
_NUMBERED = re.compile(r"^\d+(?:\.\d+)+\s+|^\d+[.)]\s+")
#: ALL CAPS title: no lowercase anywhere, at least three characters. Deliberately
#: a heuristic -- mistaking a shouted line for a heading costs a shorter chunk,
#: never a broken one.
_CAPS = re.compile(r"^[^a-z]*[A-Z0-9][A-Z0-9 &\-_/]{2,}[^a-z]*$")
#: A table row: starts and ends with a pipe.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class ChunkingError(Exception):
    """Base class for every error this component raises (FR-4).

    Catching :class:`ChunkingError` catches all of them; catching
    :class:`OverlapTooLarge` catches the overlap case alone. Nothing here is a
    bare ``ValueError``, so a caller can tell this component's failures apart
    from its own programming errors.
    """


class InvalidChunkSize(ChunkingError):
    """``size`` or ``overlap`` was not a usable configuration."""


class OverlapTooLarge(ChunkingError):
    """``overlap >= size`` (FR-7): rejected rather than clamped."""


@dataclass(frozen=True)
class Chunk:
    """One produced chunk.

    Field names match ``core_types.Chunk`` so a caller can pass these to
    embedding or indexing code unchanged. Identity fields stay optional because
    a chunker cannot know which document its text came from.
    """

    chunk_index: int
    text: str
    chunk_id: str | None = None
    document_id: str | None = None
    source_name: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ChunkingError(
                f"structure-aware-chunker: chunk_index must be >= 0, got "
                f"{self.chunk_index}"
            )
        if not self.text:
            raise ChunkingError(
                "structure-aware-chunker: a Chunk's text must not be empty; "
                "the chunker never emits one, so this is a construction bug"
            )


def _validate(size: Any, overlap: Any) -> None:
    """Reject unusable configuration before any work happens (FR-4).

    Types are checked explicitly rather than left to ``range()`` to fail:
    ``True`` is an ``int`` in Python, and ``size=True`` would silently chunk to
    width 1.
    """
    for name, value in (("size", size), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidChunkSize(
                f"structure-aware-chunker: {name} must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    if size <= 0:
        raise InvalidChunkSize(
            f"structure-aware-chunker: size must be > 0, got {size}; "
            "a chunker cannot cut text into windows of zero or negative width"
        )
    if overlap < 0:
        raise InvalidChunkSize(
            f"structure-aware-chunker: overlap must be >= 0, got {overlap}; "
            "use 0 for non-overlapping chunks"
        )
    if overlap >= size:
        raise OverlapTooLarge(
            f"structure-aware-chunker: overlap ({overlap}) must be smaller than "
            f"size ({size}). At equal, the step would be 0 and chunking would "
            f"never advance; larger, and a chunk would repeat itself. Reduce "
            f"overlap below {size}, or raise size above {overlap}."
        )


def _is_heading(line: str) -> bool:
    """True if this line reads as a section title (see module docstring)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        _ATX.match(stripped) or _NUMBERED.match(stripped) or _CAPS.match(stripped)
    )


def _lines(text: str) -> list[str]:
    """Split into lines, each keeping its trailing newline.

    Lossless by construction: the newline that ``split`` removed is handed back
    to the line it terminated, and a trailing newline yields a final empty part
    so ``"".join(_lines(text)) == text``.
    """
    parts = text.split("\n")
    if len(parts) == 1:
        return parts
    return [part + "\n" for part in parts[:-1]] + [parts[-1]]


def _units(text: str) -> list[str]:
    """Group lines into structural units: a table run counts as one unit.

    Lossless: units are built by concatenating whole lines in order, so they
    rejoin to ``text`` exactly.
    """
    lines = _lines(text)
    units: list[str] = []
    index = 0
    while index < len(lines):
        if _TABLE_ROW.match(lines[index].rstrip("\n")):
            run = ""
            while index < len(lines) and _TABLE_ROW.match(lines[index].rstrip("\n")):
                run += lines[index]
                index += 1
            units.append(run)
        else:
            units.append(lines[index])
            index += 1
    return units


def _split_words(text: str, size: int) -> list[str]:
    """Break a line that is too big: word boundaries first, then characters.

    The cut points are the byte offsets where a word *begins*, plus the end of
    the text, so every piece is a genuine slice of ``text`` and the pieces
    rejoin exactly. A piece closes at the last boundary that still fits -- and
    it has to be the boundary, not the word's own end, because the space that
    trails a word travels with the piece. Testing the word alone is what makes
    a piece come out one byte over budget.

    The character cut fires only for a single word longer than the whole
    budget, where there is nothing semantic left to cut on.
    """
    boundaries = {0, len(text)}
    pos = 0
    for word in text.split(" "):
        boundaries.add(pos)
        pos += len(word) + 1
    ordered = sorted(boundary for boundary in boundaries if boundary <= len(text))

    pieces: list[str] = []
    start = 0
    while start < len(text):
        best = None
        for boundary in ordered:
            if boundary <= start:
                continue
            if boundary - start <= size:
                best = boundary
            else:
                break
        if best is None:
            best = start + size  # one word longer than the whole budget
        pieces.append(text[start:best])
        start = best
    return [piece for piece in pieces if piece]


def _fit(units: list[str], size: int) -> list[str]:
    """Guarantee every unit fits ``size``, breaking oversized ones gently.

    A multi-line unit (a table run) breaks at line boundaries first, so rows
    survive and re-pack in :func:`_merge` instead of being cut through a cell;
    a single line breaks at word boundaries. The line branch cannot loop on
    itself: a line from :func:`_lines` carries at most its own trailing newline
    and no other, so once a piece is down to one line the word branch takes it.
    Empty pieces are dropped -- they contribute nothing to the join, so
    dropping them stays lossless.
    """
    fitted: list[str] = []
    for unit in units:
        if not unit:
            continue
        if len(unit) <= size:
            fitted.append(unit)
            continue
        lines = [line for line in _lines(unit) if line]
        if len(lines) > 1:
            for line in lines:
                if len(line) <= size:
                    fitted.append(line)
                else:
                    fitted.extend(_split_words(line, size))
        else:
            fitted.extend(_split_words(unit, size))
    return fitted


def _glue_headings(units: list[str], size: int) -> list[str]:
    """Attach each heading line to the unit under it, when both fit.

    Without this, a chunk can end with ``2.1 Scope`` and drop the section title
    into the next chunk, leaving content nobody can place. If the two do not
    fit together the heading stays alone -- a heading larger than the budget
    cannot be honoured, and swallowing the following unit would be worse than
    the seam.
    """
    glued: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if (
            _is_heading(unit)
            and index + 1 < len(units)
            and len(unit) + len(units[index + 1]) <= size
        ):
            glued.append(unit + units[index + 1])
            index += 2
        else:
            glued.append(unit)
            index += 1
    return glued


def _merge(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge units to ``size``, threading ``overlap`` between chunks.

    Every returned chunk is at most ``size`` characters: the overlap tail is cut
    down to the room the incoming unit leaves, so the size contract wins
    whenever the two disagree.
    """
    if not units:
        return []
    chunks: list[str] = []
    current = units[0]
    for unit in units[1:]:
        if len(current) + len(unit) <= size:
            current += unit
            continue
        chunks.append(current)
        room = size - len(unit)
        tail = current[-min(overlap, room) :] if overlap > 0 and room > 0 else ""
        current = tail + unit
    chunks.append(current)
    return chunks


def chunk(text: str, *, size: int, overlap: int = 0, **kw: Any) -> list[Chunk]:
    """Split a structured document into chunks that respect its structure.

    Args:
        text: Source document. ``""`` yields ``[]`` rather than an error.
        size: Maximum chunk length in characters; must be ``> 0``.
        overlap: Characters shared between consecutive chunks; must be
            ``0 <= overlap < size`` (FR-7).
        **kw: No options. Any keyword is rejected by name rather than ignored,
            because a silently dropped option reads as if it worked.

    Returns:
        One :class:`Chunk` per merged unit, in order, ``chunk_index`` from 0.
        At ``overlap == 0`` the chunk texts rejoin to ``text`` exactly.

    Raises:
        InvalidChunkSize: ``size``/``overlap`` missing the type, sign or range.
        OverlapTooLarge: ``overlap >= size`` (FR-7 -- never clamped).
        ChunkingError: ``text`` is not a string, or an unknown option was
            passed.
    """
    if kw:
        raise ChunkingError(
            f"structure-aware-chunker: unknown option(s) {', '.join(sorted(kw))}; "
            "this chunker takes only text, size and overlap"
        )
    if not isinstance(text, str):
        raise ChunkingError(
            f"structure-aware-chunker: text must be a str, got {type(text).__name__}"
        )
    _validate(size, overlap)
    if not text:
        return []

    # Lines -> structural units (tables grouped) -> units that fit -> headings
    # glued to their content -> merged to size with overlap.
    fitted = _fit(_units(text), size)
    pieces = _glue_headings(fitted, size)
    merged = _merge(pieces, size, overlap)
    return [Chunk(chunk_index=index, text=piece) for index, piece in enumerate(merged)]

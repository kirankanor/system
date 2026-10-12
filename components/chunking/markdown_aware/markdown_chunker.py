"""Split Markdown at Markdown boundaries, not at arbitrary characters.

Three things make this "markdown-aware" rather than a synonym for the
recursive chunker:

1. **Fenced code blocks are atomic.** A ``` or ~~~ fence, including any blank
   lines and prose-looking text inside it, stays in one piece. The recursive
   chunker would cheerfully cut a Python function in half at a blank line; a
   chunk like that can never be reconstructed or pasted back together.
2. **Text splits at blank lines, and only there.** A paragraph that fits stays
   whole -- a hard-wrapped sentence is not an invitation to cut.
3. **A heading is glued to the paragraph under it** when both fit in one chunk,
   so a chunk never opens with content whose title went into the previous one.

Only when a unit is genuinely larger than ``size`` does it break down further,
at line boundaries, then word boundaries, then characters -- so the last resort
still lands on something a reader would recognise.

Overlap semantics (FR-7)
------------------------
Pieces are contiguous slices of the input (``"".join(pieces) == text``), then
merged greedily into chunks of at most ``size``. Consecutive chunks share text:
when a chunk closes its final ``overlap`` characters become the head of the
next one. The tail is truncated to the room the incoming piece leaves, so
overlap yields before the size contract does -- a chunk is never larger than
``size``. ``overlap >= size`` is rejected outright and never clamped: at
equality the step is zero and chunking never advances, which is exactly the
silent failure FR-7 exists to rule out.

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

#: ATX heading: 1-6 hashes, a space, then the title.
_HEADING = re.compile(r"^#{1,6}\s")


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
                f"markdown-chunker: chunk_index must be >= 0, got {self.chunk_index}"
            )
        if not self.text:
            raise ChunkingError(
                "markdown-chunker: a Chunk's text must not be empty; "
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
                f"markdown-chunker: {name} must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    if size <= 0:
        raise InvalidChunkSize(
            f"markdown-chunker: size must be > 0, got {size}; "
            "a chunker cannot cut text into windows of zero or negative width"
        )
    if overlap < 0:
        raise InvalidChunkSize(
            f"markdown-chunker: overlap must be >= 0, got {overlap}; "
            "use 0 for non-overlapping chunks"
        )
    if overlap >= size:
        raise OverlapTooLarge(
            f"markdown-chunker: overlap ({overlap}) must be smaller than size "
            f"({size}). At equal, the step would be 0 and chunking would never "
            f"advance; larger, and a chunk would repeat itself. Reduce overlap "
            f"below {size}, or raise size above {overlap}."
        )


def _segments(text: str) -> list[str]:
    """Split so every fenced code block becomes exactly one segment.

    The invariant that matters: segments are *slices of ``text`` by index*,
    never rebuilt strings, so they rejoin to ``text`` byte for byte -- the
    newline that closes a fence is swallowed by the fence, not dropped.
    An unclosed fence runs to the end of the input rather than being guessed at.
    """
    lines = text.split("\n")
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1  # +1 for the newline that was split away

    segments: list[str] = []
    cursor = 0  # index in text where the prose segment being built begins
    fence_start: int | None = None
    marker = ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("```") or stripped.startswith("~~~")):
            continue
        if fence_start is None:
            if starts[index] > cursor:
                segments.append(text[cursor : starts[index]])
            # Move the prose cursor up to the fence's first byte. If this fence
            # is never closed, the trailing segment must begin here; leaving the
            # cursor back in the prose makes the final append re-emit the whole
            # document from offset 0 and silently duplicate everything.
            cursor = starts[index]
            fence_start, marker = starts[index], stripped[:3]
        elif stripped.startswith(marker):
            end = starts[index] + len(line)
            if end < len(text):
                end += 1  # the newline ending the closing fence line
            segments.append(text[fence_start:end])
            cursor, fence_start = end, None

    if cursor < len(text):
        segments.append(text[cursor:])
    return [segment for segment in segments if segment]


def _blocks(segment: str) -> list[str]:
    """Split a prose segment at blank lines, keeping the blank lines attached.

    Empty parts are preserved rather than dropped: ``"a\\n\\n\\n\\nb"`` must
    still rejoin to itself, so the doubled separator has to live somewhere.
    """
    parts = segment.split("\n\n")
    return [
        part if index == len(parts) - 1 else part + "\n\n"
        for index, part in enumerate(parts)
    ]


def _pack(text: str, size: int, separator: str, fallback: Any) -> list[str]:
    """Group ``separator``-separated parts so each group fits ``size``.

    Every group is a slice of ``text`` by index, so ``"".join(groups) == text``.
    ``fallback`` is called with a part that cannot fit on its own; its result
    takes that part's place. Zero-length groups are dropped -- they contribute
    nothing to the join, so dropping them stays lossless.
    """
    parts = text.split(separator)
    groups: list[str] = []
    start = 0  # index in text where the group being built begins
    pos = 0  # index in text where parts[0] begins
    tail = len(separator)

    for index, part in enumerate(parts):
        part_start, part_end = pos, pos + len(part)
        after = part_end + (tail if index < len(parts) - 1 else 0)

        if after - start <= size:
            pos = part_end + tail
            continue

        if part_start > start:
            groups.append(text[start:part_start])
            start = part_start

        if part_end - start <= size:
            if after - start <= size:
                pos = part_end + tail
                continue
            groups.append(text[start:part_end])
            start = part_end
            pos = part_end + tail
            continue

        groups.extend(fallback(part))
        start = part_end
        pos = part_end + tail

    if start < len(text):
        groups.append(text[start:])
    return [group for group in groups if group]


def _break(text: str, size: int) -> list[str]:
    """Split oversized text: lines, then words, then characters.

    Each level is only reached if the level above produced something that still
    does not fit, so the degradation is strictly gradual -- a hard character cut
    happens only for a single word longer than ``size``.
    """
    if len(text) <= size:
        return [text]

    def oversized_word(word: str) -> list[str]:
        return [word[start : start + size] for start in range(0, len(word), size)]

    def oversized_line(line: str) -> list[str]:
        if len(line) <= size:
            return [line]
        return _pack(line, size, " ", oversized_word)

    return _pack(text, size, "\n", oversized_line)


def _is_fenced(segment: str) -> bool:
    """True if this segment *is* a fenced code block rather than prose."""
    stripped = segment.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _attach_headings(blocks: list[str], size: int) -> list[str]:
    """Glue each heading-only block to the block under it, when they fit.

    Without this, a chunk can end with ``# Title`` and drop the title into the
    next chunk, leaving content whose heading it no longer sits under. If the
    two do not fit together the heading is left as-is -- a heading larger than
    the budget cannot be honoured, and silently swallowing the paragraph would
    be worse than the seam.
    """
    attached: list[str] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if (
            _HEADING.match(block.strip())
            and index + 1 < len(blocks)
            and len(block) + len(blocks[index + 1]) <= size
        ):
            attached.append(block + blocks[index + 1])
            index += 2
        else:
            attached.append(block)
            index += 1
    return attached


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge pieces to ``size``, threading ``overlap`` between chunks.

    Every returned chunk is at most ``size`` characters: the overlap tail is cut
    down to the room the incoming piece leaves, so the size contract wins
    whenever the two disagree.
    """
    if not pieces:
        return []
    chunks: list[str] = []
    current = pieces[0]
    for piece in pieces[1:]:
        if len(current) + len(piece) <= size:
            current += piece
            continue
        chunks.append(current)
        room = size - len(piece)
        tail = current[-min(overlap, room) :] if overlap > 0 and room > 0 else ""
        current = tail + piece
    chunks.append(current)
    return chunks


def chunk(text: str, *, size: int, overlap: int = 0, **kw: Any) -> list[Chunk]:
    """Split Markdown into chunks that respect fences, paragraphs and headings.

    Args:
        text: Markdown source. ``""`` yields ``[]`` rather than an error.
        size: Maximum chunk length in characters; must be ``> 0``.
        overlap: Characters shared between consecutive chunks; must be
            ``0 <= overlap < size`` (FR-7).
        **kw: No options. Any keyword is rejected by name rather than ignored,
            because a silently dropped option reads as if it worked.

    Returns:
        One :class:`Chunk` per merged piece, in order, ``chunk_index`` from 0.
        At ``overlap == 0`` the chunk texts rejoin to ``text`` exactly.

    Raises:
        InvalidChunkSize: ``size``/``overlap`` missing the type, sign or range.
        OverlapTooLarge: ``overlap >= size`` (FR-7 -- never clamped).
        ChunkingError: ``text`` is not a string, or an unknown option was
            passed.
    """
    if kw:
        raise ChunkingError(
            f"markdown-chunker: unknown option(s) {', '.join(sorted(kw))}; "
            "this chunker takes only text, size and overlap"
        )
    if not isinstance(text, str):
        raise ChunkingError(
            f"markdown-chunker: text must be a str, got {type(text).__name__}"
        )
    _validate(size, overlap)
    if not text:
        return []

    # 1. Fences become atomic segments; prose stays prose.
    # 2. Prose splits at blank lines; an oversized unit degrades
    #    line -> word -> character.
    # 3. A fence or a paragraph larger than `size` is broken only then, because
    #    exceeding the budget is worse than breaking the structure.
    blocks: list[str] = []
    for segment in _segments(text):
        if _is_fenced(segment):
            blocks.extend(_break(segment, size) if len(segment) > size else [segment])
        else:
            for block in _blocks(segment):
                blocks.extend(_break(block, size) if len(block) > size else [block])

    pieces = _attach_headings(blocks, size)
    merged = _merge(pieces, size, overlap)
    return [Chunk(chunk_index=index, text=piece) for index, piece in enumerate(merged)]

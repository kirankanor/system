"""Split text recursively on a separator priority list, then merge to size.

Where the fixed-size chunker cuts blind windows, this one walks *down* a list
of separators (paragraph, line, word) so that cuts land on meaningful
boundaries whenever the text offers one, and only hard-cuts as a last resort.

Overlap semantics (FR-7)
------------------------
Pieces produced by the recursive split are contiguous slices of the input --
they rejoin to the original text exactly, separator included. Those pieces are
then greedily merged into chunks of at most ``size`` characters. Consecutive
chunks share text: when a chunk closes, its final ``overlap`` characters become
the head of the next one.

The tail is truncated to whatever room the next piece leaves
(``size - len(piece)``), so a chunk is *never* larger than ``size`` -- overlap
gives way before the size contract does. ``overlap >= size`` is rejected up
front and never clamped: at equality the step would be zero, and a silently
accepted ``overlap == size`` is how a chunker becomes a hang.

Self-contained on purpose
-------------------------
The manifest declares no dependencies, so this file imports only the standard
library and defines the types it returns. :class:`Chunk` mirrors the field
names of the canonical ``core_types.Chunk``, and ``chunk()`` matches the shared
``ChunkingProtocol.chunk(text, *, size, overlap, **kw)`` shape structurally --
that is what makes FR-5's "swap the component, keep the caller" true without
depending on core-types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chunk",
    "ChunkingError",
    "InvalidChunkSize",
    "OverlapTooLarge",
    "SEPARATORS",
    "chunk",
]

#: Default split priority, most semantic first. The empty string is the
#: "no separator will do" sentinel that triggers a hard character cut.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")


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
    embedding or indexing code unchanged. Identity fields are optional because
    a chunker cannot know which document its text came from; the application
    fills those in afterwards.
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
                f"recursive-chunker: chunk_index must be >= 0, got {self.chunk_index}"
            )
        if not self.text:
            raise ChunkingError(
                "recursive-chunker: a Chunk's text must not be empty; "
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
                f"recursive-chunker: {name} must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    if size <= 0:
        raise InvalidChunkSize(
            f"recursive-chunker: size must be > 0, got {size}; "
            "a chunker cannot cut text into windows of zero or negative width"
        )
    if overlap < 0:
        raise InvalidChunkSize(
            f"recursive-chunker: overlap must be >= 0, got {overlap}; "
            "use 0 for non-overlapping chunks"
        )
    if overlap >= size:
        raise OverlapTooLarge(
            f"recursive-chunker: overlap ({overlap}) must be smaller than size "
            f"({size}). At equal, no chunk could advance past its predecessor; "
            f"larger, and a chunk would repeat itself. Reduce overlap below "
            f"{size}, or raise size above {overlap}."
        )


def _hard_cut(text: str, size: int) -> list[str]:
    """Last resort: cut on a character boundary. Always contiguous."""
    return [text[start : start + size] for start in range(0, len(text), size)]


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Split ``text`` into contiguous pieces, each at most ``size``.

    Walks ``separators`` in priority order. If one is present in ``text``, the
    split is delegated to :func:`_split_on` with the *remaining* (lower
    priority) separators for anything still too big. If none is present, or the
    empty-string sentinel is reached, the text is hard-cut.
    """
    if len(text) <= size:
        return [text]
    for index, separator in enumerate(separators):
        if separator == "":
            break
        if separator not in text:
            continue
        return _split_on(text, size, separator, separators[index + 1 :])
    return _hard_cut(text, size)


def _split_on(
    text: str, size: int, separator: str, remaining: tuple[str, ...]
) -> list[str]:
    """Split on ``separator`` and group parts so each group fits ``size``.

    The invariant that matters: every group is a *slice* of ``text`` by index,
    never a re-built string. Re-joining groups with ``"".join`` reproduces
    ``text`` byte for byte, separator included, so no character is lost or
    duplicated at a boundary.
    """
    parts = text.split(separator)
    groups: list[str] = []
    start = 0  # index in text where the group being built begins
    pos = 0  # index in text where parts[0] begins

    for index, part in enumerate(parts):
        part_start, part_end = pos, pos + len(part)
        # End of this part plus the separator that followed it (if any).
        after = part_end + (len(separator) if index < len(parts) - 1 else 0)

        if after - start <= size:
            pos = part_end + len(separator)
            continue

        # The part does not fit in the current group: close the group first.
        if part_start > start:
            groups.append(text[start:part_start])
            start = part_start

        if part_end - start <= size:
            if after - start <= size:
                # Part fits alongside its trailing separator; let the next
                # iteration decide whether the following part joins too.
                pos = part_end + len(separator)
                continue
            # Part fits, its separator does not: the separator becomes the
            # opening characters of the next group, which keeps the join exact.
            groups.append(text[start:part_end])
            start = part_end
            pos = part_end + len(separator)
            continue

        # The part alone is too big: recurse with lower-priority separators.
        groups.extend(_split(part, size, remaining))
        start = part_end
        pos = part_end + len(separator)

    if start < len(text):
        groups.append(text[start:])
    return [group for group in groups if group]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge pieces to ``size``, threading ``overlap`` between chunks.

    Every returned chunk is at most ``size`` characters: the overlap tail is
    cut down to the room the incoming piece leaves, so the size contract wins
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


def chunk(
    text: str, *, size: int, overlap: int = 0, separators: Any = SEPARATORS, **kw: Any
) -> list[Chunk]:
    """Split ``text`` on a separator priority, merged into chunks of ``size``.

    Args:
        text: Source text. ``""`` yields ``[]`` rather than an error.
        size: Maximum chunk length in characters; must be ``> 0``.
        overlap: Characters shared between consecutive chunks; must be
            ``0 <= overlap < size`` (FR-7).
        separators: Split priority, most semantic first. The empty string (or
            exhausting the list) means "hard-cut here"; it is not an error for
            a separator to be absent from the text, it is simply skipped.
        **kw: No further options. Any keyword is rejected by name rather than
            ignored, because a silently dropped option reads as if it worked.

    Returns:
        One :class:`Chunk` per merged piece, in order, ``chunk_index`` from 0.

    Raises:
        InvalidChunkSize: ``size``/``overlap`` missing the type, sign or range,
            or ``separators`` is not a sequence of strings.
        OverlapTooLarge: ``overlap >= size`` (FR-7 -- never clamped).
        ChunkingError: ``text`` is not a string, or an unknown option was
            passed.
    """
    if kw:
        raise ChunkingError(
            f"recursive-chunker: unknown option(s) {', '.join(sorted(kw))}; "
            "this chunker takes text, size, overlap and separators"
        )
    if not isinstance(text, str):
        raise ChunkingError(f"recursive-chunker: text must be a str, got {type(text).__name__}")
    _validate(size, overlap)
    if isinstance(separators, (str, bytes)) or not isinstance(separators, (list, tuple)):
        raise InvalidChunkSize(
            f"recursive-chunker: separators must be a sequence of str, got "
            f"{type(separators).__name__}"
        )
    if not all(isinstance(item, str) for item in separators):
        raise InvalidChunkSize(
            "recursive-chunker: every separator must be a str, got "
            + ", ".join(type(item).__name__ for item in separators)
        )
    if not text:
        return []

    pieces = _split(text, size, tuple(separators))
    merged = _merge(pieces, size, overlap)
    return [Chunk(chunk_index=index, text=piece) for index, piece in enumerate(merged)]

"""Split text into fixed-size character windows with explicit overlap.

The simplest chunker in the area, and the one the others' overlap rules are
written against: cut ``[start, start + size)`` and advance ``start`` by
``size - overlap``.

Overlap semantics (FR-7)
------------------------
Chunks are contiguous, *overlapping* windows over the input. Consecutive
chunks therefore share exactly ``overlap`` characters -- the tail of one is the
head of the next -- and no chunk ever exceeds ``size`` characters. The final
chunk is short, because it runs to the end of the text; it is not padded and
not discarded.

``overlap`` is **rejected, never clamped**, when it is ``>= size``. The spec
requires "larger than" to be rejected; equality is rejected too because the
step becomes ``size - overlap == 0`` and the loop would never advance -- a
silently clamped or silently accepted ``overlap == size`` is how a chunker
turns into a hang. Rejecting it at the call is the whole point of FR-7.

Self-contained on purpose
-------------------------
This file is copy-ready: its manifest declares no dependencies, so it imports
nothing outside the standard library and defines the types it returns rather
than importing them. :class:`Chunk` mirrors the field names of the canonical
``core_types.Chunk`` so the two are interchangeable, and the class satisfies
the shared ``ChunkingProtocol.chunk(text, *, size, overlap, **kw)`` shape
structurally -- that is what makes FR-5's "swap the component, keep the caller"
true without this file having to depend on core-types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chunk",
    "ChunkingError",
    "InvalidChunkSize",
    "OverlapTooLarge",
    "chunk",
]


class ChunkingError(Exception):
    """Base class for every error this component raises (FR-4).

    Catching :class:`ChunkingError` catches all of them; catching
    :class:`OverlapTooLarge` catches the overlap case alone. Nothing here is a
    bare ``ValueError``, so a caller can tell this component's failures apart
    from its own programming errors.
    """


class InvalidChunkSize(ChunkingError):
    """``size`` or ``overlap`` was not a usable non-negative configuration."""


class OverlapTooLarge(ChunkingError):
    """``overlap >= size`` (FR-7): rejected rather than clamped."""


@dataclass(frozen=True)
class Chunk:
    """One produced chunk.

    Field names match ``core_types.Chunk`` so a caller can pass these to
    embedding or indexing code unchanged. ``chunk_id``/``document_id``/
    ``source_name`` and the page range are optional because a chunker is called
    as ``chunk(text, size=..., overlap=...)`` and cannot know which document
    the text came from -- the application fills those in afterwards.
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
                f"fixed-size-chunker: chunk_index must be >= 0, got {self.chunk_index}"
            )
        if not self.text:
            raise ChunkingError(
                "fixed-size-chunker: a Chunk's text must not be empty; "
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
                f"fixed-size-chunker: {name} must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    if size <= 0:
        raise InvalidChunkSize(
            f"fixed-size-chunker: size must be > 0, got {size}; "
            "a chunker cannot cut text into windows of zero or negative width"
        )
    if overlap < 0:
        raise InvalidChunkSize(
            f"fixed-size-chunker: overlap must be >= 0, got {overlap}; "
            "use 0 for non-overlapping chunks"
        )
    if overlap >= size:
        raise OverlapTooLarge(
            f"fixed-size-chunker: overlap ({overlap}) must be smaller than size "
            f"({size}). At equal, the step size - overlap would be 0 and "
            f"chunking would never advance; larger, and a chunk would repeat "
            f"itself. Reduce overlap below {size}, or raise size above "
            f"{overlap}."
        )


def chunk(text: str, *, size: int, overlap: int = 0, **kw: Any) -> list[Chunk]:
    """Cut ``text`` into overlapping windows of at most ``size`` characters.

    Args:
        text: Source text. ``""`` yields ``[]`` -- there is nothing to cut --
            rather than an error, so callers need not pre-check.
        size: Window width in characters; must be ``> 0``.
        overlap: Characters shared between consecutive chunks; must be
            ``0 <= overlap < size`` (FR-7).
        **kw: No options. Any keyword is rejected by name rather than ignored,
            because a silently dropped ``separator=...`` reads as if it worked.

    Returns:
        One :class:`Chunk` per window, in order, ``chunk_index`` starting at 0.
        Concatenating the *non-overlapping* parts reproduces the input.

    Raises:
        InvalidChunkSize: ``size``/``overlap`` missing the type, sign or range.
        OverlapTooLarge: ``overlap >= size`` (FR-7 -- never clamped).
        ChunkingError: ``text`` is not a string, or an option was passed that
            this chunker does not implement.
    """
    if kw:
        raise ChunkingError(
            f"fixed-size-chunker: unknown option(s) {', '.join(sorted(kw))}; "
            "this chunker takes only text, size and overlap"
        )
    if not isinstance(text, str):
        raise ChunkingError(
            f"fixed-size-chunker: text must be a str, got "
            f"{type(text).__name__}"
        )
    _validate(size, overlap)
    if not text:
        return []

    step = size - overlap
    pieces: list[str] = []
    for start in range(0, len(text), step):
        pieces.append(text[start : start + size])
        if start + size >= len(text):
            # This window already reaches the end of the text. Any later start
            # would re-emit characters every earlier window covered, so the
            # caller would pay to embed a duplicate -- visible only when
            # overlap > 0, because with overlap == 0 the step is size and the
            # last window reaches the end anyway.
            break
    return [Chunk(chunk_index=index, text=piece) for index, piece in enumerate(pieces)]

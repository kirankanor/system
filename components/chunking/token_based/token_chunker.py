"""Split text on a *token* budget rather than a character budget.

Character windows and token windows disagree as soon as the text is anything
but ASCII, and a context window is measured in tokens -- so this is the chunker
to use when the downstream budget is a model's.

Overlap semantics (FR-7)
------------------------
Identical in shape to the fixed-size chunker, but the unit is a token: the
tokenizer's ids are cut into windows of ``size`` tokens advancing by
``size - overlap`` tokens. Consecutive chunks therefore share exactly
``overlap`` tokens, and decoding is per-window, so a chunk never splits a
character. ``overlap >= size`` is rejected and never clamped -- at equality the
step is zero and tokenisation would loop forever, which is precisely the silent
failure FR-7 exists to rule out.

Optional dependency (FR-3)
--------------------------
``tiktoken`` is declared *optional* in this component's manifest, so it is
imported lazily inside the function that needs it. If it is missing the call
raises :class:`MissingTokenizer`, an ``ImportError`` carrying the manifest's
``missing_dependency_message`` verbatim -- so the message a user sees is the
one the registry declares, not a paraphrase that drifts from it. Nothing in
this file imports ``tiktoken`` at module scope, which is what keeps the module
importable on a machine that never installed it.

Self-contained on purpose
-------------------------
The manifest declares no required packages, so beyond the guarded optional
``tiktoken`` this file imports only the standard library and defines the types
it returns. :class:`Chunk` mirrors the field names of the canonical
``core_types.Chunk``, and ``chunk()`` matches the shared
``ChunkingProtocol.chunk(text, *, size, overlap, **kw)`` shape structurally --
what makes FR-5's "swap the component, keep the caller" true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Chunk",
    "ChunkingError",
    "InvalidChunkSize",
    "InvalidEncoding",
    "MissingTokenizer",
    "OverlapTooLarge",
    "DEFAULT_ENCODING",
    "MISSING_DEPENDENCY_MESSAGE",
    "chunk",
]

#: Encoding used when the caller does not name one. Matches the manifest's
#: ``optional_packages`` entry, which pins ``tiktoken>=0.7.0``.
DEFAULT_ENCODING = "cl100k_base"

#: Copied verbatim from the component manifest's ``missing_dependency_message``.
#: ``scripts/validate_components.py`` cannot check a string inside a .py file,
#: so this is asserted against the manifest in the component's own test.
MISSING_DEPENDENCY_MESSAGE = (
    "token-chunker requires 'tiktoken', which is not installed. "
    "Install it with: uv add tiktoken"
)


class ChunkingError(Exception):
    """Base class for every error this component raises (FR-4).

    :class:`MissingTokenizer` is deliberately *not* a subclass -- FR-3 requires
    it to be an ``ImportError``, and callers guard optional components with
    ``except ImportError``.
    """


class InvalidChunkSize(ChunkingError):
    """``size`` or ``overlap`` was not a usable configuration."""


class OverlapTooLarge(ChunkingError):
    """``overlap >= size`` (FR-7): rejected rather than clamped."""


class InvalidEncoding(ChunkingError):
    """``encoding`` was not a usable tokenizer name."""


class MissingTokenizer(ImportError):
    """``tiktoken`` is not installed; carries the manifest's message verbatim (FR-3)."""


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
                f"token-chunker: chunk_index must be >= 0, got {self.chunk_index}"
            )
        if not self.text:
            raise ChunkingError(
                "token-chunker: a Chunk's text must not be empty; "
                "the chunker never emits one, so this is a construction bug"
            )


def _validate(size: Any, overlap: Any) -> None:
    """Reject unusable configuration before any work happens (FR-4).

    Types are checked explicitly rather than left to ``range()`` to fail:
    ``True`` is an ``int`` in Python, and ``size=True`` would silently chunk to
    a one-token budget.
    """
    for name, value in (("size", size), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidChunkSize(
                f"token-chunker: {name} must be an int, got "
                f"{type(value).__name__} ({value!r})"
            )
    if size <= 0:
        raise InvalidChunkSize(
            f"token-chunker: size must be > 0, got {size}; "
            "a token budget cannot be zero or negative"
        )
    if overlap < 0:
        raise InvalidChunkSize(
            f"token-chunker: overlap must be >= 0, got {overlap}; "
            "use 0 for non-overlapping chunks"
        )
    if overlap >= size:
        raise OverlapTooLarge(
            f"token-chunker: overlap ({overlap} tokens) must be smaller than "
            f"size ({size} tokens). At equal, the step size - overlap would be "
            f"0 and tokenisation would never advance; larger, and a chunk "
            f"would repeat itself. Reduce overlap below {size}, or raise size "
            f"above {overlap}."
        )


def _load_tokenizer(encoding: str) -> Any:
    """Import ``tiktoken`` and resolve ``encoding`` -- the only heavy import.

    Deliberately a function rather than a module-level import or a cached
    global: FR-3 wants the optional dependency guarded at the point of use, and
    a fresh lookup is what lets a test prove the guard fires by removing the
    module from ``sys.modules``.

    Raises:
        MissingTokenizer: ``tiktoken`` is not installed (FR-3).
        InvalidEncoding: ``tiktoken`` does not know that encoding name.
    """
    try:
        import tiktoken
    except ImportError as exc:
        raise MissingTokenizer(MISSING_DEPENDENCY_MESSAGE) from exc
    try:
        return tiktoken.get_encoding(encoding)
    except Exception as exc:  # KeyError for unknown names, ValueError for junk
        raise InvalidEncoding(
            f"token-chunker: unknown encoding {encoding!r}; pass a name "
            f"tiktoken knows, such as {DEFAULT_ENCODING!r}"
        ) from exc


def chunk(
    text: str,
    *,
    size: int,
    overlap: int = 0,
    encoding: str = DEFAULT_ENCODING,
    **kw: Any,
) -> list[Chunk]:
    """Cut ``text`` into windows of at most ``size`` tokens.

    Args:
        text: Source text. ``""`` yields ``[]`` -- there is nothing to cut --
            rather than an error, so callers need not pre-check.
        size: Token budget per chunk; must be ``> 0``.
        overlap: Tokens shared between consecutive chunks; must be
            ``0 <= overlap < size`` (FR-7).
        encoding: A tokenizer name ``tiktoken`` recognises.
        **kw: No further options. Any keyword is rejected by name rather than
            ignored, because a silently dropped option reads as if it worked.

    Returns:
        One :class:`Chunk` per window, in order, ``chunk_index`` from 0. A
        window that decodes to nothing (a tokenizer's unused or special token)
        is dropped rather than emitted as an empty chunk, and indices are
        assigned after that drop so they stay contiguous.

    Raises:
        InvalidChunkSize: ``size``/``overlap`` missing the type, sign or range.
        OverlapTooLarge: ``overlap >= size`` (FR-7 -- never clamped).
        InvalidEncoding: ``encoding`` is empty, not a string, or unknown to
            ``tiktoken``.
        MissingTokenizer: ``tiktoken`` is not installed (FR-3; an ImportError).
        ChunkingError: ``text`` is not a string, or an unknown option was
            passed.
    """
    if kw:
        raise ChunkingError(
            f"token-chunker: unknown option(s) {', '.join(sorted(kw))}; "
            "this chunker takes text, size, overlap and encoding"
        )
    if not isinstance(text, str):
        raise ChunkingError(
            f"token-chunker: text must be a str, got {type(text).__name__}"
        )
    _validate(size, overlap)
    if not isinstance(encoding, str) or not encoding:
        raise InvalidEncoding(
            f"token-chunker: encoding must be a non-empty str, got "
            f"{type(encoding).__name__} ({encoding!r})"
        )
    if not text:
        return []

    tokenizer = _load_tokenizer(encoding)
    token_ids = tokenizer.encode(text)

    step = size - overlap
    windows: list[list[int]] = []
    for start in range(0, len(token_ids), step):
        windows.append(token_ids[start : start + size])
        if start + size >= len(token_ids):
            # Stop as soon as a window reaches the end: later starts would
            # re-emit tokens already covered, producing a duplicate tail chunk
            # that costs an embedding and returns nothing new. With overlap == 0
            # the step equals size, so this condition and plain range() agree --
            # the redundancy only appears once overlap shrinks the step.
            break
    texts = [tokenizer.decode(window) for window in windows]
    # Filter before numbering: a window that decodes to nothing is dropped, and
    # numbering afterwards is what keeps chunk_index a dense 0..n-1. Filtering
    # inside the comprehension over `enumerate(texts)` would preserve the
    # *pre-drop* index and leave a hole -- chunk_index is then no longer a
    # position, and a caller that builds `chunks[chunk.chunk_index]` or a dict
    # keyed by it silently skips an entry.
    kept = [decoded for decoded in texts if decoded]
    return [
        Chunk(chunk_index=index, text=decoded)
        for index, decoded in enumerate(kept)
    ]

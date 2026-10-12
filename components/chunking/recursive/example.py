"""Runnable example for `recursive-chunker`.

    python example.py

Splits text by walking a separator priority -- paragraphs first, then lines,
then words -- and only hard-cuts characters when the text offers no separator
at all. Runs standalone: the only import is this directory's
`recursive_chunker.py`, standard library included.

The three guarantees this example demonstrates:

1. cuts land on boundaries the separator priority chooses, not blind windows;
2. at `overlap=0` the chunks rejoin to the input exactly (nothing lost);
3. `overlap >= size` is rejected with a typed error, never clamped (FR-7).
"""

from __future__ import annotations

from recursive_chunker import (
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    SEPARATORS,
    chunk,
)

ARTICLE = (
    "Retrieval augmented generation grounds an answer in retrieved text.\n"
    "\n"
    "The retriever finds candidate passages, a reranker orders them by "
    "relevance, and the context builder packs the best of them into a "
    "budget.\n"
    "\n"
    "Everything downstream depends on those stages agreeing on one type."
)


def main() -> None:
    print(f"default separator priority: {SEPARATORS}")

    # 1. Separator priority: with size=55 the first paragraph (67 chars) will
    #    not fit, so the split recurses to "\n" and then " " -- cuts land on
    #    word boundaries, never mid-word.
    chunks = chunk(ARTICLE, size=55)
    print(f"\nchunked into {len(chunks)} chunks (max size 55):")
    for c in chunks:
        print(f"  [{c.chunk_index}] ({len(c.text):2d}) {c.text[:48]!r}...")

    # 2. Lossless only at overlap=0: pieces are contiguous slices of the input,
    #    separators included, so the join is exact.
    rejoined = "".join(c.text for c in chunks)
    print(f"\nlossless join at overlap=0: {rejoined == ARTICLE}")

    # 3. Overlap: consecutive chunks share text. The tail is truncated to the
    #    room the incoming piece leaves, so a chunk NEVER exceeds size --
    #    overlap gives way before the size contract does.
    overlapped = chunk("alpha beta gamma delta", size=20, overlap=5)
    first, second = overlapped
    print(f"\noverlap example ({len(overlapped)} chunks, size=20, overlap=5):")
    print(f"  first  = {first.text!r}")
    print(f"  second = {second.text!r}")
    print(
        f"  shares tail: {second.text.startswith(first.text[-5:])}, "
        f"all <= size: {all(len(c.text) <= 20 for c in overlapped)}"
    )

    # A custom priority: hard-cut sentinel last, paragraph separator dropped
    # entirely because this text has no paragraphs.
    piped = chunk("a|b|c|d", size=5, separators=("|", ""))
    print(f"\ncustom separators ('|' then hard cut): {[c.text for c in piped]}")

    # Failure paths, shown rather than hidden.

    # FR-7: full overlap is rejected, not clamped. At equality the step would
    # be zero -- a silently accepted overlap==size is how a chunker becomes a
    # hang.
    try:
        chunk(ARTICLE, size=20, overlap=20)
    except OverlapTooLarge as exc:
        print(f"\nrejected as expected (FR-7): {exc}")

    try:
        chunk(ARTICLE, size=0)
    except InvalidChunkSize as exc:
        print(f"rejected as expected: {exc}")

    try:
        chunk(ARTICLE, size=10, step=2)
    except ChunkingError as exc:
        print(f"rejected as expected: {exc}")


if __name__ == "__main__":
    main()

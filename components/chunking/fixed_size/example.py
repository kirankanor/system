"""Runnable example for `fixed-size-chunker`.

    python example.py

Cuts fixed-width character windows and advances by `size - overlap`. Runs
standalone: the only import is this directory's `fixed_size_chunker.py`,
standard library included.

The three guarantees this example demonstrates:

1. windows advance by exactly `size - overlap`, and stop at the end of the
   text rather than emitting a duplicate tail chunk;
2. at `overlap=0` the chunks rejoin to the input exactly (nothing lost);
3. `overlap >= size` is rejected with a typed error, never clamped (FR-7).
"""

from __future__ import annotations

from fixed_size_chunker import (
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

ARTICLE = (
    "Retrieval augmented generation grounds an answer in retrieved text. "
    "The retriever finds candidate passages, a reranker orders them by "
    "relevance, and the context builder packs the best of them into a budget."
)


def main() -> None:
    # 1. The window arithmetic, stated out loud: size=60, overlap=15 -> a step
    #    of 45. Three windows, and the last one runs to the end of the text.
    chunks = chunk(ARTICLE, size=60, overlap=15)
    print(f"size=60, overlap=15 -> step = {60 - 15}")
    print(f"chunked into {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c.chunk_index}] ({len(c.text):2d}) {c.text[:44]!r}...")

    # 2. Without overlap the windows tile the input, so the join is exact.
    tiled = chunk(ARTICLE, size=60)
    rejoined = "".join(c.text for c in tiled)
    print(f"\nlossless join at overlap=0: {rejoined == ARTICLE}")

    # 3. The exact worked example from the contract. Step = 3, so the windows
    #    start at 0, 3 and 6 -- and stop there, because the window at 6 already
    #    reaches the end of the ten-character text. Stepping on to 9 would
    #    emit a one-character chunk the previous one already covers.
    worked = chunk("0123456789", size=5, overlap=2)
    print(f"\n'0123456789' size=5 overlap=2: {[c.text for c in worked]}")
    print(f"  last chunk reaches the end: {'0123456789'.endswith(worked[-1].text)}")

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
        chunk(ARTICLE, size=10, separator="|")
    except ChunkingError as exc:
        print(f"rejected as expected: {exc}")


if __name__ == "__main__":
    main()

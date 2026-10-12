"""Runnable example for `markdown-aware-chunker`.

    python example.py

Splits Markdown so that fenced code blocks, paragraphs and headings survive
the cut. Runs standalone: the only import is this directory's
`markdown_chunker.py`, standard library included.

The four guarantees this example demonstrates:

1. a fenced code block stays whole whenever it fits the budget;
2. at `overlap=0` the chunks rejoin to the input exactly (nothing lost);
3. an oversized fence is split rather than allowed to blow the budget;
4. `overlap >= size` is rejected with a typed error, never clamped (FR-7).
"""

from __future__ import annotations

from markdown_chunker import (
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

FENCE = (
    "```python\n"
    "def retrieve(query, k=5):\n"
    "    hits = index.search(query, k)\n"
    "\n"
    "    # a blank line inside the fence: prose chunkers cut here\n"
    "    return rerank(hits)[:k]\n"
    "```"
)

DOC = (
    "# Retrieval\n"
    "\n"
    "Retrieval augmented generation grounds an answer in retrieved passages.\n"
    "\n"
    + FENCE
    + "\n"
    "\n"
    "The chunker decides where those passages begin and end.\n"
)


def main() -> None:
    # 1. The fence is 154 characters, so a budget of 200 keeps it whole even
    #    though the document around it would not fit in one chunk.
    chunks = chunk(DOC, size=200)
    print(f"size=200 -> {len(chunks)} chunks")
    print(f"  fence came through whole: {any(FENCE in c.text for c in chunks)}")
    for c in chunks:
        print(f"  [{c.chunk_index}] ({len(c.text):3d}) {c.text[:46]!r}...")

    # 2. Lossless only at overlap=0: segments are contiguous slices of the
    #    input, so the join is exact, fence delimiters included.
    rejoined = "".join(c.text for c in chunks)
    print(f"\nlossless join at overlap=0: {rejoined == DOC}")

    # 3. Exceeding the budget is the worse failure, so an oversized fence
    #    gives way: it splits, and no chunk comes out over size.
    small = chunk(DOC, size=60)
    print(f"\nsize=60 -> {len(small)} chunks, fence whole: "
          f"{any(FENCE in c.text for c in small)}")
    print(f"  all <= 60: {all(len(c.text) <= 60 for c in small)}")

    # A heading is glued to the paragraph that follows it, so no chunk opens
    # with content whose title travelled on ahead.
    headed = chunk("# Alpha\n\nbody text here\n\n# Beta\n\nmore body\n", size=60)
    print("\nheading stays with its paragraph:")
    for c in headed:
        print(f"  [{c.chunk_index}] {c.text!r}")

    # Failure paths, shown rather than hidden.

    try:
        chunk(DOC, size=20, overlap=20)
    except OverlapTooLarge as exc:
        print(f"\nrejected as expected (FR-7): {exc}")

    try:
        chunk(DOC, size=0)
    except InvalidChunkSize as exc:
        print(f"rejected as expected: {exc}")

    try:
        chunk(DOC, size=10, separators=("|",))
    except ChunkingError as exc:
        print(f"rejected as expected: {exc}")


if __name__ == "__main__":
    main()

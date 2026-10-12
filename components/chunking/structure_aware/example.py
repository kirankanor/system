"""Runnable example for `structure-aware-chunker`.

    python example.py

Splits on line boundaries, keeps table rows whole, and glues headings to the
content under them. Runs standalone: the only import is this directory's
`structure_aware_chunker.py`, standard library included.

The four guarantees this example demonstrates:

1. a table stays whole while it fits, and breaks at row boundaries when it
   does not -- never at a cell;
2. at `overlap=0` the chunks rejoin to the input exactly (nothing lost);
3. a heading travels with the line beneath it rather than being left behind;
4. `overlap >= size` is rejected with a typed error, never clamped (FR-7).
"""

from __future__ import annotations

from structure_aware_chunker import (
    ChunkingError,
    InvalidChunkSize,
    OverlapTooLarge,
    chunk,
)

TABLE = (
    "| name  | qty |\n"
    "|-------|-----|\n"
    "| bolt  |  12 |\n"
    "| nut   |  30 |\n"
)

DOC = (
    "OVERVIEW\n"
    "Retrieval augmented generation grounds an answer.\n"
    "\n"
    "1. Scope\n"
    "- first item\n"
    "- second item\n"
    "\n"
    + TABLE
    + "2.3 Details\n"
    "Some trailing prose on the topic.\n"
)


def main() -> None:
    # 1. The whole table is 64 characters. At size=40 it cannot fit in one
    #    chunk, so it breaks at row boundaries -- two whole rows per chunk,
    #    never a cut through a cell.
    print(f"table length: {len(TABLE)}")
    for size in (200, 40, 30):
        chunks = chunk(DOC, size=size, overlap=0)
        print(f"  size={size:3d} -> {len(chunks)} chunks, "
              f"table whole: {any(TABLE in c.text for c in chunks)}")

    rows = chunk(TABLE, size=40, overlap=0)
    print(f"\nsize=40 over the table alone -> {len(rows)} chunks:")
    for c in rows:
        print(f"  [{c.chunk_index}] ({len(c.text):2d}) {c.text!r}")

    # 2. Lossless only at overlap=0: units are contiguous slices of the input,
    #    trailing newlines included, so the join is exact.
    chunks = chunk(DOC, size=60, overlap=0)
    rejoined = "".join(c.text for c in chunks)
    print(f"\nlossless join at overlap=0: {rejoined == DOC}")

    # 3. A heading leads its chunk and is not left travelling alone.
    print("\nheading with the line under it:")
    for c in chunk("OVERVIEW\nRetrieval grounds an answer.\n", size=60, overlap=0):
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

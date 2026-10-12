"""Runnable example for `token-chunker`.

    python example.py

Budgets a context window in *tokens* rather than characters. Runs standalone:
the only imports are this directory's `token_chunker.py` and the standard
library.

The three guarantees this example demonstrates:

1. the optional `tiktoken` dependency is guarded at the point of use, and the
   message it raises is the component manifest's, verbatim (FR-3);
2. windows advance by exactly `size - overlap` tokens and stop at the end of
   the input rather than emitting a duplicate tail chunk;
3. `overlap >= size` is rejected with a typed error, never clamped (FR-7).

Note that `tiktoken` is *not* installed in this checkout -- installing it
would modify dependency files, which this library forbids. Part 1 shows the
guard firing, which is the real behaviour here. Part 2 installs a tiny
character-level codec, one character per token, so the window arithmetic is
legible; `chunk()` itself is untouched by that substitution.
"""

from __future__ import annotations

import token_chunker
from token_chunker import (
    ChunkingError,
    InvalidChunkSize,
    MissingTokenizer,
    OverlapTooLarge,
    chunk,
)

DOC = "Retrieval augmented generation grounds an answer in retrieved text."


class CharacterCodec:
    """One character per token: makes token windows readable by hand."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)


def main() -> None:
    # 1. FR-3, on a machine without tiktoken: the call raises an ImportError
    #    carrying the manifest's message rather than a bare ModuleNotFoundError
    #    from somewhere deep in the import stack.
    try:
        chunk(DOC, size=8, overlap=2)
    except MissingTokenizer as exc:
        print("tiktoken is not installed, so the guard fired:")
        print(f"  {exc}")
        print(f"  is an ImportError: {isinstance(exc, ImportError)}")

    # 2. With a tokenizer reachable, windows are counted in tokens. Swapping
    #    the codec above for a real `tiktoken.get_encoding(...)` changes
    #    nothing about `chunk()` -- only what `encode` returns.
    token_chunker._load_tokenizer = lambda encoding: CharacterCodec()

    print(f"\n'abcdefghij' is ten tokens under a character codec")
    windows = chunk("abcdefghij", size=4, overlap=1)  # step 3
    print(f"  size=4 overlap=1 -> step 3 -> {[c.text for c in windows]}")
    print(f"  last chunk reaches the end: "
          f"{'abcdefghij'.endswith(windows[-1].text)}")

    tiled = chunk("abcdefghij", size=4, overlap=0)
    rejoined = "".join(c.text for c in tiled)
    print(f"\n  size=4 overlap=0 -> {[c.text for c in tiled]}")
    print(f"  lossless join at overlap=0: {rejoined == 'abcdefghij'}")

    # A budget, applied to real prose.
    chunks = chunk(DOC, size=12, overlap=3)
    print(f"\nprose at size=12 overlap=3 -> {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c.chunk_index}] ({len(c.text):2d}) {c.text!r}")

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

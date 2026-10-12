"""Runnable example for `batch-embeddings`.

    python example.py

Splits embedding calls into batches and checks what comes back. Runs
standalone with `embedding_protocol.py` beside it -- the manifest declares
that component a dependency, so `ragkit add` copies it along.

The three guarantees this example demonstrates:

1. one vector comes back per input text, in input order, whatever the
   batch size;
2. an empty input returns `[]` without calling the provider at all;
3. a provider that returns the wrong number of vectors, or vectors of
   differing widths, is rejected rather than silently indexed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The protocol sits in its own component directory. `ragkit add` copies it
# beside this file; here we reach across to the sibling directory so the
# example runs from the component tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embedding_protocol"))

from embedding_protocol import EmbeddingProviderError  # noqa: E402

from batch_embeddings import BatchEmbedder, MAX_BATCH_SIZE  # noqa: E402


class CountingEmbedder:
    """A conforming embedder that records how it was called.

    One three-wide vector per text, with the text's length in position 0 so
    the output below is readable by hand.
    """

    dimensions = 3
    model_id = "counting-demo"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def main() -> None:
    provider = CountingEmbedder()

    # 1. Six texts, batches of four: two calls, order preserved.
    wrapped = BatchEmbedder(provider, batch_size=4)
    texts = ["a", "bb", "ccc", "dddd", "eeeee", "f"]
    vectors = wrapped.embed(texts)

    print(f"{len(texts)} texts at batch_size=4 -> {len(provider.calls)} calls")
    for index, call in enumerate(provider.calls):
        print(f"  call {index}: {call}")
    print(f"  vectors returned: {[v[0] for v in vectors]}  (text lengths, in order)")

    # The wrapper is itself an embedder, so it composes.
    print(f"\nsatisfies the protocol: dimensions={wrapped.dimensions}, "
          f"model_id={wrapped.model_id!r}")

    # 2. Nothing to embed is not a failure, and costs no round trip.
    provider.calls.clear()
    print(f"embed([]) -> {wrapped.embed([])}, provider calls: {len(provider.calls)}")

    # 3. Failure paths, shown rather than hidden.

    # A provider returning fewer vectors than it was given would silently
    # attach the wrong vector to the wrong document further down the line.
    class TooFew(CountingEmbedder):
        def embed(self, texts):
            return [[1.0, 0.0, 1.0]]

    try:
        BatchEmbedder(TooFew()).embed(["a", "b", "c"])
    except EmbeddingProviderError as exc:
        print(f"\nrejected as expected: {exc}")

    # Vectors of differing widths cannot share an index.
    class Ragged(CountingEmbedder):
        def embed(self, texts):
            return [[1.0], [1.0, 2.0]]

    try:
        BatchEmbedder(Ragged()).embed(["a", "b"])
    except EmbeddingProviderError as exc:
        print(f"rejected as expected: {exc}")

    # A failing batch is wrapped with the batch's identity, and the original
    # exception is chained so nothing is lost.
    class Down(CountingEmbedder):
        def embed(self, texts):
            raise RuntimeError("quota exhausted")

    try:
        BatchEmbedder(Down()).embed(["a", "b"])
    except EmbeddingProviderError as exc:
        print(f"rejected as expected: {exc}")
        print(f"  cause preserved: {type(exc.__cause__).__name__}")

    # Configuration mistakes fail at construction, before any text is sent.
    try:
        BatchEmbedder(CountingEmbedder(), batch_size=0)
    except Exception as exc:
        print(f"rejected as expected: {exc}")
    print(f"\nMAX_BATCH_SIZE = {MAX_BATCH_SIZE}")


if __name__ == "__main__":
    main()

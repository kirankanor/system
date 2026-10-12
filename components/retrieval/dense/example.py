"""Runnable example for `dense-retrieval`.

    python example.py

Shows the composition this component owns, and the checks that keep it from
failing invisibly. Runs standalone with `core_types.py` beside it -- the
manifest declares that component a dependency, so `ragkit add` copies it
along.

Neither an embedder nor a store is installed here, and neither is this
component's business: both are supplied as objects honouring the protocols
from core-types, which is what the example demonstrates. The fakes below are
deliberately tiny so the wiring is visible.

The five guarantees this example demonstrates:

1. the components compose -- an EmbeddingProtocol plus a VectorStoreProtocol;
2. the query is embedded exactly once, and passed to the store unchanged;
3. asking for more results than the store holds returns what exists;
4. a blank query is rejected rather than retrieving a real-looking answer
   about nothing;
5. a broken embedder or store is wrapped with context, not leaked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory under components/storage.
# `ragkit add` copies it beside this file; here we reach across to the sibling
# so the example runs from the component tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "storage" / "core_types")
)

import core_types as core  # noqa: E402

from dense_retrieval import (  # noqa: E402
    DenseRetriever,
    RetrievalError,
    RetrievalInputError,
)


# --------------------------------------------------------------------------- #
# Fakes honouring the two protocols
# --------------------------------------------------------------------------- #


class NearestNeighbourEmbedder:
    """Maps each text to a point on a line, so 'similar text' is nearby."""

    dimensions = 1
    model_id = "toy-1d-v1"

    def embed(self, texts):
        return [[float(len(text))] for text in texts]


class TinyVectorStore:
    """Holds three chunks and answers with the nearest by 1-D distance."""

    def __init__(self):
        self._rows = [
            ("c0", 4.0, "a short passage"),
            ("c1", 11.0, "a considerably longer passage"),
            ("c2", 20.0, "the longest passage of all, by some margin"),
        ]

    def add(self, items):
        return len(items)

    def delete(self, ids):
        return len(ids)

    def count(self):
        return len(self._rows)

    def query(self, vector, k, filter=None):
        needle = vector[0]
        ranked = sorted(self._rows, key=lambda row: abs(row[1] - needle))
        return [
            core.ScoredDocument.from_chunk(
                core.Chunk(chunk_index=i, text=text, chunk_id=cid),
                score=abs(distance - needle),
            )
            for i, (cid, distance, text) in enumerate(ranked[:k])
        ]


def main() -> None:
    retriever = DenseRetriever(
        embedder=NearestNeighbourEmbedder(), store=TinyVectorStore(), k=2
    )
    print(f"configured: {retriever}")

    print("\nretrieve('a medium passage') -- the query is embedded, then asked:")
    for hit in retriever.retrieve("a medium passage"):
        print(f"  {hit.chunk_id}  score={hit.score:.1f}  {hit.text!r}")

    print("\nMore results than the store holds returns what exists:")
    print(f"  k=50 over a 3-chunk store -> "
          f"{[h.chunk_id for h in retriever.retrieve('x', k=50)]}")

    print("\nA blank query is rejected, because it would retrieve about nothing:")
    try:
        retriever.retrieve("   ")
    except RetrievalInputError as exc:
        print(f"  {exc}")

    print("\nA non-string query is rejected too:")
    try:
        retriever.retrieve(None)
    except RetrievalInputError as exc:
        print(f"  {exc}")

    print("\nA dependency that does not honour its protocol is caught at "
          "construction:")
    try:
        DenseRetriever(embedder=object(), store=TinyVectorStore())
    except RetrievalInputError as exc:
        print(f"  {exc}")

    print("\nA broken embedder is wrapped with its model id, and the cause "
          "kept:")

    class Broken:
        dimensions = 1
        model_id = "unreachable-v1"

        def embed(self, texts):
            raise RuntimeError("service is down")

    try:
        DenseRetriever(embedder=Broken(), store=TinyVectorStore()).retrieve("q")
    except RetrievalError as exc:
        print(f"  {exc}")
        print(f"  caused by: {type(exc.__cause__).__name__}")

    print("\nA store returning the wrong shape is rejected, because FR-8 says "
          "downstream components must never convert:")

    class DictStore(TinyVectorStore):
        def query(self, vector, k, filter=None):
            return [{"chunk_id": "c0"}]

    try:
        DenseRetriever(embedder=NearestNeighbourEmbedder(), store=DictStore()).retrieve("q")
    except RetrievalError as exc:
        print(f"  {exc}")

    print("\nBoth failures are catchable under one base class: "
          f"{issubclass(RetrievalInputError, RetrievalError)}")


if __name__ == "__main__":
    main()

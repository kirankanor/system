"""Runnable example for `storage-adapters`.

    python example.py

Two vector stores, neither of which needs its real backend installed to be
demonstrated. Runs standalone with `core_types.py` and
`vector_store_protocol.py` beside it -- the manifest declares both as
dependencies, so `ragkit add` copies them along.

The five guarantees this example demonstrates:

1. both adapters satisfy the same protocol, so they are interchangeable;
2. results name the backend that produced them, which is what lets a hybrid
   retriever merge two result lists;
3. a chunk's identity fields survive the round trip through flat backend
   storage, and user metadata is passed through untouched;
4. a filter naming an identity field is rewritten internally, so filtering on
   `document_id` works even though the backend stores it under a reserved key;
5. the optional backends are guarded at the point of use, and the message is
   the manifest's, verbatim (FR-3).

Neither chromadb nor faiss is installed here, and installing them would
modify dependency files, which this library forbids. Everything below is
therefore driven by injected fakes speaking the same API surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types and vector-store-protocol sit in their own component
# directories. `ragkit add` copies them beside this file; here we reach
# across to the siblings so the example runs from the component tree.
_HERE = Path(__file__).resolve().parents[1]
for sibling in ("core_types", "vector_store_protocol"):
    sys.path.insert(0, str(_HERE / sibling))

import core_types as core  # noqa: E402
from vector_store_protocol import (  # noqa: E402
    MissingStorageDependency,
    VectorStoreProtocol,
)

from storage_adapters import ChromaVectorStore, FaissVectorStore  # noqa: E402

ITEMS = [
    core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=0,
            text="alpha text",
            chunk_id="c0",
            document_id="d1",
            source_name="a.txt",
        ),
        vector=(1.0, 2.0),
        index_fingerprint="fp1",
    ),
    core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=1,
            text="beta text",
            chunk_id="c1",
            document_id="d1",
            source_name="a.txt",
        ),
        vector=(3.0, 4.0),
        index_fingerprint="fp1",
    ),
]


# --------------------------------------------------------------------------- #
# Fakes speaking the same surface as the real backends
# --------------------------------------------------------------------------- #


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def add(self, ids, embeddings, documents, metadatas):
        for row_id, vector, document, metadata in zip(ids, embeddings, documents, metadatas):
            self.rows[row_id] = {
                "vector": list(vector),
                "document": document,
                "metadata": dict(metadata),
            }

    def query(self, query_embeddings, n_results, where=None, include=None):
        needle = query_embeddings[0]
        scored = []
        for row_id, row in self.rows.items():
            if where and any(row["metadata"].get(k) != v for k, v in where.items()):
                continue
            scored.append((sum((a - b) ** 2 for a, b in zip(needle, row["vector"])), row_id))
        scored.sort()
        top = scored[:n_results]
        return {
            "ids": [[r for _, r in top]],
            "documents": [[self.rows[r]["document"] for _, r in top]],
            "metadatas": [[self.rows[r]["metadata"] for _, r in top]],
            "distances": [[d for d, _ in top]],
        }

    def delete(self, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def count(self):
        return len(self.rows)


class FakeChromaClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, metadata=None):
        return self.collections.setdefault(name, FakeCollection())


def _report(label, results):
    print(f"  {label}:")
    for r in results:
        print(f"    {r.chunk_id!r} score={r.score:.1f} from {r.source_name!r} "
              f"-- {r.text!r} metadata={r.metadata}")


def main() -> None:
    chroma = ChromaVectorStore("docs", client=FakeChromaClient())
    chroma.add(ITEMS)

    # 1 + 2. Same protocol, and results that say where they came from.
    print(f"ChromaVectorStore satisfies the protocol: "
          f"{isinstance(chroma, VectorStoreProtocol)}")
    _report("chroma query", chroma.query([1.0, 2.0], k=2))

    # 3. Identity fields survive; user metadata is untouched.
    ITEMS[0].chunk.metadata["section"] = "intro"
    chroma.add(ITEMS)
    _report("chroma query with user metadata", chroma.query([1.0, 2.0], k=1))

    # 4. Filtering on an identity field works despite the reserved key.
    hits = chroma.query([1.0, 2.0], k=2, filter={"document_id": "d1"})
    print(f"\n  filter={{'document_id': 'd1'}} -> {len(hits)} hit(s): "
          f"{[h.chunk_id for h in hits]}")
    print(f"  filter={{'document_id': 'other'}} -> "
          f"{len(chroma.query([1.0, 2.0], k=2, filter={'document_id': 'other'}))} hit(s)")

    print(f"\n  delete(['c0']) -> {chroma.delete(['c0'])}, count -> {chroma.count()}")

    # The faiss adapter, same guarantees, different constraints.
    try:
        import numpy

        class FakeFaissIndex:
            ntotal = 0

            def __init__(self):
                self.rows = {}

            def add_with_ids(self, matrix, labels):
                for vector, label in zip(matrix, labels):
                    self.rows[int(label)] = [float(x) for x in vector]
                self.ntotal = len(self.rows)

            def search(self, matrix, k):
                needle = list(matrix[0])
                scored = sorted(
                    (sum((a - b) ** 2 for a, b in zip(needle, v)), label)
                    for label, v in self.rows.items()
                )[:k]
                return (
                    numpy.array([[d for d, _ in scored]]),
                    numpy.array([[label for _, label in scored]]),
                )

            def remove_ids(self, labels):
                for label in labels:
                    self.rows.pop(int(label), None)
                self.ntotal = len(self.rows)

        faiss_store = FaissVectorStore(dimensions=2, index=FakeFaissIndex())
        print(f"\nFaissVectorStore satisfies the protocol: "
              f"{isinstance(faiss_store, VectorStoreProtocol)}")
        print(f"  add -> {faiss_store.add(ITEMS)}, count -> {faiss_store.count()}")
        _report("faiss query", faiss_store.query([1.0, 2.0], k=2))
        print(f"  delete(['c0']) -> {faiss_store.delete(['c0'])}, "
              f"count -> {faiss_store.count()}")
        _report("faiss query after delete", faiss_store.query([1.0, 2.0], k=2))

        # The faiss adapter refuses a filter rather than ignoring it.
        try:
            faiss_store.query([1.0, 2.0], k=1, filter={"document_id": "d1"})
        except Exception as exc:
            print(f"\n  a filter on faiss -> {exc}")
    except ImportError:
        print("\n  numpy is not installed, so the faiss adapter cannot be shown")

    # 5. FR-3, with both backends blocked: the manifest's message, verbatim.
    for module in ("chromadb", "faiss"):
        sys.modules[module] = None
    try:
        ChromaVectorStore("docs").count()
    except MissingStorageDependency as exc:
        print(f"\nchromadb blocked, so the guard fired:")
        print(f"  {exc}")
        print(f"  is an ImportError: {isinstance(exc, ImportError)}")
    finally:
        for module in ("chromadb", "faiss"):
            sys.modules.pop(module, None)


if __name__ == "__main__":
    main()

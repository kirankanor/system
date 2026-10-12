"""Runnable example for `sparse-retrieval`.

    python example.py

Shows Okapi BM25 working, and the three properties that make it a
reproducible retrieval model rather than a score generator. Runs standalone
with `core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

Nothing is installed to make this run: BM25 is implemented in the component,
so there is no wheel to fetch and no scorer to disagree with.

The five guarantees this example demonstrates:

1. a query ranks the corpus by BM25, best first;
2. results are ScoredDocuments carrying the chunk's own provenance;
3. asking for more results than match returns what exists;
4. an empty result set is returned for a query no document claims, not
   raised;
5. IDF is inspectable, so a caller can see why a common term barely counts.
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

from sparse_retrieval import (  # noqa: E402
    SparseRetriever,
    RetrievalInputError,
    tokenize,
)

CORPUS = [
    core.Chunk(chunk_index=0, text="Python error handling with exceptions",
               chunk_id="c0", source_name="python/errors.md"),
    core.Chunk(chunk_index=1, text="Java error handling compared to Python",
               chunk_id="c1", source_name="java/errors.md"),
    core.Chunk(chunk_index=2, text="Raising and catching Python exceptions",
               chunk_id="c2", source_name="python/raising.md"),
    core.Chunk(chunk_index=3, text="Cooking pasta with salt and patience",
               chunk_id="c3", source_name="cooking/pasta.md"),
]


def main() -> None:
    retriever = SparseRetriever(CORPUS, k=4)
    print(f"corpus: {len(retriever)} chunks, "
          f"average length {retriever.average_document_length:.1f} tokens")

    print("\nTokenization is public, because it is the vocabulary:")
    print(f"  'The ERROR_Code 42 in Python' -> {tokenize('The ERROR_Code 42 in Python')}")

    print("\nIDF is inspectable -- a rare term carries more than a common one:")
    for term in ("pasta", "handling", "python", "cooking"):
        print(f"  idf({term!r:12}) = {retriever.idf(term):.3f}")
    print("  (a stopword has no IDF here because it never enters the index)")

    for query in ("python error", "pasta", "xyzzy"):
        hits = retriever.retrieve(query)
        print(f"\nretrieve({query!r}):")
        if not hits:
            print("  (no document claims this query -- an empty list, not an error)")
            continue
        for hit in hits:
            print(f"  {hit.chunk_id}  score={hit.score:.3f}  from {hit.source_name}")

    print("\nAsking for more than matches returns what exists:")
    print(f"  retrieve('pasta', k=10) -> "
          f"{[h.chunk_id for h in retriever.retrieve('pasta', k=10)]}")

    print("\nPermuting the corpus does not change the ranking -- ties break on "
          "chunk_index:")
    shuffled = SparseRetriever(list(reversed(CORPUS)), k=4)
    print(f"  original : {[h.chunk_id for h in retriever.retrieve('python error')]}")
    print(f"  reversed : {[h.chunk_id for h in shuffled.retrieve('python error')]}")

    print("\nWhat the length parameter does:")
    two = [
        core.Chunk(chunk_index=0, text="python", chunk_id="short"),
        core.Chunk(chunk_index=1,
                   text=("filler words here and there " * 2) + "python python",
                   chunk_id="long"),
    ]
    print(f"  b=0.75 (default) -> "
          f"{[h.chunk_id for h in SparseRetriever(two).retrieve('python')]}")
    print(f"  b=0.00 (off)     -> "
          f"{[h.chunk_id for h in SparseRetriever(two, b=0.0).retrieve('python')]}")
    print("  -> with normalisation the long document does not win on length alone")

    print("\nAn empty corpus is not an error either:")
    print(f"  SparseRetriever([]).retrieve('anything') -> "
          f"{SparseRetriever([]).retrieve('anything')}")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("an out-of-range b", lambda: SparseRetriever([], b=1.5)),
        ("a negative k1", lambda: SparseRetriever([], k1=-1.0)),
        ("a non-chunk document", lambda: SparseRetriever(["not a chunk"])),
        ("a blank query", lambda: retriever.retrieve("   ")),
    ]:
        try:
            call()
        except RetrievalInputError as exc:
            print(f"  {label:<22} -> {str(exc)[:62]}...")


if __name__ == "__main__":
    main()

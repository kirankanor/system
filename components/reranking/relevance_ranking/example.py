"""Runnable example for `relevance-ranking`.

    python example.py

Shows lexical reranking working, and the properties that make the score
reproducible rather than merely sortable. Runs standalone with
`core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

Nothing is installed to make this run: the scoring is arithmetic in the
component, so there is no wheel to fetch and no model to download.

The six guarantees this example demonstrates:

1. a candidate covering more of the query ranks higher, in [0, 1];
2. a term every candidate contains cannot change their relative order;
3. ties break on chunk_index, never on input order;
4. scoring preserves input order while rerank() reorders;
5. asking for more results than exist returns what exists;
6. a blank query is rejected, because it would rank arbitrarily.
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

from relevance_ranking import (  # noqa: E402
    RelevanceRanker,
    RankingInputError,
    score_documents,
    tokenize,
)

CANDIDATES = [
    core.ScoredDocument(chunk_id="c0", text="Python error handling with exceptions",
                        score=0.0, chunk_index=0, source_name="python/errors.md"),
    core.ScoredDocument(chunk_id="c1", text="Cooking pasta with salt",
                        score=0.0, chunk_index=1, source_name="cooking/pasta.md"),
    core.ScoredDocument(chunk_id="c2", text="Java exceptions and Python compared",
                        score=0.0, chunk_index=2, source_name="java/vs.md"),
]


def main() -> None:
    ranker = RelevanceRanker()
    print(f"candidates: {len(CANDIDATES)}")

    print("\nTokenization is public, because it is the vocabulary:")
    print(f"  'The Python ERROR_Code' -> {sorted(tokenize('The Python ERROR_Code'))}")

    print("\nscore() scores and preserves input order; rerank() orders:")
    for document in ranker.score("python exception", CANDIDATES):
        print(f"  {document.chunk_id}  score={document.score:.3f}  {document.text!r}")
    print(f"  input order kept: {[d.chunk_id for d in ranker.score('python exception', CANDIDATES)]}")

    print("\nThe score is 'how much of the query's weight is covered':")
    one = core.ScoredDocument(chunk_id="full", text="a python exception and another python",
                              score=0.0, chunk_index=0)
    print(f"  a document containing every term -> {ranker.score('python exception', [one])[0].score:.3f}")

    print("\nA term every candidate contains adds the same to each, so it cannot")
    print("change their order -- it only compresses the range:")
    shared = [
        core.ScoredDocument(chunk_id="c0", text="python shared", score=0.0, chunk_index=0),
        core.ScoredDocument(chunk_id="c1", text="python shared", score=0.0, chunk_index=1),
        core.ScoredDocument(chunk_id="c2", text="gamma shared", score=0.0, chunk_index=2),
    ]
    for document in ranker.score("python shared", shared):
        print(f"  {document.chunk_id}  score={document.score:.3f}  {document.text!r}")
    print("  -> 'shared' is in all three, so it cannot separate them; 'python' decides")
    print("  -> c2 still scores > 0 because it does cover 'shared'")

    for query in ("python exception", "pasta", "xylophone"):
        print(f"\nrerank({query!r}, top_k=3):")
        for document in ranker.rerank(query, CANDIDATES, 3):
            print(f"  {document.chunk_id}  score={document.score:.3f}")

    print("\nAsking for more than exist returns what exists:")
    print(f"  rerank('python', top_k=100) -> "
          f"{[d.chunk_id for d in ranker.rerank('python', CANDIDATES, 100)]}")

    print("\nPermuting the candidates does not change the ranking -- ties break "
          "on chunk_index:")
    print(f"  original : {[d.chunk_id for d in ranker.rerank('python exception', CANDIDATES, 3)]}")
    print(f"  reversed : {[d.chunk_id for d in ranker.rerank('python exception', list(reversed(CANDIDATES)), 3)]}")

    tied = [
        core.ScoredDocument(chunk_id="z", text="identical words", score=0.0, chunk_index=1),
        core.ScoredDocument(chunk_id="a", text="identical words", score=0.0, chunk_index=0),
    ]
    print(f"  an exact tie -> {[d.chunk_id for d in ranker.rerank('identical words', tied, 2)]}")

    print("\nScoring never mutates the caller's documents (they are frozen):")
    before = [d.score for d in CANDIDATES]
    score_documents("python", CANDIDATES)
    print(f"  scores before {before} after {[d.score for d in CANDIDATES]}")

    print("\nNo stemming -- a documented limitation, not a bug:")
    docs = [core.ScoredDocument(chunk_id="c0", text="handling exceptions",
                                score=0.0, chunk_index=0)]
    print(f"  score('exception', 'handling exceptions') -> "
          f"{ranker.score('exception', docs)[0].score:.3f}  (the plural never matches)")

    print("\nAn empty candidate list is not an error:")
    print(f"  rerank('anything', [], top_k=3) -> {ranker.rerank('anything', [], 3)}")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("a blank query", lambda: ranker.rerank("   ", CANDIDATES, 1)),
        ("a non-string query", lambda: ranker.rerank(None, CANDIDATES, 1)),
        ("a top_k of 0", lambda: ranker.rerank("q", CANDIDATES, 0)),
        ("a non-document", lambda: ranker.rerank("q", CANDIDATES + ["nope"], 3)),
    ]:
        try:
            call()
        except RankingInputError as exc:
            print(f"  {label:<20} -> {str(exc)[:60]}...")

    print("\nIt satisfies RerankerProtocol, so it slots in beside a model:")
    print(f"  isinstance(RelevanceRanker(), RerankerProtocol) -> "
          f"{isinstance(ranker, core.RerankerProtocol)}")


if __name__ == "__main__":
    main()

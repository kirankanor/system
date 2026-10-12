"""Runnable example for `hybrid-retrieval`.

    python example.py

Shows two ranked lists becoming one, and the guard that stops the standard
way of getting it wrong. Runs standalone with `core_types.py` beside it --
the manifest declares that component a dependency, so `ragkit add` copies it
along.

The five guarantees this example demonstrates:

1. reciprocal rank fusion works on rank, so incomparable score scales cannot
   break it;
2. a document found by both sources outranks one found by either alone;
3. a document's text, source and metadata survive fusion, and the winning
   source is recorded;
4. weighted fusion refuses scores that are not on a shared scale, pointing
   at result-normalizer instead;
5. ordering is deterministic even when scores tie.
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
# The weighted section below shows the flow the README recommends, which uses
# the result-normalizer component. That component is a dependency of the
# *example*, not of `hybrid_retrieval.py` -- the component itself only ever
# fuses lists it is handed, so its manifest still declares core-types alone.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "result_normalizer")
)

import core_types as core  # noqa: E402

from retrieval_result_normalizer import normalize_scores  # noqa: E402

from hybrid_retrieval import (  # noqa: E402
    SOURCE_KEY,
    HybridError,
    HybridInputError,
    fuse,
)


def scored(cid: str, score: float, text: str, source: str) -> core.ScoredDocument:
    return core.ScoredDocument(
        chunk_id=cid, score=score, text=text, chunk_index=0, source_name=source
    )


DENSE = [
    scored("c0", 0.91, "the dense retriever's favourite", "chroma"),
    scored("c1", 0.62, "its second choice", "chroma"),
    scored("c3", 0.31, "a weak dense match", "chroma"),
]
SPARSE = [
    scored("c1", 38.0, "its second choice, again", "bm25"),
    scored("c2", 21.0, "a lexical-only match", "bm25"),
    scored("c0", 4.0, "the dense favourite, weakly", "bm25"),
]


def show(label: str, results) -> None:
    print(f"  {label}")
    for result in results:
        print(f"    {result.chunk_id}  score={result.score:.4f}  "
              f"via {result.metadata[SOURCE_KEY]:<12} {result.text!r}")


def main() -> None:
    print("Raw, the two scales are not comparable:")
    print(f"  dense  scores: {[d.score for d in DENSE]}")
    print(f"  sparse scores: {[s.score for s in SPARSE]}")
    print("  -> adding them would let the 38.0 decide the outcome outright.")

    print("\nrrf (the default) never looks at a score, only at rank:")
    show("fuse(dense, sparse)", fuse(DENSE, SPARSE))

    print("\nA document found by both sources outranks one found by either:")
    results = fuse(DENSE, SPARSE)
    sources = {d.chunk_id: d.metadata[SOURCE_KEY] for d in results}
    print(f"  first place {results[0].chunk_id} -> {sources[results[0].chunk_id]}")

    print("\nA document's own fields survive fusion:")
    print(f"  text={results[0].text!r}  source={results[0].source_name!r}")

    print("\nEither source may be empty:")
    print(f"  sparse empty -> {[d.chunk_id for d in fuse(DENSE, [])]}")
    print(f"  dense empty  -> {[d.chunk_id for d in fuse([], SPARSE)]}")
    print(f"  both empty   -> {fuse([], [])}")

    print("\nWeights shift the result toward the heavier source:")
    show("sparse_weight=3.0", fuse(DENSE, SPARSE, sparse_weight=3.0))

    print("\nA larger rrf_k flattens the list:")
    print(f"  rrf_k=1   spread = "
          f"{fuse(DENSE, SPARSE, rrf_k=1)[0].score - fuse(DENSE, SPARSE, rrf_k=1)[-1].score:.4f}")
    print(f"  rrf_k=100 spread = "
          f"{fuse(DENSE, SPARSE, rrf_k=100)[0].score - fuse(DENSE, SPARSE, rrf_k=100)[-1].score:.4f}")

    print("\nweighted fusion checks the scale first, and refuses raw scores:")
    try:
        fuse(DENSE, SPARSE, method="weighted")
    except HybridInputError as exc:
        print(f"  {exc}")

    print("\nNormalize both lists first, and weighted fusion works:")
    show(
        "method='weighted' over normalized lists",
        fuse(normalize_scores(DENSE), normalize_scores(SPARSE), method="weighted"),
    )

    print("\nOther failures are caught the same way:")
    for label, call in [
        ("an unknown method", lambda: fuse(DENSE, SPARSE, method="combsum")),
        ("a duplicate id", lambda: fuse([], [scored("c0", 1.0, "a", "x"), scored("c0", 0.5, "b", "x")])),
        ("a zero weight", lambda: fuse(DENSE, SPARSE, dense_weight=0.0)),
        ("a non-document", lambda: fuse(["not a document"], [])),
    ]:
        try:
            call()
        except HybridError as exc:
            print(f"  {label:<20} -> {str(exc)[:64]}...")

    print(f"\nEvery failure is catchable under one base class: "
          f"{issubclass(HybridInputError, HybridError)}")


if __name__ == "__main__":
    main()

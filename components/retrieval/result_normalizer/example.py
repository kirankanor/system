"""Runnable example for `retrieval-result-normalizer`.

    python example.py

Shows why scores from different retrievers cannot be compared raw, and what
each normalization method does about it. Runs standalone with
`core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

The four guarantees this example demonstrates:

1. scores from two sources with different scales are put on one scale;
2. score direction is corrected here, once, so the best document always wins;
3. a constant score list -- where min-max divides by zero -- returns a
   documented constant instead of NaN;
4. tied documents share a rank rather than being told apart arbitrarily.
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

from retrieval_result_normalizer import (  # noqa: E402
    CONSTANT_SCORE,
    METHODS,
    NormalizationError,
    NormalizationInputError,
    normalize_scores,
)


def scored(cid: str, score: float, text: str) -> core.ScoredDocument:
    return core.ScoredDocument(
        chunk_id=cid, score=score, text=text, chunk_index=0
    )


def show(label: str, originals, normalized) -> None:
    print(f"  {label}")
    for before, after in zip(originals, normalized):
        print(f"    {before.chunk_id}  {before.score:>7.3f} -> {after.score:.3f}")


def main() -> None:
    dense = [
        scored("c0", 0.91, "cosine distance, lower is better"),
        scored("c1", 0.44, "further from the query"),
        scored("c2", 0.12, "furthest"),
    ]
    sparse = [
        scored("c5", 42.0, "a large BM25 score"),
        scored("c6", 18.0, "a medium BM25 score"),
        scored("c7", 3.0, "a small BM25 score"),
    ]

    print("Raw, the two lists mean nothing next to each other:")
    print(f"  dense  max={max(d.score for d in dense)}")
    print(f"  sparse max={max(d.score for d in sparse)}")
    print("  -> the largest sparse number would beat the best dense hit")
    print("     on magnitude alone, whatever the query was.")

    print("\nmin_max puts both on [0, 1] with the best at 1.0:")
    show("dense  (higher_is_better=True)", dense, normalize_scores(dense))
    show("sparse (higher_is_better=True)", sparse, normalize_scores(sparse))

    print("\nDistance-like scores invert with higher_is_better=False:")
    show(
        "the same dense list, treated as distance",
        dense,
        normalize_scores(dense, higher_is_better=False),
    )

    print("\nrank ignores magnitude entirely, so an outlier cannot dominate:")
    outlier = [scored("c0", 0.01, "tiny"), scored("c1", 0.02, "small"), scored("c2", 900.0, "huge")]
    show("min_max on an outlier", outlier, normalize_scores(outlier))
    show("rank on the same list", outlier, normalize_scores(outlier, method="rank"))

    print("\nz_score reports standard deviations from the mean (unbounded):")
    show("z_score", dense, normalize_scores(dense, method="z_score"))

    print("\nThe case a naive implementation gets wrong:")
    constant = [scored("c0", 2.0, "a"), scored("c1", 2.0, "b")]
    print(f"  two documents, both scoring 2.0")
    print(f"  min_max -> {[round(d.score, 3) for d in normalize_scores(constant)]}"
          f"   (naive min-max would divide by zero)")
    print(f"  z_score -> {[round(d.score, 3) for d in normalize_scores(constant, method='z_score')]}")
    print(f"  the neutral constant is {CONSTANT_SCORE}")

    print("\nTies share a rank rather than being told apart:")
    tied = [scored("c0", 5.0, "a"), scored("c1", 5.0, "b"), scored("c2", 1.0, "c")]
    show("rank over a tie", tied, normalize_scores(tied, method="rank"))

    print("\nEvery failure is catchable under one base class:")
    for label, call in [
        ("a non-finite score", lambda: normalize_scores([scored("c0", float("nan"), "x")])),
        ("an unknown method", lambda: normalize_scores(dense, method="softmax")),
        ("a non-document", lambda: normalize_scores(["not a document"])),
    ]:
        try:
            call()
        except NormalizationError as exc:
            print(f"  {label:<22} -> {type(exc).__name__}")
            print(f"  {'':<22}    {str(exc)[:66]}...")

    print(f"\nmethods available: {', '.join(METHODS)}")


if __name__ == "__main__":
    main()

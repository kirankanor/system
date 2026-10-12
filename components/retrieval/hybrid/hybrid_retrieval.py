"""Fuse dense and sparse results into one ranked list, with the rule stated.

The problem this solves
-----------------------
Dense retrieval finds documents that *mean* the same thing as the query;
sparse finds documents that *contain* the same words. Neither wins alone --
an acronym search favours BM25, a paraphrase favours cosine, and most real
queries want both. Fusing them is the standard answer, and doing it badly is
the standard failure: adding raw cosine distance to raw BM25 produces a number
whose meaning depends on which store you used, so whichever retriever happens
to emit larger numbers wins every query silently.

Two fusion methods, and the difference that matters
---------------------------------------------------
``rrf`` (reciprocal rank fusion) is the default and the safe one. It ignores
scores entirely and works on *rank*::

    fused(doc) = SUM over sources of  weight / (rrf_k + rank(doc))

with rank counting from 1, so the top hit of a source scores ``1/61`` at the
default ``rrf_k=60``. Because it never sees a score, it cannot be broken by
two sources using incomparable scales -- which dense and sparse always do.
``rrf_k`` damps the influence of a source's very top hits; larger is flatter.

``weighted`` adds the scores directly, which is only meaningful when they are
already on one scale. It therefore *checks* that, and rejects scores outside
``[0, 1]`` rather than producing a plausible-looking wrong order. Combine it
with ``result-normalizer`` first; that is what that component exists for.

In both cases a document's own fields come through unchanged -- text, source
and metadata survive fusion, and only the score is replaced. The winning
source is recorded in ``metadata['fusion_source']`` so a caller can tell
whether a hit arrived densely, lexically, or both, which is the question that
usually explains a surprising result.

Rank comes from position, and that is deliberate
-----------------------------------------------
``rrf`` reads each list's ordering as its ranking: the first element is rank
1, the second rank 2. It does *not* sort by score, because it cannot know
which end of a score is better -- dense results are cosine distance (lower is
better) and sparse results are BM25 (higher is better), and ``ScoredDocument``
carries no direction of its own. Sorting internally would mean guessing, and
guessing wrong inverts the ranking.

The practical consequence: **each list must arrive already ordered best
first**, which is what a vector store and a BM25 retriever both return. A
permuting caller gets a permuted answer, correctly, because they permuted the
rank. ``weighted`` by contrast uses the scores themselves and is therefore
insensitive to input order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core_types import ScoredDocument

__all__ = [
    "METHODS",
    "HybridError",
    "HybridInputError",
    "fuse",
]

#: The fusion methods this component implements.
METHODS: tuple[str, ...] = ("rrf", "weighted")

#: The metadata key recording which source produced a fused document.
SOURCE_KEY = "fusion_source"

#: The conventional RRF damping constant. 60 is the value from the original
#: paper and a reasonable default; it is a parameter, not a magic number, and
#: callers are expected to tune it against their own evaluation set.
DEFAULT_RRF_K = 60


class HybridError(Exception):
    """Base for every failure in this component."""


class HybridInputError(HybridError):
    """The caller passed results this component cannot fuse."""


def _validate_list(documents: object, label: str) -> list[ScoredDocument]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise HybridInputError(
            f"hybrid-retrieval: {label} must be a sequence of ScoredDocument, "
            f"got {type(documents).__name__}"
        )
    items = list(documents)
    for position, item in enumerate(items):
        if not isinstance(item, ScoredDocument):
            raise HybridInputError(
                f"hybrid-retrieval: {label}[{position}] is "
                f"{type(item).__name__}, expected ScoredDocument"
            )
    return items


def _require_distinct_ids(items: Sequence[ScoredDocument], label: str) -> None:
    """Reject a duplicate id inside one list.

    Rank fusion sums one contribution per source, so a duplicated id inside a
    single list would be counted twice and outrank a genuinely better
    document. It is almost always an ingestion bug, and surfacing it beats
    shipping a silently distorted ranking.
    """
    seen: set[str] = set()
    for item in items:
        key = item.chunk_id or ""
        if key in seen:
            raise HybridInputError(
                f"hybrid-retrieval: {label} contains chunk_id {key!r} more "
                f"than once. A duplicate would be fused twice and outrank "
                f"better documents, so it is rejected rather than averaged"
            )
        seen.add(key)


def _rrf(
    dense: Sequence[ScoredDocument],
    sparse: Sequence[ScoredDocument],
    *,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> list[ScoredDocument]:
    scores: dict[str, float] = {}
    base: dict[str, ScoredDocument] = {}
    sources: dict[str, set[str]] = {}

    for label, items, weight in (
        ("dense", dense, dense_weight),
        ("sparse", sparse, sparse_weight),
    ):
        for rank, item in enumerate(items, start=1):
            key = item.chunk_id or ""
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            base.setdefault(key, item)
            sources.setdefault(key, set()).add(label)

    return _finish(scores, base, sources)


def _weighted(
    dense: Sequence[ScoredDocument],
    sparse: Sequence[ScoredDocument],
    *,
    dense_weight: float,
    sparse_weight: float,
) -> list[ScoredDocument]:
    """Add scores directly, after checking they are on one scale."""
    for label, items in (("dense", dense), ("sparse", sparse)):
        for position, item in enumerate(items):
            if not 0.0 <= item.score <= 1.0:
                raise HybridInputError(
                    f"hybrid-retrieval: weighted fusion needs scores on a "
                    f"shared [0, 1] scale, but {label}[{position}] scores "
                    f"{item.score!r}. Normalise both lists first with "
                    f"result-normalizer, or use method='rrf', which ignores "
                    f"scales entirely"
                )

    scores: dict[str, float] = {}
    base: dict[str, ScoredDocument] = {}
    sources: dict[str, set[str]] = {}

    for label, items, weight in (
        ("dense", dense, dense_weight),
        ("sparse", sparse, sparse_weight),
    ):
        for item in items:
            key = item.chunk_id or ""
            scores[key] = scores.get(key, 0.0) + weight * item.score
            base.setdefault(key, item)
            sources.setdefault(key, set()).add(label)

    return _finish(scores, base, sources)


def _finish(
    scores: Mapping[str, float],
    base: Mapping[str, ScoredDocument],
    sources: Mapping[str, set[str]],
) -> list[ScoredDocument]:
    """Sort deterministically and stamp the winning source onto each result."""
    ordered = sorted(
        scores.items(),
        key=lambda row: (
            -row[1],
            base[row[0]].chunk_index,
            row[0],
        ),
    )
    return [
        base[key].model_copy(
            update={
                "score": score,
                "metadata": {
                    **base[key].metadata,
                    SOURCE_KEY: "+".join(sorted(sources[key])),
                },
            }
        )
        for key, score in ordered
    ]


def _validate_weight(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridInputError(
            f"hybrid-retrieval: {label} must be a number, got "
            f"{type(value).__name__} ({value!r})"
        )
    number = float(value)
    if number < 0.0:
        raise HybridInputError(
            f"hybrid-retrieval: {label} must be non-negative, got {value}. A "
            f"negative weight would invert a source's ranking"
        )
    if number == 0.0:
        raise HybridInputError(
            f"hybrid-retrieval: {label} is 0, which discards that source "
            f"entirely. Omit the source instead, so the caller reading this "
            f"call can see what is actually being searched"
        )
    return number


def fuse(
    dense: Sequence[ScoredDocument],
    sparse: Sequence[ScoredDocument],
    *,
    method: str = "rrf",
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ScoredDocument]:
    """Fuse two ranked result lists into one, best first.

    Either list may be empty -- a caller whose lexical index is empty still
    wants its dense hits rather than an exception. Both empty returns an empty
    list.

    Ordering is fully determined by ``(fused score, chunk_index, chunk_id)``,
    so two runs over the same inputs return the same order even when scores
    tie, which they do constantly under RRF.
    """
    dense_items = _validate_list(dense, "dense")
    sparse_items = _validate_list(sparse, "sparse")
    _require_distinct_ids(dense_items, "dense")
    _require_distinct_ids(sparse_items, "sparse")

    if method not in METHODS:
        raise HybridInputError(
            f"hybrid-retrieval: unknown fusion method {method!r}. Valid "
            f"methods are: {', '.join(METHODS)}"
        )

    dense_weight = _validate_weight(dense_weight, "dense_weight")
    sparse_weight = _validate_weight(sparse_weight, "sparse_weight")

    if not dense_items and not sparse_items:
        return []

    if method == "weighted":
        return _weighted(
            dense_items, sparse_items, dense_weight=dense_weight, sparse_weight=sparse_weight
        )

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise HybridInputError(
            f"hybrid-retrieval: rrf_k must be an int of at least 1, got "
            f"{rrf_k!r}. A smaller value sharpens the top of the list at the "
            f"cost of the rest"
        )
    return _rrf(
        dense_items,
        sparse_items,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )

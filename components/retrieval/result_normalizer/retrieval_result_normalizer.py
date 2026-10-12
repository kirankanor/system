"""Normalize retrieval scores from different sources onto one scale.

Why this exists
---------------
A dense retriever returns cosine *distance* (lower is better, bounded by the
metric). A sparse retriever returns BM25 (higher is better, unbounded). Hand
both lists to a hybrid fuser and the numbers mean nothing next to each other
-- the largest BM25 value wins regardless of relevance, which looks like a
working pipeline and answers wrongly.

This component puts scores on a comparable scale so fusion has something
meaningful to combine. It is deliberately a pure function over
``ScoredDocument``: it holds no state, touches no store, and can be tested and
reasoned about on its own.

Two things a naive implementation gets wrong, and this one does not:

* **A constant score list.** Min-max divides by ``max - min``. When every
  score is equal that denominator is zero and the result is ``NaN`` or a
  ``ZeroDivisionError`` -- at the exact moment the caller most needs a sane
  answer, because an all-equal list means "nothing distinguished these".
  This returns a documented constant instead.
* **Score direction.** Cosine distance and BM25 disagree about which end is
  better, and ``core-types`` records that ``ScoredDocument.score`` carries no
  direction of its own. Pass ``higher_is_better=False`` for distance-like
  scores; the normalizer then puts the *best* (lowest) at 1.0.

Methods
-------
``min_max``   Maps to ``[0, 1]`` with the best at 1.0. Compresses outliers
              badly: one far-away document stretches everything else toward 0.
``rank``      Uses relative position instead of magnitude, so an outlier
              cannot dominate. Ties share their average rank, so two equally
              scored documents never come out differently ranked.
``z_score``   Standard deviations from the mean. Unbounded -- useful when you
              want to see how unusual a hit is rather than rank it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from core_types import ScoredDocument

__all__ = [
    "METHODS",
    "NormalizationError",
    "NormalizationInputError",
    "normalize_scores",
]

MethodName = Literal["min_max", "rank", "z_score"]

#: The fusion methods this component implements, in documentation order.
METHODS: tuple[str, ...] = ("min_max", "rank", "z_score")

#: The value every document receives when the input cannot be ranked.
#: Min-max and z-score are both undefined over a constant list; 0.5 is the
#: midpoint of [0, 1] and reads as "no signal either way", which is exactly
#: what an all-equal list means.
CONSTANT_SCORE = 0.5


class NormalizationError(Exception):
    """Base for every failure in this component."""


class NormalizationInputError(NormalizationError):
    """The caller passed something this component cannot normalize."""


def _validate_documents(documents: object) -> list[ScoredDocument]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise NormalizationInputError(
            f"result-normalizer: documents must be a sequence of "
            f"ScoredDocument, got {type(documents).__name__}"
        )
    items = list(documents)
    for position, item in enumerate(items):
        if not isinstance(item, ScoredDocument):
            raise NormalizationInputError(
                f"result-normalizer: documents[{position}] is "
                f"{type(item).__name__}, expected ScoredDocument"
            )
    return items


def _validate_scores(items: Sequence[ScoredDocument]) -> list[float]:
    scores = [float(item.score) for item in items]
    for position, score in enumerate(scores):
        if not math.isfinite(score):
            raise NormalizationInputError(
                f"result-normalizer: documents[{position}].score is {score!r}. "
                f"A non-finite score compares against nothing, so it would "
                f"poison every other document's normalized value"
            )
    return scores


def _validate_method(method: object) -> str:
    if method not in METHODS:
        raise NormalizationInputError(
            f"result-normalizer: unknown normalization method {method!r}. "
            f"Valid methods are: {', '.join(METHODS)}"
        )
    return method


def _min_max(scores: Sequence[float], higher_is_better: bool) -> list[float]:
    best, worst = max(scores), min(scores)
    if best == worst:
        return [CONSTANT_SCORE] * len(scores)
    span = best - worst
    if higher_is_better:
        return [(score - worst) / span for score in scores]
    return [(best - score) / span for score in scores]


def _rank(scores: Sequence[float], higher_is_better: bool) -> list[float]:
    count = len(scores)
    if count == 1:
        return [1.0]
    # Sort positions by score. ``sorted`` is stable, so equal scores keep
    # their input order, which makes ties deterministic.
    order = sorted(range(count), key=lambda i: scores[i], reverse=higher_is_better)
    out = [0.0] * count
    position = 0
    while position < count:
        end = position
        while end + 1 < count and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        # Average of the 0-based ranks this tie occupies, so equal scores get
        # equal normalized values.
        average = (position + end) / 2
        for slot in range(position, end + 1):
            out[order[slot]] = (count - 1 - average) / (count - 1)
        position = end + 1
    return out


def _z_score(scores: Sequence[float]) -> list[float]:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    deviation = math.sqrt(variance)
    if deviation == 0.0:
        return [0.0] * len(scores)
    return [(score - mean) / deviation for score in scores]


def normalize_scores(
    documents: Sequence[ScoredDocument],
    *,
    method: MethodName = "min_max",
    higher_is_better: bool = True,
) -> list[ScoredDocument]:
    """Return copies of ``documents`` with normalized ``score`` values.

    The best document scores highest under every method, so callers never have
    to remember which way a source points: pass ``higher_is_better=False`` for
    distance-like scores and the ordering is corrected here, once.

    Ordering of the input is preserved; only the scores change. The documents
    themselves are replaced rather than mutated because ``ScoredDocument`` is
    frozen, so a caller can keep the originals and compare.

    An empty list returns an empty list rather than raising -- "no results" is
    an expected condition in a pipeline, not a caller error.
    """
    items = _validate_documents(documents)
    chosen = _validate_method(method)
    if not items:
        return []

    scores = _validate_scores(items)

    if chosen == "min_max":
        normalized = _min_max(scores, higher_is_better)
    elif chosen == "rank":
        normalized = _rank(scores, higher_is_better)
    else:
        normalized = _z_score(scores)

    return [
        document.model_copy(update={"score": score})
        for document, score in zip(items, normalized)
    ]

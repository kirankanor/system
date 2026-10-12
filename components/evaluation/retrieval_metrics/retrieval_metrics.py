"""Score retrieval quality against labeled examples.

What this component is
----------------------
One of the three evaluation components (FR-9). It answers a single question:
*given what the retriever returned and what the labels say should have been
returned, how good was that retrieval?*

It produces three numbers, all standard, all hand-checkable:

**hit rate@k**
    The fraction of queries where at least one relevant chunk appeared in the
    top ``k``. A pass/fail measure -- did the retriever land anything at all?

**MRR@k** (mean reciprocal rank)
    The mean of ``1 / rank`` for the *first* relevant chunk in the top ``k``,
    and ``0`` when there is none. Rank 1 scores 1.0, rank 2 scores 0.5, rank 4
    scores 0.25. This is the measure that cares about *ordering*: a retriever
    that puts the right chunk second is meaningfully better than one that
    buries it at ten, even though both "hit".

**recall@k**
    The mean of ``|relevant in top k| / |relevant|``. This is the measure that
    cares about *coverage*: with several valid answers in play, how many did
    the retriever surface?

Why all three
-------------
They fail in different directions, and a single number hides which one broke.

Hit rate can sit at 1.0 while MRR collapses, because every query technically
hit -- just at rank 9. Raising ``k`` moves hit rate and recall up and says
nothing about ordering. Only MRR sees the difference between "found it" and
"found it first". Reporting one of the three is how a retrieval change looks
like an improvement in one review and a regression in the next.

Usage
-----
``k`` is fixed per instance, because an evaluation is run *at a cutoff*:

    metrics = RetrievalMetrics(k=10)
    score = metrics.score(retrieved=["c3", "c1", "c8"], relevant=["c1"])
    corpus = metrics.score_many([...])

``retrieved`` is in **ranked order**, best first. That ordering is the whole
input; nothing here re-sorts, and a caller who passes an unsorted list gets
numbers for the list they passed.

Definitions this component commits to
-------------------------------------
- **Rank is 1-based.** The first retrieved item is rank 1, not rank 0. A
  reciprocal rank of ``1/0`` is not a number, and the alternative convention
  only invites off-by-one errors into every caller.
- **Duplicates occupy one slot.** ``["c1", "c1", "c2"]`` is treated as
  ``["c1", "c2"]``. Otherwise a retriever could inflate its own recall by
  returning the same chunk twice, and the metric would reward a defect.
- **A duplicated label is a set.** ``relevant=["c1", "c1"]`` is one target,
  not two, so recall cannot exceed 1.0.
- **``k`` past the end of the list is clamped, not an error.** Asking for
  ``recall@100`` against a 10-result list is a legitimate question with a
  defined answer; refusing it would make every caller pre-clamp by hand.
- **An empty retrieval with a non-empty label is a miss**, scoring 0 on all
  three. That is a real outcome, not a mistake.
- **An empty label is a caller bug.** Nothing can be correct, so every metric
  would be 0 and the result would silently look like a broken retriever
  rather than an unlabelled query. It raises.

Nothing here reaches a network, a model, or a store. Every input is an
already-materialised list of ids, which is what makes the numbers reproducible
and the component safe to run in CI.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "DEFAULT_K",
    "QueryScore",
    "RetrievalMetrics",
    "RetrievalMetricsError",
    "RetrievalMetricsInputError",
    "RetrievalScore",
    "unique_in_order",
]

#: A conventional default, not a recommendation. An evaluation should name
#: its own cutoff: the right ``k`` is the one the application actually shows
#: a user, which differs between a chat answer and a search results page.
DEFAULT_K: Final[int] = 10


class RetrievalMetricsError(Exception):
    """Base for every failure in this component."""


class RetrievalMetricsInputError(RetrievalMetricsError):
    """The caller passed something this component cannot score."""


@dataclass(frozen=True, slots=True)
class QueryScore:
    """The three metrics for a single query."""

    hit: float
    reciprocal_rank: float
    recall: float
    rank: int | None
    """1-based rank of the first relevant chunk, or ``None`` if there was none."""

    @property
    def missed(self) -> bool:
        return self.hit == 0.0

    def __repr__(self) -> str:
        shown = f"rank={self.rank}" if self.rank is not None else "rank=None"
        return (
            f"QueryScore(hit={self.hit}, reciprocal_rank={self.reciprocal_rank}, "
            f"recall={self.recall}, {shown})"
        )


@dataclass(frozen=True, slots=True)
class RetrievalScore:
    """The corpus-level means, plus every query that produced them."""

    hit_rate: float
    mrr: float
    recall: float
    k: int
    examples: int
    per_query: tuple[QueryScore, ...]

    def as_metrics(self) -> dict[str, float]:
        """The three means as a plain ``{name: value}`` mapping.

        The shape ``EvaluationDataset.assert_no_regression`` compares. Only
        the corpus means are included: a baseline compares like with like, so
        per-query values -- which change count as soon as one case is added --
        do not belong in it.
        """
        return {"hit_rate": self.hit_rate, "mrr": self.mrr, "recall": self.recall}

    @property
    def missed_queries(self) -> tuple[QueryScore, ...]:
        """Only the queries that found nothing, for diagnosing a bad score.

        A corpus mean of 0.60 tells you something got worse; this tells you
        which queries to open.
        """
        return tuple(query for query in self.per_query if query.missed)

    def __repr__(self) -> str:
        return (
            f"RetrievalScore(hit_rate={self.hit_rate}, mrr={self.mrr}, "
            f"recall={self.recall}, k={self.k}, examples={self.examples})"
        )


def unique_in_order(items: Iterable[str]) -> tuple[str, ...]:
    """Drop repeats, keeping first-seen order.

    Exported because it is the rule this component applies to ``retrieved``,
    and a caller preparing the same input for their own report will want the
    identical treatment.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


class RetrievalMetrics:
    """Score retrieval quality at a fixed cutoff ``k``.

    ``k`` is per-instance because an evaluation is run at a cutoff. Construct
    one per cutoff you want to report, rather than threading ``k`` through
    every call.
    """

    def __init__(self, *, k: int = DEFAULT_K, name: str = "retrieval-metrics") -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise RetrievalMetricsInputError(
                f"{name}: k must be an int, got {type(k).__name__}"
            )
        if k < 1:
            raise RetrievalMetricsInputError(
                f"{name}: k must be at least 1, got {k}. A cutoff of 0 would "
                f"inspect no results at all, so every metric would be 0 and "
                f"the score would say nothing about the retriever"
            )
        if not isinstance(name, str) or not name:
            raise RetrievalMetricsInputError(
                f"{name}: name must be a non-empty string, got {name!r}"
            )
        self.k = k
        self.name = name

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def score(
        self,
        *,
        retrieved: Sequence[str],
        relevant: Sequence[str],
    ) -> QueryScore:
        """Score one query.

        ``retrieved`` is in ranked order, best first; ``relevant`` is the set
        of ids that would count as correct. Order within ``relevant`` is
        irrelevant -- a label is a set of acceptable answers, not a ranking.
        """
        top = unique_in_order(self._ids(retrieved, "retrieved"))[: self.k]
        targets = set(self._ids(relevant, "relevant"))
        if not targets:
            raise RetrievalMetricsInputError(
                f"{self.name}: relevant was empty. With nothing to be right "
                f"about, every metric scores 0 and the result looks like a "
                f"broken retriever rather than an unlabelled query"
            )

        found = [item for item in top if item in targets]
        hit = 1.0 if found else 0.0
        # ``index`` is the 0-based position; rank is 1-based by definition.
        rank = next(
            (index + 1 for index, item in enumerate(top) if item in targets), None
        )
        reciprocal_rank = (1.0 / rank) if rank is not None else 0.0
        recall = len(found) / len(targets)
        return QueryScore(
            hit=hit,
            reciprocal_rank=reciprocal_rank,
            recall=recall,
            rank=rank,
        )

    def score_many(
        self,
        examples: Iterable[tuple[Sequence[str], Sequence[str]] | dict[str, Any]],
    ) -> RetrievalScore:
        """Score a corpus and average.

        Each example is either a ``(retrieved, relevant)`` pair or a mapping
        with ``retrieved`` and ``relevant`` keys -- the shape a JSON dataset
        round-trips to, so loading a dataset and scoring it needs no adapter.
        """
        per_query: list[QueryScore] = []
        for position, example in enumerate(examples):
            if isinstance(example, dict):
                try:
                    retrieved, relevant = example["retrieved"], example["relevant"]
                except KeyError as exc:
                    raise RetrievalMetricsInputError(
                        f"{self.name}: example {position} is missing the "
                        f"{exc.args[0]!r} key. Expected 'retrieved' and 'relevant'"
                    ) from exc
            else:
                try:
                    retrieved, relevant = example
                except (TypeError, ValueError) as exc:
                    raise RetrievalMetricsInputError(
                        f"{self.name}: example {position} must be a "
                        f"(retrieved, relevant) pair or a mapping, got "
                        f"{type(example).__name__}"
                    ) from exc
            per_query.append(self.score(retrieved=retrieved, relevant=relevant))

        if not per_query:
            raise RetrievalMetricsInputError(
                f"{self.name}: there were no examples to score. An empty "
                f"corpus would average over nothing, and 0.0 would read as a "
                f"score rather than as an absence of one"
            )

        count = len(per_query)
        return RetrievalScore(
            hit_rate=sum(item.hit for item in per_query) / count,
            mrr=sum(item.reciprocal_rank for item in per_query) / count,
            recall=sum(item.recall for item in per_query) / count,
            k=self.k,
            examples=count,
            per_query=tuple(per_query),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ids(value: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)):
            raise RetrievalMetricsInputError(
                f"{label} must be a sequence of ids, got "
                f"{type(value).__name__}. A string is iterable, so passing "
                f"one would be read as a list of single characters"
            )
        try:
            items = tuple(value)
        except TypeError as exc:
            raise RetrievalMetricsInputError(
                f"{label} must be a sequence of ids, got {type(value).__name__}"
            ) from exc
        for item in items:
            if not isinstance(item, str):
                raise RetrievalMetricsInputError(
                    f"{label} must contain only id strings, got {item!r} "
                    f"({type(item).__name__})"
                )
        return items

    def __repr__(self) -> str:
        return f"RetrievalMetrics(k={self.k}, name={self.name!r})"

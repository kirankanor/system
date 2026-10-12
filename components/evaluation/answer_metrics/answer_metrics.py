"""Score an answer for groundedness and relevance, from supplied text only.

What this component is
----------------------
One of the three evaluation components (FR-9). It turns the two questions a
RAG evaluation actually asks into numbers in ``[0, 1]``:

**groundedness** -- *does the evidence support this answer?*
    The share of the answer's distinctive vocabulary that appears in the
    evidence, weighted by how distinctive each term is.

**answer relevance** -- *does the answer address the question?*
    The same measure run the other way: the share of the query's distinctive
    vocabulary that appears in the answer.

Both are computed **only from the text you pass in**. No model is called, no
lookup happens, nothing is fetched. That is not a limitation to apologise for
-- it is the property that makes the number trustworthy. A "groundedness"
score that consulted an external model would be a second model's opinion
about the first, and it would change when the vendor changed their weights,
which is the opposite of what a regression test needs.

How this differs from ``grounded-answer-contract``
--------------------------------------------------
They overlap on purpose, and serve different jobs:

| | ``grounded-answer-contract`` | ``answer-metrics`` |
|---|---|---|
| answers | *does this pass?* | *how well did it do?* |
| output | a boolean gate | a continuous score |
| use | block a bad answer in a pipeline | compare two versions of a pipeline |

A gate cannot tell you that v2 of a retriever is 4% better than v1 -- it
returns the same ``True`` for both. A score cannot tell a caller to reject
this answer right now -- 0.62 is not a decision. Each component is
standalone and defines its own error hierarchy, because in a real project
you install the one you need.

Why the score is IDF-weighted, not a plain overlap
--------------------------------------------------
``the, a, of`` appear in almost every answer and in almost every passage.
Counting them equally means a fluent paragraph of filler scores highly
grounded, because its stopwords are all present in the evidence.

Weighting by inverse document frequency fixes this: a term that appears
everywhere carries little weight, while a term that appears in one passage
carries a lot. An answer made of distinctive terms that the evidence
actually contains scores high; an answer padded with generic words scores
low, where it belongs.

The unseen-term rule is the important one
-----------------------------------------
A term the evidence never mentions takes the **maximum** available IDF, not
zero.

That looks like a detail and is the whole ballgame. Give it zero weight and
a fabricated claim scores *perfectly grounded*, because every one of its
invented terms contributes ``0/0`` -- the sum vanishes, the ratio reads 1.0,
and the most confabulated answer in the batch wins the comparison. Weighting
unseen terms at the top instead means a claim built from vocabulary the
evidence has never heard of scores at or near 0, which is the direction that
error should fail.

Definitions this component commits to
-------------------------------------
- **Scores are in ``[0, 1]``**, and 1.0 means every distinctive term matched.
- **Only distinctive terms count.** Both texts reduced to the same
  stopwords-only string is an error, not a 1.0 -- there is nothing being
  claimed.
- **No stemming, no lemmatising, no fuzzy matching.** ``exception`` is not
  ``exceptions`` and ``run`` is not ``running``. That is a deliberate gap:
  it means the score never claims a match the text does not contain, and
  anyone who wants stemming can stem before calling.
- **Tokenising is whitespace and punctuation, lowercased.** Enough to be
  stable and reproducible across machines; not a linguistics claim.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "AnswerMetrics",
    "AnswerMetricsError",
    "AnswerMetricsInputError",
    "AnswerScore",
    "DEFAULT_STOPWORDS",
    "tokenize",
]

#: A small, closed English list. Deliberately not a data file: a component
#: whose score depends on a downloaded list is not reproducible, and the
#: error this introduces is bounded -- a few common words weigh slightly
#: more than they should, uniformly, in every run.
DEFAULT_STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a an and are as at be but by for from has have if in into is it its of on
    or that the their then there these this to was were what when where which
    who how why will with would your you we they he she i not no nor so than
    too very can could should may might must do does did done been being our
    us them his her hers its own same s t don now
    """.split()
)

_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


class AnswerMetricsError(Exception):
    """Base for every failure in this component."""


class AnswerMetricsInputError(AnswerMetricsError):
    """The caller passed something this component cannot score."""


def tokenize(text: str, *, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> frozenset[str]:
    """Distinct lowercase terms, with stopwords removed.

    A set, because these metrics measure *which* terms are shared, never how
    often. Repeating a word three times does not make an answer three times
    more grounded -- it makes it three times longer.
    """
    if not isinstance(text, str):
        raise AnswerMetricsInputError(
            f"text must be a string, got {type(text).__name__}"
        )
    return frozenset(word for word in _WORD.findall(text.lower()) if word not in stopwords)


@dataclass(frozen=True, slots=True)
class AnswerScore:
    """Both metrics for one answer."""

    groundedness: float
    relevance: float
    answer_terms: int
    """How many distinctive terms the answer contributed. 0 for an empty answer."""

    def as_metrics(self) -> dict[str, float]:
        """The score as a plain ``{name: value}`` mapping.

        The shape ``EvaluationDataset.assert_no_regression`` compares, and
        what most reporting wants. There is deliberately no combined single
        number: groundedness and relevance fail in different directions, and
        averaging them would let a perfectly faithful answer that ignores the
        question look the same as an irrelevant answer that happens to be
        copied from the evidence -- which is exactly the hiding place a
        regression looks for.
        """
        return {"groundedness": self.groundedness, "relevance": self.relevance}

    def __repr__(self) -> str:
        return (
            f"AnswerScore(groundedness={self.groundedness}, "
            f"relevance={self.relevance}, answer_terms={self.answer_terms})"
        )


class AnswerMetrics:
    """Score answers for groundedness and relevance.

    Construct once and reuse: the configuration is only the stopword list,
    which almost nobody changes.
    """

    def __init__(
        self,
        *,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
        name: str = "answer-metrics",
    ) -> None:
        if not isinstance(stopwords, frozenset) or not all(
            isinstance(word, str) for word in stopwords
        ):
            raise AnswerMetricsInputError(
                f"{name}: stopwords must be a frozenset of strings, got "
                f"{type(stopwords).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise AnswerMetricsInputError(
                f"{name}: name must be a non-empty string, got {name!r}"
            )
        self.stopwords = stopwords
        self.name = name

    # ------------------------------------------------------------------ #
    # The two metrics
    # ------------------------------------------------------------------ #

    def groundedness(self, answer: str, evidence: Sequence[str]) -> float:
        """How much of the answer is backed by ``evidence``.

        ``evidence`` is a sequence of passages -- the exact text that was put
        in front of the model, not a document id and not a summary of it.
        Passing ids would measure nothing: the point is that the terms are
        physically present in what the answer could have seen.
        """
        terms = tokenize(answer, stopwords=self.stopwords)
        passages = self._passages(evidence)
        if not terms:
            # An answer that says nothing distinctive (empty, or pure
            # stopwords) makes no claim, so there is nothing to ground. It
            # scores 0 rather than raising: an evaluation has to be able to
            # score a model that produced nothing, and crashing the whole run
            # on one empty output would be worse than the score.
            return 0.0

        weights, unseen = self._weights(passages)
        total = sum(weights.get(term, unseen) for term in terms)
        if total == 0.0:  # pragma: no cover - guarded by _weights returning > 0
            return 0.0
        present = sum(
            weights.get(term, unseen) for term in terms if self._in(term, passages)
        )
        return self._ratio(present, total)

    def relevance(self, answer: str, query: str) -> float:
        """How much of the query the answer actually addresses.

        Run in the groundedness direction: the query's terms are the target,
        and the answer supplies the evidence for them. A query term absent
        from the answer takes the maximum weight, so an answer that ignores
        the question scores low instead of quietly scoring well.
        """
        terms = tokenize(query, stopwords=self.stopwords)
        if not terms:
            raise AnswerMetricsInputError(
                f"{self.name}: query contained no distinctive terms after "
                f"stopwords were removed ({query!r}). There is no question "
                f"here to measure an answer against"
            )
        answer_terms = tokenize(answer, stopwords=self.stopwords)
        passages = [answer_terms]
        weights, unseen = self._weights(passages)
        total = sum(weights.get(term, unseen) for term in terms)
        if total == 0.0:  # pragma: no cover
            return 0.0
        present = sum(weights.get(term, unseen) for term in terms if term in answer_terms)
        return self._ratio(present, total)

    def score(
        self, *, answer: str, query: str, evidence: Sequence[str]
    ) -> AnswerScore:
        """Both metrics for one answer, in one call."""
        terms = tokenize(answer, stopwords=self.stopwords)
        return AnswerScore(
            groundedness=self.groundedness(answer, evidence),
            relevance=self.relevance(answer, query),
            answer_terms=len(terms),
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _passages(self, evidence: Sequence[str]) -> list[frozenset[str]]:
        if isinstance(evidence, (str, bytes)):
            raise AnswerMetricsInputError(
                f"{self.name}: evidence must be a sequence of passages, got "
                f"{type(evidence).__name__}. A string is iterable, so passing "
                f"one would be read as a list of single characters"
            )
        try:
            items = tuple(evidence)
        except TypeError as exc:
            raise AnswerMetricsInputError(
                f"{self.name}: evidence must be a sequence of passages, got "
                f"{type(evidence).__name__}"
            ) from exc
        for item in items:
            if not isinstance(item, str):
                raise AnswerMetricsInputError(
                    f"{self.name}: evidence must contain only strings, got "
                    f"{type(item).__name__}"
                )
        passages = [tokenize(item, stopwords=self.stopwords) for item in items]
        if not passages:
            raise AnswerMetricsInputError(
                f"{self.name}: there is no evidence. Groundedness computed "
                f"against nothing would score every answer 0, which reports "
                f"the caller's setup rather than the answer"
            )
        if not any(passages):
            raise AnswerMetricsInputError(
                f"{self.name}: the evidence contained no distinctive terms. "
                f"Every answer would score 0 against it, which is a property "
                f"of this evidence, not of the answer"
            )
        return passages

    def _weights(
        self, passages: Sequence[frozenset[str]]
    ) -> tuple[dict[str, float], float]:
        """IDF of every known term, plus the weight of an unseen one.

        The unseen weight is the *maximum* this corpus can produce: a term
        the evidence has never heard of is the most surprising thing in it.
        Returning zero instead would make a fabricated claim score 1.0
        grounded, which is backwards.
        """
        count = len(passages)
        vocabulary = frozenset().union(*passages) if passages else frozenset()
        weights = {
            term: _idf(count, self._containing(term, passages))
            for term in vocabulary
        }
        return weights, _idf(count, 0)

    @staticmethod
    def _containing(term: str, passages: Sequence[frozenset[str]]) -> int:
        return sum(1 for terms in passages if term in terms)

    @staticmethod
    def _in(term: str, passages: Sequence[frozenset[str]]) -> bool:
        return any(term in terms for terms in passages)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:  # pragma: no cover - guarded by callers
            return 0.0
        return max(0.0, min(1.0, numerator / denominator))

    def __repr__(self) -> str:
        return f"AnswerMetrics(name={self.name!r}, stopwords={len(self.stopwords)})"


def _idf(count: int, containing: int) -> float:
    """Smoothed IDF, always positive. ``containing`` 0 gives the maximum.

    The same smoothing ``grounded-answer-contract`` uses, so the two
    components weigh a corpus identically.
    """
    return math.log((count - containing + 0.5) / (containing + 0.5) + 1.0)

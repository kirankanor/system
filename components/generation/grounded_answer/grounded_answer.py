"""Reject an answer whose claims the supplied evidence does not support.

The failure this exists to catch
--------------------------------
A retrieval pipeline's most damaging failure is not returning nothing. It is
returning a fluent, confident, *fabricated* answer. The model fills a gap with
prior knowledge, and because the surrounding sentences are well-grounded the
whole answer reads as sourced. Nobody notices until the invented detail is
acted on.

This component is a guard against that specific failure. It is deliberately
not a quality judge.

How it decides, stated plainly
------------------------------
The answer is split into **claims** -- one per sentence, because a sentence is
the unit a reader would attribute to a source. Each claim is scored for how
much of its distinctive vocabulary the evidence actually contains::

    support(c) = SUM of idf(t) for claim terms t present in the evidence
                 / SUM of idf(t) for all claim terms t

IDF is computed over the evidence set, so a term the evidence never mentions
carries the **maximum** weight and sits in the denominator alone. That is the
intended behaviour: an invented identifier, model name, version or number is
the most distinctive thing in the sentence, and it is weighted accordingly. A
claim whose support falls below ``min_support`` is unsupported, and a single
unsupported claim fails the contract.

This is lexical, and that is a limit worth stating
--------------------------------------------------
Support is measured in shared vocabulary, not meaning. A faithful paraphrase
scores lower than a near-quote, and a claim that is true but absent from the
evidence scores zero -- correctly, from this component's point of view: it was
not grounded in what the caller supplied.

So the result is a **floor**, not a verdict. Passing it does not mean the
answer is good. Failing it means some sentence is not backed by the evidence
in front of it, which is worth a human look before the answer is trusted.

No model, no network (``FR-10``). This is arithmetic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from core_types import Context, ScoredDocument

__all__ = [
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_STOPWORDS",
    "Claim",
    "GroundednessContract",
    "GroundednessResult",
    "GroundingError",
    "GroundingInputError",
    "tokenize",
]

DEFAULT_MIN_SUPPORT: Final[float] = 0.5

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words carrying no evidence of anything. Counted in neither side of the
#: ratio, so "the answer is blue" is judged on "answer" and "blue".
DEFAULT_STOPWORDS: Final[frozenset[str]] = frozenset(
    "a an and are as at be but by for if in into is it of on or such that the "
    "their then there these they this to was will with".split()
)

#: Sentence boundaries: a terminator followed by whitespace, so the split
#: happens after a full stop rather than inside one. Deliberately naive --
#: "Dr. Smith" splits, because a correct splitter needs an abbreviation list.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class GroundingError(Exception):
    """Base for every failure in this component."""


class GroundingInputError(GroundingError):
    """The caller passed something this component cannot judge."""


@dataclass(frozen=True)
class Claim:
    """One sentence of the answer, and whether the evidence supports it."""

    text: str
    support: float
    supported: bool


@dataclass(frozen=True)
class GroundednessResult:
    """The outcome of checking an answer. ``grounded`` is the verdict."""

    grounded: bool
    claims: tuple[Claim, ...]

    @property
    def unsupported(self) -> tuple[str, ...]:
        """Just the text of the claims that failed, for a readable report."""
        return tuple(c.text for c in self.claims if not c.supported)

    def report(self) -> str:
        """A short human-readable summary, one line per claim."""
        if not self.claims:
            return "no claims"
        lines = [
            f"{'ok  ' if c.supported else 'FAIL'} {c.support:.2f}  {c.text}"
            for c in self.claims
        ]
        verdict = "GROUNDED" if self.grounded else "NOT GROUNDED"
        return "\n".join([*lines, verdict])


def tokenize(text: str, *, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> frozenset[str]:
    """Distinct lowercase terms. Public: tokenization *is* the vocabulary."""
    if not isinstance(text, str):
        raise GroundingInputError(
            f"grounded-answer-contract: tokenize expects a string, got "
            f"{type(text).__name__}"
        )
    return frozenset(t for t in _TOKEN.findall(text.lower()) if t not in stopwords)


def _split_claims(answer: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(answer) if part.strip()]


class GroundednessContract:
    """Check an answer's claims against the evidence in a ``Context``.

    ``min_support`` is the fraction of a claim's distinctive vocabulary that
    must appear in the evidence. The default of ``0.5`` tolerates connective
    phrasing while failing a sentence built around one invented term. Lower it
    for terse, technical answers; raise it for prose.
    """

    def __init__(
        self,
        *,
        min_support: float = DEFAULT_MIN_SUPPORT,
        stopwords: frozenset[str] = DEFAULT_STOPWORDS,
        name: str = "grounded-answer-contract",
    ) -> None:
        if isinstance(min_support, bool) or not isinstance(min_support, (int, float)):
            raise GroundingInputError(
                f"grounded-answer-contract: min_support must be a number in "
                f"[0, 1], got {type(min_support).__name__} ({min_support!r})"
            )
        value = float(min_support)
        if not 0.0 <= value <= 1.0 or math.isnan(value):
            raise GroundingInputError(
                f"grounded-answer-contract: min_support must be in [0, 1], "
                f"got {min_support!r}"
            )
        if not isinstance(stopwords, frozenset):
            raise GroundingInputError(
                f"grounded-answer-contract: stopwords must be a frozenset, "
                f"got {type(stopwords).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise GroundingInputError(
                f"grounded-answer-contract: name must be a non-empty string, "
                f"got {name!r}"
            )
        self.min_support = value
        self.stopwords = stopwords
        self.name = name

    def _weights(self, passages: list[frozenset[str]]) -> tuple[dict[str, float], float]:
        """IDF of every evidence term, plus the weight of an unseen term.

        ``passages`` is a list, not a set: two passages with identical
        vocabulary are still two passages, and collapsing them would corrupt
        the document frequency this whole score rests on.

        The second value is the weight of a term appearing in **no** passage
        -- df 0, so the highest IDF available. Returning it matters more than
        anything else here: a term the evidence never mentions must carry full
        weight in the denominator, or a sentence built entirely from invented
        vocabulary would sum to zero and, divided by zero, look perfectly
        supported.
        """
        count = len(passages)
        vocabulary: set[str] = set()
        for terms in passages:
            vocabulary |= terms
        weights = {
            term: _idf(count, _containing(term, passages)) for term in vocabulary
        }
        return weights, _idf(count, 0)

    def _validate(
        self, answer: object, context: object
    ) -> tuple[str, list[ScoredDocument]]:
        if not isinstance(answer, str):
            raise GroundingInputError(
                f"grounded-answer-contract: answer must be a string, got "
                f"{type(answer).__name__}"
            )
        if not answer.strip():
            raise GroundingInputError(
                f"grounded-answer-contract: answer is empty or whitespace. "
                f"There is nothing to check; an empty answer makes no claims "
                f"and would pass vacuously"
            )
        if not isinstance(context, Context):
            raise GroundingInputError(
                f"grounded-answer-contract: context must be a Context, got "
                f"{type(context).__name__}"
            )
        if not context.chunks:
            raise GroundingInputError(
                f"grounded-answer-contract: context has no passages. There is "
                f"no evidence to ground an answer in, so every claim would "
                f"fail -- and a uniform failure tells the caller nothing they "
                f"did not already know"
            )
        return answer, list(context.chunks)

    def check(self, answer: str, context: Context) -> GroundednessResult:
        """Score every claim. Raises only on unusable input."""
        text, chunks = self._validate(answer, context)
        per_passage = [tokenize(c.text, stopwords=self.stopwords) for c in chunks]
        evidence: set[str] = set().union(*per_passage) if per_passage else set()
        weights, unseen = self._weights(per_passage)

        claims: list[Claim] = []
        for sentence in _split_claims(text):
            terms = tokenize(sentence, stopwords=self.stopwords)
            if not terms:
                # Nothing but stopwords; no distinctive claim is being made.
                claims.append(Claim(text=sentence, support=1.0, supported=True))
                continue
            # An unseen term takes the maximum weight, not zero -- see
            # _weights. That is what makes an invented identifier expensive.
            total = sum(unseen if t not in weights else weights[t] for t in terms)
            covered = sum(weights[t] for t in terms if t in evidence)
            support = covered / total if total else 1.0
            claims.append(
                Claim(
                    text=sentence,
                    support=support,
                    supported=support >= self.min_support,
                )
            )

        return GroundednessResult(
            grounded=all(c.supported for c in claims),
            claims=tuple(claims),
        )

    def is_grounded(self, answer: str, context: Context) -> bool:
        """``True`` only when every claim is supported."""
        return self.check(answer, context).grounded

    def __repr__(self) -> str:
        return (
            f"GroundednessContract(min_support={self.min_support}, "
            f"name={self.name!r})"
        )


def _idf(count: int, containing: int) -> float:
    """Smoothed IDF, always positive. ``containing`` 0 gives the maximum."""
    return math.log((count - containing + 0.5) / (containing + 0.5) + 1.0)


def _containing(term: str, passages: Sequence[frozenset[str]]) -> int:
    return sum(1 for terms in passages if term in terms)

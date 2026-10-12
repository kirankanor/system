"""Rank candidates by how much of the query they actually cover.

Why a second ranking method exists
----------------------------------
A cross-encoder is the accurate way to rerank and needs a wheel, a model
download and a GPU or a lot of patience. This component is the other end of
that trade: no model, no dependency, no corpus index -- a score you can
compute in your head from the query and the passage.

It is what you reach for when a reranker must run on every request, when the
model is unavailable, or when you need a ranking you can *explain*. It is not
a substitute for a cross-encoder on paraphrase; it matches words, not meaning.

The score
---------
IDF-weighted query coverage, in ``[0, 1]``::

    score(d) = SUM of idf(t) for query terms t present in d
               / SUM of idf(t) for all query terms t

Read it as *"how much of the query's weight does this passage cover?"* -- a
document containing every query term scores 1.0, one containing none scores
0.0, and a partial match lands in between, weighted toward the distinctive
words.

Two properties worth naming:

* **Bounded and comparable.** Unlike BM25 this is not a raw log-sum, so 0.63
  means the same thing across candidate sets and can be thresholded.
* **IDF is computed over the candidate set, not a corpus.** A reranker is
  handed a handful of candidates and no index, so rarity is estimated from
  those candidates. That is the right denominator anyway: a term appearing in
  *every* candidate discriminates between none of them, however rare it is in
  the wider world.

Ties are broken by ``chunk_index``, then ``chunk_id`` -- never by input
order, so the same query over the same candidates returns the same ranking
even if the caller's list was permuted.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from core_types import ScoredDocument

__all__ = [
    "DEFAULT_STOPWORDS",
    "RankingError",
    "RankingInputError",
    "RelevanceRanker",
    "score_documents",
    "tokenize",
]

_TOKEN = re.compile(r"[a-z0-9]+")

#: Words that appear in nearly every passage and so discriminate nothing.
#: Counted toward neither the numerator nor the denominator.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be but by for if in into is it of on or such that the "
    "their then there these they this to was will with".split()
)


class RankingError(Exception):
    """Base for every failure in this component."""


class RankingInputError(RankingError):
    """The caller passed something this component cannot rank."""


def tokenize(
    text: str, *, stopwords: frozenset[str] | None = None
) -> frozenset[str]:
    """Split ``text`` into the distinct lowercase terms this scorer uses.

    Public for the same reason as in sparse retrieval: a caller who cannot see
    how text becomes terms cannot debug why a passage did not match.
    """
    if not isinstance(text, str):
        raise RankingInputError(
            f"relevance-ranking: tokenize expects a string, got "
            f"{type(text).__name__}"
        )
    drop = DEFAULT_STOPWORDS if stopwords is None else stopwords
    return frozenset(term for term in _TOKEN.findall(text.lower()) if term not in drop)


def _validate_documents(documents: object) -> list[ScoredDocument]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise RankingInputError(
            f"relevance-ranking: documents must be a sequence of ScoredDocument, "
            f"got {type(documents).__name__}"
        )
    items = list(documents)
    for position, item in enumerate(items):
        if not isinstance(item, ScoredDocument):
            raise RankingInputError(
                f"relevance-ranking: documents[{position}] is "
                f"{type(item).__name__}, expected ScoredDocument"
            )
    return items


def _validate_query(query: object, stopwords: frozenset[str]) -> frozenset[str]:
    if not isinstance(query, str):
        raise RankingInputError(
            f"relevance-ranking: query must be a string, got "
            f"{type(query).__name__}"
        )
    if not query.strip():
        raise RankingInputError(
            f"relevance-ranking: query is empty or whitespace. There is no "
            f"query to cover, so ranking it would order candidates arbitrarily"
        )
    return tokenize(query, stopwords=stopwords)


def _validate_top_k(top_k: object, available: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RankingInputError(
            f"relevance-ranking: top_k must be an int, got "
            f"{type(top_k).__name__} ({top_k!r})"
        )
    if top_k < 1:
        raise RankingInputError(
            f"relevance-ranking: top_k must be at least 1 to return anything, "
            f"got {top_k}"
        )
    return min(top_k, available)


def _weight_by_rarity(query_terms: Iterable[str], document_terms: Sequence[frozenset[str]]) -> dict[str, float]:
    """IDF of each query term, estimated across the candidate set."""
    count = len(document_terms)
    containing = {term: 0 for term in query_terms}
    for terms in document_terms:
        for term in query_terms:
            if term in terms:
                containing[term] += 1
    if count == 0:
        return {term: 0.0 for term in query_terms}
    return {
        term: math.log((count - seen + 0.5) / (seen + 0.5) + 1.0)
        for term, seen in containing.items()
    }


def score_documents(
    query: str,
    documents: Sequence[ScoredDocument],
    *,
    stopwords: frozenset[str] | None = None,
) -> list[ScoredDocument]:
    """Return copies of ``documents`` carrying a relevance score in ``[0, 1]``.

    Input order is preserved; only the scores change. The documents are
    replaced rather than mutated because ``ScoredDocument`` is frozen, so a
    caller can keep the retriever's original ordering and scores to compare.

    Scoring and ordering are separate on purpose: this scores, and
    :meth:`RelevanceRanker.rerank` orders and truncates.
    """
    drop = DEFAULT_STOPWORDS if stopwords is None else stopwords
    terms = _validate_query(query, drop)
    items = _validate_documents(documents)
    if not items:
        return []

    per_document = [tokenize(item.text, stopwords=drop) for item in items]
    weights = _weight_by_rarity(terms, per_document)
    total = sum(weights.values())

    if total == 0.0:
        # Every query term is absent from every candidate, or all of them were
        # stopwords. A flat zero is the honest answer: nothing was covered.
        return [item.model_copy(update={"score": 0.0}) for item in items]

    return [
        item.model_copy(
            update={
                "score": sum(weights[t] for t in terms if t in document_terms) / total
            }
        )
        for item, document_terms in zip(items, per_document)
    ]


class RelevanceRanker:
    """Deterministic lexical reranker implementing ``RerankerProtocol``.

    Holds no model and no index, so a single instance is safe to share
    between threads and costs nothing to construct.
    """

    def __init__(
        self, *, stopwords: frozenset[str] | None = None, name: str = "relevance"
    ) -> None:
        if not isinstance(name, str) or not name:
            raise RankingInputError(
                f"relevance-ranking: name must be a non-empty string, got {name!r}"
            )
        self.stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self.name = name

    def score(
        self, query: str, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        """Score every candidate, preserving input order."""
        return score_documents(query, documents, stopwords=self.stopwords)

    def rerank(
        self, query: str, documents: Sequence[ScoredDocument], top_k: int
    ) -> list[ScoredDocument]:
        """Return the best ``top_k`` candidates in the new order.

        Ties resolve by ``chunk_index`` then ``chunk_id``, so the result never
        depends on the order the candidates arrived in. Asking for more than
        exist returns every candidate rather than raising -- a caller should
        not have to know how many results a retriever produced.
        """
        scored = self.score(query, documents)
        limit = _validate_top_k(top_k, len(scored))
        if not scored:
            return []
        ordered = sorted(
            scored, key=lambda d: (-d.score, d.chunk_index, d.chunk_id or "")
        )
        return ordered[:limit]

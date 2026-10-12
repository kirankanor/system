"""Lexical retrieval with Okapi BM25, implemented here rather than imported.

Why BM25 is implemented rather than delegated
---------------------------------------------
The design table for this project lists ``rank-bm25`` as this component's
optional dependency. It is deliberately not used, for two reasons that both
come down to determinism:

* **Two scorers would disagree.** ``rank-bm25`` computes IDF differently from
  the formulation below (it omits the ``+1`` smoothing on some paths), so a
  corpus scored with one and scored with the other ranks differently. A
  retriever whose results depend on which packages happen to be installed
  cannot be regression-tested, and every "why did this change?" investigation
  would start by asking what got installed.
* **The formula is thirty lines.** Okapi BM25 is fully specified, has no
  external state, and needs no service. Making a well-specified local
  computation wait on a third-party wheel is the kind of coupling this library
  exists to remove.

The cost is that this scorer is *this* scorer: it does not implement BM25+
extensions, parameter-free variants, or per-field weighting.

The formula
-----------
For a query with terms ``t`` and a document ``d``::

    score(d, q) = SUM over t in q of
        IDF(t) * f(t,d) * (k1 + 1)
        / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

``k1`` saturates term frequency (the tenth occurrence of a word counts for
much less than the second); ``b`` length-normalises, so a long document does
not win by sheer word count. The ``+1`` inside the IDF keeps it positive even
for a term appearing in more than half the corpus -- without it, common terms
score *negatively* and a document can end up worse off for containing the
query's main word.

Only documents scoring above zero are returned. A document sharing no term
with the query has no lexical claim on it, and returning it with score 0.0
would push real matches down the ranked list.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from core_types import Chunk, ScoredDocument

__all__ = [
    "SparseRetriever",
    "RetrievalError",
    "RetrievalInputError",
    "tokenize",
]

#: Words BM25 scores best when they *discriminate*. These are stripped
#: before scoring because they appear in nearly every document, so their IDF
#: is near zero and they contribute noise rather than signal -- but they still
#: count toward document length, which is what the length normalisation needs.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be but by for if in into is it of on or such that the "
    "their then there these they this to was will with".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


class RetrievalError(Exception):
    """Base for every failure in this component."""


class RetrievalInputError(RetrievalError):
    """The caller passed something this component cannot retrieve with."""


def tokenize(
    text: str, *, stopwords: frozenset[str] | None = None
) -> list[str]:
    """Split ``text`` into the lowercase terms BM25 scores.

    Public because tokenization *is* the retrieval model's vocabulary: a
    caller who cannot see how text becomes terms cannot debug why a document
    did not match. Numbers are kept -- a query for an error code or a version
    is common and must not be stripped.
    """
    if not isinstance(text, str):
        raise RetrievalInputError(
            f"sparse-retrieval: tokenize expects a string, got "
            f"{type(text).__name__}"
        )
    drop = DEFAULT_STOPWORDS if stopwords is None else stopwords
    return [term for term in _TOKEN.findall(text.lower()) if term not in drop]


class SparseRetriever:
    """Score a fixed corpus of chunks against a query with BM25.

    The corpus is indexed once, at construction, because BM25 is a
    collection-wide statistic: a term's IDF depends on how many documents
    contain it, so a document cannot be scored against a corpus that is still
    arriving. Rebuild the retriever to change the corpus.
    """

    def __init__(
        self,
        documents: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: frozenset[str] | None = None,
        k: int = 5,
        name: str = "sparse-bm25",
    ) -> None:
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise RetrievalInputError(
                f"sparse-retrieval: documents must be a sequence of Chunk, got "
                f"{type(documents).__name__}"
            )
        items = list(documents)
        for position, item in enumerate(items):
            if not isinstance(item, Chunk):
                raise RetrievalInputError(
                    f"sparse-retrieval: documents[{position}] is "
                    f"{type(item).__name__}, expected Chunk"
                )
        for label, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RetrievalInputError(
                    f"sparse-retrieval: {label} must be a number, got "
                    f"{type(value).__name__} ({value!r})"
                )
            if not math.isfinite(float(value)):
                raise RetrievalInputError(
                    f"sparse-retrieval: {label} must be finite, got {value!r}"
                )
        if not (0.0 <= float(b) <= 1.0):
            raise RetrievalInputError(
                f"sparse-retrieval: b must be between 0 (no length "
                f"normalisation) and 1 (full), got {b}"
            )
        if float(k1) < 0.0:
            raise RetrievalInputError(
                f"sparse-retrieval: k1 must be non-negative, got {k1}"
            )
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise RetrievalInputError(
                f"sparse-retrieval: k must be an int of at least 1, got {k!r}"
            )
        if not isinstance(name, str) or not name:
            raise RetrievalInputError(
                f"sparse-retrieval: name must be a non-empty string, got "
                f"{name!r}"
            )

        self.k1 = float(k1)
        self.b = float(b)
        self.stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self.k = k
        self.name = name
        self._documents = items
        self._terms = [tokenize(item.text, stopwords=self.stopwords) for item in items]
        self._lengths = [len(terms) for terms in self._terms]
        self._frequencies = [Counter(terms) for terms in self._terms]
        self._total = len(items)
        self._average = (
            sum(self._lengths) / self._total if self._total else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            self._document_frequency.update(frequencies.keys())

    def __len__(self) -> int:
        return self._total

    @property
    def average_document_length(self) -> float:
        """Mean token count, which the length normalisation divides by."""
        return self._average

    def idf(self, term: str) -> float:
        """Return the IDF ``term`` carries across this corpus.

        Exposed so a caller can see why a common term contributes almost
        nothing, which is otherwise an invisible property of the score.
        """
        if not isinstance(term, str):
            raise RetrievalInputError(
                f"sparse-retrieval: idf expects a string term, got "
                f"{type(term).__name__}"
            )
        seen = self._document_frequency.get(term, 0)
        if self._total == 0:
            return 0.0
        return math.log(
            (self._total - seen + 0.5) / (seen + 0.5) + 1.0
        )

    def retrieve(self, query: str, *, k: int | None = None) -> list[ScoredDocument]:
        """Return the chunks matching ``query``, best first.

        Ties are broken by ``chunk_index`` then ``chunk_id``, so the same
        query over the same corpus always returns the same order -- including
        when several documents share an exact score, which BM25 produces
        readily on short documents.
        """
        if not isinstance(query, str):
            raise RetrievalInputError(
                f"sparse-retrieval: query must be a string, got "
                f"{type(query).__name__}"
            )
        if not query.strip():
            raise RetrievalInputError(
                f"sparse-retrieval: query is empty or whitespace"
            )
        limit = self.k if k is None else k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise RetrievalInputError(
                f"sparse-retrieval: k must be an int of at least 1, got "
                f"{limit!r}"
            )
        if self._total == 0:
            return []

        query_terms = tokenize(query, stopwords=self.stopwords)
        scored: list[tuple[float, int, str]] = []
        for position, frequencies in enumerate(self._frequencies):
            length = self._lengths[position]
            normaliser = (
                self.k1
                * (1.0 - self.b + self.b * (length / self._average))
                if self._average
                else self.k1
            )
            score = 0.0
            for term in query_terms:
                count = frequencies.get(term)
                if not count:
                    continue
                score += (
                    self.idf(term) * count * (self.k1 + 1.0) / (count + normaliser)
                )
            if score > 0.0:
                item = self._documents[position]
                scored.append((score, item.chunk_index, item.chunk_id or "", position))

        # Tie-break on the document's own index, then its id, so the same
        # corpus always ranks equally-scored documents the same way regardless
        # of the order they were passed in. The input position is carried only
        # to look the document back up; it is never part of the ordering.
        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        top = scored[:limit]

        return [
            ScoredDocument.from_chunk(self._documents[position], score=score)
            for score, _chunk_index, _chunk_id, position in top
        ]

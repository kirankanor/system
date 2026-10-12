# sparse-retrieval: Okapi BM25, implemented rather than imported

```
sparse_retrieval.py   SparseRetriever, tokenize(), the error vocabulary
example.py            runnable: BM25 ranking, IDF, length normalisation
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; NO optional packages, core-types dependency
```

## Purpose

Lexical retrieval over a fixed corpus. A query is tokenized, every document
is scored on how well it matches, and the best `k` come back as
`ScoredDocument` values -- provenance intact, because the chunk already knows
where it came from.

## Why BM25 is implemented here rather than delegated

The design table for this project lists `rank-bm25` as this component's
optional dependency. **It is deliberately not used**, and the manifest says
so. Two reasons, both about determinism:

- **Two scorers would disagree.** `rank-bm25` computes IDF differently from
  the formulation below (it omits the `+1` smoothing on some paths), so a
  corpus scored with one and scored with the other ranks differently. A
  retriever whose results depend on which packages happen to be installed
  cannot be regression-tested, and every "why did this change?" would start
  by asking what got installed.
- **The formula is thirty lines.** Okapi BM25 is fully specified, has no
  external state, and needs no service. Making a well-specified local
  computation wait on a third-party wheel is the kind of coupling this
  library exists to remove.

The cost is that this is *this* scorer: no BM25+ extensions, no parameter-free
variants, no per-field weighting.

## The formula

```
score(d, q) = SUM over t in q of
    IDF(t) * f(t,d) * (k1 + 1)
    / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

`k1` saturates term frequency (the tenth occurrence counts for much less than
the second); `b` length-normalises, so a long document does not win by sheer
word count. The `+1` inside the IDF keeps it positive even for a term in more
than half the corpus -- **without it, common terms score *negatively*** and a
document can end up worse off for containing the query's main word.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `SparseRetriever(documents, *, k1=1.5, b=0.75, stopwords=None, k=5, name="sparse-bm25")` | | corpus indexed once, at construction |
| `.retrieve(query, *, k=None)` | `-> list[ScoredDocument]` | best first, positive scores only |
| `.idf(term)` | `-> float` | exposed, so a caller can see why a common term contributes little |
| `.average_document_length` | `float` | the length normalisation divides by this |
| `tokenize(text, *, stopwords=None)` | `-> list[str]` | public: tokenization *is* the vocabulary |
| `RetrievalError` | base exception | catches this component |
| `RetrievalInputError` | `RetrievalError` | the caller passed something unusable |

### Invariants the tests pin down

- **The corpus is indexed at construction, not lazily.** BM25 is a
  *collection-wide* statistic: a term's IDF depends on how many documents
  contain it, so a document cannot be scored against a corpus that is still
  arriving. Rebuild the retriever to change the corpus.
- **Only documents scoring above zero are returned.** A document sharing no
  term with the query has no lexical claim on it, and returning it at 0.0
  would push real matches down the list.
- **Ties break on `chunk_index`, then `chunk_id`** -- not on the order the
  corpus was passed in. The same query over the same corpus returns the same
  order, and permuting the input does not change the ranking.
- **Stopwords are dropped from scoring but kept in document length.** They
  contribute no signal (IDF near zero) but they are still text, and length
  normalisation needs the real length.
- **Numbers are kept.** A query for an error code or a version is common and
  must not be stripped.
- **`b` must be in `[0, 1]`** and `k1` non-negative; both reject `NaN` and
  `inf`, which would make every score non-finite.

### What `b` is for, proven by turning it off

Given a short document that is exactly `"python"` and a long one with the
same word buried in filler, the default `b=0.75` ranks the short one first.
With `b=0.0` (no length normalisation) the longer one wins on repetition
alone. The component tests assert both, so the parameter's effect is
demonstrated rather than asserted.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `Chunk` and `ScoredDocument`.
- **Packages:** **none**, optional or required. `rank-bm25` is not used and
  not declared.
- **External services:** none.

## Trade-offs

**Use it when** you want lexical recall -- exact terms, identifiers, error
codes -- alongside or instead of dense retrieval. Almost always as one input
to `hybrid-retrieval`.

**It costs** one corpus pass at construction (tokenize every document, count
term frequencies) and one pass per query.

**When not to use it:** a corpus that changes per query. The index is built
once; rebuilding per query is a sign you want a different tool. Also,
cross-lingual or paraphrase queries -- BM25 matches words, not meaning, and
will happily return nothing for a synonym.

## Known limitations

**The index is in memory and rebuilt wholesale.** There is no incremental
update. A corpus of hundreds of thousands of chunks will cost real memory and
a real pause on construction.

**Tokenization is a regex, not a word segmenter.** `[a-z0-9]+` lowercased.
That is wrong for CJK, misleading for agglutinative languages, and adequate
for English source code and prose.

**No phrase queries.** `"error handling"` matches documents containing *both*
words anywhere, not the phrase. Phrase scoring needs positional indexes,
which this does not build.

**No field weighting.** A match in a title and a match in body text score
identically. Fields are not separate here; a caller wanting weighting should
index them as separate chunks.

**Stopword removal is fixed and English-only.** Pass `stopwords=` to override,
but there is no language detection.

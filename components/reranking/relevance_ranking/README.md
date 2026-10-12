# relevance-ranking: reranking you can do on a napkin

```
relevance_ranking.py  RelevanceRanker, score_documents(), tokenize(), errors
example.py            runnable: coverage scoring, IDF weighting, determinism
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; NO packages, core-types dependency
```

## Purpose

Rerank a short list of retrieved candidates by how much of the query each one
actually covers. No model, no wheel, no index, no network — a score you can
reproduce by hand from the query and the passage.

It exists because a cross-encoder is not always available: not on a machine
without the wheel, not on every request when a forward pass per candidate is
too slow to afford, and not when someone asks *why is this first?* and the
answer has to be more than "the model said so".

## The score

```
score(d) = SUM of idf(t) for query terms t present in d
           / SUM of idf(t) for all query terms t

idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

Read it as *"how much of the query's weight does this passage cover?"* — 1.0
means every query term is present, 0.0 means none are, and a partial match
lands in between weighted toward the distinctive words.

**The result is bounded to `[0, 1]`.** That is the deliberate difference from
BM25 in `sparse-retrieval`: this is not a raw log-sum, so `0.63` means the
same thing across candidate sets and can be thresholded.

**IDF is estimated over the candidate set, not a corpus.** A reranker is
handed a handful of candidates and no index, so rarity has to come from
somewhere — it comes from the list itself. That is the right denominator
anyway: a term appearing in *every* candidate adds the same constant to each
one's numerator, so it cannot separate them, however rare it is in the wider
world. It still raises every candidate's floor (the `+1` smoothing keeps it
positive), which compresses the range without reordering it.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `RelevanceRanker(*, stopwords=None, name="relevance")` | | holds nothing but a stopword set |
| `.score(query, documents)` | `-> list[ScoredDocument]` | scores in **input order** |
| `.rerank(query, documents, top_k)` | `-> list[ScoredDocument]` | best first, truncated to `top_k` |
| `score_documents(query, documents, *, stopwords=None)` | `-> list[ScoredDocument]` | the function form |
| `tokenize(text, *, stopwords=None)` | `-> frozenset[str]` | public: tokenization *is* the vocabulary |
| `DEFAULT_STOPWORDS` | `frozenset[str]` | the fixed English set, override with `stopwords=` |
| `RankingError` | base exception | catches this component |
| `RankingInputError` | `RankingError` | the caller passed something unusable |

### Invariants the tests pin down

- **Ties break on `chunk_index`, then `chunk_id`** — never on the order the
  candidates arrived in. Permuting the input does not permute a tied ranking,
  so the same query over the same candidates returns the same order.
- **`score()` preserves input order; `rerank()` orders.** Separating them
  means a caller can inspect scores against the retriever's own ordering
  before committing to a new one.
- **Scoring replaces, never mutates.** `ScoredDocument` is frozen; the
  retriever's originals are left untouched so both can be compared.
- **A blank query is rejected.** There is no query to cover, so ranking would
  order candidates arbitrarily — and an arbitrary order looks exactly like a
  confident one.
- **Every candidate is scored, then truncated** — not the other way round.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `ScoredDocument`.
- **Packages:** **none**, optional or required.
- **External services:** none.

## Trade-offs

**Use it when** a reranker must run cheaply and always, when the model wheel
is not installed, or when a ranking has to be explainable to a person. It is
also a sensible first stage: cut a longer list lexically, then let
`cross-encoder-reranker` order what remains.

**It costs** one tokenize pass over the candidates plus one sort. No I/O, no
load time, no model memory.

**When not to use it:** paraphrase or semantic similarity. This matches
*words*. "car" and "automobile" score 0.0 against each other however clearly
they mean the same thing, and no amount of IDF weighting changes that — which
is precisely what the cross-encoder is for.

## Known limitations

**No stemming, so `exception` does not match `exceptions`.** The tokenizer is
`[a-z0-9]+` lowercased, the same one as `sparse-retrieval`, and it does no
morphology. The gate tests pin this behaviour deliberately so it reads as a
documented limit rather than a bug. A caller who needs it should stem or
lemmatize before calling.

**A query whose terms all appear in every candidate scores everything
equally.** The weights are all zero, the denominator is zero, and every
candidate scores `0.0`. That is the honest answer — nothing was covered — but
it means this cannot break a tie the retriever could not.

**English-only stopword set, fixed by default.** Pass `stopwords=` to
override; there is no language detection.

**It is not a relevance model.** It measures lexical coverage. Do not read
`0.8` as "80% relevant" — read it as "covers 80% of the query's IDF weight".

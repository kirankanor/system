# hybrid-retrieval: fuse two ranked lists without pretending scales match

```
hybrid_retrieval.py   fuse() and the error vocabulary
example.py            runnable: RRF vs weighted, weights, the guard
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; core-types as a component dependency
```

## Purpose

Dense retrieval finds documents that *mean* the same thing as the query;
sparse finds documents that *contain* the same words. Neither wins alone --
an acronym search favours BM25, a paraphrase favours cosine, and most real
queries want both.

Fusing them badly is the standard failure: **adding raw cosine distance to
raw BM25 produces a number whose meaning depends on which store you used**, so
whichever retriever happens to emit larger numbers wins every query, silently,
forever. The pipeline runs, the ranking looks plausible, and no error will
ever reveal it.

## Two fusion methods

**`rrf`** (reciprocal rank fusion) is the default and the safe one. It
ignores scores entirely and works on *rank*:

```
fused(doc) = SUM over sources of  weight / (rrf_k + rank(doc))
```

Rank counts from 1, so a source's top hit scores `1/61` at the default
`rrf_k=60`. Because it never sees a score, **it cannot be broken by two
sources using incomparable scales** -- which dense and sparse always do.
Larger `rrf_k` flattens the list, reducing the influence of a source's very
top hits.

**`weighted`** adds scores directly, which is only meaningful when they are
already on one scale. It therefore *checks* that, and rejects scores outside
`[0, 1]` with a message pointing at `result-normalizer` -- because that is
what that component exists for. Combining unnormalized scores is not a
reasonable thing to do, so it is not allowed to happen quietly.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `fuse(dense, sparse, *, method="rrf", dense_weight=1.0, sparse_weight=1.0, rrf_k=60)` | `-> list[ScoredDocument]` | best first |
| `METHODS` | `("rrf", "weighted")` | |
| `SOURCE_KEY` | `"fusion_source"` | the metadata key recording the winning source |
| `HybridError` | base exception | catches this component |
| `HybridInputError` | `HybridError` | the caller passed something unusable |

## Invariants the tests pin down

- **Ordering is fully determined by `(fused score, chunk_index, chunk_id)`.**
  Two runs over the same inputs return the same order even when scores tie,
  which they do constantly under RRF.
- **A document's own fields survive fusion.** Text, source name and metadata
  come through unchanged; only the score is replaced. The winning source is
  recorded in `metadata['fusion_source']` as `"dense"`, `"sparse"` or
  `"dense+sparse"` -- which is the question that usually explains a surprising
  result.
- **Matching is by `chunk_id`.** It is the stable identity both retrievers
  agree on.
- **Either list may be empty.** A caller whose lexical index is empty still
  gets their dense hits rather than an exception.
- **A duplicate `chunk_id` inside one list is rejected.** RRF sums one
  contribution per source, so a duplicate would be counted twice and outrank
  a genuinely better document. That is almost always an ingestion bug, and
  surfacing it beats shipping a distorted ranking.
- **A zero weight is rejected**, because it silently discards a source while
  leaving the caller's code reading as though both are searched. Omit the
  source instead.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `ScoredDocument`.
- **Packages:** standard library only.
- **External services:** none. It does not run the retrievers; it fuses what
  they return.

## Trade-offs

**Use it when** you have two ranked lists from two different signals.

**It costs** one pass over each list and a dict of the union.

**When not to use it:** a single source (nothing to fuse), or when you need
the fused score to *mean* something. RRF scores are comparable within one
result set and meaningless outside it -- do not threshold on them, and do not
compare them between queries.

## Known limitations

**`rrf` output is not on a natural scale.** Scores cluster near
`1/(rrf_k + 1)` and the range depends on how many sources matched. This is
inherent to RRF and is the price of being immune to scale mismatch.

**Only two sources are supported.** `fuse` takes exactly `dense` and
`sparse`. A third retriever would mean a different signature, and the callers
who need three-way fusion generally know their weights well enough to want
`weighted` anyway.

**No score propagation on a miss.** A document present in one list and absent
from the other is fused from the one contribution. RRF's usual variant would
add a floor score for the missing source; this does not, so a document found
by only one retriever is ranked strictly below any document found by both
when weights are equal.

**`weighted` requires the caller to have normalized.** It checks the range but
cannot check that both lists were normalized *the same way* -- normalizing
one with `min_max` and the other with `rank` gives scores that pass the check
and mean different things.

**`chunk_id` must be stable and shared.** If the two retrievers return
different ids for the same passage, fusion treats them as two documents. That
is a data problem, not something this component can detect.

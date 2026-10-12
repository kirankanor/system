# retrieval-result-normalizer: scores that can finally be compared

```
retrieval_result_normalizer.py   normalize_scores() and the error vocabulary
example.py                       runnable: three methods, direction, the NaN case
tests/                           success paths and deliberate failure paths
manifest.json                    schema-valid; core-types as a component dependency
```

## Purpose

A dense store returns cosine **distance** (lower is better). A sparse
retriever returns BM25 (higher is better, unbounded). Add those together and
the largest number wins regardless of relevance -- the pipeline runs, answers
confidently, and is wrong in a way no error will ever reveal.

This component puts scores on one comparable scale, so fusion has something
meaningful to combine. It is a pure function over `ScoredDocument`: no state,
no store, no I/O.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `normalize_scores(documents, *, method="min_max", higher_is_better=True)` | `-> list[ScoredDocument]` | input order preserved |
| `METHODS` | `("min_max", "rank", "z_score")` | |
| `CONSTANT_SCORE` | `0.5` | what a constant list returns |
| `NormalizationError` | base exception | catches this component |
| `NormalizationInputError` | `NormalizationError` | the caller passed something unusable |

### The three methods

| Method | Range | Best | Survives an outlier? |
|---|---|---|---|
| `min_max` | `[0, 1]` | 1.0 | No -- one far-away document flattens everything else toward 0 |
| `rank` | `[0, 1]` | 1.0 | Yes -- uses position, never magnitude |
| `z_score` | unbounded | highest | Mostly -- reported in standard deviations |

### Invariants the tests pin down

- **A constant score list does not become `NaN`.** Min-max divides by
  `max - min`. When every score is equal that denominator is zero, and a
  naive implementation returns `NaN` or raises `ZeroDivisionError` -- at the
  moment the caller most needs a sane answer, because an all-equal list means
  "nothing distinguished these". This returns `CONSTANT_SCORE` (0.5), which
  reads as *no signal either way*.
- **`higher_is_better=False` inverts direction once, here.** Cosine distance
  and BM25 disagree about which end is better, and `ScoredDocument.score`
  carries no direction of its own. Under every method the *best* document
  ends up with the highest normalized value, so a caller never has to
  remember which way a source points.
- **Tied scores share an average rank.** Under `rank`, two equally scored
  documents get *equal* normalized values. Assigning them different ranks
  would invent a distinction the retriever did not make.
- **Ordering of the input is preserved.** Only the scores change. The
  documents are replaced rather than mutated because `ScoredDocument` is
  frozen, so a caller can keep the originals and compare.
- **A non-finite incoming score is rejected** before it can poison every
  other document's result.
- **An empty list returns an empty list.** "No results" is an expected
  condition in a pipeline, not a caller error.
- **An unknown method raises and lists the valid ones**, rather than falling
  back to a default the caller did not choose.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it -- the manifest declares that
component a dependency, so `ragkit add` copies it along.

## Dependencies

- **Components:** `core-types` — `ScoredDocument` and its frozen-model
  semantics. Copy it alongside.
- **Packages:** standard library only (`math`).
- **External services:** none.

## Trade-offs

**Use it when** combining results from two retrievers with different score
scales, or when you want a rank-only ordering that an outlier cannot distort.

**It costs** one pass over the list per call, plus a copy of each document.

**Alternatives:** `method="rank"` when you care only about order and want to
avoid scale entirely (or let `hybrid-retrieval`'s RRF do it for you).

**When not to use it:** a single-source result list. Normalizing one
retriever's scores against themselves tells you nothing -- it is a no-op that
costs a copy.

## Known limitations

**`min_max` is extremely sensitive to outliers.** One document scoring
1000.0 next to others scoring 0.1 and 0.2 leaves the latter two both below
0.001, indistinguishable after normalization. `rank` exists for exactly this.

**`z_score` is not bounded**, so its output cannot be fed to
`hybrid-retrieval`'s `weighted` method, which checks for `[0, 1]`. Use
`min_max` or `rank` before weighted fusion.

**Normalization is computed per call over the list you pass.** There is no
corpus-level or running statistic, so the same document normalized in two
different result sets can get two different values. That is inherent to
normalizing a *result set* rather than a model.

**`rank` with many ties is coarse.** Ten documents where nine share a score
and one is better gives the nine an identical value, so downstream tie-breaking
falls to whatever sorts them next.

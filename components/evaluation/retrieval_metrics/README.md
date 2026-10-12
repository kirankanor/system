# retrieval-metrics: three numbers that fail in different directions

```
retrieval_metrics.py  RetrievalMetrics, RetrievalScore, QueryScore, unique_in_order()
example.py            runnable: hand-computed values, degenerate k, diagnosis
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; NO packages; NO dependencies
```

## Purpose

Score a retriever against labelled examples: **hit rate@k**, **MRR@k** and
**recall@k**, over one query or a whole corpus.

## Why three metrics

They fail in different directions, and a single number hides which one broke.

| metric | cares about | can sit at 1.0 while... |
|---|---|---|
| **hit rate@k** | *did anything land?* | the right chunk arrived at rank 9 |
| **MRR@k** | *did it land first?* | recall is fine but ordering regressed |
| **recall@k** | *did we cover all answers?* | one query missed entirely |

A change that raises `k` moves hit rate and recall up and says nothing at all
about ordering. A reranker that puts the right chunk first moves MRR and may
not move recall. Reporting one of the three is how a retrieval change looks
like an improvement in one review and a regression in the next — which is
exactly the failure mode `evaluation-dataset` exists to prevent, and it
refuses to compare a baseline whose metrics your new run stopped reporting.

## Definitions it commits to

- **Rank is 1-based.** The first retrieved item is rank 1. A reciprocal rank
  of `1/0` is not a number, and the alternative convention only invites
  off-by-one errors into every caller.
- **Duplicates occupy one slot.** `["c1", "c1", "c2"]` is treated as
  `["c1", "c2"]`, so a retriever cannot inflate its own recall by returning
  the same chunk twice. The metric would otherwise be rewarding a defect.
- **A duplicated label is a set.** `relevant=["c1", "c1"]` is one target, so
  recall cannot exceed 1.0.
- **`k` past the end of the list is clamped, not an error.** Asking for
  `recall@100` against a 10-result list is a legitimate question with a
  defined answer; refusing it would make every caller pre-clamp by hand.
- **An empty retrieval with a non-empty label is a miss**, scoring 0 on all
  three. That is a real outcome, not a mistake.
- **An empty label is a caller bug**, and it raises. Nothing can be correct,
  every metric would be 0, and the result would silently look like a broken
  retriever rather than an unlabelled query.
- **`k < 1` raises.** A cutoff of 0 inspects no results, so every metric is 0
  and the score says nothing about the retriever.

### Degenerate `k`, in one table

| `k` | behaviour | why |
|---|---|---|
| `0`, `-1` | `RetrievalMetricsInputError` | meaningless cutoff, silently zero |
| `True`, `1.5`, `"10"` | `RetrievalMetricsInputError` | `bool` is not a cutoff; neither is a float |
| `1` | first result only | the strictest meaningful question |
| `100` against 10 results | clamped to 10 | a real question, already answered |

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `RetrievalMetrics(*, k=10, name=...)` | | one instance per cutoff |
| `.score(retrieved=, relevant=)` | `-> QueryScore` | one query |
| `.score_many(examples)` | `-> RetrievalScore` | corpus means |
| `RetrievalScore` | frozen dataclass | `.hit_rate`, `.mrr`, `.recall`, `.k`, `.examples`, `.per_query` |
| `RetrievalScore.as_metrics()` | `-> dict[str, float]` | feeds `EvaluationDataset.assert_no_regression` |
| `RetrievalScore.missed_queries` | `-> tuple[QueryScore, ...]` | which queries found nothing |
| `QueryScore` | frozen dataclass | `.hit`, `.reciprocal_rank`, `.recall`, `.rank`, `.missed` |
| `unique_in_order(items)` | `-> tuple[str, ...]` | the de-duplication rule, exported |
| `RetrievalMetricsError` | base exception | catches this component |
| `RetrievalMetricsInputError` | `RetrievalMetricsError` | the caller passed something unusable |

### Invariants the tests pin down

- **Every value matches a hand-computed one.** The gate corpus is three
  queries worked out on paper — `2/3`, `(0.5+0+0.5)/3`, `(1+0+0.5)/3` — and
  compared exactly, not approximately.
- **`as_metrics()` returns exactly `hit_rate`, `mrr`, `recall`.** The names
  are the contract with `evaluation-dataset`; a rename would silently stop a
  regression check from seeing a metric.
- **`score_many` accepts pairs *and* dicts**, because that is the shape a
  JSON dataset round-trips to. Loading a dataset and scoring it needs no
  adapter.
- **Meaning does not depend on example order.**

## Example

```bash
python example.py
```

Standalone — this component has no dependencies at all.

## Dependencies

- **Components:** **none.**
- **Packages:** **none**, optional or required.
- **External services:** **none.**

## Usage

```python
from retrieval_metrics import RetrievalMetrics

metrics = RetrievalMetrics(k=10)          # the cutoff your app actually shows
score = metrics.score_many([
    (["doc3", "doc1"], ["doc1"]),
    (["doc7", "doc8"], ["doc1"]),
])

print(score.hit_rate, score.mrr, score.recall)
for query in score.missed_queries:        # open these first
    print("missed:", query)

baseline = score.as_metrics()             # keep this from the last good run
```

Pick `k` as the number of results your application shows a user, not as a
round number. A chat answer that surfaces four passages is evaluated at 4; a
search page showing twenty is evaluated at 20. Evaluating at 10 when the app
shows 4 is measuring a system that does not exist.

## Trade-offs

**Use it** after every retrieval change, over a labelled set you trust.

**It costs** one pass per query, no I/O, no model, no network. The inputs are
already-materialised lists of ids, which is what makes the numbers
reproducible and the component safe to run in CI.

**When not to use it:** as a *relevance* judgement. It measures whether the
right chunk was retrieved, never whether it was any good. For the quality of
the text that reached the model, see `answer-metrics`; for the quality of the
answer that came back, `answer-metrics` again.

## Known limitations

**No graded relevance.** A chunk that is *partly* right scores the same as
one that is exactly right: `relevant` is a set of ids, and an id is either a
target or it is not. Graded judgements need a different metric (nDCG) and a
different label format.

**No averaging over `k`.** Each instance scores at one cutoff. Report several
by constructing several instances — a mean over `k` hides where the
retriever's quality actually falls off.

**Rank is taken from input position, never recomputed.** The component trusts
the order it is handed. If a caller passes an unsorted list it scores the
list they passed, which is correct but not what they meant — sorting is the
caller's decision, because only the caller knows the score's direction.

**Zero-label coverage is not detected.** If your corpus labels a query with
ids your retriever could never return (a stale index, a re-keyed corpus),
recall falls and the metric correctly reports that. It cannot tell you the
*label* was wrong; that is what `scripts/` verification is for.

**No confidence intervals.** On a 20-query set, a one-query difference moves
hit rate by 0.05. The means are exact arithmetic over what you supplied and
carry no claim about significance.

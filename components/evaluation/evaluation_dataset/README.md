# evaluation-dataset: labelled cases, and the assertion that nothing regressed

```
evaluation_dataset.py  EvaluationDataset, EvaluationCase, RegressionError, Regression
example.py             runnable: a baseline, a regression, and the failure it raises
tests/                 success paths and deliberate failure paths
manifest.json          schema-valid; NO packages; NO dependencies
```

## Purpose

Two things, both named in FR-9: **a dataset format** for labelled evaluation
cases, and **regression-test support** that fails when a change makes a metric
worse.

## Where the numbers come from

This component computes **no metrics**. It holds labels and compares scores;
the scoring belongs to `retrieval-metrics` and `answer-metrics`.

That is not an omission. All three manifests declare `dependencies: []`, so
each component copies out alone — and if this one imported the metric
components, adding a dataset to a project would drag two metric
implementations in with it whether or not the caller wanted them. Wiring is
three lines:

```python
from evaluation_dataset import EvaluationDataset
from retrieval_metrics import RetrievalMetrics

dataset = EvaluationDataset.from_json(Path("eval.json").read_text())
baseline = RetrievalMetrics(k=10).score_many(dataset.retrieval_examples()).as_metrics()

# ... after your change ...
current = RetrievalMetrics(k=10).score_many(new_run).as_metrics()
dataset.assert_no_regression(current=current, baseline=baseline)
```

## The regression rule

A regression is a metric that got **worse by more than `tolerance`**.
`tolerance` is absolute and in metric units, so `0.01` on `hit_rate` means
"a one-point drop is noise".

Three properties make it trustworthy rather than merely convenient:

1. **Every baseline metric must still be reported.** If the baseline had
   `hit_rate`, `mrr` and `recall` and the new run reports only `hit_rate`,
   this *raises* rather than comparing the one metric that happened to
   survive. Comparing a subset is precisely how a real regression hides:
   hit rate can hold steady while MRR collapses, and a subset comparison
   calls it a pass.

2. **Metrics absent from the baseline are not regressions.** A new metric has
   no history to regress from, so it is reported but never fails the check.

3. **Lower-is-better metrics are opt-in** via `lower_is_better=`. `latency`
   and `loss` improve as they fall, so they must be named — otherwise the
   check would fail every improvement.

A regression raises `RegressionError` carrying **every** offending metric,
not just the first, so one run reports the whole story instead of a
whack-a-mole sequence of fixes. Each `Regression` exposes `.baseline`,
`.current`, `.delta` and `.worsening` (how far past the tolerance it moved),
so a report can say *"hit rate 0.80 → 0.70, past tolerance by 0.10"* rather
than "something changed".

## Why there is no `to_file`

The component returns and accepts JSON **strings**, and where they are
written is the caller's decision. A library that writes files on request has
to answer *where it is allowed to write* — and that question belongs to the
application, not to a metrics helper. Keeping the I/O out also keeps this
component clear of the path-safety invariant that governs `ragkit`'s own
writes.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `EvaluationCase(id, query, retrieved=(), relevant=(), answer=None, evidence=())` | frozen | one labelled example |
| `EvaluationCase.has_retrieval` / `.has_answer` | `-> bool` | which metrics a case can feed |
| `EvaluationCase.to_dict()` / `.from_dict()` | | the JSON shape |
| `EvaluationDataset(cases=, name=)` | frozen | ordered, de-duplicated by id |
| `.get(case_id)` | `-> EvaluationCase` | raises naming what is available |
| `.retrieval_examples()` | `-> list[tuple]` | ready for `RetrievalMetrics.score_many` |
| `.assert_no_regression(*, current, baseline, tolerance=0.0, lower_is_better=())` | `-> tuple` | empty tuple when clean |
| `.to_json()` / `.from_json(text)` | `-> str` / `-> EvaluationDataset` | versioned format |
| `.to_dict()` / `.from_dict()` | | |
| `FORMAT_VERSION` | `1` | bumped only on an incompatible shape change |
| `EvaluationDatasetError` | base exception | catches this component |
| `EvaluationInputError` | `EvaluationDatasetError` | malformed data or configuration |
| `RegressionError` | `EvaluationDatasetError` | carries `.regressions` |

### Invariants the tests pin down

- **JSON round-trips exactly**: `from_json(to_json(d)).to_dict() == d.to_dict()`.
- **A known-worse result fails the assertion** — the headline property. A
  baseline of `{0.80, 0.60, 0.70}` against a run of `{0.70, 0.60, 0.70}`
  raises, naming `hit_rate`.
- **An improvement and an exact tie both pass.**
- **A missing baseline metric is refused**, never silently skipped.
- **Duplicate case ids are refused** — a failure must name one example.
- **An unknown `version` is refused**, not guessed at.
- **An empty dataset is refused** — a baseline over nothing is `0.0`, which
  would read as a measurement rather than an absence.

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
from evaluation_dataset import EvaluationCase, EvaluationDataset

dataset = EvaluationDataset(
    cases=(
        EvaluationCase(id="q1", query="how to parse", retrieved=("a", "b"), relevant=("a",)),
        EvaluationCase(id="q2", query="why json", retrieved=("x",), relevant=("b",)),
    ),
    name="search-v1",
)

baseline = {"hit_rate": 0.80, "mrr": 0.60, "recall": 0.70}
current = {"hit_rate": 0.75, "mrr": 0.62, "recall": 0.70}

dataset.assert_no_regression(current=current, baseline=baseline, tolerance=0.05)
```

Store `baseline` as JSON beside the code that produced it. A baseline you
cannot find is a baseline you cannot regress against, and the value of this
component is entirely in making that comparison happen on every change.

## Trade-offs

**Use it** in a test, once per change, against a checked-in baseline.

**It costs** a dict comparison. Everything else is the caller's scoring step.

**When not to use it:** as a *test-data* format or a *benchmark* harness. It
holds labelled cases and compares scores; it does not shuffle, split,
bootstrap or estimate significance. For measuring a single run's quality,
score it with `retrieval-metrics` directly.

## Known limitations

**No statistical significance.** A drop of one query on a 20-query set is
noise, and this component cannot tell noise from signal — `tolerance` is how
you say so, and choosing it is your judgement, not its. Setting a tolerance
loose enough to hide a real regression is the failure mode; there is no
default that is right for every corpus, which is why the default is `0.0`.

**Baselines are absolute numbers, not recomputed from the dataset.** The
dataset stores *labels*, not the retriever's output at baseline time. Store
the scored metrics from your last good run alongside it.

**The dataset does not validate label quality.** A stale label — ids a
retriever could never return — lowers recall and the component correctly
reports the drop. It cannot tell you the label was wrong.

**One flat namespace for cases.** Cases are identified by a string `id` and
must be unique across the whole dataset. There is no grouping by suite or
domain; split those into separate datasets.

**`retrieval_examples()` drops answer-only cases**, because a case with no
`relevant` label has nothing to retrieve against. Score those with
`answer-metrics` over `case.answer` and `case.evidence` instead.

# answer-metrics: score groundedness and relevance from text you supply

```
answer_metrics.py   AnswerMetrics, AnswerScore, tokenize()
example.py          runnable: the unseen-term rule, no stemming, evidence that moves the score
tests/              success paths and deliberate failure paths
manifest.json       schema-valid; NO packages; NO dependencies
```

## Purpose

Two numbers in `[0, 1]` over one answer:

- **groundedness** — *does the evidence support this answer?*
- **answer relevance** — *does the answer address the question?*

Both computed **only from the text you pass in**. No model, no lookup, no
network.

## Why "only from supplied evidence" is the point

A groundedness score that consulted an external model would be a second
model's opinion about the first. It would change when the vendor changed
their weights, so two runs a week apart would disagree for reasons that have
nothing to do with your change — which is the opposite of what a regression
test needs.

Keeping it to string comparison makes the number **reproducible, auditable
and attributable**: the score moves only when the answer or the evidence
moved. The tests assert this directly — the same answer scored against
different evidence gives different numbers, and scoring the same pair twice
never differs.

## Why IDF-weighted, not plain overlap

`the, a, of` appear in almost every answer and in almost every passage.
Counting terms equally means a fluent paragraph of filler scores highly
grounded, because its stopwords are all present in the evidence.

Inverse document frequency fixes this: a term that appears everywhere carries
little weight, while a term that appears in one passage carries a lot. An
answer made of distinctive terms the evidence contains scores high; an answer
padded with generic words scores low, where it belongs.

## The unseen-term rule — the whole ballgame

A term the evidence never mentions takes the **maximum** available IDF, not
zero.

That looks like a detail and is not. Give it zero weight and a fabricated
claim scores *perfectly grounded*, because every invented term contributes
`0/0` — the sum vanishes, the ratio reads `1.0`, and the most confabulated
answer in the batch wins the comparison.

Weighting unseen terms at the top instead means a claim built from vocabulary
the evidence has never heard of scores at or near 0. The gate asserts this
direction explicitly: an answer about *quantum tunnelling* scored against a
passage about `json` comes back at `0.0`, while the same passage quoted
exactly comes back at `1.0`.

```
groundedness = Σ idf(t) for answer terms t present in evidence
             ÷ Σ idf(t) for all answer terms t
```

With one passage in evidence the two weights are closed form —
`idf(1,1) = log(4/3)` for a term present, `idf(1,0) = log(4)` for one absent
— so "one of two terms is grounded" is `log(4/3) / (log(4/3) + log(4)) =
0.1718555…` and the tests assert exactly that, not an approximate guess.

## How this differs from `grounded-answer-contract`

They overlap on purpose and serve different jobs:

| | `grounded-answer-contract` | `answer-metrics` |
|---|---|---|
| answers | *does this pass?* | *how well did it do?* |
| output | a boolean gate | a continuous score |
| use | block a bad answer in a pipeline | compare two versions of a pipeline |

A gate cannot tell you that v2 is 4% better than v1 — it returns the same
`True` for both. A score cannot tell a caller to reject this answer right now
— `0.62` is not a decision. Each component stands alone and defines its own
error hierarchy, because a real project installs the one it needs.

There is deliberately **no combined single number**. Groundedness and
relevance fail in different directions: averaging them would let a perfectly
faithful answer that ignores the question look identical to an irrelevant
answer copied from the evidence — exactly the hiding place a regression looks
for. `AnswerScore.as_metrics()` returns both, separately.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `AnswerMetrics(*, stopwords=DEFAULT_STOPWORDS, name=...)` | | construct once, reuse |
| `.groundedness(answer, evidence)` | `-> float` | `evidence` is a sequence of passages |
| `.relevance(answer, query)` | `-> float` | |
| `.score(answer=, query=, evidence=)` | `-> AnswerScore` | both in one call |
| `AnswerScore` | frozen dataclass | `.groundedness`, `.relevance`, `.answer_terms`, `.as_metrics()` |
| `tokenize(text, *, stopwords=...)` | `-> frozenset[str]` | the exact tokenizer, exported |
| `AnswerMetricsError` | base exception | catches this component |
| `AnswerMetricsInputError` | `AnswerMetricsError` | the caller passed something unusable |
| `DEFAULT_STOPWORDS` | `frozenset[str]` | a small closed English list |

### Invariants the tests pin down

- **`groundedness` answers that duplicate the evidence are `1.0`; answers
  sharing nothing with it are `0.0`.** Both endpoints, exactly.
- **An empty answer scores `0.0` on both**, and does not raise. An evaluation
  must be able to score a model that produced nothing; crashing the run on
  one empty output would be worse than the score.
- **The score depends only on the supplied text.** Swapping the evidence
  changes the answer's score; nothing else can.
- **No stemming.** `exception` does not match `exceptions`, `run` does not
  match `running`. A test pins both.
- **A `bool` is rejected for `stopwords`** — as is a list or a set.

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
from answer_metrics import AnswerMetrics

metrics = AnswerMetrics()
score = metrics.score(
    answer="The json module serialises objects into JSON text.",
    query="how does json serialise objects",
    evidence=retrieved_passages,        # the exact text the model saw
)

print(score.groundedness, score.relevance)
print(score.as_metrics())               # feeds EvaluationDataset.assert_no_regression
```

Pass the **passages as they were given to the model**, not document ids. The
point is that the terms are physically present in what the answer could have
seen; scoring against an id would measure nothing.

## Trade-offs

**Use it** once per (answer, query, evidence) triple, over a labelled set.

**It costs** one tokenization per string plus a linear pass. No I/O, no
model, no network — which is the property that makes the number trustworthy
in CI.

**When not to use it:** as a *factual verification* check. It establishes that
the answer's vocabulary is present in the evidence, not that the evidence
itself is true or that the answer drew the right inference. For a pass/fail
gate on an answer before it reaches a user, see `grounded-answer-contract`,
which fails loudly rather than scoring softly.

## Known limitations

**No stemming, no lemmatising, no fuzzy matching.** `exception` is not
`exceptions` and `run` is not `running`. A deliberate gap: it means the score
never claims a match the text does not contain, and anyone who wants stemming
can stem before calling. The cost is real and visible — an answer that says
"serialised" for a query asking about "serialise" loses that term entirely,
which the tests demonstrate rather than hide.

**`relevance` is a lexical proxy, not a semantic one.** It asks whether the
answer uses the query's vocabulary. An answer that addresses the question in
entirely different words scores low. The LLM-based alternatives (generate
questions the answer could answer; embed and compare) require a model, which
would forfeit the reproducibility this component is built on.

**IDF is computed over what you pass, not over a corpus.** With one passage
of evidence every present term shares the same weight, so "how distinctive"
within a single passage cannot be expressed. Pass more passages and the
weighting starts to discriminate.

**An answer of pure stopwords scores `0.0`, not `1.0`.** Vacuously grounded
is not the same as grounded, and `1.0` would let an empty string top a
leaderboard.

**The stopword list is a closed English list, and deliberately not a data
file.** A component whose score depends on a downloaded list is not
reproducible. The error it introduces is bounded and uniform: a few common
words weigh slightly more than they should, in exactly the same way, in
every run.

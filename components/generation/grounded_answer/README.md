# grounded-answer-contract: catch the confident, fluent, fabricated answer

```
grounded_answer.py  GroundednessContract, Claim, GroundednessResult, tokenize
example.py          runnable: supported claims, fabrication caught, thresholds
tests/              success paths and deliberate failure paths
manifest.json       schema-valid; NO packages; core-types dependency
```

## Purpose

Reject an answer whose claims the supplied evidence does not support.

## The failure this exists to catch

A retrieval pipeline's most damaging failure is not returning nothing. It is
returning a fluent, confident, **fabricated** answer.

The model fills a gap with prior knowledge, and because the surrounding
sentences are well-grounded the whole answer reads as sourced. Nobody notices
until the invented detail is acted on. The cost is not a bad score on a
benchmark — it is a decision made on something nobody ever wrote down.

This component is a guard against that specific failure. It is deliberately
**not** a quality judge.

## How it decides, stated plainly

The answer is split into **claims** — one per sentence, because a sentence is
the unit a reader would attribute to a source. Each claim is scored for how
much of its distinctive vocabulary the evidence actually contains:

```
support(c) = SUM of idf(t) for claim terms t present in the evidence
             -----------------------------------------------
             SUM of idf(t) for all claim terms t

idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

IDF is computed over the **evidence set**, so a term the evidence never
mentions has `df = 0` and therefore carries the **maximum** possible weight —
and it lands in the denominator alone.

That is the intended behaviour, and it is the crux of the component. An
invented identifier, model name, version number or figure is the most
distinctive thing in the sentence, and it is weighted accordingly. A claim
whose support falls below `min_support` is unsupported, and a single
unsupported claim fails the contract.

**An entirely invented sentence scores `0.0`.** Not near zero — zero, because
not one of its terms is in evidence.

## This is lexical, and that is a limit worth stating

Support is measured in shared vocabulary, not meaning.

- A faithful paraphrase scores **lower** than a near-quote.
- A claim that is true but absent from the evidence scores **zero** — which is
  correct from this component's point of view: it was not grounded in what the
  caller supplied.
- `min_support=1.0` will fail prose that paraphrases, because paraphrase is
  exactly where vocabulary stops overlapping.

So the result is a **floor, not a verdict**. Passing does not mean the answer
is good. Failing means *some sentence is not backed by the evidence in front
of it*, which is worth a human look before the answer is trusted.

No model, no network (`FR-10`). This is arithmetic.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `GroundednessContract(*, min_support=0.5, stopwords=DEFAULT_STOPWORDS, name=...)` | | |
| `.check(answer, context)` | `-> GroundednessResult` | raises only on unusable input |
| `.is_grounded(answer, context)` | `-> bool` | the boolean form |
| `Claim` | frozen dataclass | `text`, `support`, `supported` |
| `GroundednessResult` | frozen dataclass | `.grounded`, `.claims`, `.unsupported`, `.report()` |
| `tokenize(text, *, stopwords=...)` | `-> frozenset[str]` | public: tokenization *is* the vocabulary |
| `GroundingError` | base exception | catches this component |
| `GroundingInputError` | `GroundingError` | the caller passed something unusable |

### Invariants the tests pin down

- **A fabricated claim scores exactly `0.0`** and is named in `.unsupported`.
- **One bad claim fails the whole answer.** `.grounded` requires every claim.
- **Support stays in `[0, 1]`.**
- **A `0.0`-support claim fails at any positive threshold.** At `min_support=0.0`
  the check is disabled — see the threshold table; that is an opt-out, not a
  strictest setting.
- **Identical passages count twice in IDF.** Two passages with the same
  vocabulary are still two passages; collapsing them would corrupt the
  document frequency this score rests on.
- **An empty answer is rejected**, not passed vacuously — an answer making no
  claims would trivially "pass" and tell the caller nothing.

## Choosing a threshold

`min_support` is the fraction of a claim's distinctive vocabulary that must
appear in the evidence.

| Value | Behaviour |
|---|---|
| `0.0` | **disables the check** — every claim passes, including one scoring `0.0` |
| `0.01` | catches only total fabrication: any shared vocabulary passes |
| `0.5` (default) | tolerates connective phrasing; fails a sentence built around one invented term |
| `1.0` | requires near-exact vocabulary overlap; will fail paraphrase |

The predicate is a plain threshold, `support >= min_support`, with no special
case for zero. That makes `min_support` mean exactly what it says — and it
means `0.0` is an explicit opt-out, not a strictest-possible setting. Want to
catch only complete fabrication? Set `0.01`, not `0.0`.

Lower it for terse, technical answers where almost every word is load-bearing.
Raise it for prose. There is no universally right value, which is why it is a
constructor argument and not a constant.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `Context`, `ScoredDocument`.
- **Packages:** **none**, optional or required.
- **External services:** **none.**

## Network-free by design

`FR-10` and `DECISION-e907e1a6`: v1 generation ships a prompt builder and an
answer contract, **no LLM clients**. A test asserts neither module imports
`openai`, `anthropic`, `httpx` or any transport. An evaluator that needs a
model to judge a model is a different, larger product; this is a floor you can
compute on any machine, offline, in microseconds.

## Trade-offs

**Use it** as a final gate before an answer reaches a person — especially
where a wrong answer is worse than a refusal.

**It costs** one tokenisation of the evidence and one of the answer.

**When not to use it:** as a measure of answer *quality*. It measures
grounding in supplied text, and nothing else. A high score means "backed by
the evidence", not "correct" or "well written".

## Known limitations

**Paraphrase is penalised.** See above; this is the honest cost of being
dependency-free. A semantic judge would do better and would need a model.

**Sentence splitting is naive.** `re.split(r"(?<=[.!?])\s+")` — so
`"Dr. Smith"` splits, and so does any abbreviation followed by a space. A
correct splitter needs an abbreviation list, and this component does not carry
one. It affects claim *boundaries*, not the per-claim arithmetic.

**Stopwords are a fixed English set.** Pass `stopwords=` to override; there is
no language detection. For non-English answers the default set does little,
which makes the score noisier rather than wrong.

**No stemming.** `exception` and `exceptions` are different terms, the same
posture as `sparse-retrieval` and `relevance-ranking`. A paraphrase that
inflects differently costs support, on top of the paraphrase cost already.

**No citation-span validation.** `[3]` with two sources is `citation-formatter`'s
job, not this one. The two are meant to run alongside: resolve the spans, then
score the claims.

# token-budget: measure, don't guess

```
token_budget.py   TokenBudget, word_count(), the error vocabulary
example.py        runnable: counting, truncation, packing, the guarded wheel
tests/            success paths and deliberate failure paths
manifest.json     schema-valid; tiktoken OPTIONAL; no component dependencies
```

## Purpose

Answer "how much context fits in this model's window?" correctly, and cut text
to fit when it does not.

## Why a guess is not good enough

`len(text) // 4` is the estimate that appears in a great many RAG pipelines.
It is fine for English prose and wrong nearly everywhere else — code, tables,
JSON, and most non-English text all tokenize far more densely — and the error
is not symmetric. It overshoots, which means the model truncates the **far
end** of the context, which is where the answer usually is. A budget that is
wrong by 20% does not degrade quality evenly; it deletes the evidence you
retrieved.

So this component does not guess, and does not ship a guess. It counts with a
real tokenizer or with one you supply.

## The counter is yours

```
TokenBudget()                          # tiktoken, cl100k_base, lazily loaded
TokenBudget(encoding="p50k_base")      # a different tiktoken encoding
TokenBudget(counter=my_tokenizer)      # anything callable: str -> int
```

`counter` wins when given, and then tiktoken is never imported. That makes
this usable with HuggingFace tokenizers, a remote counting service, or a word
count in tests — and it is why the missing-wheel path is testable at all.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `TokenBudget(*, encoding="cl100k_base", counter=None, name="token-budget")` | | loads lazily |
| `.count(text)` | `-> int` | also a valid `counter` for another `TokenBudget` or a `ContextSelector` |
| `.fits(text, budget)` | `-> bool` | |
| `.remaining(text, budget)` | `-> int` | **negative when already over** |
| `.truncate(text, budget, *, at_word_boundary=True)` | `-> str` | never returns more than asked for |
| `.pack(texts, budget, *, separator="\n\n")` | `-> list[int]` | indices of the items that fit |
| `.is_loaded()` | `-> bool` | inspect the cost before paying it |
| `word_count(text)` | `-> int` | a **named**, honest approximation — explicitly chosen, never inherited |
| `TokenBudgetError` | base exception | catches this component |
| `TokenBudgetInputError` | `TokenBudgetError` | the caller passed something unusable |
| `MissingDependency` | `TokenBudgetError, ImportError` | no wheel and no `counter` |

### Invariants the tests pin down

- **`truncate` and `pack` never exceed the budget.** Enforced by measuring
  the result, not by predicting it.
- **The separator counts.** Passages that fit plus joins that do not is the
  classic way a context overshoots.
- **`truncate` cuts at a word boundary** when one is available, because a
  context ending in `handl` costs tokens and reads badly. `at_word_boundary=False`
  opts out.
- **`truncate` may be shorter than the theoretical maximum, never longer.**
  Binary search assumes token count rises with text length; BPE merges make
  that very nearly but not strictly true. See below.
- **`remaining` can be negative.** "Already over" is a state worth reporting
  rather than clamping to zero.
- **`pack` stops at the first item that does not fit.** Skipping ahead to a
  shorter later item would reorder evidence, which is the caller's call.

## The dependency is optional, and genuinely guarded

`tiktoken` is declared under `optional_packages`, never `required_packages`,
and is imported **lazily, inside `count`** (`DECISION-bf2566ea`). Importing
this module never touches the wheel.

With no wheel and no `counter`, `count` raises `MissingDependency` — an
`ImportError` — whose message is the manifest's `missing_dependency_message`
**verbatim**.

## Example

```bash
python example.py
```

Runs standalone. It never requires the wheel: the counting path is
demonstrated with an injected counter, and the missing-dependency path is
demonstrated for real, because tiktoken is not installed here.

## Dependencies

- **Components:** **none.** This module imports nothing from the library, so
  it copies out and runs anywhere Python does.
- **Packages:** `tiktoken` `>=0.7.0`, **optional**.
- **External services:** none. tiktoken ships its vocab locally; there is no
  network call.

## Trade-offs

**Use it when** a budget must be right for a specific model, or text must be
cut to fit without ending mid-word.

**It costs** one encode per `count`. `truncate` performs `O(log n)` counts,
and `pack` one per item plus one per separator.

**When not to use it:** when you already know the text fits, in which case you
do not need to measure it.

## Known limitations

**The real tokenizer is unexecuted in this repository.** `tiktoken` is not
installed here, and installing it would modify dependency files — which this
library forbids. The gate tests prove everything that does not need it: the
module imports, loads lazily, counts via an injected counter, never exceeds a
budget, and fails with the manifest's message verbatim when the wheel is
absent. **They do not prove a real encoding's counts.** Run `example.py` on a
machine with the wheel for that.

**Truncation is character-based with a measured bound, not token-precise.**
Finding the exact longest-fitting token prefix needs the encoder's own API
surface, which differs per tokenizer; a generic `counter` cannot offer it. The
binary search returns a prefix that *fits*, verified by measurement. On
pathological input it can be shorter than optimal. It is never too long.

**`pack` is greedy and order-preserving.** It does not solve a knapsack
problem. That is deliberate: an optimal packing that reorders evidence is
worse for generation than a slightly shorter context in the right order.

**One encoding per `TokenBudget`.** Budgeting for two models means two
objects, which is the honest shape — a budget belongs to a model.

# context-selector: from a ranked list to one context that fits

```
context_selector.py  ContextSelector, the error vocabulary
example.py           runnable: dedup, budget packing, the empty context
tests/               success paths and deliberate failure paths
manifest.json        schema-valid; NO packages; core-types dependency
```

## Purpose

Implement `ContextBuilderProtocol` from core-types: turn a ranked list of
separate passages into **one string** that fits a token budget, and report
exactly which passages made it in.

Three things go wrong between retrieval and generation, and this fixes all
three:

1. **The same passage arrives twice.** Dense and sparse retrieval each return
   it; a reranker does not know they are the same. Showing a model its own
   evidence twice is not harmless — it double-counts in attention and inflates
   the impression that several independent sources agree.
2. **The list does not fit.** Something has to stop, and it should stop in
   rank order rather than squeezing in a shorter later passage.
3. **The joins push it over.** Every passage fits on its own; the separators
   between them are the last few tokens that overflow.

## It selects; it does not truncate

A passage too large to fit is **skipped** and `truncated` is set, rather than
cut mid-sentence. The model cannot tell it is looking at a fragment, and a
fragment is usually worse than nothing. Cutting text is `token-budget`'s job;
truncate first and pass the result if you want an oversized passage included.

## The counter is required, on purpose

```python
budget = TokenBudget()
ContextSelector(counter=budget.count)
```

There is no default counter. `len(text) // 4` is a plausible-looking
approximation that overshoots on code, tables and non-English text — precisely
where overshooting is most expensive, because models drop the far end of the
context. So the caller must say what a token is.

`Context.token_count` and `Context.budget` are in the counter's unit, so a
returned `Context` is self-describing about what it was measured in.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `ContextSelector(*, counter, separator="\n\n", name="context-selector")` | | `counter` required |
| `.select(query, chunks, budget)` | `-> Context` | deduplicate, then pack |
| `.build(query, chunks, budget)` | `-> Context` | `ContextBuilderProtocol`'s name for the same thing |
| `.deduplicate(chunks)` | `-> list[ScoredDocument]` | callable on its own |
| `ContextError` | base exception | catches this component |
| `ContextInputError` | `ContextError` | the caller passed something unusable |

### Invariants the tests pin down

- **`token_count` never exceeds `budget`**, for any budget, verified exhaustively
  over a range.
- **`token_count` is the real cost of the returned text** under the caller's
  counter — not an estimate of it.
- **The separator is counted.**
- **Order is the order given.** The caller already ranked these.
- **Deduplication happens before the budget is spent**, so a duplicate cannot
  consume the room real evidence needed.
- **A duplicate keeps its first occurrence** — the best-ranked copy.
- **An empty result is an empty `Context`, not an error.** Whether an empty
  context means abstain is the answer stage's decision, not this component's.

## What counts as a duplicate

Two things, and both matter:

- the `chunk_id` has already been seen;
- the **text, whitespace-collapsed**, has already been seen.

The second catches the real case: the same passage returned by two retrievers
under two different ids.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `ScoredDocument`, `Context`.
- **Packages:** **none**, optional or required. It counts with whatever
  callable you hand it.
- **External services:** none.

## Trade-offs

**Use it** after retrieval and reranking, immediately before generation.

**It costs** one dedup pass and one counter call per passage plus per
separator.

**When not to use it:** when the passages already fit and are already unique —
in which case joining them with `separator.join` is three lines and this
component is overhead. Also, when you need query-aware reordering; see below.

## Known limitations

**The query is not used.** `build` takes one because the protocol says so, and
it is validated — but selection is driven by the order retrieval already
produced. Reordering here would contradict every ranking done upstream. It is
said plainly rather than hidden behind a parameter that does nothing.

**Greedy, not optimal.** `pack` takes passages in order until one does not fit.
It does not solve a knapsack problem, deliberately: an optimal packing that
reorders evidence is worse for generation than a slightly shorter context in
the right order.

**Deduplication is exact, not semantic.** Two passages that say the same thing
in different words are both kept. Near-duplicate detection needs embeddings or
a similarity threshold, and is out of scope here.

**One counter, one unit.** Budgeting for two models means two selectors.

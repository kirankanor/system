# prompt-builder: the prompt that makes citations possible

```
prompt_builder.py  PromptBuilder, render_sources(), the error vocabulary
example.py         runnable: numbering, guarantees, rejected inputs
tests/             success paths and deliberate failure paths
manifest.json      schema-valid; NO packages; core-types dependency
```

## Purpose

Turn a query and a built `Context` into one grounded prompt, in which every
passage carries a number the model can cite.

## Why the numbering is the whole design

Between context assembly and generation there is a formatting step that is
easy to get wrong in a way that *silently* costs grounding: the passages have
to be labelled so the model can refer to them.

An unlabelled context is a wall of text. A model that cannot point at
something must quote it verbatim or paraphrase it — and paraphrasing is how an
answer drifts away from its evidence while sounding perfectly confident. The
reader gets a fluent sentence with no way to check it.

So this component exists to make `[n]` mean something:

- every passage in `context.chunks` gets **exactly one** number;
- the numbers are **sequential from 1**, with no gaps;
- they appear in the order the passages appear in `context.text`.

That is precisely the convention `citation-formatter` validates against, so a
prompt built here and an answer checked there agree **without either component
knowing about the other**. No shared registry, no ordering contract, no
coupling — just the same numbering rule, stated once in each place.

## No context, no prompt

A missing `Context` — and an empty one — are both rejected with a typed error
rather than producing a prompt.

This is the one refusal worth explaining. The instruction says *"Answer using
ONLY the numbered sources below."* With nothing below it, that instruction is
a trap: the model will answer anyway, and every word of it will be unsourced.
A caller who wants an open-ended answer wants a different prompt, and should
not reach for a grounded one to get it.

An empty `Context` most often means the retriever found nothing — which is a
signal to abstain, decided at the answer stage, not a moment to hand the model
a prompt that quietly licenses invention.

## The instruction contains no literal citation

The default instruction tells the model to cite, but never demonstrates with
`[1]` — because once the prompt is assembled, an example number reads exactly
like a real one. A test asserts the built prompt's only bracketed numbers come
from actual sources.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `PromptBuilder(*, instruction=DEFAULT_INSTRUCTION, question_label="Question", name=...)` | | `instruction` **replaces**, never appends |
| `.build(query, context)` | `-> str` | the grounded prompt |
| `.numbers(context)` | `-> list[int]` | citations available, before any model runs |
| `render_sources(chunks)` | `-> list[str]` | public: the same numbering, for a UI |
| `PromptError` | base exception | catches this component |
| `PromptInputError` | `PromptError` | the caller passed something unusable |

### Invariants the tests pin down

- **Every supplied source is cited**, and nothing beyond the last one appears.
- **`[n]` and its text share a line**, so a reader (and a validator) can
  associate them unambiguously.
- **Sources render in `context.chunks` order** — never re-sorted by score.
- **The prompt is deterministic.** Same query, same context, same string. No
  timestamps, no randomness, no ordering by score.
- **`instruction` replaces wholesale**, so a house style is not fighting a
  template.
- **A blank query is rejected.** A grounded prompt with no question gives the
  model nothing to answer.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `Context`, `ScoredDocument`.
- **Packages:** **none**, optional or required.
- **External services:** **none.** This returns a string.

## Network-free by design

`FR-10` and `DECISION-e907e1a6`: v1 generation ships a prompt builder and an
answer contract, **no LLM clients**. Nothing in this module imports `openai`,
`anthropic`, `httpx` or any transport — a test asserts it, and the absence is
the point. Sending the string to a model is the caller's business, which keeps
this library usable with any provider, local model, or none at all.

## Trade-offs

**Use it** whenever a generated answer should be traceable to its sources —
which is every time the answer will be trusted.

**It costs** one string join. No I/O, no tokenisation, no model.

**When not to use it:** when you do not want citations. A plain
`f"{instruction}\n\n{context.text}\n\n{query}"` is three lines; this component
buys you the numbering, and the numbering is what makes validation possible.

## Known limitations

**The prompt is not measured against a model's token budget.** It renders the
context it is given. What fits is `context-selector`'s decision; the instruction
itself adds a few dozen tokens on top, which the caller should leave room for.

**No chat-template formatting.** No role markers, no system/user split, no
provider-specific wrapper. Those belong to whatever calls the provider, and
encoding them here would make the output specific to one vendor.

**The instruction is a fixed default, not a tuned one.** It is explicit about
citing and abstaining because a model not told to cite will not. It has not
been benchmarked against any particular model, and callers with a house style
should replace it — wholesale, which is what `instruction=` does.

**It does not validate anything.** Rendering and checking are separate jobs:
this builds, `citation-formatter` resolves, `grounded-answer-contract` scores.
Keeping them apart is what lets any one be used without the others.

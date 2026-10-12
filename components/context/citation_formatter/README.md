# citation-formatter: a span that does not resolve is a source the model was never shown

```
citation_formatter.py  CitationFormatter, the error vocabulary
example.py             runnable: spans, orphans, bibliography, custom styles
tests/                 success paths and deliberate failure paths
manifest.json          schema-valid; NO packages; core-types dependency
```

## Purpose

Resolve `[n]` citation spans against a `Context`'s own chunks, and refuse to
render text whose spans do not resolve.

## The failure this prevents

A generator writes `[3]` because that is the number that looked right. If only
two passages were in the context, `[3]` names nothing — and a reader who
follows it finds either a dead end or, far worse, a *different* source that
happens to be third elsewhere. Citation numbers are an interface between the
model and the reader; an interface with no validation is a lie waiting to
happen.

The check is cheap and the failure it catches is not: a confidently cited
wrong source is worse than an uncited answer.

## Numbering is 1-based

`[1]` means `chunks[0]`. Nobody writes `[0]` in prose, and the off-by-one
between "first item" and "index zero" is the single most common defect in
citation code. It is decided here, once, and stated in every error message.

Pass `one_based=False` if a house style genuinely uses `[0]`.

## `format` appends nothing

A natural thing for a "formatter" to do is append a numbered bibliography. It
does not, deliberately: appending text to a `Context` makes it longer than the
budget `context-selector` just enforced, so the caller would send more than
the model has room for and never find out.

Sources come back separately from `.bibliography()`, to be rendered wherever
the application wants them — a footnote, a sidebar, a hover card — by code
that knows what those extra tokens cost. **Text is returned unchanged, so
token counts stay true.**

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `CitationFormatter(*, pattern=r"\[(\d+)\]", one_based=True, name=...)` | | exactly one capture group |
| `.validate(context)` | `-> None` | raises `CitationOrphanSpanError` |
| `.format(context)` | `-> str` | validates, then returns `context.text` **unchanged** |
| `.spans(text)` | `-> list[tuple[int, int]]` | `(number, character offset)` |
| `.references(context)` | `-> list[int]` | cited numbers, in order, with repeats |
| `.bibliography(context)` | `-> list[str]` | one label per distinct **cited** passage |
| `CitationError` | base exception | catches this component |
| `CitationInputError` | `CitationError` | the caller passed something unusable |
| `CitationOrphanSpanError` | `CitationError` | a span names a passage not in the context |

### Invariants the tests pin down

- **`format` returns the text unchanged.** Never grows a context.
- **Every orphan is reported at once**, carried on `error.spans` as
  `(number, position)` pairs — fixing one per run is a bad loop to be in.
- **The message says what *is* valid** (`[1] through [3]`), not just what
  failed.
- **Text with no spans is valid.** A model may answer without citing, and
  inventing citations to satisfy a validator would be worse.
- **`[0]` is an orphan** under 1-based numbering.
- **A bracketed non-number is ordinary text.** `[note]` and `[1.5]` are left
  alone.
- **The bibliography omits uncited passages**, and never renders a blank label
  — falling back through `source_name` → `document_id` → `chunk_id` →
  `passage n`.
- **An unreachable span does not reach the bibliography.** `bibliography()` is
  safe on text with orphans; `validate()` and `format()` are where it raises.

## Custom styles

`pattern` must contain exactly one capture group yielding the number, so a
house style is a constructor argument rather than a fork:

```python
CitationFormatter(pattern=r"\((\d+)\)")  # (1)
CitationFormatter(pattern=r"§\s*(\d+)")  # §1
```

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `Context`, `ScoredDocument`.
- **Packages:** **none**, optional or required.
- **External services:** none.

## Trade-offs

**Use it** whenever a generated answer carries citations, and before rendering
one. It is a guard, so it belongs at the boundary where text becomes visible
to a person.

**It costs** one regex scan per call. Negligible.

**When not to use it:** text with no citation convention at all — there are no
spans to resolve, so there is nothing to check.

## Known limitations

**Spans are resolved positionally.** `[3]` means "the third passage shown",
which is only meaningful while the context's `chunks` order is the order the
text presents them in. `context-selector` guarantees that; a caller assembling
a `Context` by hand must too.

**No span matching across a rewritten context.** If text is edited after
selection and passages are reordered, the numbers stop meaning what they did.
Validate after the last edit.

**The pattern is a regex, not a parser.** Nested or escaped brackets in
unusual styles may need a different `pattern`; this component will not grow a
grammar.

**Bibliography labels are flat strings.** No page ranges, no authors, no
link rendering — `ScoredDocument` carries `source_name`, `document_id` and
`chunk_id`, and this component reports the first it finds. Richer citation
styles belong in the application.

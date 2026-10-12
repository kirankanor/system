# metadata-filter: a criteria language that fails loudly

```
metadata_filter.py   filter_documents(), apply_to_chunks(), the error vocabulary
example.py           runnable: every operator, a typo, a type mismatch
tests/               success paths and deliberate failure paths
manifest.json        schema-valid; core-types as a component dependency
```

## Purpose

Filtering is where a RAG pipeline quietly loses recall. This component exists
because all three of the ways it happens end in the same place -- an empty
list that looks like a correct answer:

| What went wrong | What usually happens | What happens here |
|---|---|---|
| A typo'd operator (`"gtr"`) | Treated as "never matches" | Raises, naming the operator and listing the valid ones |
| A typo'd field (`"langauge"`) | Matches nothing, forever | Matches nothing -- but `exists` lets you ask about presence |
| Incomparable operands (`page > "three"`) | `TypeError` traceback escapes | Raises a typed error naming both types |

The first is the one worth being angry about: a filter that silently accepts
an operator it cannot evaluate turns a configuration mistake into a
production answer.

## The criteria language

Each key is a field; each value is either a literal (meaning `eq`) or a
one-key mapping naming an operator:

```python
filter_documents(docs, {"lang": "en"})                  # equality
filter_documents(docs, {"page": {"gte": 3}})            # range
filter_documents(docs, {"title": {"contains": "setup"}})# substring
filter_documents(docs, {"lang": {"in": ["en", "de"]}})  # membership
filter_documents(docs, {"title": {"startswith": "SET"}})# prefix
filter_documents(docs, {"reviewed": {"exists": True}})  # presence
```

Criteria combine with **AND**, not OR, because a caller narrowing a result
set expects every condition to hold. Want OR? Filter twice and concatenate --
an explicit `|` operator would be a second thing to get wrong.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `filter_documents(documents, criteria)` | `-> list[ScoredDocument]` | order preserved |
| `apply_to_chunks(chunks, criteria)` | `-> list[Chunk]` | same language, for pre-filtering a corpus |
| `OPERATORS` | the ten valid operator names | |
| `MetadataFilterError` | base exception | catches this component |
| `MetadataFilterInputError` | `MetadataFilterError` | the caller passed something unusable |

### How a field is resolved

A field is looked up **first among the document's own fields**
(`chunk_id`, `document_id`, `source_name`, `chunk_index`, `text`) **and then
in its `metadata`**. So a caller writes the name they see in the result, not
the storage path it happens to live under -- the same accommodation
`storage-adapters` makes for its reserved keys.

### Invariants the tests pin down

- **A document missing the field does not match**, rather than raising. Not
  carrying a field is legal; `exists` is how you ask about presence.
- **Ordering is preserved.** A caller's ranking survives the filter.
- **`in` requires a sequence**, so `{"lang": {"in": "en"}}` is rejected --
  otherwise it would mean per-character membership, matching any document
  whose language is `e`, `n`.
- **`contains` and `startswith` operate on text only.** Metadata values are
  scalars by contract (`core-types` types them as text, number or bool), so
  membership in a *list* is not expressible -- use `in`.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `ScoredDocument` and `Chunk`.
- **Packages:** standard library only.
- **External services:** none.

## Trade-offs

**Use it when** narrowing retrieval results before context building, or
pre-narrowing a corpus before indexing.

**It costs** one pass over the list, and one criterion evaluation per
document per field. Criteria are resolved once, outside the loop.

**When not to use it:** filtering *inside* a vector store. Pass the flat
filter through to `VectorStoreQuery` instead -- pushing the narrowing down to
the backend means it never leaves the store.

## Known limitations

**No nested criteria.** You cannot express `(a AND b) OR (c AND d)`. The
criteria language is a conjunction; expressible queries are the ones a flat
backend filter can also represent, which is deliberate.

**`ne` on a missing field does not match.** A document without the field
fails every value operator including `ne`. Use `{"field": {"exists": False}}`
to mean "absent", which is unambiguous.

**Comparisons are Python's.** `"10" < "9"` is `True` for text, because
lexicographic ordering is what strings do. The component rejects *mixed*
type comparisons, but cannot tell you that comparing two strings numerically
was probably a mistake.

**No regex.** `contains` is a substring test. A pattern-matching filter
invites catastrophic backtracking on user-supplied input, and this library
does not run user-supplied regexps anywhere.

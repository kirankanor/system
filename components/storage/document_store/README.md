# document-store: the contracts a document backend owes you

```
document_store.py   errors + input checks + the re-exported protocol
example.py          runnable: valid documents, flat filters, the None rule
tests/              success paths and deliberate failure paths
manifest.json       schema-valid; core-types as a component dependency
```

## Purpose

`core-types` declares `DocumentStoreProtocol` — `put`, `get`, `list` — and
`SourceDocument`. What it cannot declare is the *checking* those methods owe
their callers, and one behaviour worth stating out loud. Both are here.

**The behaviour is the important part.** `get` returns `None` for an unknown
id. A missing document is an *expected* condition in a RAG pipeline — a chunk
referring to a document that was never ingested, or was deleted since — so
raising forces every caller to wrap a plain lookup in `try`/`except`.
Returning `None` puts the decision where it belongs: at the call site, where
the caller knows whether a gap is fatal.

Alongside that sit the error vocabulary (`DocumentStoreError` and its two
subclasses) and the input checks (`validate_filter`, `validate_ids`,
`validate_source_document`), so a backend implementation can be written
against this file alone.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `DocumentStoreProtocol` | re-export of `core_types.DocumentStoreProtocol` | satisfied structurally |
| `DocumentStoreError` | base exception | catches this component's failures |
| `DocumentStoreInputError` | `DocumentStoreError` | the caller passed something unusable |
| `DocumentStoreBackendError` | `DocumentStoreError` | the backend failed |
| `validate_filter(filter)` | `-> dict \| None` | rejects nested values |
| `validate_ids(ids)` | `-> list[str]` | rejects a bare `str` and empty ids |
| `validate_source_document(doc)` | `-> SourceDocument` | re-runs the model's own rules |

### Invariants the tests pin down

- **Filters are flat, and nested values are rejected rather than dropped.**
  A backend that silently dropped a nested value would return a *narrower*
  result set than the caller asked for, with no signal that anything was
  ignored.
- **`validate_source_document` re-runs the model's own validation** rather
  than duplicating the field rules, so the rules live in one place and this
  component cannot drift from them. The wrapper adds the component's name to
  the message, so a caller reads `document-store: ...` rather than a pydantic
  traceback.
- **`validate_ids` rejects a bare `str`.** A `str` *is* a `Sequence[str]`, so
  accepting it would mean `["a", "b", "c"]` for `"abc"`.
- **Errors form a hierarchy** catchable under `DocumentStoreError`, and are
  deliberately *not* `ValueError`, so a caller can tell this component's
  failures from its own programming errors.

## Example

See `example.py`, which runs standalone — from this directory, with
`core_types.py` beside it:

```bash
python example.py
```

## Dependencies

- **Components:** `core-types` — `SourceDocument`, `Metadata` and the
  protocol. Copy it alongside (the manifest declares it, so `ragkit add`
  brings it in).
- **Packages:** standard library only. `core-types` imports `pydantic`, which
  the library's runtime requires anyway.
- **External services:** none. This component never touches a backend.

## Trade-offs

**Use it when** writing or reviewing a document-store implementation. Call the
validators at the top of each method and return `None` from `get` on a miss.

**It costs** nothing at runtime beyond a few `isinstance` checks and one
re-validation per write.

**Why the errors are not imported from somewhere central:** a contract
component has to be copy-ready on its own, and importing an error hierarchy
from elsewhere means copying *that* file too. The names deliberately mirror
`vector-store-protocol`'s so the duplication is familiar rather than
surprising.

## Known limitations

**This component defines no store.** It is the contract and the checks; a
concrete backend lives in `storage-adapters` or in your own code.

**`validate_filter` allows any scalar value**, including one a backend may not
support. Type portability is the backend's concern; this checks shape.

**No paging or ordering guarantee in `list`.** Ordering is the backend's to
define, and a caller needing a stable order should sort by `document_id`.

**`validate_source_document` re-validates on every call**, which is a
`model_dump` plus a `model_validate` per write. For a bulk ingest that cost is
real; for a pipeline writing documents one at a time it is noise.

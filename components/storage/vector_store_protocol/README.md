# vector-store-protocol: the operations every backend owes you

```
vector_store_protocol.py   errors + input checks + the re-exported protocol
example.py                 runnable: flat filters, finite vectors, bad ids
tests/                     success paths and deliberate failure paths
manifest.json              schema-valid; core-types as a component dependency
```

## Purpose

`core-types` declares `VectorStoreProtocol` — `add`, `query`, `delete`,
`count` — and `ScoredDocument`. What it cannot declare is the *checking* those
methods owe their callers, because that checking is the same whichever backend
sits underneath. That is this component:

- **The error vocabulary** — `StorageError` catches everything; `StorageInputError`
  means the *caller* is at fault, `StorageBackendError` means the *backend* is,
  and `MissingStorageDependency` is an `ImportError` so `ragkit add` users see
  it under `except ImportError`.
- **The input checks** — `validate_filter`, `validate_ids`, `validate_vector`,
  `validate_k`, `validate_embedded_chunks`. Each raises a message naming the
  component and the field, so a caller never has to read a traceback to learn
  which value was wrong.
- **The protocol, re-exported** — so an implementation can be written against
  this file alone.

Two of these checks exist because the alternative fails *silently*, which is
the failure mode this library exists to remove.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `VectorStoreProtocol` | re-export of `core_types.VectorStoreProtocol` | satisfied structurally |
| `StorageError` | base exception | catches the whole area |
| `StorageInputError` | `StorageError` | the caller passed something unusable |
| `StorageBackendError` | `StorageError` | the backend failed |
| `MissingStorageDependency` | **`ImportError`** | an optional backend is absent (FR-3) |
| `validate_filter(filter)` | `-> dict \| None` | rejects nested values |
| `validate_ids(ids)` | `-> list[str]` | rejects a bare `str` and empty ids |
| `validate_vector(vector)` | `-> list[float]` | rejects empty and non-finite |
| `validate_k(k)` | `-> int` | rejects non-positive and non-int |
| `validate_embedded_chunks(items)` | `-> list[EmbeddedChunk]` | rejects mixed widths |

### Invariants the tests pin down

- **Filters are flat, and nested values are rejected rather than dropped.**
  A filter is a mapping of field to value. A backend that silently dropped a
  nested value would return a *narrower* result set than the caller asked for,
  with no signal that anything was ignored. The message says so.
- **`None` means "no filter"** and passes through untouched.
- **`validate_vector` rejects `NaN` and `±inf`.** These compare against
  nothing, so a `NaN` in an index poisons a nearest-neighbour search without
  failing visibly — the distance to everything becomes `NaN`, and the result
  order becomes whatever the sort happened to do. Values are also normalised
  to `float`, so an `int` vector becomes usable.
- **A vector must be non-empty.** An empty vector has no position in a space.
- **`validate_ids` rejects a bare `str`.** A `str` *is* a `Sequence[str]`, so
  accepting it would mean `["a", "b", "c"]` for `"abc"` — three deletes where
  the caller meant one.
- **`validate_embedded_chunks` rejects mixed widths** within one `add`. One
  call must add one vector space; otherwise the index has no single geometry
  to search against.

## Example

See `example.py`, which runs standalone — from this directory, with
`core_types.py` beside it:

```bash
python example.py
```

## Dependencies

- **Components:** `core-types` — `Chunk`, `EmbeddedChunk`, `ScoredDocument`,
  `Metadata` and the protocol itself. Copy it alongside (the manifest declares
  it, so `ragkit add` brings it in).
- **Packages:** standard library only. `core-types` imports `pydantic`, which
  the library's runtime requires anyway.
- **External services:** none. This component never touches a backend.

## Trade-offs

**Use it when** writing or reviewing a vector-store implementation. Call the
validators at the top of each method and let the messages do the rest.

**It costs** nothing at runtime beyond a few `isinstance` checks.

**Why the errors are not imported from somewhere central:** a contract
component has to be copy-ready on its own, and importing an error hierarchy
from elsewhere means copying *that* file too. The names deliberately mirror
`document-store`'s so the duplication is familiar rather than surprising.

## Known limitations

**`validate_filter` allows any scalar value**, including one the backend may
not support (a `datetime`, say). Type portability is the backend's concern;
this checks shape, not type.

**No paging, no cursors, no ordering guarantee beyond what `query` returns.**
`ScoredDocument`'s ordering encodes rank; this component does not add a
second notion of order on top.

**`validate_k` has no upper bound.** Asking for `k=10**9` is allowed, because
a store may legitimately hold fewer and returning what exists is correct.

**`validate_embedded_chunks` checks widths against each other, not against a
declared dimension.** A store knows its own width; that check belongs to the
implementation (and is asserted for both adapters in `test_storage.py`).

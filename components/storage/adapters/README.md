# storage-adapters: two vector stores, each behind a lazy import

```
storage_adapters.py   ChromaVectorStore + FaissVectorStore
example.py            runnable: both adapters against fake backends
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; chromadb + faiss optional
```

## Purpose

Two vector-store implementations, both satisfying `VectorStoreProtocol` and
both carrying the same guarantees, so a caller swaps one for the other without
touching its own code (FR-5):

- **`ChromaVectorStore`** — over a `chromadb` collection. Supports metadata
  filtering, because a collection holds metadata.
- **`FaissVectorStore`** — over a `faiss` index with an explicit id map.
  Refuses a filter rather than ignoring it, because a raw faiss index holds
  no metadata to filter on.

**Neither backend is imported at module scope.** Each is imported inside the
function that first needs it, and a missing package raises
`MissingStorageDependency` — an `ImportError` carrying the manifest's
`missing_dependency_message` *verbatim*, so the message a user sees is the one
the registry declares rather than a paraphrase that drifts from it. Importing
this module therefore succeeds on a machine with neither installed, which is
what lets the tests run at all.

## Why faiss needs an id map

A raw faiss index is addressed by row number, which cannot survive a delete:
remove row 3 and every later row now addresses the wrong id, silently. This
adapter keeps its own `id -> row` mapping and rebuilds it after a delete, so
`delete(["c0"])` followed by `query` returns the right documents rather than
whatever happened to move.

## Results name their backend

Every `ScoredDocument` returned from `query` carries the store's name in
`source_name`. `core-types` calls this out as a conformance requirement the
*type* cannot enforce, and it is the one that matters most: a hybrid retriever
merging a dense result list with a sparse one has nothing to merge on if the
results do not say where they came from. It is asserted for both adapters.

## Reserved metadata keys

A vector backend stores flat key/value pairs, so a chunk's identity fields
(`chunk_id`, `document_id`, `source_name`, `chunk_index`) travel in backend
metadata under `_rag_`-prefixed keys. User metadata is passed through
untouched, and the prefix keeps the two from colliding.

The consequence is handled rather than left to the caller: **a filter naming
one of those identity fields is rewritten to its reserved key** before it
reaches the backend. The caller sees the *unprefixed* name in the results, so
filtering on `document_id` would otherwise match nothing and come back empty —
which reads as "no such document" rather than "wrong field name".

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `ChromaVectorStore` | `ChromaVectorStore(collection_name, *, metadata=None, client=None)` | `client` is injectable |
| `FaissVectorStore` | `FaissVectorStore(*, dimensions, index=None)` | `index` is injectable |
| `add(items)` | `-> int` | count added |
| `query(vector, k, filter=None)` | `-> list[ScoredDocument]` | nearest first |
| `delete(ids)` | `-> int` | count deleted; unknown ids are a no-op |
| `count()` | `-> int` | items held |
| `IDENTITY_FIELDS` | the four reserved field names | |
| `MISSING_DEPENDENCY_MESSAGE` | `str` | copied verbatim from the manifest |

## Dependencies

- **Components:** `core-types` and `vector-store-protocol` — the types, the
  protocol and the error vocabulary this file raises. Copy them alongside (the
  manifest declares both, so `ragkit add` brings them in).
- **Packages:** standard library only, plus **optional** `chromadb>=0.4.0` and
  `faiss>=1.5.0`, each imported lazily. The faiss path also uses `numpy`,
  which is not declared separately because faiss cannot be installed without
  it.
- **External services:** none. Backends are in-process libraries.

## Trade-offs

**Use it when** you want a vector store without writing one.

**It costs** a dependency and its memory: chromadb is an embedded database,
faiss is a native library. Both hold their index in memory.

**Alternatives:** `vector-store-protocol` alone, if you are writing your own
backend; a hosted store, if the index must outlive the process.

**When not to use it:** an index larger than memory, or one that must be
shared across processes — both of these are single-process, in-memory stores.

## Known limitations

**Neither real backend is exercised here.** `chromadb` and `faiss` are not
installed in this checkout, and installing them would modify dependency files
— which this library forbids (FR-2). Every test and example runs both
adapters against injected fakes that speak the same API surface. Treat real
backend integration as unexecuted until a run on a machine that has them
installed.

**The faiss adapter supports no filters.** It raises rather than ignoring one,
because an ignored filter looks like an empty result rather than an error. Use
`ChromaVectorStore`, or filter the returned documents yourself.

**`delete` on chroma reports how many ids were passed, not how many existed.**
A collection's `delete` returns nothing, so an unknown id counts as deleted
anyway. This is deliberate — deleting what is not there is a no-op, which is
what makes the call safe after a failed re-ingest — but the number is "ids
submitted", not "rows removed".

**No persistence configuration.** `ChromaVectorStore` builds an in-process
`chromadb.Client()`; pointing it at a persistent or server-backed collection
means passing your own `client=`.

**Not `isinstance`-compatible with `core_types.VectorStoreProtocol`.** The
protocol is re-exported through `vector-store-protocol` and these classes
satisfy it structurally; `isinstance` against the `core_types` symbol passes
because it is the same object, but the adapters themselves import the area's
own copy of the errors.

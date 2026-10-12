# dense-retrieval: embed a query, ask a store, get ScoredDocuments back

```
dense_retrieval.py   DenseRetriever and the error vocabulary
example.py           runnable: a composed retrieval with a broken embedder
tests/               success paths and deliberate failure paths
manifest.json        schema-valid; core-types as a component dependency
```

## Purpose

The three-step pipeline, composed and checked:

```
embed query  ->  ask the vector store  ->  hand back ScoredDocument
```

Every part is already a contract from another component. The embedder is an
`EmbeddingProtocol` and the store is a `VectorStoreProtocol`, both from
`core-types`. **This component owns neither and re-implements neither.** It
validates that what it was handed honours those contracts, does the
composition, and fails with a message saying which part broke.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `DenseRetriever` | `DenseRetriever(*, embedder, store, k=5, name="dense")` | both dependencies validated at construction |
| `.retrieve(query, *, k=None, filter=None)` | `-> list[ScoredDocument]` | best first, as the store orders them |
| `.k`, `.name`, `.embedder`, `.store` | attributes | readable, so a caller can inspect the wiring |
| `RetrievalError` | base exception | catches this component |
| `RetrievalInputError` | `RetrievalError` | the caller passed something unusable |

## What it checks, and why each earns its place

| Check | Failure it prevents |
|---|---|
| `embedder` satisfies `EmbeddingProtocol` | A duck-typed object missing `dimensions` failing deep inside a call |
| `store` satisfies `VectorStoreProtocol` | Same, one layer later and harder to read |
| `embed([query])` returns **exactly one** vector | A batch embedder returning two vectors, leaving the caller matched against the wrong one |
| the query vector's width equals `embedder.dimensions` | A store indexed at width *N* queried at width *M* |
| the store returns `ScoredDocument` values | Pushing a conversion onto every downstream caller (FR-8) |
| the query is a non-empty string | An empty string embeds to a real vector and retrieves a real-looking answer about nothing |

The **vector-space mismatch** is the one worth dwelling on. An embedder and a
store only agree when they were built with the same model *and* the same
width. Query a store indexed with model A using vectors from model B and you
get confident, well-formed, entirely wrong neighbours -- no exception
anywhere in the system. This component checks the width, which is the half it
can see, and exposes `embedder.model_id` on the retriever so the other half
is at least visible in the object the caller holds.

## Requesting more results than exist

Not an error. A store with three chunks, asked for ten, returns three, and
this passes them through. A retriever that raised would force every caller to
know its index size before asking.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it.

## Dependencies

- **Components:** `core-types` — `EmbeddingProtocol`, `VectorStoreProtocol`,
  `ScoredDocument`. Copy it alongside.
- **Packages:** standard library only. The embedder and store you supply may
  of course have their own; those are their components' concern.
- **External services:** none of its own. Whatever backend the store wraps.

## Trade-offs

**Use it when** you have an embedder and a store and want the composition
without writing the glue for the fourth time.

**It costs** one `isinstance` check per construction and a handful per call --
nothing measurable next to the embedding and the search.

**Alternatives:** call `embedder.embed` and `store.query` yourself, which is
about six lines. Use this when you want the validation, the wrapping of
failures with context, and one place to change when the composition changes.

**When not to use it:** a store that embeds internally (some managed vector
databases take text directly). Then this component adds a hop you do not
need.

## Known limitations

**It cannot detect a model mismatch, only a width mismatch.** Two models
that both produce 384-wide vectors are indistinguishable here, and querying
across them is silently wrong. Storing `model_id` next to the index and
checking it at load time is the caller's job; this component makes the value
available but does not persist it.

**It does not retry.** A transient embedder or store failure is wrapped and
raised. Retrying belongs to the caller, who knows which failures are
transient and which are permanent.

**No batching across queries.** `retrieve` handles one query. Callers wanting
to embed many queries should use the embedder directly, or a batch wrapper
from the embeddings area.

**`filter` is passed through unvalidated.** The store owns filter validation
(and `vector-store-protocol` provides it), so validating here would be a
second, possibly divergent copy.

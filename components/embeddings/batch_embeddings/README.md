# batch-embeddings: split the calls, and check what comes back

```
batch_embeddings.py   BatchEmbedder; depends on embedding-protocol (copy it first)
example.py            runnable: batching, order, and the failures it refuses
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; zero packages, one component dependency
```

## Purpose

Most embedding APIs cap how many texts one request may carry, and a naive
`[embed([t]) for t in texts]` turns a corpus into thousands of round trips.
This component wraps any object satisfying `EmbeddingProtocol` and splits the
work into batches, returning one vector per input text, in input order.

The slicing is the easy half. The other half — and the reason this component
exists rather than a three-line loop — is **checking what comes back**. A
provider that returns a different number of vectors than it was asked for, or
vectors of differing widths, produces an index that looks perfectly healthy
while every retrieval from that moment on returns the wrong document. Both are
caught where they happen.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `BatchEmbedder` | `BatchEmbedder(embedder, *, batch_size=32)` | wraps a protocol-shaped embedder |
| `embed(texts)` | `-> list[Vector]` | one vector per text, in input order |
| `embed_one(text)` | `-> Vector` | convenience over `embed` |
| `dimensions` / `model_id` | delegated properties | so the wrapper is itself an embedder |
| `DEFAULT_BATCH_SIZE` | `32` | deliberately modest |
| `MAX_BATCH_SIZE` | `1024` | the largest `batch_size` accepted |

Because `dimensions` and `model_id` are delegated, the wrapper still satisfies
`EmbeddingProtocol` — so it composes: a batching wrapper around a remote
embedder is still an embedder, and can be wrapped again.

### Invariants the tests pin down

- **Count.** `len(result) == len(texts)`, always, in input order.
- **Empty input returns `[]`** without ever calling the wrapped embedder.
- **A provider returning the wrong count is an error**, not a silent
  truncation. The message names the batch and both counts, because this is the
  failure that corrupts an index without visibly failing.
- **Vectors of differing widths are an error**, and so is a width that
  disagrees with what the embedder declares.
- **A failing batch is re-raised with context**: the batch index, how many
  texts it carried, and the original exception chained as `__cause__`.
- **Bad configuration fails at construction**, before any text is sent:
  a non-protocol object, or a `batch_size` that is not an int in
  `1..MAX_BATCH_SIZE`.

## Dependencies

- **Components:** `embedding-protocol` — the contract and error types this
  file imports. Copy it alongside (the manifest declares it, so `ragkit add`
  brings it in).
- **Packages:** standard library only (`numbers`, `typing`). No optional
  packages, no external services, no network — it never talks to a provider
  itself, only to the object you hand it.

## Example

See `example.py`, which runs standalone — from this directory, with
`embedding_protocol.py` beside it:

```bash
python example.py
```

## Trade-offs

**Use it when** you have an embedder and a corpus larger than one request.

**It costs** one extra object and a few per-vector checks. Those checks are
cheap (`len()` on each vector) compared with the network call they guard.

**Alternatives:** calling `embed` yourself in a loop — fine for a script, and
you then own the count and width checks; the provider's own batch endpoint, if
it has one and you are already within its limit.

**When not to use it:** single texts. `embed_one` on the underlying embedder
is simpler and avoids the wrapper entirely.

## Known limitations

**No concurrency.** Batches are sent one after another. Adding a thread pool
would be a real feature, and a real source of ordering bugs and rate-limit
storms; it is deliberately not here. If you need it, wrap this component's
`_call` rather than the whole class, so the count and width checks still run.

**No retry or backoff.** A transient failure surfaces immediately, wrapped.
Retrying is a policy that belongs to the caller, who knows whether the
provider charges per attempt.

**It trusts the wrapped object's declared `dimensions`.** If an embedder
declares one width and serves another, that is caught — but a wrong declaration
that matches wrong output is not something this component can detect.

**Not `isinstance`-compatible with `core_types.EmbeddingProtocol`.** This file
imports the area's own `embedding_protocol` contract, which agrees on attribute
names but is a separate type — the same deliberate duplication `core-types`
documents for its own `Chunk`.

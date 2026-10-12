# local-embeddings: run the model here, and name the model yourself

```
local_embeddings.py   LocalEmbedder; depends on embedding-protocol (copy it first)
example.py            runnable: no-default model, mocked encode, the FR-3 guard
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; sentence-transformers optional
```

## Purpose

Embed text with a model running inside this process. No request leaves the
machine, no API key is needed, and the only network access that can happen at
all is the *initial* model download.

**There is no default model, on purpose.** A hidden default means the same
call returns different vectors on different machines — and vectors from
different models cannot share an index, so the damage is silent and permanent.
The model id must come from the caller or from `$RAG_LOCAL_EMBEDDINGS_MODEL`;
a missing one raises at construction naming the variable to set.

Set `TRANSFORMERS_OFFLINE=1` to forbid even the initial download. On an
air-gapped host the component then fails rather than reaching out, which is
the correct behaviour — it is `transformers` that enforces the flag, and this
component reports the failure with the model id and a hint about the cache.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `LocalEmbedder` | `LocalEmbedder(model_name=None, *, env=os.environ)` | `env` is injectable so tests need not touch the real environment |
| `embed(texts)` | `-> list[Vector]` | one vector per text, in input order |
| `embed_one(text)` | `-> Vector` | convenience over `embed` |
| `dimensions` | property | read from the model itself (`get_sentence_embedding_dimension`) |
| `model_id` | attribute | the resolved model name |
| `MODEL_ENV_VAR` | `"RAG_LOCAL_EMBEDDINGS_MODEL"` | where the default comes from |
| `MISSING_DEPENDENCY_MESSAGE` | `str` | copied verbatim from the manifest |

### Invariants the tests pin down

- **The model id is resolved from the argument first, then the environment,
  then nowhere** — the third case raises `EmbeddingInputError` naming
  `$RAG_LOCAL_EMBEDDINGS_MODEL`.
- **`embed` returns one vector per text, in order**, and an empty input
  returns `[]` without encoding anything.
- **A model returning the wrong count is an error**, not a silent truncation.
- **`dimensions` comes from the model**, and a model reporting a non-positive
  width is rejected rather than trusted.
- **FR-3: the guard is the manifest's message, verbatim**, as an
  `ImportError`.

## Dependencies

- **Components:** `embedding-protocol` — the contract and error types this
  file imports. Copy it alongside (the manifest declares it, so `ragkit add`
  brings it in).
- **Packages:** standard library only, plus **optional**
  `sentence-transformers>=2.2.0`, imported lazily inside `_load_model`. It is
  never imported at module scope, which is what keeps this module importable —
  and its tests runnable — on a machine that never installed it.
- **External service:** the Hugging Face model hub, for the initial download
  only. `$TRANSFORMERS_OFFLINE=1` forbids it.

## Example

See `example.py`, which runs standalone — from this directory, with
`embedding_protocol.py` beside it:

```bash
python example.py
```

It demonstrates the no-default failure, then the encode path through an
injected fake model, then the FR-3 guard.

## Trade-offs

**Use it when** data must not leave the machine, calls must be free, or the
corpus is small enough that per-request latency does not dominate.

**It costs** memory and start-up time: the model loads into this process on
first use and stays there. It also puts model management — downloads, cache
directories, offline flags — on the caller.

**Alternatives:** `remote-embeddings` (no local memory, costs money and sends
text out); the `batch-embeddings` wrapper around either, if a corpus is
larger than one call.

**When not to use it:** a large corpus on a small machine, or anywhere the
same model is already served by an API you trust.

## Known limitations

**The real `SentenceTransformer` path is not exercised here.**
`sentence-transformers` is not installed in this checkout, and installing it
would modify dependency files — which this library forbids (FR-2). Every
encode test therefore runs against an injected fake model. Treat
`sentence-transformers` integration as unexecuted until a run on a machine
that has it installed.

**`dimensions` triggers the model load.** Reading it on a fresh instance
loads the model, which is expensive. It is cached afterwards.

**No batching of its own.** `SentenceTransformer.encode` batches internally
with its own defaults. For explicit control, wrap this in
`batch-embeddings`.

**Not `isinstance`-compatible with `core_types.EmbeddingProtocol`.** This file
imports the area's own `embedding_protocol` contract, which agrees on attribute
names but is a separate type.

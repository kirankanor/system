# remote-embeddings: call a provider, and distrust its response

```
remote_embeddings.py   RemoteEmbedder; depends on embedding-protocol (copy it first)
example.py             runnable: required settings, mocked responses, rejections
tests/                 success paths and deliberate failure paths
manifest.json          schema-valid; httpx optional
```

## Purpose

Embed text through a provider's HTTP API — no local model, no memory held by
this process.

**Nothing about the provider is assumed.** There is no default endpoint, no
default model and no default key: all three must be supplied as arguments or
read from `$RAG_EMBEDDINGS_URL`, `$RAG_EMBEDDINGS_API_KEY` and
`$RAG_EMBEDDINGS_MODEL`. A hidden default here would mean an accidental
request to somebody else's service carrying the caller's text, so a missing
setting raises at construction, naming the variable.

**What it checks on the way back is most of the work.** A response that parses
as JSON is not a response that is usable. This component verifies, in order:

1. the payload carries a `data` list;
2. every entry carries both `index` and `embedding`;
3. the indices are exactly `0..n-1` — no gaps, no repeats;
4. the count matches the number of texts sent;
5. every value is numeric.

Providers *do* reorder entries, and a reordered response that is not noticed
attaches every vector to the wrong document. So the indices are honoured
rather than assumed to arrive in order: entries are sorted by `index` before
being returned.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `RemoteEmbedder` | `RemoteEmbedder(url=None, *, api_key=None, model=None, dimensions=None, timeout=30.0, env=os.environ)` | `env` is injectable so tests need not touch the real environment |
| `embed(texts)` | `-> list[Vector]` | one vector per text, in input order |
| `embed_one(text)` | `-> Vector` | convenience over `embed` |
| `dimensions` | property | declared at construction, else measured from the first response and cached |
| `model_id` | attribute | the resolved model name |
| `DEFAULT_TIMEOUT` | `30.0` | explicit, so a hung provider fails the pipeline |
| `MISSING_DEPENDENCY_MESSAGE` | `str` | copied verbatim from the manifest |

The request body is `{"model": ..., "input": [...]}`, sent as a JSON POST with
`Authorization: Bearer <key>` — the shape the majority of providers accept.

### Invariants the tests pin down

- **Reordered responses are corrected.** A payload whose entries arrive as
  indices `[2, 0, 1]` comes back in `[0, 1, 2]` order.
- **Duplicate or gapped indices are rejected**, because no vector can then be
  safely matched to its text.
- **A missing `index` is rejected** rather than assumed to be the position.
- **Non-200 responses report the status**, with a truncated body.
- **A non-JSON body is rejected** with the reason.
- **`dimensions` before the first call raises** rather than guessing; after
  it, it is measured and cached, and a disagreement with a declared width is
  caught.
- **An empty input never opens a client.**

## Dependencies

- **Components:** `embedding-protocol` — the contract and error types this
  file imports. Copy it alongside (the manifest declares it, so `ragkit add`
  brings it in).
- **Packages:** standard library only, plus **optional** `httpx>=0.27.0`,
  imported lazily inside `_load_client`. Never imported at module scope.
- **External service:** a provider's HTTP API, at whatever URL you configure.
  The component makes no request until you call `embed` with real text.

## Example

See `example.py`, which runs standalone — from this directory, with
`embedding_protocol.py` beside it:

```bash
python example.py
```

Every response in it is mocked; nothing reaches the network.

## Trade-offs

**Use it when** a hosted model is the right tool: no local memory, models too
large to run here, or a team-standard endpoint.

**It costs** data leaving the machine, per-request latency, and per-request
money. It also couples your index to a provider's vocabulary: switching
providers changes the vectors, which means re-indexing.

**Alternatives:** `local-embeddings` (nothing leaves the machine); the
`batch-embeddings` wrapper around either, for a corpus larger than one call.

**When not to use it:** text that may not leave the machine, or a workload
where request latency dominates and a local model would be faster.

## Known limitations

**The response shape is one convention.** `{"data": [{"index": ..., "embedding": ...}]}` is the common one, but providers differ. Adapting a provider
that uses another shape means editing `_interpret`, not configuring it.

**`httpx` is installed in this checkout, but no real provider was called.**
Installing `httpx` would modify dependency files — which this library forbids
(FR-2) — so it is not a project dependency here; it happens to be present in
the environment. Every test and every example in this component mocks the
client, so the HTTP path is unexecuted end to end. Treat real-provider
integration as unexecuted until a run against an actual endpoint.

**No retry, backoff or streaming.** A transient failure surfaces immediately,
wrapped, with the status and a body fragment. Retrying is a policy the caller
owns, since only the caller knows whether the provider charges per attempt.

**The API key appears in the request headers**, never in an error message.
Response bodies are truncated to 200 characters in messages, but they are not
scrubbed — a provider that echoes a secret back in an error body would have it
printed. Check `check_no_secrets.py` output for this repository.

**Not `isinstance`-compatible with `core_types.EmbeddingProtocol`.** This file
imports the area's own `embedding_protocol` contract, which agrees on attribute
names but is a separate type.

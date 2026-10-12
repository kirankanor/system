# cross-encoder-reranker: the accurate reranker, with the wheel kept at arm's length

```
cross_encoder_reranker.py  CrossEncoderReranker, the error vocabulary
example.py                 runnable: guarded loading, fallback, determinism
tests/                     success paths and deliberate failure paths
manifest.json              schema-valid; sentence-transformers OPTIONAL
```

## Purpose

Rerank a short list of candidates with a cross-encoder. A **bi-encoder**
(`embeddings`) embeds the query and each passage *separately* and compares
vectors; a **cross-encoder** reads the query and one passage **together**, in a
single forward pass, and scores how well they match. That joint attention is
why it is markedly more accurate — and why it cannot pre-compute anything:
every query costs one pass per candidate.

**That cost is the whole design of this component.** It reranks a handful of
candidates. It is not a retriever and must not be pointed at a corpus.

## The dependency is optional, and genuinely guarded

`sentence-transformers` is heavy — it drags in PyTorch and downloads a model
on first use. This library's rule (`DECISION-bf2566ea`) is that such libraries
are never imported at module top level by shared code:

- importing this module never touches the wheel, and works without it;
- the import happens inside `CrossEncoderReranker.rerank`, on first use;
- a missing wheel raises `MissingDependency` — an `ImportError` — whose
  message is the manifest's `missing_dependency_message` **verbatim**.

So a caller can list this reranker among several strategies, decide at runtime
whether the machine can run it, and fall back to `relevance-ranking` when it
cannot.

## Score direction

`CrossEncoder.predict` returns a relevance logit per pair: **higher is more
relevant**. No normalisation is applied — cross-encoder logits are not
comparable across models, so returning them untouched is more honest than
rescaling them into a range that implies precision the model does not have.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `CrossEncoderReranker(model_name, *, model=None, device=None, name=None)` | | `model_name` required, never defaulted |
| `.rerank(query, documents, top_k)` | `-> list[ScoredDocument]` | best first, truncated to `top_k` |
| `.is_loaded()` | `-> bool` | inspect the cost before paying it |
| `.model_name` `.device` `.name` | `str` | plain attributes |
| `RerankingError` | base exception | catches this component |
| `RerankingInputError` | `RerankingError` | the caller passed something unusable |
| `MissingDependency` | `RerankingError, ImportError` | the wheel is not installed |
| `MANIFEST_MESSAGE` | `str` | the manifest text, read without importing the wheel |

### Invariants the tests pin down

- **`model_name` is required and never defaulted.** Picking a model is a
  quality decision; a hidden default would make results depend on which
  version of this file you have.
- **The model loads lazily, on first `rerank`.** Constructing the object is
  free and never touches the dependency, so a strategy list can be registered
  before the machine is known to be able to run it.
- **Every candidate is scored, then truncated.** Scoring a pre-truncated list
  would bake the first retriever's ordering into the answer, which defeats
  the point of reranking.
- **A model returning the wrong number of scores is an error.** Silently
  taking the first `n` would drop candidates from the ranking without a word.
- **A broken model is wrapped with the model's name, and keeps its cause.**
  The original exception is chained, never swallowed.
- **Ties break on `chunk_index`, then `chunk_id`.** Cross-encoders emit
  exactly equal logits often enough that this is not theoretical.
- **`MissingDependency` is an `ImportError`, not a `ValueError`.** The machine
  is the problem, not the caller's arguments.

## Supplying your own model

Pass `model=` with an already-constructed cross-encoder. This is how callers
who manage their own models — shared across rerankers, loaded at startup,
pinned to a device — avoid a second load, and how the tests run at all on a
machine without the wheel.

## Example

```bash
python example.py
```

Runs standalone with `core_types.py` beside it. It never requires the wheel:
the loading path is demonstrated with an injected stand-in, and the
missing-dependency path is demonstrated for real, because the wheel is not
installed here.

## Dependencies

- **Components:** `core-types` — `ScoredDocument`.
- **Packages:** `sentence-transformers` `>=2.2.0`, **optional**. The manifest
  declares it under `optional_packages`, never `required_packages`.
- **External services:** none, but a model download happens on first load for
  any model name not already cached.

## Trade-offs

**Use it when** retrieval quality is the bottleneck rather than latency, and
you have a handful of candidates that a lexical or bi-encoder ranking has
already narrowed down. On MS MARCO-style benchmarks a cross-encoder typically
moves nDCG by more than any amount of tuning the retriever.

**It costs** one forward pass per candidate per query, on whatever device
`device=` names. Ten candidates is comfortable; a thousand is not.

**When not to use it:** over a corpus (it cannot index), under a tight
per-request budget, or on a machine where you cannot install a
PyTorch-sized dependency. In all three, `relevance-ranking` is the
dependency-free answer.

## Known limitations

**The real model is unexecuted in this repository.** `sentence-transformers`
is not installed here, and installing it would modify dependency files — which
this library forbids. The gate tests prove the properties that do not need it:
the module imports, loads lazily, scores and orders an injected model
correctly, breaks ties deterministically, and fails with the manifest's
message verbatim when the wheel is absent. **They do not prove a real
cross-encoder scores well.** Run `example.py` on a machine with the wheel to
exercise the real path.

**Logits are not comparable across models.** Do not threshold them, and do not
compare a score from `ms-marco-MiniLM` with one from a larger model.

**No batching or truncation control.** Candidates are passed to `predict` as
one batch with no `max_length` handling, so very long passages rely on the
model's own defaults. Callers with long documents should chunk first — which
`chunking` already did.

**No quantisation, no device management beyond a `device=` passthrough.** The
component names a device; it does not discover one, fall back from CUDA, or
manage memory.

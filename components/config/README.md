# config/: shared layer for your RAG pipeline

```
config/
├── __init__.py     public exports (import everything from `config`)
├── settings.py     typed env-driven settings + index_fingerprint()
├── models.py       data contracts between stages (Chunk, Query, RagAnswer, ...) + id helpers
├── errors.py       one exception hierarchy (stage, retryable, code, safe details)
├── enums.py        PipelineStage, DistanceMetric
├── prompts.py      versioned grounded-QA prompt + context packing
├── .env.example    every env var, required ones first
└── tests/test_config.py
```

## Rule: who may import this
Your **orchestration / app code** imports `config`. Your **standalone library components**
(e.g. `pdf-extraction-pymupdf`) must not, or they stop being copy-ready. Map their output
into `config.models` at the boundary, e.g. `PageResult -> PageText`.

## Quick use
```python
from config import load_settings, ExtractionError, is_retryable

settings = load_settings(".env")               # explicit file; process env wins over it
print(settings.safe_dump())                    # secrets masked; never log model_dump()
stored = settings.index_fingerprint()          # save next to your vectors
# later: if load_settings().index_fingerprint() != stored -> re-embed/re-chunk
```

## Key decisions (WHY)
- **No defaults for embedding/LLM/vector-store provider or model.** The embedding model + chunking +
  metric define what's in your index. A silent default can mix incompatible vectors.
- **`index_fingerprint()`** hashes only index-defining settings (embedding, chunking, metric), not tuning
  knobs (top_k, timeouts, LLM). A mismatch means the stored index is stale.
- **Deterministic ids** (`make_document_id`, `make_chunk_id`) make re-ingestion an upsert, not a duplicate.
  Trade-off: identical bytes under two filenames share one document id.
- **Flat metadata only** (str/int/float/bool): most vector DBs can't filter nested data.
- **Errors carry `stage` + `retryable`.** Wrap library exceptions with `raise EmbeddingError(...) from exc`.
  `details` must never hold secrets, document text or queries.
- **Prompts are versioned** (`GROUNDED_QA_VERSION`): log it with each answer. Retrieved text is
  wrapped in `<context>` and told to be treated as data. This reduces prompt-injection risk; it does not remove it.

## Install
`pip install pydantic pydantic-settings` (Python 3.11+). Tested only with pydantic 2.14.0 and
pydantic-settings 2.15.0 on Python 3.12/Linux; older versions are unverified.

## Tests
Run from the folder that *contains* `config/`: `pytest config/tests`

## Limitations
- `chunking.strategy` and provider names are free-form strings; nothing here implements chunkers or providers.
- `chunking.unit="tokens"` is only a setting: your chunker must honor it.
- Metadata flatness is enforced on models, not by your vector store; check its own limits.
- Folder name `config` is generic and can collide with another top-level `config` module in a client project.
  Rename the folder (e.g. `rag_config`) if so; imports inside are relative and keep working.

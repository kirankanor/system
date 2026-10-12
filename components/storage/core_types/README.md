# core-types: the contracts every other component speaks

```
core_types.py     7 protocols + the shared result types they name
example.py        runnable: chunk -> score -> build context, with nothing but this file
tests/            success paths and deliberate failure paths
manifest.json     schema-valid; zero dependencies
```

## Purpose

Most areas in this library hold more than one implementation that has to stay
interchangeable — chunking alone has five. For that to work, the *contract* has
to live in exactly one place — otherwise each chunker invents its own return
shape and callers end up writing adapters between stages. `core-types` is that
one place.

It answers two specification requirements:

- **FR-5 — interchange.** Implement one of the seven protocols and your
  component drops into any caller written against it.
- **FR-8 — compose without adapters.** Retrieval, reranking and context
  building all pass the same `ScoredDocument` around, so no stage needs to
  convert its input before it can start.

## Public API

### Result types

| Type | Carries | Constructed by |
|---|---|---|
| `Chunk` | `text`, `chunk_index`, optional ids/pages, `metadata` | chunkers |
| `ScoredDocument(Chunk)` | the above plus `score` | retrieval, reranking |
| `SourceDocument` | `document_id`, `source_name`, `sha256`, `page_count`, `metadata` | document store |
| `EmbeddedChunk` | a `Chunk` + its `vector` + `index_fingerprint` | embedding + indexing |
| `Context` | assembled `text`, the `chunks` actually used, `token_count`, `budget`, `truncated` | context builders |
| `EvaluationExample` | query, relevant/retrieved ids, expected/actual answers | evaluation datasets |
| `MetricResult` | `name`, `value`, `higher_is_better`, `sample_count`, `details` | metrics |

All are frozen pydantic models with `extra="forbid"`: validate at
construction, then trust the value.

### Protocols

| Protocol | Signature |
|---|---|
| `ChunkingProtocol` | `chunk(text, *, size, overlap, **kw) -> list[Chunk]` |
| `EmbeddingProtocol` | `embed(texts) -> list[list[float]]`, `dimensions: int`, `model_id: str` |
| `VectorStoreProtocol` | `add(items)`, `query(vector, k, filter)`, `delete(ids)`, `count()` |
| `DocumentStoreProtocol` | `put(doc) -> str`, `get(id)`, `list(filter)` |
| `RerankerProtocol` | `rerank(query, documents, top_k) -> list[ScoredDocument]` |
| `ContextBuilderProtocol` | `build(query, chunks, budget) -> Context` |
| `MetricProtocol` | `compute(dataset) -> MetricResult` |

Each is `@runtime_checkable`, so `isinstance(obj, ChunkingProtocol)` works as a
cheap capability check — but only that: it confirms the members *exist*, not
that their signatures or types are right (see the tests, which assert the real
contract separately). `issubclass` works for the six method-only protocols and
raises `TypeError: Protocols with non-method members don't support issubclass()`
for `EmbeddingProtocol`, whose `dimensions`/`model_id` are data members — a
limitation of `typing`, not of these definitions.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

It chunks a document, scores the chunks as a fake retriever would, and hands
them to a context builder — demonstrating that the three stages pass one type
around with no conversion in between.

```python
from core_types import ChunkingProtocol, ContextBuilderProtocol, ScoredDocument

def serve(chunker: ChunkingProtocol, builder: ContextBuilderProtocol) -> None:
    chunks = chunker.chunk(TEXT, size=80, overlap=10)
    # `from_chunk` is the supported way to score a chunk: spreading
    # `**chunk.model_dump()` alongside an id the chunk already carries
    # raises TypeError, and every retriever needs this conversion.
    candidates = [
        ScoredDocument.from_chunk(c, score=float(i), document_id="doc-1")
        for i, c in enumerate(chunks)
    ]
    ctx = builder.build("what is this about?", candidates, budget=120)
    print(ctx.truncated, len(ctx.chunks))
```

## Dependencies

- **Components:** none. This is a leaf — nothing else has to be copied first.
- **Packages:** `pydantic>=2.13` only (the same constraint `pyproject.toml`
  declares, so dependency validation sees one version floor). No optional
  packages, no external services, no network.

## Trade-offs

**Use it when** you are adding any component to a project, or you need to know
the shape of data crossing a stage boundary.

**It costs** one model layer: values are validated and frozen at construction
rather than being plain dicts. That is deliberate — a typo'd field fails loudly
at the boundary instead of becoming `None` three stages later — but it means
you cannot mutate a chunk in place; build a new one.

**Alternatives:** plain dataclasses (cheaper, but you write and maintain the
validation yourself); dicts (no contract at all, which is exactly what FR-5 and
FR-8 exist to prevent).

**When not to use it:** as an application-facing settings object. The
destination project's `config` component owns settings, prompts and the error
hierarchy — this file is the *component interchange* vocabulary, not your
pipeline configuration.

## Known limitation: two `Chunk` definitions exist today

`config/models.py` (the `settings` component) defines a near-identical `Chunk`
and `SourceDocument`. That is intentional and temporary:

- `config/__init__.py` states that standalone library components must **not**
  depend on `config`, because components have to be copyable on their own.
  Importing `config` here would drag the application's settings, prompts and
  error hierarchy in behind a type definition.
- `config` is owned by a different task than this one, so it cannot be edited
  here without breaking task scope.

Field names are mirrored deliberately so the two can later be collapsed into
one definition as a mechanical move. The single intentional difference: ids
(`chunk_id`, `document_id`, `source_name`) and the page range are **optional**
in `core-types.Chunk`, because a chunker is called as
`chunk(text, size=..., overlap=...)` and genuinely does not know which document
its text came from — ingestion fills those in afterwards.

Until they are reconciled, do not copy both layers into one project and assume
the two `Chunk` classes are interchangeable; they are different classes.

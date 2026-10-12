# index-writer

The write path: bounded batches, refusal before writing, and records that
prove where their bytes came from.

## What it protects against

**Silent data loss on create.** Rebuilding an index to "fix" it is the most
natural impulse there is, and if it works by wiping the collection first, the
rows are gone before anyone notices the rebuild was unnecessary. Creation and
replacement are therefore separate operations: `create()` refuses to sit on
top of rows that exist, and `replace()` is the only way through — it demands
a `reason` and writes the loss into an `IndexDestruction` record.

**A payload the index cannot hold, discovered after the write.** A vector of
the wrong width, or a record missing a field the index is meant to be
queryable on, is only caught once rows are already landing.
`validate_payload()` checks the whole payload *before* the first write.

**A big batch that half-fails.** One `add()` of ten thousand rows either
succeeds or throws, and a throw says nothing about which of the ten thousand
landed. Writes are split into bounded batches, and a batch the backend rejects
is retried item by item — which is how the report can say, per item, what
actually happened.

## Why per-item retry is safe

The retry writes an id that may already exist. That is harmless precisely
because every write is an **upsert**: the same content lands in the same
place. Without upsert semantics this recovery step would risk doubling every
row the backend had already accepted before failing — which is why the two
requirements are specified together.

## API

```python
writer = IndexWriter(store, dimensions=1536, batch_size=128,
                     required_fields=("document_id", "chunk_id"))
report = writer.upsert(items)
report.raise_if_failed()      # optional; successes are already written
```

| Name | Purpose |
| --- | --- |
| `IndexWriter(store, *, dimensions, batch_size, required_fields)` | Writes in bounded, validated, upserting batches. |
| `IndexWriter.upsert(items) -> BatchReport` | Validates first, then writes; never raises for a per-item backend failure. |
| `BatchReport` | `.ok`, `.succeeded`, `.failed`, `.failures`, `.results`, `.raise_if_failed()`. |
| `validate_payload(items, *, dimensions, ...)` | Pre-write check; raises `PayloadMismatchError`. |
| `bounded_batches(items, batch_size)` | Plain splitting, with its own validation. |
| `record_from(item) -> StoredChunkRecord` | Build the provenance-carrying record. |
| `IndexLifecycle(count=...)` | `create` / `replace` / `destructions`. |

## The stored record (FR-48)

`StoredChunkRecord` carries `document_id`, `chunk_id`, `source_name`,
`chunk_index`, `start_offset`, `end_offset`, `page_start`, `page_end`,
`section`, `text`, `vector`, `index_fingerprint` and `metadata`.

**Every provenance field is present on every record.** Where the chunker
supplied no value the field is `None` rather than absent — a field that only
exists sometimes cannot be asserted on, and AC-26 asks that every record
carry these.

`core-types:Chunk` has no dedicated offset or section fields, so those are
read from the chunk's metadata under the conventional keys `start_offset`,
`end_offset` and `section`. `record.provenance` returns just the citation
fields, and `record.is_traceable` says whether enough survives to find the
text in its source again (needs the document, plus a page range, an offset
range, or a section).

## Errors

| Exception | Raised when |
| --- | --- |
| `IndexWriterError` | Bad configuration: non-positive dimensions or batch size, a `required_fields` name that is not a record field, a store with no `add`. |
| `PayloadMismatchError` | Vector width or schema does not match the target index — always **before** anything is written. Carries `chunk_id` and `document_id`. |
| `DestructiveOperationRefused` | `create()` would have discarded stored rows. |
| `BatchWriteError` | Raised by `BatchReport.raise_if_failed()`; carries the report. |

## Example

```
python example.py
```

## Not implemented

No retries with backoff, no journaling, no transactional write. A batch that
fails is retried exactly once, item by item, and the outcome is reported —
the decision to retry further, or to abort the run, belongs to the
orchestration layer rather than to a component writing rows.

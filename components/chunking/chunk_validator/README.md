# chunk-validator: the gate between a chunker and an embedder

## Purpose

Chunking produces *candidates*. Something has to decide which of them are safe
to embed, and the honest answer is that the failures are expensive in
different ways:

- a whitespace-only chunk costs an embedding call and stores nothing,
- a chunk with no `document_id` can never be cited, so it is retrievable but
  useless,
- a chunk four times the context window fails at the provider, after the
  network round trip,
- a chunk with an inverted page range silently cites the wrong page.

This component is that gate. It returns **one result per chunk, always**, and
guarantees the property the orchestration layer actually depends on:

> Nothing reaches `outcome.accepted` carrying an unaddressed fatal finding.

"Unaddressed" is the important word. A fatal finding can be *repaired* when
the repair is derivable from the chunk itself — deriving a missing `chunk_id`
is a derivation, not a fabrication. Repairing a missing page range by writing
`page 1` would be a fabrication, so it is not offered: the chunk is held,
with the reason recorded.

## Public API

| Name | What it is |
|---|---|
| `ChunkValidator(*, min_chars=1, max_chars=8000, required_metadata=(), severities=None, fatal_policy=FatalPolicy.QUARANTINE, anomaly_markers=..., size_of=None)` | Configure. Severities are overridable per rule. |
| `ChunkValidator.validate(chunks) -> ValidationOutcome` | The run. One `ValidationResult` per chunk, in order. |
| `ChunkValidator.inspect(chunk, *, index=0) -> ValidationResult` | One chunk, without partitioning it. |
| `ValidationOutcome` | `accepted`, `rejected`, `quarantined`, `repairs`, `results`, plus `.fatal_count`, `.warning_count`, `.is_clean`, `.held()`. |
| `ValidationResult` | `index`, `chunk_id`, `findings`, `.is_fatal`, `.is_valid`, `.rules`. |
| `Finding` | `rule`, `severity`, `message`. |
| `Rule` | `empty-text`, `whitespace-only-text`, `text-too-short`, `text-too-long`, `missing-document-id`, `missing-chunk-id`, `invalid-chunk-id`, `missing-source-location`, `invalid-page-range`, `missing-required-metadata`, `extraction-anomaly`. |
| `Severity` | `WARNING` or `FATAL`. |
| `FatalPolicy` | `QUARANTINE` (default), `REJECT`, `REPAIR`. |

## Example

Run `python example.py` from this directory.

```python
from chunk_validator import ChunkValidator, FatalPolicy

validator = ChunkValidator(min_chars=10, max_chars=500, fatal_policy=FatalPolicy.REPAIR)
outcome = validator.validate(chunks)

for held in outcome.held():
    print(f"held chunk {held.index}: {[f.message for f in held.findings]}")
for repair in outcome.repairs:
    print(f"repaired: {repair.description}")

# The contract: accepted is safe to embed.
assert all(v.is_fatal is False for v in outcome.results if v.index in ...)
```

## Dependencies

None. Standard library only.

## Trade-offs

- **Warning is not fatal, by default.** A short chunk or one with no page
  range is still perfectly retrievable — it just cannot cite a page. Those are
  warnings and never remove a chunk from `accepted`. Override the severity per
  rule if you disagree; the override is recorded in `validator.severities`.
- **Repairs are few, on purpose.** Only `missing-chunk-id`, `invalid-chunk-id`
  and an *inverted* page range are repairable, because only those are
  derivable. Everything else is held rather than guessed at.
- **A frozen chunk is never mutated.** Repair uses `dataclasses.replace`, so
  the caller's original object is untouched and a new one is returned. Only a
  plainly mutable object is edited in place.
- **Duck typing over typing.** Chunks are read by attribute, matching the
  `core-types` contract, so this module imports neither `rag_code_library` nor
  any chunker. Both `core_types.Chunk` and a chunker's local `Chunk` — whose
  identity fields default to `None` — validate as they are.
- **`size_of` replaces the size function entirely.** Pass a token counter and
  you get token bounds; the validator will not hand a token-aware caller a
  character count.

## Known limitations

- **One chunk's findings never consider its neighbours.** Duplicate content,
  overlapping windows and corpus-level anomalies are `chunk-deduplicator`'s
  job, not this one's.
- **`extraction-anomaly` is metadata-driven.** It fires on the explicit keys
  in `anomaly_markers` (`extraction_warning`, `extraction_failed`,
  `ocr_failed`). A chunker that records its extraction problems under some
  other name will not be detected, and this component will not guess.
- **`min_chars`/`max_chars` are measured by `size_of`, which defaults to
  `len`.** Characters are not tokens. If your bound must be token-accurate,
  pass a counter.
- **A chunk that is not a dataclass and not mutable cannot be repaired.** It
  is held, with `repair_failed` set, rather than mutated behind the caller's
  back.

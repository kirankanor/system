# chunk-deduplicator

Two answers to one question: **is this work already done?**

An ingestion pipeline re-does work in two ways, and both cost the same thing.

It re-embeds **content it has already embedded** — two chunks carrying identical
text, a document re-ingested unchanged, the same template paragraph on a
thousand pages. That is the deduplication half (FR-43).

It re-embeds **content whose stored vector came from a different
configuration** — someone swapped the embedding model, or changed the chunk
size, and the vectors already in the index no longer match what the current
code would produce. That is the fingerprint half (FR-44).

Both fail silently when they go wrong. A pipeline that skips a chunk it should
have re-embedded does not error; it returns degraded answers for weeks before
anyone notices. So both halves here default to **not reusing** when the
evidence is missing or unclear.

## What "exact" means here

Duplicate detection hashes *normalized* text, where the normalization is
declared and deliberately narrow:

1. Unicode NFC
2. every run of whitespace collapsed to a single space
3. strip

**Case is preserved on purpose.** Folding case would merge content that
genuinely differs, which is similarity matching by another name — and
near-duplicate matching is out of scope (FR-43). This component will happily
report no duplicates in a corpus full of near-misses, because claiming
otherwise would be a lie about what it computes.

The normalizer is injectable, so a caller who has already run
`text-normalizer` can pass that same function rather than normalizing twice
under two different rules.

## Scopes

| Scope | Effect |
| --- | --- |
| `DOCUMENT` | Duplicates collapse only within a document. Identical text in two documents is retained in both, so no document silently loses content. |
| `CORPUS` | Everything identical collapses to one survivor, whichever document it came from. |
| `BOTH` | The document pass runs first, then the corpus pass. |

Each group records a `level` describing the **data** rather than the
configured scope: `document` when every appearance shared one document,
`corpus` when the shared content crossed documents or its origin was unknown.

## Provenance is never thrown away

Removing a duplicate does not mean pretending it was not there. Every removal
records the chunk it collapsed *into* and the document each appearance came
from, so an index that drops a duplicate can still say "this content also
appears in X". Without that, corpus-wide dedup would quietly erase the fact
that a shared template appeared in a thousand documents.

Retained chunks are returned as the **original objects**, not wrappers — what
goes in comes out unchanged.

## The fingerprint

`IndexFingerprint` is a parseable string —
`model=...|dims=...|chunking=...` — not an opaque hash. An opaque hash would
tell you *that* something differed; these three fields tell you **which** one
did, which is the difference between a log line a human can act on and one
they cannot.

```python
gate = EmbeddingReuseGate.configured(
    model="text-embedding-3-small",
    dimensions=1536,
    chunking_config={"size": 512, "overlap": 64},
)

gate.value                     # the value to store alongside every vector
gate.may_reuse(stored)         # -> bool, never raises
gate.check(stored)             # raises FingerprintMismatch naming the field
```

`chunking_config` may be any JSON-able value; it is canonically serialized, so
the same configuration hashes identically regardless of key order.

## Example

```
python example.py
```

## Errors

| Exception | Raised when |
| --- | --- |
| `DedupError` | Invalid construction (bad scope, unknown hash algorithm). |
| `MissingChunkText` | A chunk has no readable `text`, so there is nothing to compare. |
| `EmbeddingReuseError` | Invalid fingerprint configuration (empty model, non-positive dimensions). |
| `FingerprintMismatch` | A stored vector must not be reused — unparseable, absent, or from a different configuration. |

## Not implemented

Near-duplicate similarity matching (cosine, MinHash, shingling) is **not**
implemented and is not claimed. Exact content equality only.

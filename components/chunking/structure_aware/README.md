# structure-aware-chunker: lines, tables and headings as indivisible units

```
structure_aware_chunker.py   chunk() + the Chunk it returns; stdlib only, zero dependencies
example.py                   runnable: table atomicity, heading glue, FR-7 rejection
tests/                       success paths and deliberate failure paths
manifest.json                schema-valid; zero dependencies
```

## Purpose

Text without blanks lines does not segment the way prose does. This component
cuts on **line boundaries** first, which keeps a list item, a shell command or
a log line intact, and then applies two structure rules on top:

1. **Table rows are kept whole, and tables are packed together.** A row never
   breaks at a cell boundary; a table larger than the budget breaks *at its row
   separators*.
2. **A heading is glued to the unit under it**, so a chunk never opens with
   content whose title travelled on ahead.

It recognises three heading shapes — ATX (`## Title`), numbered
(`1. Scope`, `2.3 Details`, `12) Note`) and ALL-CAPS (`OVERVIEW`) — and
deliberately ignores everything else, because the ALL-CAPS rule in particular
is a heuristic that would rather miss a heading than mistake a shouted line
for one.

It satisfies the library's shared chunking contract structurally:
`chunk(text, *, size, overlap, **kw) -> list[Chunk]` with the same field names
as `core_types.Chunk`, so a caller written against `ChunkingProtocol` swaps in
this component unchanged (FR-5) — despite this file importing nothing beyond
the standard library.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `chunk` | `chunk(text, *, size, overlap=0) -> list[Chunk]` | the one entry point |
| `Chunk` | frozen dataclass: `chunk_index`, `text`, optional `chunk_id`/`document_id`/`source_name`/`page_start`/`page_end`, `metadata={}` | identity fields are `None` — a chunker does not know its document |
| `ChunkingError` | base exception | catching it catches every failure below |
| `InvalidChunkSize` | `ChunkingError` | unusable `size`/`overlap` |
| `OverlapTooLarge` | `ChunkingError` | `overlap >= size` (FR-7) |

### Invariants the tests pin down

- **Lossless at `overlap=0`.** Units are contiguous *slices* of the input —
  including the trailing newline of every line — so joining the chunks
  reproduces `text` byte for byte, pipe characters and all.
- **A table that fits is one chunk.** Table rows are grouped into a single unit
  while that unit fits `size`.
- **An oversized table breaks at rows, never at cells.** Each emitted piece is
  a whole row, still within `size`.
- **A line is atomic.** A line shorter than `size` is not cut just because its
  neighbours are long; it is only broken down when the line *itself* exceeds
  the budget.
- **A heading leads its chunk.** When a heading and the following unit both
  fit, they travel together.
- **`Section One` is not a heading** and `3 items here` is not either — the
  first has no leading number or marker, the second is a list item, not a
  section. Testing the negative case is what stops the heuristic from silently
  over-matching.
- **Size cap always**, at any `size` down to 1.
- **FR-7: `overlap >= size` is rejected, never clamped.**
- `""` returns `[]`. Unknown `**kw` is rejected by name rather than dropped.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

```python
from structure_aware_chunker import OverlapTooLarge, chunk

chunks = chunk(DOC, size=80, overlap=0)
assert "".join(c.text for c in chunks) == DOC     # lossless at overlap=0
assert any(TABLE in c.text for c in chunks)       # rows stay together

try:
    chunk(DOC, size=20, overlap=20)
except OverlapTooLarge as exc:
    print(exc)
```

## Dependencies

- **Components:** none. Self-contained on purpose — copy the one `.py` file.
- **Packages:** standard library only (`re`, `dataclasses`, `typing`). No
  optional packages, no external services, no network.

## Trade-offs

**Use it when** the source is line-oriented rather than paragraph-oriented —
numbered specs, runbooks, log excerpts, or anything with tables and list
items, where cutting mid-row or mid-item destroys the meaning of the line.

**It costs** evenness and prose quality. Lines are atomic, so a document made
of very short lines packs slowly and a document made of very long lines falls
back to word boundaries; and because there is no paragraph rule, two
sentences that a reader would call one paragraph can land in different chunks
if they are on different lines.

**Alternatives:** the markdown chunker (paragraph-first, understands fenced
code — better for prose-shaped documents with code in them); the recursive
chunker (cheapest boundary-aware option, understands no structure at all);
the fixed-size chunker (cheapest, understands nothing).

**When not to use it:** hard-wrapped prose where newlines are cosmetic. Line
atomicity buys you nothing when lines wrap at 80 columns for typesetting
reasons — you would be chunking on the wrapper's decision, not the author's.

## Known limitations

**ALL-CAPS detection is a heuristic.** It reads a line as a heading when it
contains no lowercase at all and is at least three characters, so a shouted
line (`DO NOT MERGE`) is mistaken for a section title. The failure mode was
chosen deliberately: the mistake makes one chunk shorter, never splits
anything that was otherwise whole.

**Numbered headings require a bracket or a multi-level number.** `1. Scope`
and `2.3 Details` are recognised; `3 items here` is not, because it is a list
item rather than a section. A numbered heading written as `1 Scope` (no dot)
will not be detected.

**Tables are recognised by shape, not by a parser.** A line beginning and
ending with a pipe is a row, so a fenced code block containing ASCII-art
pipes is treated as a table. Splitting it at line boundaries is still
lossless, so the cost is a shorter chunk, not a broken one — but if your
documents mix both, the markdown chunker is the better choice.

**Not `isinstance`-compatible with `core_types.Chunk`.** This file's `Chunk`
mirrors that model's field names but is a separate stdlib dataclass. Callers
written against the field names work unchanged; `isinstance(chunk,
core_types.Chunk)` does not pass. That duplication is the same deliberate,
temporary split `core-types` documents for its own `Chunk`.

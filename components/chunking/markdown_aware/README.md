# markdown-chunker: respect the fence, the paragraph and the heading

```
markdown_chunker.py   chunk() + the Chunk it returns; stdlib only, zero dependencies
example.py            runnable: fence atomicity, paragraph splits, FR-7 rejection
tests/                success paths and deliberate failure paths
manifest.json         schema-valid; zero dependencies
```

## Purpose

A prose chunker that does not know about Markdown will cut through a fenced
code block the moment a blank line falls inside it, and the result is a
function split across two chunks that can neither be executed nor pasted back
together. This component makes three structural promises the generic chunkers
do not:

1. **Fenced code blocks are atomic.** A ``` or ~~~ fence — including any blank
   lines and prose-looking text inside it — stays in one piece.
2. **Text splits at blank lines, and only there.** A paragraph that fits stays
   whole; a hard-wrapped sentence is not an invitation to cut.
3. **A heading is glued to the paragraph under it** when both fit, so a chunk
   never opens with content whose title went into the previous chunk.

Only when a unit is genuinely larger than `size` does it break down further —
at line boundaries, then word boundaries, then characters — so the last resort
still lands on something a reader would recognise.

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

- **Lossless at `overlap=0`.** Segments are contiguous *slices* of the input
  (cut by index, never rebuilt from parts), so `"".join(c.text for c in
  chunk(text, size=N))` reproduces `text` byte for byte — fence delimiters and
  blank lines included.
- **A fence stays whole whenever it fits the budget.** The unit is the fence
  *plus the newline that terminates its closing delimiter* — that newline has
  to be emitted somewhere, and stranding it in the following prose would break
  the exact join — so a 61-character fence needs a budget of 62 to come
  through whole. When the fence alone exceeds `size` it *is* split, because
  exceeding the budget is the worse failure; that case is asserted separately
  rather than hidden.
- **An unclosed fence still rejoins exactly.** A fence that runs to the end of
  the document is treated as one segment to the end. (This was a real defect:
  the prose cursor was not advanced to the fence's first byte, so the trailing
  segment re-emitted the entire document from offset 0.)
- **Size cap always**, at any `size` down to 1.
- **FR-7: `overlap >= size` is rejected, never clamped.**
- `""` returns `[]`. Unknown `**kw` is rejected by name rather than dropped.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

```python
from markdown_chunker import OverlapTooLarge, chunk

chunks = chunk(DOC, size=80)
assert "".join(c.text for c in chunks) == DOC       # lossless at overlap=0
assert any("```python" in c.text for c in chunks)   # the fence came through whole

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

**Use it when** the source is Markdown or anything fenced — READMEs,
notebooks' exported docs, issue threads, code-heavy runbooks — and losing a
code block across a chunk boundary would be worse than an uneven chunk.

**It costs** evenness. Paragraphs are kept whole even when that closes a chunk
well short of `size`, and a heading is pulled forward to stay with its
content, so chunk lengths vary more than a fixed-size pass would give you. It
also runs a segmentation pass before the merge, which is more work than
walking windows — still linear in the size of the text.

**Alternatives:** the recursive chunker (splits at `\n\n` too, but will happily
cut inside a fence — cheaper, less aware); the structure-aware chunker
(line-atomic, recognises numbered and ALL-CAPS headings and keeps tables
together); the fixed-size chunker (cheapest, respects nothing).

**When not to use it:** non-Markdown text. The fence and ATX rules are inert on
plain prose, so you would be paying for structure detection you do not use —
the recursive chunker is the better default there.

## Known limitations

**A fence larger than `size` is split.** There is no way to keep a 4 kB code
block in one piece inside a 512-character budget. Splitting it at line/word
boundaries is chosen deliberately over emitting an over-budget chunk, since an
over-budget chunk breaks every downstream caller's contract.

**A heading can still close a chunk** when the heading and its paragraph do
not both fit — gluing them would itself produce an over-budget unit, and
exceeding `size` is the worse failure. In practice this needs a heading
longer than the remaining room.

**`#`-style headings only.** Numbered (`2.1 Scope`) and ALL-CAPS titles are
not treated as headings here; the structure-aware chunker recognises those.

**Not `isinstance`-compatible with `core_types.Chunk`.** This file's `Chunk`
mirrors that model's field names but is a separate stdlib dataclass. Callers
written against the field names work unchanged; `isinstance(chunk,
core_types.Chunk)` does not pass. That duplication is the same deliberate,
temporary split `core-types` documents for its own `Chunk`.

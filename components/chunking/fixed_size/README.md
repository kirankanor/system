# fixed-size-chunker: blind character windows, exact arithmetic

```
fixed_size_chunker.py   chunk() + the Chunk it returns; stdlib only, zero dependencies
example.py              runnable: window arithmetic, overlap, and FR-7 rejection
tests/                  success paths and deliberate failure paths
manifest.json           schema-valid; zero dependencies
```

## Purpose

The simplest chunker in the area, and the one every other chunker's overlap
rules are written against: cut `[start, start + size)` and advance `start` by
`size - overlap`. Because the arithmetic is this plain, it is also the cheapest
— a single pass over the text, no separators, no parser, no tokenizer.

It satisfies the library's shared chunking contract structurally:
`chunk(text, *, size, overlap, **kw) -> list[Chunk]` with the same field names
as `core_types.Chunk`, so a caller written against `ChunkingProtocol` swaps in
this component unchanged (FR-5) — despite this file importing nothing beyond
the standard library.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `chunk` | `chunk(text, *, size, overlap=0) -> list[Chunk]` | the one entry point |
| `Chunk` | frozen dataclass: `chunk_index`, `text`, optional `chunk_id`/`document_id`/`source_name`/`page_start`/`page_end`, `metadata={}` | identity fields are `None` — a chunker does not know its document; ingestion fills them in |
| `ChunkingError` | base exception | catching it catches every failure below |
| `InvalidChunkSize` | `ChunkingError` | unusable `size`/`overlap` |
| `OverlapTooLarge` | `ChunkingError` | `overlap >= size` (FR-7) |

### Invariants the tests pin down

- **Windows advance by `size - overlap`.** `chunk("0123456789", size=5,
  overlap=2)` steps by 3, giving `["01234", "34567", "6789"]`.
- **No duplicate tail.** The loop stops at the first window that reaches the
  end of the text. Without that stop, a non-zero overlap keeps stepping while
  the text is already exhausted and the caller gets a final chunk that is a
  subset of its predecessor — a wasted embedding that returns nothing new. The
  redundancy is invisible at `overlap=0` (the step equals `size`, so the last
  window reaches the end anyway), which is exactly why it is worth a test.
- **Lossless at `overlap=0`.** The windows tile the input exactly:
  `"".join(c.text for c in chunk(text, size=N))` reproduces `text`.
- **Size cap always.** Every chunk is at most `size` characters — the final
  window runs to the end rather than being padded.
- **FR-7: `overlap >= size` is rejected, never clamped.** At equality the step
  would be zero and chunking would never advance. You get `OverlapTooLarge`
  with a message naming both numbers.
- `""` returns `[]`. Unknown `**kw` is rejected by name rather than dropped.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

```python
from fixed_size_chunker import OverlapTooLarge, chunk

chunks = chunk(DOC, size=60)                       # windows of 60
assert "".join(c.text for c in chunks) == DOC      # lossless at overlap=0

overlapped = chunk("0123456789", size=5, overlap=2)
assert [c.text for c in overlapped] == ["01234", "34567", "6789"]

# FR-7: full overlap is rejected, not clamped behind your back.
try:
    chunk(DOC, size=20, overlap=20)
except OverlapTooLarge as exc:
    print(exc)
```

## Dependencies

- **Components:** none. Self-contained on purpose — copy the one `.py` file.
- **Packages:** standard library only (`dataclasses`, `typing`). No optional
  packages, no external services, no network.

## Trade-offs

**Use it when** the text is already roughly uniform — sentence-per-line data,
log lines, rows of a table — or when you need a chunking pass that is cheap
enough to run inside a tight loop and predictable enough to reason about
without running it.

**It costs** quality on prose. Windows are blind: a cut lands wherever the
arithmetic puts it, so a paragraph can be sliced in half, a word split across
two chunks, and a heading separated from the section it titles. Each of those
makes a retrieved chunk harder to interpret on its own, and for retrieval that
is the whole game.

**Alternatives:** the recursive chunker (same zero dependencies, cuts on
paragraph/line/word boundaries instead of blind windows — usually the better
default for prose); the markdown-aware chunker (keeps fenced code intact); the
structure-aware chunker (keeps tables and list items intact); the token-based
chunker (budgets in tokens, but pulls in an optional tokenizer this component
refuses to have).

**When not to use it:** prose with structure you care about preserving. If the
document has paragraphs, headings or code, a boundary-aware chunker costs
nothing extra and produces chunks that stand alone better.

## Known limitations

**Characters, not tokens.** `size` and `overlap` are character counts. For a
caller budgeting an embedding model's context window, characters are a proxy:
~4 chars per token for English prose, far less for code or CJK text. Either
pick a conservative character size and let the embedding component truncate the
rare long chunk, or budget with the token-based chunker instead. Shipping a
tokenizer here would violate the zero-dependency manifest this component is
specified under.

**Structural boundaries are ignored.** That is the defining property, not a
bug — but it means the component will happily cut between a heading and its
paragraph. See the trade-offs above.

**Not `isinstance`-compatible with `core_types.Chunk`.** This file's `Chunk`
mirrors that model's field names but is a separate stdlib dataclass. Callers
written against the field names work unchanged; `isinstance(chunk,
core_types.Chunk)` does not pass. That duplication is the same deliberate,
temporary split `core-types` documents for its own `Chunk`.

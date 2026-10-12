# recursive-chunker: cut on meaning, hard-cut only as a last resort

```
recursive_chunker.py   chunk() + the Chunk it returns; stdlib only, zero dependencies
example.py             runnable: paragraph-aware splits, overlap, and FR-7 rejection
tests/                 success paths and deliberate failure paths
manifest.json          schema-valid; zero dependencies
```

## Purpose

A fixed-size chunker cuts blind windows: a paragraph boundary or a word can be
sliced in half simply because it straddled the window edge. This component
walks *down* a separator priority — `("\n\n", "\n", " ", "")` — so cuts land on
paragraph boundaries first, then lines, then words, and only hard-cut on a
character boundary when the text offers nothing better. A separator absent from
the text is skipped, not an error.

It satisfies the library's shared chunking contract structurally:
`chunk(text, *, size, overlap, **kw) -> list[Chunk]` with the same field names
as `core_types.Chunk`, so a caller written against `ChunkingProtocol` swaps in
this component unchanged (FR-5) — despite this file importing nothing beyond
the standard library.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `chunk` | `chunk(text, *, size, overlap=0, separators=SEPARATORS) -> list[Chunk]` | the one entry point |
| `Chunk` | frozen dataclass: `chunk_index`, `text`, optional `chunk_id`/`document_id`/`source_name`/`page_start`/`page_end`, `metadata={}` | identity fields are `None` — a chunker does not know its document; ingestion fills them in |
| `SEPARATORS` | `("\n\n", "\n", " ", "")` | default priority; the empty string is the "hard-cut here" sentinel |
| `ChunkingError` | base exception | catching it catches every failure below |
| `InvalidChunkSize` | `ChunkingError` | unusable `size`/`overlap`/`separators` |
| `OverlapTooLarge` | `ChunkingError` | `overlap >= size` (FR-7) |

### Invariants the tests pin down

- **Lossless at `overlap=0`.** Pieces are contiguous *slices* of the input, so
  `"".join(c.text for c in chunk(text, size=N))` reproduces `text` exactly —
  separators included, nothing lost or duplicated at a boundary.
- **Size cap always.** Every chunk is at most `size` characters, whatever the
  separators or overlap do.
- **Overlap never breaks the cap.** When a chunk closes, its final `overlap`
  characters become the head of the next one — truncated to whatever room the
  incoming piece leaves (`size - len(piece)`). Overlap gives way before the
  size contract does; if the incoming piece fills the chunk, the overlap is
  truncated to zero rather than the chunk growing past `size`.
- **FR-7: `overlap >= size` is rejected, never clamped.** At equality the step
  would be zero (a hang); a silently accepted `overlap == size` is how a
  chunker becomes an infinite loop. You get `OverlapTooLarge` with a message
  telling you both numbers.
- `""` returns `[]`. Unknown `**kw` is rejected by name rather than dropped.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

```python
from recursive_chunker import OverlapTooLarge, chunk

chunks = chunk(DOC, size=60)                      # cuts land on \n\n / \n / " "
assert "".join(c.text for c in chunks) == DOC     # lossless at overlap=0

# FR-7: full overlap is rejected, not clamped behind your back.
try:
    chunk("alpha beta gamma delta", size=20, overlap=20)
except OverlapTooLarge as exc:
    print(exc)  # "overlap (20) must be smaller than size (20)..."
```

## Dependencies

- **Components:** none. Self-contained on purpose — copy the one `.py` file.
- **Packages:** standard library only (`dataclasses`, `typing`). No optional
  packages, no external services, no network.

## Trade-offs

**Use it when** your text has structure (prose, markdown, logs with blank-line
separated records) and you want cuts to respect it without pulling in a
tokenizer or a parser.

**It costs** two things a fixed-size chunker does not charge: the split is
recursive, so pathological inputs (a separator repeated thousands of times) do
more work than a single pass; and chunk *lengths* are uneven — greedy merging
packs pieces up to `size` and can close a chunk well short of it. Both are
bounded and deliberate: even splits are not worth an uneven cut through a word.

**Alternatives:** a fixed-size window chunker (cheaper, dumber — fine for
already-uniform text like sentence-per-line datasets); token-based chunkers
(honest about model context windows, but every one drags in a tokenizer
dependency this component refuses to have); semantic chunkers (embeddings at
split time — strictly more machinery than a boundary walk).

**When not to use it:** when the budget is measured in tokens, not characters.
This component counts characters; see the limitation below.

## Known limitation: characters, not tokens

`size` and `overlap` are character counts. For a caller budgeting an embedding
model's context window or an LLM prompt, characters are a proxy: ~4 chars per
token for English prose, far less for code or CJK text. Two honest options:

- pick a conservative character size for your content and let the embedding
  component truncate the rare long chunk, or
- post-filter with a tokenizer before indexing.

The alternative — shipping a tokenizer — would violate the zero-dependency
manifest this component is specified under, so the trade is made explicitly
here rather than hidden in an import.

Also note: this file's `Chunk` mirrors `core_types.Chunk`'s field names but is
a separate stdlib dataclass, not that pydantic model. Callers written against
the field names work unchanged; `isinstance(chunk, core_types.Chunk)` does not
pass. That duplication is the same deliberate, temporary split `core-types`
documents for its own `Chunk`.

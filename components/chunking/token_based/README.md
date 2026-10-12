# token-chunker: budget in tokens, the unit a model actually counts

```
token_chunker.py   chunk() + the Chunk it returns; stdlib only, tiktoken optional
example.py         runnable: the FR-3 guard, token windows, FR-7 rejection
tests/             success paths and deliberate failure paths
manifest.json      schema-valid; tiktoken declared optional, never required
```

## Purpose

Character windows are a proxy for a model's budget, and the proxy drifts the
moment the text is anything but ASCII: 400 characters of English is roughly
100 tokens, while 400 characters of Japanese can be several hundred. When the
downstream limit is a *context window*, it is measured in tokens — so this is
the chunker that talks to it in its own units.

The window arithmetic is identical in shape to the fixed-size chunker's,
with tokens in place of characters: cut `[start, start + size)` and advance by
`size - overlap`. Decoding is per-window, so a chunk never splits a
character mid-way.

It satisfies the library's shared chunking contract structurally:
`chunk(text, *, size, overlap, **kw) -> list[Chunk]` with the same field names
as `core_types.Chunk`, so a caller written against `ChunkingProtocol` swaps in
this component unchanged (FR-5) — despite this file importing nothing beyond
the standard library *and* one lazily-guarded optional package.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `chunk` | `chunk(text, *, size, overlap=0, encoding=DEFAULT_ENCODING) -> list[Chunk]` | the one entry point |
| `DEFAULT_ENCODING` | `"cl100k_base"` | used when the caller names no encoding |
| `MISSING_DEPENDENCY_MESSAGE` | `str` | copied verbatim from the manifest; asserted against it |
| `Chunk` | frozen dataclass: `chunk_index`, `text`, optional `chunk_id`/`document_id`/`source_name`/`page_start`/`page_end`, `metadata={}` | identity fields are `None` — a chunker does not know its document |
| `ChunkingError` | base exception | catching it catches every failure below |
| `InvalidChunkSize` | `ChunkingError` | unusable `size`/`overlap` |
| `OverlapTooLarge` | `ChunkingError` | `overlap >= size` (FR-7) |
| `InvalidEncoding` | `ChunkingError` | `encoding` empty, not a string, or unknown to `tiktoken` |
| `MissingTokenizer` | **`ImportError`** | `tiktoken` is not installed (FR-3) |

`MissingTokenizer` is deliberately *not* a `ChunkingError`: FR-3 requires the
missing-optional-dependency case to be an `ImportError`, so that callers
guarding optional components with `except ImportError` catch it.

### Invariants the tests pin down

- **Windows advance by `size - overlap` tokens**, and stop at the first window
  that reaches the end of the input. Without that stop, a non-zero overlap
  keeps stepping while the tokens are already exhausted and the caller gets a
  final chunk that is a subset of its predecessor — a wasted embedding that
  returns nothing new. The redundancy is invisible at `overlap=0`, which is
  exactly why it is worth a test.
- **Lossless at `overlap=0`.** Decoding the windows in order reproduces the
  input exactly.
- **Size cap always**, measured in tokens.
- **An empty window is dropped, not emitted.** A tokenizer can map a window to
  nothing (an unused or special token); `chunk_index` is assigned *after* that
  drop, so indices stay contiguous.
- **FR-7: `overlap >= size` is rejected, never clamped.** At equality the step
  would be zero and tokenisation would never advance.
- **`""` returns `[]` without ever loading the tokenizer** — so it succeeds on
  a machine that has no `tiktoken` at all.
- **FR-3: the guard is the manifest's message, verbatim**, as an
  `ImportError`. Not a paraphrase that could drift from what the registry
  declares.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

It demonstrates the FR-3 guard first (the real behaviour on a machine without
`tiktoken`), then the windowing arithmetic using a tiny character-level
codec injected exactly the way a caller would supply a tokenizer.

## Dependencies

- **Components:** none. Self-contained on purpose — copy the one `.py` file.
- **Packages:** standard library only, plus **optional** `tiktoken>=0.7.0`,
  declared in the manifest and imported lazily inside `_load_tokenizer`. It is
  never imported at module scope, which is what keeps this module importable —
  and its tests runnable — on a machine that never installed it.

## Trade-offs

**Use it when** the budget is a model's context window rather than a rough
size, or the corpus is multilingual, code-heavy, or otherwise far from ASCII
where characters-per-token varies by an order of magnitude.

**It costs** a dependency and a speed penalty. `tiktoken` is a compiled
tokenizer that must load before the first call, and encoding is slower than
slicing characters. It also couples the component to a tokenizer vocabulary:
chunks change if the encoding does, which matters if you are re-indexing
against a stable corpus.

**Alternatives:** the fixed-size chunker (zero dependencies, characters not
tokens — fine for roughly uniform ASCII prose); the structure-aware chunker
(keeps tables and lines intact, still characters); the markdown chunker
(keeps fenced code intact).

**When not to use it:** when you cannot install the optional dependency, or
when character counts already describe your budget well enough. If the corpus
is plain English and the limit is generous, the extra precision buys little.

## Known limitations

**The real `tiktoken` path is not exercised in this checkout.** `tiktoken` is
not installed here, and installing it would modify dependency files — which
this library forbids (FR-2). The windowing logic is therefore tested against a
character-level codec that maps one character to one token, which makes the
arithmetic exact and predictable but is not `tiktoken`. Treat `tiktoken`
integration as unexecuted until a run on a machine that has it installed.

**Windows are fixed-size; there is no boundary awareness.** A window can land
inside a sentence, a paragraph or a fenced code block. If that matters, this
chunker is the wrong tool regardless of its unit.

**Not `isinstance`-compatible with `core_types.Chunk`.** This file's `Chunk`
mirrors that model's field names but is a separate stdlib dataclass. Callers
written against the field names work unchanged; `isinstance(chunk,
core_types.Chunk)` does not pass. That duplication is the same deliberate,
temporary split `core-types` documents for its own `Chunk`.

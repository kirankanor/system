# embedding-protocol: one contract, one error hierarchy, for the whole area

```
embedding_protocol.py   EmbeddingProtocol + the shared error types; stdlib only
example.py              runnable: the contract, input validation, the failures
tests/                  success paths and deliberate failure paths
manifest.json           schema-valid; zero dependencies
```

## Purpose

Three adapters embed text three ways — locally, remotely, and batched — and if
each invented its own errors a caller would have to learn three failure
vocabularies. This component defines them once:

- **`EmbeddingProtocol`** — the shape every embedder satisfies: `dimensions`,
  `model_id` and `embed(texts)`.
- **The error hierarchy** — `EmbeddingError` catches everything in the area;
  `EmbeddingInputError` means the *caller* is at fault, `EmbeddingProviderError`
  means the *provider* is, and `MissingEmbeddingDependency` is an `ImportError`
  so `ragkit add` users see it under `except ImportError`.
- **`validate_texts`** — the input check all three adapters run before
  touching a provider.

`dimensions` and `model_id` describe the vector space, and they are the reason
the contract is worth writing down: two embedders are comparable only when both
match, and a store indexed with one space must never be queried with another.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `EmbeddingProtocol` | `runtime_checkable` Protocol: `dimensions: int`, `model_id: str`, `embed(texts) -> list[Vector]` | satisfied structurally, so any duck-typed object conforms |
| `Vector` | `list[float]` | an alias, not a `NewType`, so it stays interchangeable |
| `EmbeddingError` | base exception | catching it catches the whole area |
| `EmbeddingInputError` | `EmbeddingError` | the caller passed something unusable |
| `EmbeddingProviderError` | `EmbeddingError` | the provider failed |
| `MissingEmbeddingDependency` | **`ImportError`** | an optional package is absent (FR-3) |
| `validate_texts(texts)` | `-> list[str]` | validates and normalises input |

### Invariants the tests pin down

- **A bare `str` is rejected.** `str` *is* a `Sequence[str]`, so accepting it
  would embed one vector per character and hand the caller four hundred
  vectors for a one-word query. The message says so and suggests the fix.
- **A bare `bytes` is rejected**, for the same reason.
- **An empty list is not an error** — it is the caller saying there is nothing
  to embed, and the answer is `[]`. Note the distinction from a list
  *containing* an empty string, which **is** rejected: an empty chunk means the
  pipeline upstream produced nothing, and embedding it would add a vector with
  no meaning to the index.
- **A non-string element is rejected by position** (`texts[1]`), so the message
  points at the offending item.
- **`validate_texts` returns a `list`** even when handed a tuple, so callers
  can index and slice it safely.

## Example

See `example.py`, which runs standalone — from this directory:

```bash
python example.py
```

## Dependencies

- **Components:** none.
- **Packages:** standard library only (`typing`). No optional packages, no
  network, no imports from the rest of the library.

## Trade-offs

**Use it when** you are writing or consuming anything that embeds: type
against the Protocol, catch `EmbeddingError`.

**It costs** nothing at runtime — Protocols are erased at runtime, and
`runtime_checkable` only adds an attribute-presence check for `isinstance`.

**Why not reuse `core_types.EmbeddingProtocol`?** This component's manifest
declares no dependencies, so it must be copy-ready on its own; importing
`core_types` would drag that component along. The two agree on the attribute
and method names, so they are interchangeable — the same deliberate,
temporary duplication `core-types` documents for its own `Chunk`.

## Known limitations

**`runtime_checkable` checks presence, not signatures.** `isinstance(x,
EmbeddingProtocol)` confirms `dimensions`, `model_id` and `embed` exist; it
cannot confirm `embed` accepts a sequence of strings. A wrong signature fails
at call time, not at the check.

**`validate_texts` is a policy, not a standard.** It rejects empty strings on
the grounds that an empty chunk means something went wrong upstream. If your
pipeline deliberately embeds empty text, do not use it.

**No retry, backoff or rate-limit logic.** Those belong to the adapter that
knows the provider, not to the contract.

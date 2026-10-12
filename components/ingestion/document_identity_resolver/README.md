# document-identity-resolver: the filename is not an identity

```
document_identity_resolver.py   hash_file() and resolve(); stdlib only
example.py                      runnable: a duplicate, a rename, and an edit
tests/                          success paths and deliberate failure paths
manifest.json                   schema-valid; no dependencies at all
```

## Purpose

Two questions get collapsed into one far too often: *which bytes are these*,
and *which file is this*. They have different answers, and the difference is
the whole subject of this component.

- **Content identity** is a SHA-256 of the file's bytes. Identical bytes give
  an identical identity wherever they sit on disk, which is what makes
  re-ingestion idempotent -- a second ingest replaces rather than duplicates --
  and what makes cross-file duplicates findable at all.
- **File identity** is the resolved path, size and mtime. This is what changes
  when a document is edited, and no filename has ever been able to answer it:
  two different files can share a name, and one file can be renamed without a
  single byte changing.

`document_id` is `sha256[:32]`, which is the same rule as `make_document_id`
in the `settings` component and the same rule `core-types` redefines for the
storage contracts. See "Known limitations" for why it is written out a third
time rather than imported.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `make_document_id(sha256) -> str` | | `sha256[:32]`; refuses anything but 64 lowercase hex |
| `DocumentIdentityResolver()` | | holds an in-memory index of what it has seen |
| `.hash_file(path) -> ContentIdentity` | | reads in 1 MiB chunks; never loads the file whole |
| `.same_content(a, b) -> bool` | | the direct duplicate question |
| `.resolve(path) -> Resolution` | | identify, and report duplicates and prior versions |
| `.paths_for_document(document_id) -> tuple[str, ...]` | | every path holding that identity |
| `ContentIdentity` | frozen: `sha256`, `size_bytes`, `.document_id` | which bytes these are |
| `FileIdentity` | frozen: `path`, `size_bytes`, `mtime_ns`, `content` | which file this is |
| `Resolution` | frozen: `identity`, `version`, `previous_sha256`, `identical_content_at` | identity plus history |
| `IdentityError` | base exception | catches every failure below |
| `MissingFile` / `UnreadableFile` | `IdentityError` | could not be read |
| `InvalidDigest` | `IdentityError` | a supplied digest was not 64 lowercase hex |

## Example

```python
from document_identity_resolver import DocumentIdentityResolver

resolver = DocumentIdentityResolver()

first = resolver.resolve("uploads/report.pdf")
print(first.identity.document_id, first.version)     # e.g. 4f2a... 1

# The same bytes, saved elsewhere.
again = resolver.resolve("archive/report (copy).pdf")
print(again.identical_content_at)                    # ('...uploads/report.pdf',)

# The original is edited. It is now a different document, and says so.
edited = resolver.resolve("uploads/report.pdf")
print(edited.version, edited.previous_sha256)        # 2, the digest being retired
```

## Dependencies

None. Not one package, required or optional -- `hashlib`, `re`, `dataclasses`
and `pathlib` are all standard library.

## Trade-offs

**Use it when** you ingest from a directory that may be re-scanned, synced or
duplicated, and you need re-ingesting to be a no-op rather than a pile-up.

**It costs** an in-memory index that is per-instance and disposable. There is no
`to_file`, deliberately: a persisted identity index would become a second
source of truth competing with the manifests, and anything that can be rebuilt
by re-resolving should be. If you need the index to survive a process, persist
the digests yourself and rebuild from them.

**Alternatives:** hashing the (path, size, mtime) tuple instead. That is
cheaper -- no read -- but it identifies a *file*, not a document, so re-saving
a file without changing its content mints a new identity and re-ingests
everything. It is the wrong trade for this library.

**When not to use it:** as a content-addressed store reference. `document_id`
is 32 hex characters, not the full digest; if you need collision resistance
beyond what a truncated SHA-256 gives you across your corpus, key on
`content.sha256` instead.

## Known limitations

- **The id rule is written out three times.** `make_document_id` also exists in
  `config/models.py` (the `settings` component) and, field-for-field, in
  `storage/core_types/core_types.py`. AC-13 forbids a component importing
  another component, and this component has to be usable on its own -- so the
  truncation is restated here, documented as equivalent, and pinned by
  `test_the_document_id_matches_the_library_rule`. If that test ever fails, three
  components have started disagreeing about what a document id is, and every
  vector stored under the earlier rule is orphaned.
- **Version is per path, not per document.** Because the id is content-derived,
  editing a file yields a *new* id, so "version 2 of document X" is not a thing
  this component can express -- the document is Y. What it reports instead is
  how many distinct states that path has been in, plus the digest being
  retired, so a caller can retire the old one deliberately.
- **A rename resets the history.** File identity is (path, size, mtime) by
  design, so a renamed file is a *new* path and resolves as version 1 with
  `previous_sha256 = None` -- even though its content never changed. Continuity
  across a rename has to come from the content identity instead, which does
  survive: `resolve()` before and after returns the same `document_id`. This is
  why `example.py` renames before it edits rather than after. If you need
  edit-history across renames, key your own history on `document_id`.
- **Going back to a previous state does not renumber.** A path that has held
  three distinct contents reports version 3 even after reverting to its first,
  because the number counts states held rather than edits made.
- **The index is not a lock.** Two processes resolving concurrently each hold
  their own view; neither sees the other's duplicates.
- **Hashing reads the whole file.** There is no fast path, because a partial
  hash is a different identity. Large files cost a full read.

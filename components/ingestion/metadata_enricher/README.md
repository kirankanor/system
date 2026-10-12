# metadata-enricher: provenance it can account for, and nothing else

```
metadata_enricher.py   enrich() is pure; stdlib only in the required path
example.py             runnable: an enriched record and four dropped fields
tests/                 success paths and deliberate failure paths
manifest.json          schema-valid; pypdf optional, never required
```

## Purpose

An ingestion pipeline has to answer *where did this come from*, and the answer
has two different sources that are usually stirred together: what the document
said about itself, and what the pipeline knows because it is the one doing the
ingesting. This component keeps them apart and refuses to let the second
overwrite the first.

The load-bearing rule is negative: **it never supplies a value it cannot
account for.** No default title, no "unknown" author, no timestamp read from a
clock. An invented author is worse than a missing one, because it arrives
looking like evidence. A field that could not be used is dropped and named in
`MetadataBundle.omitted`, so nothing disappears silently either.

`ingested_at` is a **required constructor argument** rather than something this
component reads. A wall-clock value would make re-ingesting the same file
produce different metadata on every run, which contradicts the one property
content-derived document ids exist to guarantee: that re-ingesting replaces
rather than duplicates. Passing the time in keeps enrichment pure, and leaves
the question of *when* ingestion happened to whoever ran it.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `MetadataEnricher(*, ingested_at, system_prefix="system_")` | | required timestamp; no clock, no default |
| `.enrich(*, source_name, source_metadata=None, title=None, author=None, page_number=None, content_sha256=None) -> MetadataBundle` | | pure: no file I/O, no clock, no network |
| `read_document_info(path) -> dict` | | optional; pulls title/author from a PDF's own info dictionary |
| `MetadataBundle` | frozen: `extracted`, `system`, `source`, `merged`, `omitted` | the four bags plus what was dropped |
| `MetadataBundle.title` / `.author` | `-> str \| None` | read through `extracted`; `None` when absent |
| `MetadataValue` | `str \| int \| float \| bool` | flat scalars only |
| `SYSTEM_PREFIX` | `"system_"` | where every computed key is namespaced |
| `MetadataError` | base exception | catches both failures below |
| `MalformedSourceMetadata` | `MetadataError` | an unusable container, or a bad `ingested_at` |
| `MissingOptionalDependency` | `MetadataError` | `pypdf` asked for and absent |

## Example

```python
from metadata_enricher import MetadataEnricher

enricher = MetadataEnricher(ingested_at="2026-10-11T09:30:00+00:00")

bundle = enricher.enrich(
    source_name="uploads/report.pdf",
    source_metadata={"language": "en", "pages": 12},
    author="A. Author",
    page_number=3,
)

bundle.extracted   # {'author': 'A. Author', 'page_number': 3}
bundle.system      # {'system_source_path': ..., 'system_ingested_at': ...}
bundle.source      # {'language': 'en', 'pages': 12}  -- verbatim
bundle.merged      # all four, with source winning a collision
bundle.omitted     # ('title: not supplied',) is NOT here: absent is not a fault

# title was never supplied, so it is absent rather than blank:
assert "title" not in bundle.merged
```

## Dependencies

None required in the enrichment path -- `enrich()` does not touch a file, so it
cannot need a parser. `pypdf` (>=4.0.0) is optional and reached only through
`read_document_info()`, which is a way to *obtain* a title or author before
passing it in, not something enrichment calls behind your back. Without it the
refusal names the package and the install command.

## Trade-offs

**Use it when** anything downstream filters, groups or cites on metadata, and
you need to be able to say which fields came from the document and which the
pipeline asserted.

**It costs** four bags instead of one flat dict. `merged` is the flat dict for
storage, so the cost is paid only if you want the provenance -- but the
provenance is the point. A single merged dict cannot tell you afterwards
whether an author was the document's or yours.

**Malformed fields are dropped, not repaired.** A title of `42` becomes an
omitted entry rather than `"42"`. Coercing it would be inventing a value in
disguise, which is the failure this component exists to avoid.

**Alternatives:** enriching in the extractor, which would couple metadata shape
to a particular file format; or stamping defaults so every record is
complete-looking, which is what this component deliberately will not do.

**When not to use it:** as a validator. It reports what it dropped, but
`security/metadata-validation` is the component that decides whether a bundle
meets a declared schema. Enrichment produces; validation checks.

## Known limitations

- **It validates shape, not truth.** A caller may pass `author="Nobody"`, and
  that is stored. Nothing here checks whether a supplied value is correct,
  because nothing here can know.
- **`ingested_at` is not parsed.** It must be a non-empty string and is stored
  verbatim; whether it is ISO-8601 is the caller's business. That keeps the
  component from assuming a timezone or a clock it does not have.
- **Source metadata that is not a flat scalar is dropped, not stored.** Nested
  lists and dicts cannot be filtered on by most vector databases, so they are
  removed and named in `omitted` rather than accepted and rejected at query
  time. A whole entry is kept if the *container* is bad only when the
  container is not a mapping at all, which does raise.
- **Source wins a collision, always.** If the caller passes `title="X"` and the
  document says `title="Y"`, `merged` holds `"Y"`. This is the requirement
  taken literally -- preserve original source metadata -- but it means a
  corrected title must be recorded in the source bag, not passed as an argument.
- **`page_number` refuses `True`.** `bool` subclasses `int`, so `True` would
  otherwise be accepted as page 1. This matches the rule
  `security/metadata-validation` already enforces, so a bundle produced here
  validates there.
- **Optional PDF reading assumes a well-formed file.** `read_document_info()`
  reports an unreadable PDF as `MalformedSourceMetadata`; it does not attempt
  repair.

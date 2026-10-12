# Component authoring

What a component is made of, and what it has to satisfy before it is accepted.

## Layout

Every component is one directory under `components/<category>/<name>/`:

```
components/ingestion/text-normalizer/
  text_normalizer.py        the implementation -- one file
  manifest.json             the canonical description
  README.md                 purpose, API, example, dependencies, trade-offs
  example.py                runnable, no arguments, exits 0
  tests/test_component.py   standalone; imports the component, not the package
```

The module name matches the directory name with hyphens replaced by
underscores. `example.py` and the test file are conventional rather than
required by the schema, but every shipped component has both, and the cookbook
links to them.

## The manifest

`manifest.json` is the canonical description. Fifteen fields are required and
none of them has a silent default — a missing field is an error naming the
field, not an empty value.

| Field | Notes |
|---|---|
| `id` | lowercase kebab-case: `^[a-z0-9]+(?:-[a-z0-9]+)*$` |
| `name` | human title, e.g. `Text Normalizer` |
| `category` | one of eleven: `ingestion chunking embeddings retrieval reranking context generation evaluation security storage config` |
| `source` | the module file, relative to the manifest |
| `version` | `^\d+\.\d+\.\d+$` |
| `dependencies` | ids of other components this one needs |
| `description` | one sentence; it appears in `ragkit list` |
| `destination` | where `add` copies the source; relative, forward slashes, no `..` |
| `status` | `stable`, `experimental`, or `v2-deferred` |
| `requires_python` | e.g. `{"min": "3.12"}` |
| `required_packages` | packages a user must install to use it |
| `optional_packages` | packages needed only for an optional feature |
| `external_services` | any network service it expects |
| `install` | the instruction a user follows |
| `missing_dependency_message` | what to say when an optional import fails |

`documentation`, `tests` and `tradeoffs` are optional paths.

The `destination` follows `src/{project_package}/rag/<category>/<name>.py`.
`{project_package}` is substituted with the destination project's own package
at add time.

### Dependency declarations are data, not instructions

`required_packages` and `optional_packages` exist so `ragkit add` can *print*
what you would need. `add` never installs anything and never edits a dependency
file — that is a hard rule, not a default.

An optional package must be paired with a **specific** `ImportError`. When a
user reaches an optional feature without the package, the message has to name
the package and the install command. A bare `ImportError` is not acceptable:
the user cannot act on it.

```python
try:
    import magic
except ImportError as exc:
    raise MissingOptionalDependency(
        "python-magic is optional and is not installed. Install it with "
        "`pip install python-magic`."
    ) from exc
```

## The README

The cookbook website parses each component's README into sections. Five of them
are read out of the prose and rendered as tabs:

| Site section | Heading it reads | Matching |
|---|---|---|
| Purpose | `## Purpose` | also accepts `## Quick use`, `## Usage` |
| Public API | `## Public API` | exact |
| Example | `## Example` | exact |
| Trade-offs | `## Trade-offs` | exact |
| Limitations | `## Known limitation`, `## Limitations` | prefix |

The prefix and alias matching exists because the READMEs agree on the words
but not the punctuation: two components head the section
`## Known limitation: characters, not tokens` rather than
`## Known limitations`. Matching exactly would drop both silently and the site
would render "Not recorded" for a section that plainly *is* recorded — the
failure mode that matters, because nothing else would flag it.

`## Dependencies` is worth having for a reader, but it is not read into the
site: dependency data comes from the manifest, where it can be validated.

### A missing heading is silent

A heading the cookbook cannot find yields an empty section — **the build does
not fail.** That is worse than an error, because an empty tab looks like a
finished page.

This is not hypothetical. Two of the 38 components render an empty section
today:

- `settings` (`components/config/README.md`) — has no `## Public API`,
  `## Example` or `## Trade-offs`, so three tabs are empty.
- `storage-adapters` (`components/storage/adapters/README.md`) — has no
  `## Example`, so that tab is empty.

They are recorded here rather than quietly patched, because the point of NFR-8
is that a claimed feature either exists or has an explicitly recorded status.

### Limitations

`## Known limitations` is not optional in spirit. All 38 components record one
in one of the two accepted forms, and a README without one is either wrong or
describing a trivial component.

Prose with a heading but no content counts as present to the parser and empty
to the reader — so write the section, do not just label it.

## The tests

Component tests must be **standalone**:

```console
$ python -m pytest components/ingestion/text_normalizer/tests -q
passed
```

Two rules:

1. **They may not import `rag_code_library`**, and may not import another
   component. A component test that reaches into the package would hide the
   very coupling the components forbid. Each test file inserts its own
   component directory on `sys.path` and imports the module by name.
2. **At least one success path and at least one failure path.** A test module
   that only checks the happy path cannot tell you the component refuses what
   it claims to refuse.

Tests are not run against an installed copy; they run against the working tree,
so there is no build step between editing and testing.

## Validation

Before anything else, the manifest has to be valid:

```console
$ python scripts/validate_components.py
manifest(s) valid
```

That check covers the schema, unique ids, that every declared file actually
exists, that dependencies name loaded components, and that destinations are
safe.

If you add a component, the committed cookbook goes stale, because the
generator enumerates every registered component. Regenerate it:

```console
$ python scripts/build_cookbook.py --check
OK: data.js matches the canonical manifests byte-for-byte
```

If that fails, run `python scripts/build_cookbook.py` to rebuild, then check
again.

## Review

Two things a reviewer looks for that no gate can catch:

- **Does the component refuse what its README says it refuses?** A failure path
  that is not tested is a claim, not a behaviour.
- **Does it do one job?** A component that validates *and* normalises *and*
  persists is three components, and a user cannot take one of them.

AI authors the code; a human reviews it and requires independent verification.
That is a standing rule for this repository, not a step in a checklist.

See also: [`architecture.md`](architecture.md), [`security.md`](security.md).

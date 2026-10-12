# RAG Code Library

Copy-ready RAG components, a `ragkit` CLI that copies them into a project and
runs them as an ingestion pipeline, and a local cookbook website that documents
every one of them.

The library is a **collection of independent components**, not a framework. Each
component is a single Python file with no required dependency beyond the standard
library, a manifest describing it, its own README, an example, and tests. You copy
what you need into your project and own it from then on.

This is a personal, local-use tool. It is not published to PyPI and it has no
hosted backend.

## Contents

- [Setup](#setup)
- [Commands](#commands)
- [Examples](#examples)
- [Architecture](#architecture)
- [The cookbook website](#the-cookbook-website)
- [Testing](#testing)
- [Adding a component to your project](#adding-a-component-to-your-project)
- [Limitations](#limitations)

## Setup

Python 3.12 or newer is required. The project uses [uv](https://docs.astral.sh/uv/)
for its own environment, but nothing about it requires uv.

```console
$ python --version
Python 3.
```

The example above only asserts that a Python 3 interpreter is present, not
which one: the suite runs under 3.12 and 3.14, and pinning a version string in
the documentation would make the docs fail on a machine that is merely newer.

To build the development environment:

```console
$ uv sync
```

`uv sync` is not run by the documentation checker because it mutates your
environment and needs the network. Run it once, by hand.

The `ragkit` console script is installed into the virtual environment by
`uv sync`. From here on, examples show `ragkit` as though it were on your
`PATH`; on your machine it is `uv run ragkit`, or `.venv\Scripts\ragkit` on
Windows.

**`ragkit` must be run from inside this repository**, because it locates the
component manifests by walking up from the current directory. Run from anywhere
else it exits `1` and says so:

```console
$ ragkit list --category nope # exit:1
ragkit: unknown category 'nope'; known categories: chunking, config, context
```

## Commands

`ragkit` has four commands. `--version` and `--help` work as expected.

### `ragkit list`

Lists every component in the canonical registry, which is the single source of
component metadata. `--category` narrows the listing.

```console
$ ragkit list --category ingestion
id                          category     status         description
```

### `ragkit info ID`

Shows one component in full: description, dependencies, what it requires, and
where it would land.

```console
$ ragkit info text-normalizer
text-normalizer - Text Normalizer
```

An unknown id exits `1` and lists every id that does exist, so a typo is
self-correcting:

```console
$ ragkit info text-normaliser # exit:1
ragkit: unknown component 'text-normaliser'; known ids: answer-metrics
```

### `ragkit add ID`

Copies a component into a destination project. This is the only command that
writes files.

```
usage: ragkit add [-h] [--destination PATH] [--package NAME] [--dry-run]
                  [--force]
                  ID
```

- `--destination PATH` — project root to copy into (default: current directory).
- `--package NAME` — the Python package the component lands in. Detected from
  the destination's `pyproject.toml` when omitted; pass it explicitly for a
  destination that has no package yet.
- `--dry-run` — print the file operations and exit without writing.
- `--force` — overwrite files that already exist.

### `ragkit ingest SOURCE DESTINATION`

Runs the ingestion pipeline over every document under `SOURCE`, writing the
index and a run summary into `DESTINATION`.

```
usage: ragkit ingest [-h] [--stages MODULE:FACTORY] [--failure-policy POLICY]
                     [--fingerprint TEXT] [--summary PATH] [--quiet]
                     SOURCE DESTINATION
```

- `SOURCE` — directory holding the documents to ingest.
- `DESTINATION` — directory the pipeline writes into. Created if absent, and
  the place the run lock and the resume checkpoints live.
- `--stages MODULE:FACTORY` — the stages to run, as a dotted reference to a
  function that returns them. Overrides `RAG_PIPELINE__STAGES`.
- `--failure-policy POLICY` — `stop`, `skip` or `quarantine`: what one
  document's failure does to the rest of the run.
- `--fingerprint TEXT` — the configuration fingerprint recorded against each
  document, so a later run can tell whether its setup changed.
- `--summary PATH` — where to write the run summary (default
  `DESTINATION/run-summary.json`).

**This command ships no default pipeline, and that is deliberate.** The
orchestration layer imports no component and no parsing library, so there is
nothing built in for it to import back. The stages come from your project, as
a function that returns them, named in configuration:

```
export RAG_PIPELINE__STAGES="my_project.pipeline:build_stages"
```

With nothing configured it says so, rather than inventing a pipeline or
silently doing nothing:

```console
$ ragkit ingest {scratch} {scratch}/index # exit:1
ragkit: no stages are configured; set RAG_PIPELINE__STAGES
```

A stage receives the pipeline's state, reads what an earlier stage produced,
and returns its own output. Discovery, loading, chunking, embedding, writing
and verification are all just stages, so a project picks its own parser,
chunker and vector store by returning them from its factory. The repository's
own thirteen-stage assembly, built entirely from the shipped components, is
`tests/integration/pipeline_stages.py` — a worked example, not a dependency.

A run is resumable. `DESTINATION` holds a checkpoint per document recording
its content hash, the configuration fingerprint, and how far it got; re-running
after an interruption picks up where the work stopped, and re-running over
unchanged content does nothing at all. Two runs over the same source at once
are refused rather than interleaved.

## Examples

### Dry-run first

`--dry-run` shows exactly what would be copied and writes nothing:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg --dry-run
would copy 4 file(s) for text-normalizer:
Installed nothing, and changed no dependency file.
```

`ragkit add` **never installs packages and never edits a dependency file.** It
copies source and prints what you would need to install yourself. This is
deliberate: a tool that silently edits your `pyproject.toml` or `uv.lock` is a
tool you cannot trust in CI.

### Add for real

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
```

The component's source lands on the path its manifest declares, and its
README, tests and manifest land in a namespace beside it:

```
{scratch}/src/mypkg/rag/ingestion/text_normalizer.py
{scratch}/src/mypkg/rag/ingestion/text-normalizer/README.md
{scratch}/src/mypkg/rag/ingestion/text-normalizer/manifest.json
{scratch}/src/mypkg/rag/ingestion/text-normalizer/tests/test_component.py
```

You now own those files.

### Existing files are never overwritten silently

Adding a component whose files are already there fails rather than clobbering
them, and tells you which files are in the way. The destination is left
untouched, so a refusing `add` needs no cleanup. Overwriting is opt-in:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
$ ragkit add text-normalizer --destination {scratch} --package mypkg # exit:1
already exist
$ ragkit add text-normalizer --destination {scratch} --package mypkg --force
copied 4 file(s) for text-normalizer:
```

### Adding a component with dependencies

`add` resolves the whole dependency graph. `dense-retrieval` depends on
`core-types`, so adding it copies both, source and metadata for each:

```console
$ ragkit add dense-retrieval --destination {scratch} --package mypkg --dry-run
would copy 8 file(s) for dense-retrieval:
```

## Architecture

Five things, each with one owner:

| Piece | Location | Owns |
|---|---|---|
| Components | `components/<category>/<name>/` | implementation, README, example, tests |
| Manifests | `components/<category>/<name>/manifest.json` | the canonical description of one component |
| Registry | `src/rag_code_library/registry/` | discovering and validating manifests |
| CLI | `src/rag_code_library/cli.py` | `list`, `info`, `add` |
| Cookbook generator | `scripts/build_cookbook.py` | turning manifests + READMEs into the website |

**Client independence is the central design rule.** A component may not import
another component or the `rag_code_library` package. That is what "copy-ready"
means: copying one file into a project has to be enough. Where a type has to
exist in two places — as it does for the storage contracts — the duplication is
deliberate, documented in both copies, and pinned by a test.

Every filesystem write from the CLI goes through one path-safety module, which
rejects traversal, project-root escape and symlink escape.

The detailed version is in [`docs/architecture.md`](docs/architecture.md).

## The cookbook website

`scripts/build_cookbook.py` generates a static site from the manifests and the
component READMEs. It is deterministic: the same sources always produce
byte-identical output, and there are no timestamps in the generated files.

```console
$ python scripts/check_cookbook_counts.py
OK: every section lists the components the registry declares
```

```console
$ python scripts/build_cookbook.py --check
OK: data.js matches the canonical manifests byte-for-byte
```

To view it:

```console
$ python -m http.server 8000
```

then open <http://localhost:8000/cookbook/> in a browser. Serving it is a
long-running command, so it is the one documented example the checker does not
execute — it is also the manual check owed for AC-15.

## Testing

The suite runs against the working tree, not an installed copy, so it needs no
build step.

```console
$ python -m pytest -q
passed
```

Component tests live beside their component and deliberately do not import
`rag_code_library` — importing the package from a component test would hide the
very coupling the components forbid.

The gates that stand behind the components:

```console
$ python scripts/validate_components.py
manifest(s) valid
```

## Adding a component to your project

`ragkit add` is for copying *out of* this library. To add a component *to* the
library, see [`docs/component_authoring.md`](docs/component_authoring.md).

## Limitations

These are real and current, not a roadmap.

**Deferred to v2.** The `observability` and `integrations` categories exist in
the category enumeration and are marked `v2-deferred`. No component ships in
them, and `ragkit list` shows none.

**No hosted anything.** There is no server, no API, no database and no
telemetry. The cookbook is files on disk.

**Not published.** No PyPI package, no semantic versioning contract, no
compatibility guarantee between versions. `version` in a manifest describes a
component's own maturity, not a release of the library.

**`ragkit` is not installed system-wide**, and it must run from inside this
repository because it discovers manifests by walking up from the current
directory.

**Optional dependencies are your problem, by design.** A component that needs a
heavy library declares it as optional and raises a specific `ImportError`
naming the package when you use that feature without installing it. `ragkit add`
will not install it for you.

**Documentation is checked, but the browser is not.** Every shell command in
this README and in `docs/` is extracted and executed by
`scripts/check_documented_commands.py`. The one thing that cannot be verified
unattended is opening the cookbook in a browser, which remains a human check.

## License

Personal project. No license is granted.

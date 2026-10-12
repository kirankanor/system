# CLI reference

`ragkit` lists, inspects and adds copy-ready RAG components, and runs them as
an ingestion pipeline.

```
usage: ragkit [-h] [--version] <command> ...
```

**Run it from inside this repository.** `ragkit` finds the component manifests
by walking up from the current directory; run from anywhere else it exits `1`.

## Exit codes

| Code | Meaning | Example |
|---|---|---|
| `0` | success | any command that did what was asked |
| `1` | runtime error | unknown component, files already exist, not run from the repository |
| `2` | usage error | bad flags, or a required argument missing |

A usage error means the command was *written* wrong, and argparse reports it:

```console
$ ragkit info # exit:2
usage: ragkit info [-h] ID
```

A runtime error means the command was written correctly and something about
the world was not what it assumed:

```console
$ ragkit info definitely-not-a-component # exit:1
ragkit: unknown component 'definitely-not-a-component'; known ids: answer-metrics
```

Both go to stderr. Both are single-line `ragkit: ...` messages rather than
tracebacks, so they are readable in a log.

## `ragkit --version`

```console
$ ragkit --version
ragkit 0.1.0
```

## `ragkit list [--category NAME]`

Lists every component in the canonical registry.

```
usage: ragkit list [-h] [--category NAME]
```

`--category` narrows the listing to one of the eleven categories.

```console
$ ragkit list --category ingestion
id                          category     status         description
```

An unknown category exits `1` and names every category that does exist:

```console
$ ragkit list --category nope # exit:1
ragkit: unknown category 'nope'; known categories: chunking, config, context
```

## `ragkit info ID`

Shows one component in full: its description, dependencies, what it requires,
where it would land, and its install instructions.

```
usage: ragkit info [-h] ID
```

```console
$ ragkit info text-normalizer
text-normalizer - Text Normalizer
```

An unknown id exits `1` and lists every id that exists, so a typo is
self-correcting without a second command:

```console
$ ragkit info text-normaliser # exit:1
ragkit: unknown component 'text-normaliser'; known ids: answer-metrics
```

## `ragkit add ID`

Copies a component into a destination project. This is the only command that
writes files.

```
usage: ragkit add [-h] [--destination PATH] [--package NAME] [--dry-run]
                  [--force]
                  ID
```

| Argument | Effect |
|---|---|
| `ID` | the component to add; required |
| `--destination PATH` | project root to copy into; defaults to the current directory |
| `--package NAME` | Python package the component lands in; detected from the destination's `pyproject.toml` when omitted |
| `--dry-run` | print the file operations, write nothing |
| `--force` | overwrite files that already exist |

### Package detection

When `--package` is omitted, `add` works out the package from the destination's
`pyproject.toml` (`[project].name`) or, failing that, from a single package
under `src/`. If it cannot tell, it stops and asks rather than guessing — a
guess here would scatter files into a directory that is not your package:

```console
$ ragkit add text-normalizer --destination {scratch} # exit:1
cannot tell what package
```

Passing `--package` always works:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
```

### Dry-run semantics

`--dry-run` prints the same plan `add` would execute, in the same order, and
then writes nothing at all. It is not a partial apply and it creates no
directories — the destination is exactly as it was.

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg --dry-run
would copy 4 file(s) for text-normalizer:
Installed nothing, and changed no dependency file.
```

The plan names each file with the component that owns it, so a dry run over a
destination you do not fully trust is worth reading.

### Transitive dependencies

`add` resolves the dependency graph before planning, so adding one component
copies everything it needs:

```console
$ ragkit add dense-retrieval --destination {scratch} --package mypkg --dry-run
would copy 8 file(s) for dense-retrieval:
```

`dense-retrieval` depends on `core-types`, so both arrive — source, README,
tests and manifest for each.

### Overwrite safeguards

If any file `add` would write already exists, the command stops **before
writing anything**, lists the conflicting files, and reports who owns each and
whether the existing file already matches. The destination is left untouched —
`add` is all-or-nothing, so a refusing run needs no cleanup.

The three states, in sequence against one destination:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
$ ragkit add text-normalizer --destination {scratch} --package mypkg # exit:1
already exist
$ ragkit add text-normalizer --destination {scratch} --package mypkg --force
copied 4 file(s) for text-normalizer:
```

Overwriting is explicit and opt-in with `--force`. A file that is already
byte-identical to the component's is reported as such rather than as a
conflict, so re-adding an unchanged component tells you nothing has drifted.

### What `add` will never do

- **It never installs packages.** No `pip`, no `uv add`, no package manager is
  invoked, ever.
- **It never edits a dependency file.** `pyproject.toml` and `uv.lock` are
  never written to. Required and optional packages are *printed* so you can act
  on them yourself.
- **It never writes outside the destination root.** Every write is checked by
  the path-safety module, which rejects traversal, root escape and symlink
  escape.

The second point is deliberate. A tool that silently edits your lockfile is a
tool you cannot run in CI without reviewing a diff first.

## `ragkit ingest SOURCE DESTINATION`

Runs the ingestion pipeline over every document under `SOURCE`, writing the
index and a run summary into `DESTINATION`.

```
usage: ragkit ingest [-h] [--stages MODULE:FACTORY] [--failure-policy POLICY]
                     [--fingerprint TEXT] [--summary PATH] [--quiet]
                     SOURCE DESTINATION
```

Exit codes are the same as every other command: `0` when every document
completed, `1` when the configuration was unusable or a document failed.

### Where the stages come from

`ingest` runs a list of stages that a factory function returns. Naming that
factory is the one required piece of configuration:

```
RAG_PIPELINE__STAGES   module:factory   dotted path to the function
RAG_PIPELINE__FAILURE_POLICY   stop | skip | quarantine
RAG_PIPELINE__FINGERPRINT       text recorded against each document
```

Every one of them has a flag equivalent; the flag wins over the environment.

No stages are configured by default, and this is not an omission. The
orchestration layer imports no component and no parsing library, so a built-in
pipeline would have to import them, which is precisely the coupling the design
exists to avoid. Without configuration the command says so:

```console
$ ragkit ingest {scratch} {scratch}/index # exit:1
ragkit: no stages are configured; set RAG_PIPELINE__STAGES
```

The factory returns `Stage` objects, each a name and a callable. Stages are
either corpus-scoped, running once over the whole source, or document-scoped,
running once per document. Discovery is where a document can be refused;
refusal recorded there means no later stage is ever handed the file.

The repository's own assembly of the shipped components into thirteen stages
lives in `tests/integration/pipeline_stages.py`. It is a worked example of the
convention, not something `ingest` reaches for.

### What a run writes

| Path | Contents |
|---|---|
| `DESTINATION/run-summary.json` | counts, per-stage status and timings, per-document outcome |
| `DESTINATION/_ragkit.checkpoints.json` | one record per document: content hash, fingerprint, stages completed |
| `DESTINATION/_ragkit.lock` | present only while a run is in progress |

The summary carries no document text and no credentials — counts, names,
durations and short failure reasons only. A document is identified by the hash
of its content, so renaming a file does not reprocess it and editing one does.

### Resuming

A run interrupted partway through leaves its checkpoints behind. Running again
picks up at the first stage that never finished, rather than starting over. A
fingerprint that no longer matches the recorded one invalidates the record and
forces the document to be reprocessed, which is the behaviour to expect after
changing chunking or embedding configuration.

A second run while one is already in progress is refused, naming the run that
holds the lock and the process holding it.

### What `ingest` will never do

- **It never installs packages.** No extraction, embedding or vector library is
  installed, and no dependency file is touched.
- **It never guesses a format.** A file whose type the configured stages cannot
  handle is refused with a reason, not parsed on a hunch.
- **It never writes outside `DESTINATION`.** Every write goes through the same
  path-safety checks `add` uses.

## Where files land

The component's source goes to the path its manifest declares, which follows
the pattern `src/<package>/rag/<category>/<name>.py`. Its README, tests and
manifest land in a namespace beside it:

```
src/mypkg/rag/ingestion/text_normalizer.py
src/mypkg/rag/ingestion/text-normalizer/README.md
src/mypkg/rag/ingestion/text-normalizer/manifest.json
src/mypkg/rag/ingestion/text-normalizer/tests/test_component.py
```

After that they are yours. Upgrading means re-adding with `--force` and
reviewing the result.

See also: [`architecture.md`](architecture.md),
[`troubleshooting.md`](troubleshooting.md).

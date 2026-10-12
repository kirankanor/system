# Troubleshooting

Symptoms you can actually hit, and what each one means. Every message below is
reproduced from a real run.

## Setup

### `ragkit` is not recognized

`ragkit` is a console script inside the project's virtual environment, not a
global program. It becomes available after `uv sync`, and only while that
environment is active — `uv run ragkit`, or `.venv\Scripts\ragkit` on Windows.

It is deliberately not installed system-wide.

### `python: command not found` during setup

Python 3.12 or newer is required. If `python` resolves to an older interpreter,
point `uv` at the right one explicitly rather than relying on the default.

### Which Python is the suite actually running under

```console
$ python --version
Python 3.
```

The suite is exercised under both 3.12 and 3.14. If a test passes on one and
fails on the other, that is a real portability bug and should be reported as
one, not worked around.

## CLI

### `no 'components' directory found above ...`

```
ragkit: no 'components' directory found above <path>; run ragkit from inside the
RAG Code Library repository
```

`ragkit` discovers the component manifests by walking **up** from the current
directory. It has no install-time knowledge of where the repository is, so
running it from a scratch project or anywhere outside the repository exits `1`:

```console
$ cd {scratch} && ragkit list # exit:1
no 'components' directory found above
```

**The fix is to invoke `ragkit` from inside the repository and point
`--destination` outward.** The destination may be anywhere; the invocation may
not. This is the single most common mistake with this tool.

### `unknown component '<id>'`

```
ragkit: unknown component 'text-normaliser'; known ids: answer-metrics, ...
```

Exit `1`. The full list of known ids is printed in the error, so the correct
one is in front of you without a second command:

```console
$ ragkit info text-normaliser # exit:1
ragkit: unknown component 'text-normaliser'; known ids: answer-metrics
```

If the component genuinely is missing, check whether it belongs to a
`v2-deferred` category — those exist in the enumeration but ship no
components, so they never appear.

### `unknown category '<name>'`

Exit `1`, listing all eleven valid categories:

```console
$ ragkit list --category nope # exit:1
ragkit: unknown category 'nope'; known categories: chunking, config, context
```

### The command printed a usage error

Exit `2` rather than `1` means the command itself was malformed — a missing
required argument or an unrecognised flag — and `argparse` reports it:

```console
$ ragkit info # exit:2
usage: ragkit info [-h] ID
```

The distinction matters when scripting: `1` means the world was not what the
command assumed, `2` means the command was written wrong.

### `cannot tell what package ...`

```
ragkit: cannot tell what package <destination> uses: pyproject.toml has no
[project].name and src/ holds no single package. Pass --package NAME to say
where the component should land.
```

Exit `1`. `add` resolves the destination package from `pyproject.toml`, or from
a single unambiguous package under `src/`. A destination that is empty, or has
several packages, gives it nothing to go on.

It refuses rather than guessing, because a wrong guess scatters source files
into a directory that is not your package — and unlike a bad copy, that is
awkward to notice later.

Pass the package explicitly:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
```

### `N file(s) already exist in ...`

Exit `1`. Nothing was written. `add` detects conflicts **before** it starts
copying, so a refusing run leaves the destination exactly as it was and needs
no cleanup.

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
$ ragkit add text-normalizer --destination {scratch} --package mypkg # exit:1
already exist
```

Decide deliberately: keep what is there, or overwrite with `--force`. Each
conflicting file is listed with its owner, and files already byte-identical to
the component's are called out separately — those are not drift.

### The command seemed to work but no files appeared

Run it again with `--dry-run`. A dry run prints the same plan and writes
nothing; if the plan is empty, the component resolved to no files, which means
the id or destination is not what you expected.

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg --dry-run
would copy 4 file(s) for text-normalizer:
```

## Generated code

### `ModuleNotFoundError` for an optional package

Expected, and the message is designed to tell you what to do. Optional
dependencies are never installed by `ragkit` — no copier should introduce a
package into your project on its own. For example, `pdf-extraction-pymupdf`
reports:

```
pdf-extraction-pymupdf requires 'pymupdf', which is not installed. Install it
with: uv add pymupdf
```

The message names the package *and* the command, because an `ImportError` that
does not say what is missing is a dead end.

`file-type-validator` is the deliberate counter-example: it needs no package by
default and runs on its own signature table, only asking for `python-magic` if
you construct it with `use_magic=True`.

### `libmagic is a system library rather than a wheel`

Installing `python-magic` is not sufficient — the `libmagic` C library has to
be present too, from your OS package manager. This is the case for any optional
dependency that wraps a native library, and it is why such a dependency cannot
be resolved by `ragkit` even in principle.

### `tesseract binary on PATH`

Same category of problem as above: `pdf-extraction-ocr` needs `pytesseract`
*and* the `tesseract` executable. Installing the Python package alone leaves a
runtime failure with a message that names the binary it wants.

## Cookbook

### `data.js` does not match the canonical manifests

```
OK: data.js matches the canonical manifests byte-for-byte
```

is what you want. If instead the check reports drift, a component was added,
removed or edited without regenerating the site. The generator enumerates every
registered component, so a stale `data.js` cannot pass:

```console
$ python scripts/build_cookbook.py --check
OK: data.js matches the canonical manifests byte-for-byte
```

Run `python scripts/build_cookbook.py` to rebuild, then re-run the check.

### The site loads but a component's tabs are empty

The cookbook parses each component README into `Purpose`, `Public API`,
`Example`, `Dependencies`, `Trade-offs` and `Known limitations`. A missing
heading yields an empty section — no error, which makes it look finished. Add
the heading.

### The browser check is skipped

```console
$ python -m http.server 8000
```

A long-running server never exits, so there is no exit code to assert and the
documentation checker records it as a manual check with a reason rather than
pretending to have verified it. Opening `http://localhost:8000/cookbook/` and
walking the page is a human check, and it is owed.

## Tests

### A component test fails only when the package is installed

Component tests must not import `rag_code_library`. If adding the package to
`sys.path` changes a component test's result, the test was reaching across a
boundary the components are forbidden to cross, and the failure is correct.

### A test passes locally and fails elsewhere

Tests run against the working tree, not an installed copy, so a stale editable
install is a real suspect. Re-run `uv sync`, then confirm the interpreter
matches `python --version`.

See also: [`cli_reference.md`](cli_reference.md), [`security.md`](security.md).

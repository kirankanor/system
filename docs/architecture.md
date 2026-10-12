# Architecture

How the pieces fit, who owns what, and the rules that keep a component usable
on its own.

## The five pieces

| Piece | Location | Owns |
|---|---|---|
| Components | `components/<category>/<name>/` | implementation, README, example, tests |
| Manifests | `components/<category>/<name>/manifest.json` | the canonical description of one component |
| Registry | `src/rag_code_library/registry/` | discovering and validating manifests |
| CLI | `src/rag_code_library/cli.py` | `list`, `info`, `add` |
| Cookbook generator | `scripts/build_cookbook.py` | manifests + READMEs → the website |

There are eleven categories and 38 components in them:

```console
$ ragkit list
category     status         description
```

## Client independence

The central rule, and the one every other decision follows from:

> A component may not import another component, and may not import
> `rag_code_library`.

This is what "copy-ready" has to mean. If a component needed a sibling, then
copying one file would not be enough, and `ragkit add` would be lying when it
says it copied what you need.

The cost is real and it is paid knowingly: some types exist in more than one
place. `SourceDocument`, for instance, is defined in the `settings` component
and again, field for field, in `core-types`. Both copies say so in a docstring,
and a test pins the equivalence. Where a value is derived rather than typed —
a document id being `sha256[:32]` — the rule is restated and a test asserts it
matches, because a silent divergence there would orphan every stored vector.

That duplication is preferred to a dependency for a specific reason: a
dependency would make the component unusable by itself, which is the entire
point of the library. The alternative — importing — is not available, because
AC-13 forbids it and it is checked, not merely requested.

This is enforced rather than assumed:

```console
$ python scripts/validate_components.py
manifest(s) valid
```

## The registry is the single source of metadata

`rag_code_library.registry` discovers manifests under `components/`, validates
each against the schema, and resolves cross-manifest questions that a single
file cannot answer: whether an id is a duplicate, whether a declared source
exists, whether a dependency names something loaded.

It returns `Issue` objects rather than raising. Every check reports, so one run
surfaces *all* the reasons something is wrong rather than the first, which is
what makes the output actionable instead of a sequence of re-runs.

The public surface is small on purpose: `load`, `ok`, `issues`, `manifests`,
`ids`, `count`, `get`, `resolve`. `get` returns `None` for an unknown id; only
`resolve` raises. Rendering belongs to the CLI, not to the registry — the
registry answers questions about components, it does not decide how they look
in a terminal.

## CLI generation and writing

`ragkit add` is the only command that writes files, and every write it performs
goes through `src/rag_code_library/security/path_safety.py`. That module
rejects traversal, project-root escape and symlink escape. No component and no
command writes outside the resolved destination root.

The write path is layered:

| Module | Responsibility |
|---|---|
| `project/detector.py`, `package_detector.py` | find the destination and the package it uses |
| `project/path_resolver.py` | turn a manifest's declared destination into a real path |
| `generator/planner.py` | decide what would be copied |
| `generator/conflict_detector.py` | find files that already exist |
| `generator/copier.py` | perform the writes, through path safety |
| `generator/provenance.py` | record which component owns a copied file |

Note the ordering: conflicts are detected *before* anything is written, which
is why a refusing `add` leaves the destination untouched rather than half
populated.

`generator/dependency_manager.py` resolves the transitive dependency graph, so
adding one component copies what it needs:

```console
$ ragkit add dense-retrieval --destination {scratch} --package mypkg --dry-run
would copy 8 file(s) for dense-retrieval:
```

## The cookbook build

`scripts/build_cookbook.py` reads the manifests and parses each component's
README into sections, then emits `cookbook/data.js` and the static site.

It is **deterministic**: components are sorted by id, categories are emitted in
a fixed pipeline order, and no timestamp or machine-specific path appears in
the output. The same sources always produce byte-identical bytes, which is what
makes drift checkable rather than arguable:

```console
$ python scripts/build_cookbook.py --check
OK: data.js matches the canonical manifests byte-for-byte
```

Because the generator enumerates every registry component, adding a component
makes a stale `cookbook/data.js` fail that check. Regenerating is therefore
mandatory whenever a component is added — not optional, and not something that
can be deferred.

Section counts are cross-checked separately:

```console
$ python scripts/check_cookbook_counts.py
OK: every section lists the components the registry declares
```

## Where the code is deliberately still thin

Honesty about this matters more than a tidy diagram. Outside `cli.py`,
`registry/`, `security/path_safety.py`, `generator/` and `project/`, much of
`src/rag_code_library/` is a stub. That is not an accident or an oversight: the
package exists to *deliver* components, and the components themselves are
standalone files that do not live in the package at all. Growing the package
without a requirement to would be inventing surface area for its own sake.

See also: [`cli_reference.md`](cli_reference.md),
[`component_authoring.md`](component_authoring.md),
[`security.md`](security.md).

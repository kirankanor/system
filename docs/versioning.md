# Versioning

What version numbers here mean, what they do not, and how to update a
component you have already copied out.

## Each component has its own version

Every component manifest carries a `version` in `major.minor.patch` form:

```
"version": "0.1.0"
```

It describes **that component's own maturity**, not a release of the library.
Components are independent — they may not import each other — so their versions
move independently too. `sparse-retrieval` can reach `1.0.0` while
`metadata-enricher` is still at `0.1.0`, and neither change says anything about
the other.

## There is no library-level release

The library itself has no semantic-versioning contract:

- **It is not published to PyPI.** There is no package to upgrade.
- **There are no releases.** No tags, no changelog, no compatibility promise
  between two points in time.
- **There is no migration path to follow**, because there is nothing to migrate
  between.

A version bump inside a manifest is a note to whoever reads the manifest, not a
commitment to downstream users. Treating it as a public API guarantee would be
a promise nobody made.

## Status says more than the number

`status` is the field that actually governs how much to trust a component:

| Status | Meaning |
|---|---|
| `stable` | the behaviour is settled and the tests pin it |
| `experimental` | it works and is tested, but the shape may still change |
| `v2-deferred` | declared in the category enumeration, ships no code |

Most components here are `experimental`, which is the honest label for code
that has never been asked to do anything outside its own tests. `v2-deferred`
is how `observability` and `integrations` exist as categories without anything
pretending to live in them — no placeholder files, no empty directories, just
an enum value marked as such.

## Provenance: by content, not by a record

When `ragkit add` reports who owns a file already in your destination, it
answers by **comparing bytes against the registry**, not by reading a record
file it wrote earlier.

This was a deliberate choice, and the reason is worth stating because the
record-file approach looks more obvious:

A `.ragkit.json` sidecar listing what was added drifts. It goes stale the
moment anyone edits a copied file, re-adds with `--force`, copies a component
by hand, or checks out an older revision. Every one of those leaves the file
asserting something untrue — and "which component owns this file?" is precisely
the question asked in the middle of a confusing conflict, where a confidently
wrong answer costs the most time.

There is a second, tighter reason: provenance cannot live *inside* the copied
files either. The copies are required to be **byte-for-byte identical** to
their sources, so adding a provenance header would break the check that proves
the copy was faithful.

Comparing hashes needs no state, cannot drift, and answers the question for any
file in the destination — whoever put it there, however it arrived.

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg
copied 4 file(s) for text-normalizer:
$ ragkit add text-normalizer --destination {scratch} --package mypkg # exit:1
already exist
```

The `owned by:` line in that refusal comes from the content comparison, so it
is correct even for a component you copied into the project by hand months ago.

## Updating a component you have copied

Once copied, the files are **yours**. Upstream changes do not reach you, and
there is no notification that they exist — that is the trade you accepted when
you took a copy instead of depending on a package.

To update deliberately:

1. Note the component id and what you have changed locally. Your edits are the
   part most likely to be lost, and there is no merge to catch it.
2. Re-add with `--force`:

```console
$ ragkit add text-normalizer --destination {scratch} --package mypkg --force
copied 4 file(s) for text-normalizer:
```

3. **Review the result as a diff, not as an upgrade.** `--force` overwrites;
   it does not merge. Anything you changed in those files is gone unless you
   diffed first.

Because the copy is byte-identical to its source, step 3 is tractable: any
difference you see is either your own edit or the upstream change, with no
noise from reformatting or injected headers.

## Compatibility

Two components that do not import each other cannot break each other through an
API change. That is a structural property of the independence rule, not a
discipline anyone has to maintain.

What *can* still break is a shared assumption — a value derived the same way in
two places, such as a document id being `sha256[:32]`. Where that duplication
exists it is documented in both copies and pinned by a test that asserts the
copies agree, because a silent divergence there would orphan everything stored
under the old rule.

See also: [`component_authoring.md`](component_authoring.md),
[`architecture.md`](architecture.md).

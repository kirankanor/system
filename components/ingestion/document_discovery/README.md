# document-discovery: an incremental scan that survives a broken filesystem

## Purpose

Discovery answers one question — *which files should the pipeline look at this
time?* — and it has to answer it without dying on the way there.

The naive version is `Path.rglob("*")`. That works right up until the tree
contains a symlink pointing outside the root, a directory the process cannot
open, or a socket file someone left behind, at which point it either walks off
into the filesystem or raises and returns nothing.

This component:

- **reports** every candidate file with its size and modification time,
- **records** every entry it declines to return, with a reason,
- **never follows** a symlink or junction that resolves outside the source root,
- **classifies** a new scan against a previous one as added, modified, deleted
  or unchanged.

It holds no state between calls. A previous scan comes back in as a plain
mapping, so a scan can run in a fresh process, be persisted as JSON, and be
diffed later without this component knowing or caring.

## Public API

| Name | What it is |
|---|---|
| `Discovery(source_root, *, recursive=True, allowed_extensions=None, exclude_patterns=(), sort_by="relative_path")` | Configure a scan. Validates its options at construction, so `discover()` cannot fail for a reason you could have known earlier. |
| `Discovery.discover() -> DiscoveryReport` | Walk the root. Raises only if the root itself is unavailable. |
| `Discovery.delta(previous) -> DiscoveryDelta` | Scan now, classify against a snapshot. |
| `DiscoveryReport.stamps() -> dict[str, tuple[int, int]]` | The snapshot to persist and hand back next run. |
| `classify(previous, current) -> DiscoveryDelta` | Pure classification of two snapshots, usable on JSON read back from disk. |
| `DiscoveredFile` | `path`, `relative_path`, `size_bytes`, `modified_ns`, `.stamp()`. |
| `SkippedEntry` | `relative_path`, `reason` (a `SkipReason`), `detail`. |
| `DiscoveryDelta` | `added`, `modified`, `deleted`, `unchanged`, plus `.state(path)` and `.changed()`. |
| `SkipReason` | `EXCLUDED`, `SYMLINK_ESCAPE`, `UNREADABLE`, `NOT_A_FILE`, `DUPLICATE`. |

`allowed_extensions=None` accepts everything. An empty collection accepts
nothing, which is reported rather than silently matching a whole tree.

## Example

Run `python example.py` from this directory.

```python
from pathlib import Path

from document_discovery import Discovery

docs = Path("docs")
scan = Discovery(docs, allowed_extensions={".md", ".txt"}, exclude_patterns=["drafts/*"])

report = scan.discover()
print(f"{len(report.files)} candidates, {len(report.skipped)} skipped")

for entry in report.skipped:
    print(f"  {entry.relative_path}: {entry.reason.value} ({entry.detail})")

# Persist this between runs -- it is JSON-serialisable as-is.
previous = report.stamps()

# ...later, in a fresh process:
delta = scan.delta(previous)
for path in delta.added:
    print(f"new: {path}")
for path in delta.changed():
    print(f"needs work: {path} ({delta.state(path).value})")
```

## Dependencies

None. Standard library only — `os.scandir`, `fnmatch`, `pathlib`. There is
nothing to install and no optional extra to enable.

## Trade-offs

- **Cheap facts, not content.** Size and mtime are enough to answer "what
  changed", and reading every candidate's bytes here would make an incremental
  scan as expensive as a full one. Content identity is
  `document-identity-resolver`'s job; format detection is
  `file-type-validator`'s.
- **Filtered entries are counted, not listed.** A repository routinely holds
  thousands of non-matching files. `filtered_by_extension` and
  `filtered_by_pattern` are integers; `skipped` holds only entries that were
  genuinely notable — an escaping link, an unreadable path, a non-file.
- **The component remembers nothing.** Callers own the snapshot. That is what
  makes it safe across processes and trivially testable, at the cost of the
  caller having to persist one small mapping.
- **Duplicates collapse by resolved path.** A file reached two ways — a symlink
  and its target, or a junction and the directory it points at — is reported
  once. The second name is kept as a `DUPLICATE` record naming the one that
  won, so nothing disappears silently. This is path deduplication, not content
  deduplication, and does not pretend to be.

## Known limitations

- **Change detection compares `(size_bytes, modified_ns)`.** An edit that
  preserves both — a same-length rewrite with the timestamp restored, or a
  filesystem with a coarse clock — is reported as unchanged. Send the changed
  candidates to `document-identity-resolver` for a content hash when that
  matters; discovery deliberately does not hash.
- **`relative_path` is the stable key.** Moving the whole tree does not change
  it; moving one file looks like a delete plus an add. That is the intended
  reading.
- **Exclusion patterns are `fnmatch` globs** matched against both the entry
  name and its path relative to the root. They are not regular expressions and
  not gitignore syntax.
- **A non-recursive scan silently ignores subdirectories.** They are not
  candidates, so they are not recorded as skipped.
- **When one file is reachable two ways, the name the walk reached first wins.**
  Walk order is name-sorted within each directory, so this is deterministic,
  but it is not a choice you control.

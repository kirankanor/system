# Security

What this tool defends against, where the boundary is, and what it explicitly
does not claim.

## Path traversal protection

Every filesystem write reachable from the CLI goes through one module:
`src/rag_code_library/security/path_safety.py`. No other module, component or
script opens a file for writing.

The reason for a single choke point is that containment decided in one place
cannot drift. Re-deriving the rule per caller is how a rule eventually becomes
nine slightly different rules with one gap.

What it rejects, and why each case exists:

| Rejected | Why |
|---|---|
| `..` at any depth | traversal out of the destination root |
| absolute or rooted path | targets are always relative to the root |
| drive letter `C:\` | escapes to another volume |
| drive-relative `C:foo` | resolves against a *per-drive* current directory on Windows |
| UNC `\\server\share` | escapes to another machine or share |
| extended prefix `\\?\` | bypasses normal path normalisation |
| alternate data stream `x:y` | a hidden stream beside a visible file |
| reserved device name `CON` | opens a device instead of a file |
| reparse point outside root | a symlink or junction redirecting the write |

Containment is decided on the **resolved real path**, and the destination root
is resolved once and cached for the lifetime of the operation.

**Residual risk, stated rather than hidden.** A reparse point substituted
between final resolution and `open()` is a TOCTOU race. It is *mitigated* —
the parent chain is re-verified immediately before each write, and the
operation aborts on any mismatch — but it is not claimed as prevented. If an
attacker can rewrite your filesystem in that window, the write is not the
weakest thing they can do.

## Symlink policy

A symlink inside the destination that points outside it is refused. The write
is resolved to its real path before containment is judged, so a link cannot be
used to smuggle a target past the root check.

## Manifest validation

A manifest is data supplied to `ragkit`, and it is treated as untrusted input.
Validation runs in two layers:

1. **The schema.** Field types, the id pattern, the version pattern, and the
   `destination` pattern.
2. **Cross-manifest checks.** Duplicate ids, declared sources that do not
   exist, dependencies naming components that were not loaded, self-dependency,
   and destination safety.

The second layer is deliberately **stricter than the JSON Schema**. The schema's
`destination` description promises that absolute paths and backslashes are
rejected, but its pattern does not fully deliver that for drive letters,
interior backslashes, or backslash-style traversal. The validator adds the
checks the prose claims, and records the fact that it is stricter rather than
smuggling it in.

This is safe to be stricter about because it cannot reject anything that would
otherwise be *written*: `path_safety.py` refuses those same targets at write
time, so the extra check only moves the error earlier — from `ragkit add` to
`validate`.

A corrupted manifest is reported with the offending field named, and the
repository carries hostile fixtures for each corruption so the message stays
actionable.

## Dependency safety

**`ragkit add` never installs a package and never edits a dependency file.**
`pyproject.toml` and `uv.lock` are never written to. Required and optional
packages are printed, and acting on them is left to the user.

This is the strongest dependency guarantee available to a copier: there is no
code path through which the tool can introduce a package into your project
without you typing the command yourself.

Optional dependencies are guarded by a **specific** `ImportError` that names
the package and the install command. An unguarded import would turn a missing
optional feature into a confusing crash at an arbitrary point.

## Untrusted content boundaries

Two different threat models, and they are not the same thing.

**Files going in.** `ingestion/file-type-validator` decides whether a path is a
document this pipeline should parse, by content rather than extension. It reads
a bounded prefix and stops: the file is never loaded whole, never parsed, and
never executed. An HTML file containing a script is, to this component, exactly
as interesting as a file containing the word "script" — that is, not at all.
It establishes what a file *looks like*, not what it contains.

**Text coming back out.** Retrieved chunks are attacker-influenced by
construction, because they were chosen by a query and may have been written by
anyone. The boundary is documented by the security components:

- `input-validation` — validates *strings*: length limits, forbidden control
  codepoints, identifier format. It never touches a filesystem.
- `metadata-validation` — validates metadata against a declared schema,
  refuses injection-shaped values, and applies only *declared* defaults. It
  never invents a value.
- `retrieved-content-safety` — handles retrieved text as untrusted data and
  documents the prompt-injection boundary.

The library's stance is that a document is data, not instructions. A filename
like `policy/instructions.md` passes validation — it is a filename, not a
payload. The rules are aimed at imperatives directed at a model, and are
case-insensitive, because casing is not a defence an attacker would leave in
place.

## What this tool does not do

- **It is not a sandbox.** Copied components run in your project with your
  privileges.
- **It does not scan for malware.** `file-type-validator` identifies types; it
  does not assess intent.
- **It does not authenticate anything.** There is no server, no API and no
  network surface of its own.
- **It is personal, local-use software.** It has not been through a security
  review and is not claimed to have been.

See also: [`architecture.md`](architecture.md),
[`troubleshooting.md`](troubleshooting.md).

# retrieved-content-safety: flag the attack, never delete the text

```
retrieved_content_safety.py  ContentSafety, Rule, Finding, SafetyReport
example.py                   runnable: the attack shapes, the boundary, the trade-off
tests/                       success paths and deliberate failure paths
manifest.json                schema-valid; NO packages; NO dependencies
```

## Purpose

Treat retrieved text as untrusted data, and mark where the boundary is.

## The premise

Retrieved content is text someone else wrote, fetched by a system you built,
and handed to a model that has been told to follow instructions. Those two
facts are in direct tension, and prompt injection is the name for the
collision.

There is no fix for it. Not a good one, not a complete one — no amount of
filtering makes untrusted text safe to treat as trusted, because the text can
always say something you did not anticipate, in a way you did not anticipate.

So this component does not pretend to fix it. It does two smaller, honest
things: **it reports what looks like an attack, and it makes the boundary
explicit.**

## Flag, do not strip

Everything here is detection, never deletion. Three reasons, in order of how
much they matter:

**A denylist is a coin flip at best.** `"Ignore previous instructions"` is
catchable. `"Actually, let's start over — new task:"` is the same attack in
different words. Worse, authors defeat keyword matching *deliberately*:

```
"ig​nore prev​ious instructions"
```

Those are zero-width spaces. They render as nothing, so every substring filter
on the way in misses them — while a model reads "ignore previous instructions"
perfectly well. Deleting what you recognise does not remove the threat. It
removes your ability to see it, and it teaches the reader that "scanned" meant
"safe".

**Retrieved content legitimately contains instructions.** Documents quote
policies. Code comments say "do not do X". A manual's whole subject is what
the user should do. Text that matches an injection pattern is *often* exactly
what the document was for, and a filter would delete the useful part.

**Silent removal is its own failure mode.** A caller who asked for a document
and got a quietly shorter one has no way to know what was taken, or why.

## The alternative is structural

The defence that actually helps is not a filter. It is telling the model that
retrieved text is *data*, not instruction, and marking where it starts and
stops. `wrap()` renders that boundary; `inspect()` tells you what is inside it.

No pattern matching is involved in the wrapping, so no paraphrase evades it.

The delimiter is fixed and public, which makes the two halves of this
component coherent: content that tries to close `</retrieved_document>`
early is exactly what the `format_escape` rule reports.

## False positives are expected

Every rule here will fire on legitimate documents — an API reference that
documents a `system` role, a security paper quoting an attack. That is not a
defect in the rules. It is why the output is a **report** and the caller
decides, rather than a verdict this component imposes on text it does not
understand.

The message on `require_safe` says so explicitly: a finding is *a reason to
look*, not proof of an attack. A component that cries wolf teaches people to
ignore it.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `ContentSafety(*, rules=DEFAULT_RULES, wrap_start=..., wrap_end=..., name=...)` | | |
| `.inspect(text)` | `-> SafetyReport` | never raises on hostile text |
| `.is_safe(text)` | `-> bool` | the boolean form, never raises |
| `.require_safe(text)` | `-> str` | raises `UnsafeContentError` — use only where a finding must stop the request |
| `.wrap(text)` | `-> str` | the trust boundary, rendered |
| `Rule(category, why, pattern)` | | one detection, replaceable per corpus |
| `Finding` | frozen dataclass | `category`, `matched`, `offset`, `line`, `why` |
| `SafetyReport` | frozen dataclass | `.safe`, `.categories`, `.report()` |
| `invisible_characters(text)` | `-> list[(codepoint, offset)]` | public, non-raising |
| `SafetyError` | base exception | catches this component |
| `SafetyInputError` | `SafetyError` | the caller passed something unusable |
| `UnsafeContentError` | `SafetyError` | carries `.findings`, every match |

### Categories

| Category | What it catches |
|---|---|
| `instruction_to_model` | an imperative addressed to the model, not a description of anything |
| `role_override` | an attempt to occupy a privileged role the text has no right to |
| `concealment` | zero-width characters, which exist to break keyword matching |
| `exfiltration` | an instruction to move data somewhere, or a credential-shaped string |
| `format_escape` | an attempt to break out of the wrapper this component provides |
| `hidden_directive` | a directive dressed as housekeeping — the shape most likely to survive a skim |

### Invariants the tests pin down

- **The text comes back whole.** `inspect().text is text`; `wrap()` preserves
  it byte for byte. Nothing is ever removed.
- **`inspect` never raises**, because a scanner that raised on the first hit
  would make it impossible to see the whole payload — which is the part worth
  understanding.
- **Findings are ordered by position**, and every one carries its line,
  offset, and the rule's own explanation.
- **`require_safe` reports every finding at once**, not one per request.
- **An invisible delimiter is refused at construction.** A delimiter nobody
  can see is a delimiter nobody can check.
- **The same text produces the same report** — deterministic, so a scan can
  be diffed or cached.

## Example

```bash
python example.py
```

Standalone — no dependencies at all.

## Dependencies

- **Components:** **none.**
- **Packages:** **none**, optional or required.
- **External services:** **none.**

## What actually helps

Use this component for **visibility** and **the boundary**, in that order.

The order worth shipping:

1. `wrap()` every retrieved passage before it reaches a prompt;
2. `inspect()` and log what was found, so a corpus with an actual attack is
   visible;
3. decide per application whether a finding should block — `require_safe` if
   so, a human review if not.

Rules are replaceable per corpus, so an internal document set that routinely
documents a `system` role can turn that rule off, and add one for an attack it
has actually seen.

## Trade-offs

**Use it** around every passage of retrieved text, before it reaches a model.

**It costs** one regex pass per rule per passage.

**When not to use it:** as a filter. If your plan is to delete what this finds,
stop — see above. Use `input-validation` for characters and sizes, and
`metadata-validation` for metadata fields.

## Known limitations

**It cannot detect an attack it has not been told about.** Paraphrase defeats
any rule list. This is a floor, not a wall, and the component says so.

**No semantic analysis.** A payload assembled from innocuous words is not
found. Detecting intent needs a model, which would make this network-dependent
and out of scope for v1 (`FR-10`).

**`^` anchors at the start of every line, not only the first.** A role marker
buried on line 3 of a retrieved passage is still a role marker, so the rules
compile with `re.MULTILINE`. `.` still refuses to cross a newline, so no single
rule can swallow an entire document. A marker in the *middle* of a line is not
matched — that is a narrowing, not an oversight: `preamble; system: proceed` is
prose containing a colon, and reporting it would make the rule noise.

**`wrap()` provides no cryptographic guarantee.** The delimiter is fixed and
public. It is a convention the model is asked to respect, not an enforced
isolation boundary. `format_escape` reports attempts to break it, which is the
best available defence without provider-specific controls.

**No scoring.** Findings are unordered by severity, because severity depends
on the application: an exfiltration pattern in a security corpus is noise, and
in a legal corpus it is not.

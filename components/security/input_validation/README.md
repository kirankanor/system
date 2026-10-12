# input-validation: refuse the text that changes meaning on the way through

```
input_validation.py  InputValidator, normalise(), what_is_forbidden()
example.py           runnable: the four hostile families, size limits, recovery
tests/               success paths and deliberate failure paths
manifest.json        schema-valid; NO packages; NO dependencies
```

## Purpose

Check size and character safety of a string before it goes anywhere.

## The problem, stated as four characters

Everything this component rejects is invisible in a normal editor and
survives copy-paste. That is exactly why a validator, not a human, is the
right place to catch it.

| Family | Character | What it does | Real attack |
|---|---|---|---|
| NUL | `\x00` | truncates strings at any C boundary | `report.pdf\x00.jpg` — one string to Python, two filenames to `open()` |
| Control | `\x1b`, `\x07`, `\r` | **are** commands | `\x1b[2J` clears a terminal; `\x1b]0;pwned\x07` rewrites its title; `\r` overwrites a log line's start |
| Bidi override | `\u202E` | renders everything after it right-to-left | `exe.txt` displays as `txt.exe` — the CVE-2017-1000112 filename spoof |
| Separator | `\u2028`, `\u2029` | a newline to some parsers, an ordinary character to others | a string containing a "line break" a downstream check never sees |

None of these is offensive or wrong. They are text that *changes meaning
downstream* in a way the caller cannot see by reading it.

## Refuse, do not repair

`validate_text` **raises** and returns the input unchanged. It never strips.

A filter that quietly removes a NUL produces text the caller did not ask for,
with no record that anything changed — and a NUL-stripping filter is itself
the bypass, because `a\x00b` becomes `ab` and the caller never learns the
input was malformed at all.

If you want to normalise, call `normalise()` explicitly and look at what it
returned. That is the difference between a transformation and a side effect.

## Size limits are a safety property

An unbounded string is a resource problem long before it is a security one:
it turns a text field into a memory-exhaustion vector, and it makes every
downstream pass over the text cost whatever the input decides it should.

The default is `1,000,000` characters — a ceiling meaning *"this is not a real
document"*, not a formatting policy. Set your own per validator or per call.

Size is checked **before** the character scan, so an oversized hostile string
reports the size problem rather than making you scroll to the first control
character.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `InputValidator(*, max_text_length=1_000_000, name=...)` | | the ceiling, and a name that appears in every message |
| `.validate_text(text, *, field, max_length=None)` | `-> str` | returns `text` unchanged |
| `.validate_non_empty(text, *, field, max_length=None)` | `-> str` | as above, but blank is also an error |
| `.validate_identifier(value, *, field, max_length=256)` | `-> str` | ASCII alnum, `_`, `-`, `.` only |
| `.check_size(text, *, field, max_length=None)` | `-> int` | size only; raises `OversizedInputError` |
| `what_is_forbidden(text)` | `-> list[(codepoint, name, offset)]` | public, non-raising |
| `contains_forbidden(text)` | `-> bool` | public, non-raising |
| `normalise(text)` | `-> str` | NFKC, drops nothing |
| `InputError` | base exception | catches this component |
| `OversizedInputError` | `InputError` | over the limit |
| `ForbiddenCharacter` | `InputError` | carries `.characters`, every offender |

### Invariants the tests pin down

- **Every offender is reported**, not just the first, each with its offset —
  discovering them one per request is a slow way to read an input.
- **`validate_text` returns the same object**, so nothing can be silently
  altered and then passed off as the original.
- **Exactly the limit passes**; one over does not.
- **`bool` is not an int** when a limit is asked for — `max_text_length=True`
  is rejected at construction, because that is how a bound gets quietly
  defeated.
- **NUL is described as a control character**, not "unassigned" — it is
  assigned, just unnamed in `UnicodeData.txt`, and saying otherwise is a small
  lie in a security message.

## Example

```bash
python example.py
```

Standalone — this component has no dependencies at all.

## Dependencies

- **Components:** **none.**
- **Packages:** **none**, optional or required.
- **External services:** **none.**

## Why it is not a sanitiser

Sanitisers try to make hostile text safe. This refuses it. The reason is
simple: everything here has a *legitimate* form in some corpus, and every
attempt to preserve the useful version while blocking the abuse is another
bypass to find. Rejecting and letting the caller decide is the only posture
that does not quietly lose.

## Trade-offs

**Use it** at every boundary where text arrives from outside — a query, a
filename, a log line, a value about to be written to disk.

**It costs** one pass over the string.

**When not to use it:** as a filter in front of a model. A model reads
`\x1b` as nothing much; the danger is a *terminal* or a *log viewer* reading
it as a command. For untrusted text heading into a prompt, see
`retrieved-content-safety`, which reports rather than refuses.

## Known limitations

**No homoglyph or confusable detection.** `раypthon` (Cyrillic `а`) is not
flagged. Detecting it properly needs a confusables table and a comparison
against a known-good list, which is a different, much larger component.

**No Unicode-script mixing check.** Mixed-script identifiers are a real
phishing shape and this component does not look for them.

**`normalise` is NFKC only.** It unifies compatibility forms and leaves
controls and bidi overrides alone — deliberately, so it cannot be mistaken
for a cleaning function.

**No length limit in bytes, only characters.** A 100,000-character string of
emoji is far more than 100,000 bytes. If you are bounding what reaches a disk
or a network frame, bound bytes too.

**ASCII-only identifiers.** `validate_identifier` accepts letters, digits,
`_`, `-` and `.` — all ASCII. An identifier with an accented letter is
refused. That is intentional for values interpolated into paths and queries,
but it is a limit, not a universal rule.

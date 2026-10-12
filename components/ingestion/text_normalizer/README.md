# text-normalizer: make extracted text comparable, and say what changed

```
text_normalizer.py  TextNormalizer, NormalizationResult, normalize_text()
example.py          runnable: every rule, idempotence, NFC vs NFKC
tests/              success paths and deliberate failure paths
manifest.json       schema-valid; NO packages; NO dependencies
```

## Purpose

Normalize extracted text with explicit, idempotent, reported rules.

## Why this strips what `input-validation` refuses

Both components look at the same characters and do opposite things. That is
not a contradiction, it is the two jobs:

| | `input-validation` | `text-normalizer` |
|---|---|---|
| guards | a **boundary** | **already-ingested** text |
| on a control character | **raises** | **removes** |
| because | text arriving from outside has no business containing a NUL or a bidi override, and silent stripping is itself the bypass | a PDF legitimately hands you a form feed at every page break and a soft hyphen at every line end; refusing them would make this useless |
| posture | refuse | repair, and report the repair |

So the rule here is not "reject the hostile" but "make the text comparable" —
and every transformation is *named* rather than applied silently.

## Why idempotence is a hard requirement

Normalization runs inside ingestion pipelines, where a stage is very often
re-run: a retry, a re-index, a resumed job over a file that was already
processed. If `normalize` were not idempotent, the second pass would produce
different text than the first, the index would disagree with what a fresh run
produced, and two documents ingested at different times would not be
comparable — defeating the purpose of normalizing at all.

Every rule was chosen with that in mind, and the tests assert it over a corpus
of nasty inputs across all four Unicode forms: double pass, triple pass,
each form.

## The rules, in order

1. **Line endings.** `\r\n` and lone `\r` become `\n`. The `\r\n` pass runs
   first — doing it the other way round would turn every Windows newline into
   a blank line.
2. **Vertical tab** `\x0b` becomes `\n`.
3. **Form feed** `\x0c` becomes `\n\n`. A form feed in extracted text *is* a
   page break, and collapsing it to a space would delete the only signal that
   a new page began. Two newlines keep it visible and survive step 7.
4. **Remaining C0 controls, DEL and C1** are removed. `\t` and `\n` are kept —
   indentation and structure are content.
5. **Invisible format characters** — soft hyphen `U+00AD`, byte-order mark
   `U+FEFF` and zero-width space `U+200B` — are removed. These are Unicode
   category `Cf`, not controls, so rule 4 never caught them; they render as
   nothing while defeating an exact match, which for an index is a
   correctness problem rather than a cosmetic one. `U+200D` (ZWJ) is
   deliberately left alone — removing it changes what an emoji *says*, and
   this component reports repairs rather than reinterpreting content. Neither
   is `U+00A0`; that one is whitespace, and the whitespace rule owns it.
6. **Unicode normalization** in the chosen form.
7. **Horizontal whitespace** runs collapse to one space, and trailing
   whitespace is dropped per line. *Leading* indentation is preserved,
   because in a code corpus it is meaning. (Mid-line tabs do collapse — a tab
   is horizontal whitespace, not layout that carries meaning.)
8. **Three or more blank lines** collapse to two. The rule caps the *run*; it
   does not delete every blank line.

Steps 1–4 and 6–8 each report themselves only if they actually changed
something, so `result.rules` is a record of work done, not a checklist.

## NFC by default, on purpose

`NFC` preserves compatibility distinctions: `U+FB01` (the `fi` ligature)
stays a ligature. `NFKC` folds it to `fi`, and folds full-width `ｐｙｔｈｏｎ`
to `python`.

NFC is the default because it is the *least destructive* choice — a component
that quietly changes what a document says is a component you cannot trust.

But if you index for **matching**, NFKC is very likely what you want: without
it, a full-width and a half-width spelling of the same term never match, and
every query containing one silently returns nothing. `form="NFKC"` is the
one-word change, and the trade-off is yours to make rather than one this
component makes for you.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `TextNormalizer(*, form="NFC", collapse_whitespace=True, collapse_blank_lines=True, max_length=5_000_000, name=...)` | | |
| `.normalize(text)` | `-> NormalizationResult` | the text **and** what changed |
| `.clean(text)` | `-> str` | just the normalized text |
| `.is_idempotent(text)` | `-> bool` | so a caller can assert it about *its own* corpus |
| `normalize_text(text, *, form="NFC")` | `-> str` | one-call convenience |
| `NormalizationResult` | frozen dataclass | `.text`, `.original`, `.rules`, `.changed`, `.report()` |
| `NormalizerError` | base exception | catches this component |
| `NormalizerInputError` | `NormalizerError` | the caller passed something unusable |
| `FORMS` | `("NFC", "NFD", "NFKC", "NFKD")` | the four `unicodedata.normalize` accepts |

### Invariants the tests pin down

- **`normalize(normalize(x)) == normalize(x)`** for every nasty input, in
  every form, three passes deep.
- **The result carries `original`**, so a caller can diff before and after.
  A component that rewrote your corpus without saying so would be a component
  you could not debug.
- **`rules` is empty when nothing fired**, and `report()` says so — an honest
  "no rules applied" rather than a list of rules that did nothing.
- **A `bool` is rejected for `max_length`** — `True` is not a limit.

## Example

```bash
python example.py
```

Standalone — this component has no dependencies at all.

## Dependencies

- **Components:** **none.**
- **Packages:** **none**, optional or required.
- **External services:** **none.**

## Usage

```python
from text_normalizer import TextNormalizer

normalizer = TextNormalizer(form="NFKC")   # index for matching
result = normalizer.normalize(extracted_page)

print(result.report())   # applied line-endings, controls, ...; 4,102 -> 4,088 characters (-14)
chunk(result.text)        # what you actually index
```

Use `normalize` rather than `clean` if you ever want to know what was
touched — and you will, the first time a document comes back shorter than
expected.

## Trade-offs

**Use it** once per extracted page, before chunking.

**It costs** one pass over the string plus a per-character rebuild.

**When not to use it:** as a security filter. It removes control characters
because they are noise in *ingested* text, not because it can tell a hostile
one from an innocent one — and it says so by never claiming to. For untrusted
input arriving from outside, see `input-validation`, which refuses; for text
about to reach a model, see `retrieved-content-safety`, which flags.

## Known limitations

**No stemming, no case folding, no accent stripping.** This makes text
*structurally* comparable (whitespace, controls, Unicode form). Deciding that
`Résumé` and `resume` are the same term is a linguistic decision belonging to
the analyzer, not to a normalizer — and doing it here would silently change
what is stored.

**Only three invisible characters are removed.** Rule 5 names `U+00AD`,
`U+FEFF` and `U+200B` and no others. The rest of the zero-width family —
`U+200D` (ZWJ), `U+200C` (ZWNJ), `U+2060` (word joiner) — is left alone,
because those carry meaning in emoji and script shaping, and removing them
changes what the text *says* rather than how it is spelled. A zero-width
character outside the three still defeats naive substring matching; if you
need them surfaced rather than assumed absent, `retrieved-content-safety`
reports them under `concealment` at query time.

**Removal is by codepoint, not by analysis.** The three are dropped wherever
they appear. This component cannot tell an artifact from content — no
normalizer can — so the trade is made in favour of exact matching, and the
input is always retained on `result.original` for a caller that disagrees.

**Leading whitespace is preserved, always.** A paragraph indented with four
spaces keeps them. That is the cost of not destroying code indentation, and
there is no way to distinguish the two without knowing the document's genre.

**`max_length` bounds characters, not bytes.** Bound bytes separately if you
are writing to a frame or a disk quota.

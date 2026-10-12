# pdf-extraction-pymupdf: real text out of a born-digital PDF

```
pymupdf.py       PdfExtractor, ExtractedPdf, PdfEncryptedError
example.py       runnable: extraction, page selection, the encrypted case
tests/           success paths and deliberate failure paths
manifest.json    schema-valid; optional `pymupdf`; NO component dependencies
```

## Purpose

Extract text from born-digital PDFs, and fail in a way a caller can act on.

## One of two selectable extractors (FR-6)

The two PDF extractors differ by **technique**, so you pick one and take only
that one's dependency:

| | `pdf-extraction-pymupdf` | `pdf-extraction-ocr` |
|---|---|---|
| reads | a PDF's embedded text layer | page **images** |
| needs | `pymupdf` | `pytesseract` + `pillow` |
| for | born-digital documents | scans, and anything already rasterized |
| speed | milliseconds | seconds per page |
| quality | exact | approximate |

Neither imports the other. Neither needs the other's packages. Both return the
same result shape, so whichever you chose is interchangeable downstream of the
call. That is what makes them *selectable* rather than merely separate.

**How to choose.** If the PDF has a text layer, use this one. OCR on a
document that already has real text is slow, expensive and strictly worse —
it recognises glyphs that a machine already transcribed perfectly. Reach for
the OCR sub-component only when there is no text layer to read.

## Why `pymupdf`, and why it is optional

There is no PDF text extractor in the standard library, and writing one is not
a weekend project: a PDF's text is a stream of positioning operators that have
to be reconstructed into reading order, through ligatures, embedded font
tables and per-document encodings.

So it is an optional dependency — and being optional is a real constraint on
the *code*, not just a declaration on the manifest:

**the import happens lazily, on first use, never at module import.** That
means `ragkit add pdf-extraction-pymupdf` into a project that has not
installed `pymupdf` still yields an importable module: one that tells you
exactly what to install when you first call it, instead of breaking every
import in the package at load time.

### `pymupdf` before `fitz`

pymupdf renamed its import module from `fitz` to `pymupdf` in 1.24 and emits
`The fitz API is deprecated` on every import of the old name. The component
tries `pymupdf` first and falls back to `fitz` for older installs, because a
library that warns on import turns a caller's clean stderr into noise they
cannot silence from here. The missing-dependency error names both, so a stack
trace is readable either way.

## The encrypted case, which is the interesting one

An encrypted PDF raises `PdfEncryptedError`. It is deliberately its own type
rather than a generic failure, because the remedies are different:

| failure | type | remedy |
|---|---|---|
| encrypted | `PdfEncryptedError` | you have the password — decrypt, then retry |
| corrupt / not a PDF | `PdfInputError` | the file is wrong; nothing you can supply will fix it |
| missing / a directory | `PdfInputError` | the path is wrong |
| selection out of range | `PdfInputError` | your indices are wrong |

`PdfEncryptedError` carries `.path` and `.needs_pass`, so a caller can branch
on the fact rather than string-matching a message.

A PDF encrypted with an **empty** password is a real thing — many tools
produce them — and pymupdf opens those transparently. Only a document that
actually demands a password raises here, which is the honest boundary.

The error never echoes a password, and the document is closed before raising
rather than left open waiting for a finalizer.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `PdfExtractor(*, document_factory=None, name=...)` | | `document_factory` is a test/injection seam |
| `.extract(path, *, pages=None)` | `-> ExtractedPdf` | `pages` are **zero-based** |
| `.page_count(path)` | `-> int` | without extracting any text |
| `.open(path)` | `-> document` | caller closes; for inspecting metadata cheaply |
| `ExtractedPdf` | frozen dataclass | `.pages`, `.page_count`, `.text`, `.metadata`, `.source`, `.character_count` |
| `PdfExtractionError` | base exception | catches this component |
| `PdfInputError` | `PdfExtractionError` | the path or the selection is unusable |
| `PdfEncryptedError` | `PdfExtractionError` | carries `.path`, `.needs_pass` |
| `MissingDependency` | `PdfExtractionError`, `ImportError` | `pymupdf` is not installed |

### Invariants the tests pin down

- **Page selection is zero-based**, matching pymupdf and every other PDF
  library. A caller holding a printed page number wants `page_number - 1`;
  that conversion is theirs, because guessing it here would silently drop a
  page.
- **An out-of-range selection raises** rather than returning a short result.
  A truncated extraction is indistinguishable from a document that ended
  early.
- **An empty selection raises.** Producing nothing would be a caller bug
  wearing the costume of a successful call.
- **A negative index raises** rather than selecting from the end, which is
  almost never what a caller means.
- **Pages are joined by a blank line**, not a bare newline — a page boundary
  is a real break, and `\n` alone would run the last line of one page into
  the first line of the next.
- **The document is closed** on success, on failure, and before raising.
- **Empty and non-string metadata values are dropped**, so `metadata` is
  always a usable `dict[str, str]`.

## Example

```bash
python example.py
```

## Dependencies

- **Components:** **none** — the result type is defined here, so this
  component copies out alone.
- **Packages:** `pymupdf` (optional, `>=1.24.0`). Install with
  `uv add pymupdf`. Without it the module still imports; the first call tells
  you what is missing.
- **External services:** **none.**

## Usage

```python
from pdf_extraction.pymupdf import PdfExtractor, PdfEncryptedError

extractor = PdfExtractor()

try:
    document = extractor.extract("report.pdf")
except PdfEncryptedError as exc:
    print(f"{exc.path} needs a password")     # branch on the fact
    raise

for number, page in enumerate(document.pages, start=1):
    chunk(page)                               # one chunk boundary per page

print(document.metadata.get("title"))
```

### Before you add it

`ragkit add pdf-extraction-pymupdf` copies the file. It does **not** install
`pymupdf` and does not edit any dependency file. Install it yourself, in the
environment you actually run in.

## Trade-offs

**Use it** for any PDF produced by a tool rather than a scanner — reports,
papers, exported documentation, anything you can select text out of.

**It costs** one `pymupdf` import on first call, then one pass per page.

**When not to use it:** for scans. A scanned page has no text layer, and the
extractor returns an empty string for it rather than guessing. That is
detection, not failure — but if your corpus is mostly scans, this is the wrong
sub-component and `pdf-extraction-ocr` is the right one.

## Known limitations

**A scanned page yields an empty string, not an error.** The page loaded
fine; it simply has no text. A caller must check for that themselves — a
document of 40 empty pages is the signal that you wanted OCR. Detecting
"this page is an image" would need the same PDF introspection that makes
`pymupdf` an optional dependency in the first place.

**The component does not sniff the file signature.** pymupdf has robust
detection of its own and its error is already wrapped with the path and the
cause. The OCR sub-component *does* sniff, because it has no library to fall
back on. The two differ here on purpose.

**Layout, columns and reading order are not preserved.** Text comes back in
pymupdf's reading order. A multi-column PDF may interleave columns; tables
come out as loose lines. Fixing that needs layout analysis, which is a
different and much larger component.

**No de-hyphenation.** A word broken across a line break stays broken across
a line break — `text-\nnormalizer` rather than `text-normalizer`. Running the
text through `text-normalizer` afterwards fixes the invisible characters it
produces, not this.

**`metadata` is whatever the document declared.** It is untrusted data,
straight from a file you did not write. Validate it with
`metadata-validation` before it reaches a prompt or a log.

**`document_factory` is not a supported abstraction.** It exists so tests and
unusual hosts can substitute an opener. It carries no compatibility promise.

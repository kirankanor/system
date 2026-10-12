# pdf-extraction-ocr: read page images with no PDF library at all

```
ocr.py           OcrExtractor, OcrResult, ExtractedPage, PdfEncryptedError
example.py       runnable: image OCR, the PDF rejection, the encrypted case
tests/           success paths and deliberate failure paths
manifest.json    schema-valid; optional `pytesseract`+`pillow`; NO deps
```

## Purpose

Read text out of page **images** — scans, photographs, faxes — using
Tesseract, and refuse a PDF with a typed error that names what it would need.

## One of two selectable extractors (FR-6)

| | `pdf-extraction-pymupdf` | `pdf-extraction-ocr` |
|---|---|---|
| reads | a PDF's embedded text layer | page **images** |
| needs | `pymupdf` | `pytesseract` + `pillow` + the tesseract **binary** |
| for | born-digital documents | scans, and anything already rasterized |
| speed | milliseconds | seconds per page |
| quality | exact | approximate |

Neither imports the other. Neither needs the other's packages. Both return
the same result shape, so whichever you chose is interchangeable downstream.

**How to choose.** If the PDF has a text layer, use the pymupdf sub-component
— OCR on text a machine already transcribed perfectly is slow, expensive and
strictly worse. Use this one when there is nothing but pixels.

## Why it refuses a PDF

This component reads images. A PDF is not an image. Turning a PDF page into
pixels needs a renderer, and a renderer is precisely the dependency this
sub-component exists to avoid — so claiming to accept PDFs would mean
depending on a PDF library while saying it does not have one.

Rather than fail somewhere deep inside Pillow with `cannot identify image
file`, the component looks at the first five bytes, sees `%PDF-`, and raises
`OcrInputError` explaining both remedies: rasterize first (`pdftoppm`, or
`pymupdf`), or use `pdf-extraction-pymupdf`.

The check is on **bytes, not the filename**, so a PDF named `page.png` is
still caught.

## The encrypted case

An encrypted PDF raises `PdfEncryptedError` — a distinct type, because the
remedy differs: decrypt first, then rasterize.

It is detected from the **trailer, without a PDF library**: the `/Encrypt`
dictionary is required to appear within the first 1024 bytes of an encrypted
PDF, so a small read is enough and no parser is involved. That matters
because the whole point of this sub-component is having no PDF dependency —
detecting encryption with one would reintroduce exactly what it removed.

It is checked **before any library touches the file**, so a hostile encrypted
PDF never reaches Pillow at all.

## The dependency is two things, not one

`pytesseract` and `pillow` are Python packages you install. **Tesseract
itself is a binary**, and no Python installer provides it — on Windows it
comes from UB Mannheim's installer, on macOS from Homebrew, on Debian from
`apt`. So `uv add pytesseract pillow` is necessary and *not sufficient*, and
the missing-dependency error says so explicitly rather than letting you
discover it from a `TesseractNotFoundError` three frames down.

That is also why this file cannot be verified end to end here: no Tesseract
binary has ever been run against it. Every behavioural test goes through the
`image_factory` injection seam, and the dependency paths run through the real
`ImportError` guard.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `OcrExtractor(*, language="eng", dpi=300, image_factory=None, name=...)` | | `image_factory` is a test/injection seam |
| `.extract(path)` | `-> OcrResult` | one image, one page |
| `.extract_one(path)` | `-> ExtractedPage` | just the page |
| `.extract_many(paths)` | `-> OcrResult` | several images as consecutive pages |
| `OcrResult` | frozen dataclass | `.pages`, `.page_count`, `.text`, `.metadata`, `.source`, `.character_count` |
| `ExtractedPage` | frozen dataclass | `.text`, `.is_blank` |
| `PdfExtractionError` | base exception | catches this component |
| `OcrInputError` | `PdfExtractionError` | the path, language, dpi or input is unusable |
| `PdfEncryptedError` | `PdfExtractionError` | carries `.path` |
| `MissingDependency` | `PdfExtractionError`, `ImportError` | the packages or the binary are absent |

### Invariants the tests pin down

- **`language` is validated as a command-line argument, not as data.**
  Tesseract is shell-invoked, so only letters, digits, `+`, `-` and `_` are
  accepted — `eng`, `eng+fra` and `chi_sim` yes, `eng; rm -rf /` no. It is
  not escaped, it is refused: an allow-list is the only defensible answer
  when the value is going to be an argument. The `_` matters — real Tesseract
  codes use it, so an allow-list without it would refuse languages the engine
  ships with.
- **`dpi` is bounded to 72–1200.** Below 72 there are too few pixels to
  recognise; above 1200 the accuracy gain is negligible and the memory cost
  is not.
- **A `bool` is not an `int`** — `dpi=True` is refused.
- **A bare string is refused by `extract_many`.** A string is iterable, so
  passing one would OCR it character by character.
- **An empty `paths` is refused.** An empty result would be indistinguishable
  from a caller bug.
- **`extract_many` raises on the first failure** rather than returning what
  succeeded. A half-extracted document is worse than none: the caller would
  have no way to tell which page is missing.
- **The image is closed** on success and on failure.
- **Pages are joined by a blank line**, matching the pymupdf sub-component.

## Example

```bash
python example.py
```

Runs without Tesseract — it demonstrates the real behaviour through the
injection seam, and shows the exact message you get if the dependency is
absent.

## Dependencies

- **Components:** **none.**
- **Packages:** `pytesseract>=0.3.10`, `pillow>=10.0.0` (both optional), plus
  the tesseract **binary** on `PATH`. Install with `uv add pytesseract pillow`.
- **External services:** **none.** OCR runs locally.

## Usage

```python
from pdf_extraction.ocr import OcrExtractor, PdfEncryptedError

ocr = OcrExtractor(language="eng+deu", dpi=300)

try:
    result = ocr.extract_many(sorted(scanned_pages))
except PdfEncryptedError as exc:
    print(f"{exc.path} is encrypted; decrypt before rasterizing")
    raise

for page in result.pages:
    if page.is_blank:
        continue          # a blank scan is a scan that failed, not empty text
    chunk(page.text)
```

### Before you add it

`ragkit add pdf-extraction-ocr` copies the file. It does **not** install
`pytesseract` or `pillow`, does **not** install the tesseract binary, and
does **not** edit any dependency file.

## Trade-offs

**Use it** for scanned paper, photographs of documents, and anything whose
text exists only as pixels.

**It costs** seconds per page, and needs a system binary that no `requirements`
file can declare.

**When not to use it:** for born-digital PDFs. Those have real text, and the
pymupdf sub-component reads it exactly, in milliseconds.

## Known limitations

**No OCR confidence is surfaced.** `image_to_string` does not report it, and
getting at it means calling `image_to_data` instead — a second pass over every
page, on the slowest component in this library. It is left out rather than
shipped as a field that is always `None`, which would read as "the engine was
certain of nothing". If you need it, swap the one call in `extract_one`.

**Never verified against a real Tesseract binary.** Every test runs through
`image_factory`. That is honest about what has been executed, and it is the
reason the language and dpi checks are enforced *before* the libraries are
even imported.

**Layout, columns and reading order are not preserved.** Tesseract's reading
order is what you get; multi-column pages may interleave. Fixing that needs
layout analysis, a different and much larger component.

**One image is one page.** This component does not know what a multi-page
image format is. Feed it a directory of pages with `extract_many`.

**`dpi` is declared, not measured.** It is passed through for the caller's
records; nothing here resamples the image to match it.

**Recognition quality depends on the input.** A 72 dpi scan of faded carbon
copy will come back as noise. No configuration in this component fixes a
bad source image.

# document-loader: one shape out, whatever went in

## Purpose

Every parser wants to return something different. `pymupdf` gives you pages as
strings; `ocr` gives you objects with a `.text`. Plain text has no pages at
all. HTML is a tree. Left alone, every stage downstream ends up writing
`page if isinstance(page, str) else page.text` forever.

This component ends that. It:

- **detects the format from bytes**, not from the filename, and records which
  evidence decided,
- **runs one adapter** and returns one `LoadedDocument` shape with numbered,
  citation-ready pages,
- **adapts either PDF extractor by duck typing**, so this module imports
  neither of them and neither can break the other,
- **refuses** a format it cannot open, by name, instead of substituting a
  parser that might work.

## Public API

| Name | What it is |
|---|---|
| `DocumentLoader(*, adapters=None, ocr_fallback=None, prefix_bytes=8192)` | Configure. Ships with `text`, `markdown` and `html` adapters. |
| `DocumentLoader.load(path) -> LoadedDocument` | Detect, parse, return. Typed errors only. |
| `DocumentLoader.formats` | Every format currently openable, sorted. |
| `detect_format(path) -> Detection` | Detection alone, without parsing. |
| `PdfAdapter(extractor, *, name="pdf")` | Wraps *either* PDF extractor into the common shape. |
| `TextAdapter` / `MarkdownAdapter` / `HtmlAdapter` | The built-ins, each with a `.format` and `.extract(path)`. |
| `LoadedDocument` | `source_path`, `format`, `detection`, `pages`, `warnings`, `loader`, `metadata`, plus `.text`, `.page_count`, `.blank_pages`. |
| `Page` | `number` (1-based), `text`, `source_locator` (e.g. `"page 3"`), `.is_blank`. |
| `Detection` | `format`, `evidence`, `detail`. |
| `Evidence` | `CONTENT_SIGNATURE`, `DECLARED_EXTENSION`, `DEFAULTED_TO_TEXT`, `NO_MATCH`. |
| `LoaderError` | Root of `UnreadableDocument`, `EmptyDocument`, `UnsupportedFormat`, `MissingAdapter`, `AdapterError`. |

## Example

Run `python example.py` from this directory. It parses real text, Markdown and
HTML, and shows both PDF extractor shapes normalising identically.

```python
from pathlib import Path

from document_loader import DocumentLoader, PdfAdapter

loader = DocumentLoader()
doc = loader.load(Path("guide.md"))
print(doc.format, doc.page_count, doc.detection.evidence.value)

# Wire a PDF extractor without this module importing it.
class FakeExtractor:            # stands in for pdf-extraction-pymupdf
    def extract(self, path):
        class Result:
            pages, metadata, source = ("page one", "page two"), {}, str(path)
        return Result()

loader = DocumentLoader(adapters={"pdf": PdfAdapter(FakeExtractor())})
pdf = loader.load(Path("report.pdf"))
print([page.source_locator for page in pdf.pages])
```

## Dependencies

None required. Text, Markdown and HTML are parsed with the standard library
(`html.parser`, `re`).

PDF is **not** parsed here, on purpose. Wrap whichever extractor your project
uses in `PdfAdapter` — `pdf-extraction-pymupdf` for born-digital PDFs, or
rasterise yourself and feed `pdf-extraction-ocr`. The loader imports neither
and requires neither.

## Trade-offs

- **Detection names its evidence.** `Detection.evidence` is always set, so a
  caller that cannot act on an inference can refuse it. The cost is one extra
  field; the benefit is that "we thought it was text" is never invisible.
- **Markdown is detected by extension, and says so.** Markdown *is* text —
  there is no byte signature, so claiming content-based detection would be a
  lie. The evidence reads `declared-extension` and a warning lands on the
  document.
- **`PdfAdapter` duck types the extractor's result.** It accepts a bare `str`
  page or anything with a `.text` string, and raises `AdapterError` on
  anything else. That keeps the two extractors decoupled at the cost of not
  being able to guarantee a shape at import time.
- **OCR is a policy, not an implementation.** Supply `ocr_fallback` and it
  runs *only* on blank pages. The loader will not rasterise anything, because
  a renderer is exactly the dependency `pdf-extraction-ocr` exists to avoid.
- **No chunking.** `MarkdownAdapter` does not split on headings. Where a
  document breaks is a chunking decision and does not belong at load time.

## Known limitations

- **A binary file with no signature is refused, not guessed at.** Bytes that
  are not valid UTF-8 and match no signature raise `UnsupportedFormat` naming
  the adapters that exist. DOCX, XLSX, RTF and ODF are therefore refused, and
  are not claimed as supported anywhere.
- **`prefix_bytes` bounds detection, not parsing.** Adapters read the whole
  file. Loading a very large document into memory is the caller's call to
  make; this component does not stream.
- **Line endings are preserved, not normalised.** A CRLF file comes back with
  CRLF. That is deliberate: `text-normalizer` owns whitespace and line
  endings, and rewriting them at load time would destroy the provenance back
  to the original bytes before anything had a chance to record it.
- **Page boundaries are the adapter's, not the reader's.** Plain text has one
  page unless it contains form feeds. That is honest, not a bug — but it means
  page locators for a `.txt` file are not very interesting.
- **An OCR fallback that raises is swallowed into a warning.** The blank page
  is kept as blank and the failure is recorded, because one unreadable image
  should not cost the whole document.

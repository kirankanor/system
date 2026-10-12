# file-type-validator: the extension is a claim, the bytes are the evidence

```
file_type_validator.py   assess() reports, validate() refuses; stdlib only
example.py               runnable: the four refusals, and a mismatch caught
tests/                   success paths and deliberate failure paths
manifest.json            schema-valid; python-magic optional, never required
```

## Purpose

An extension is something the *sender* chose. This component answers a
different question -- what do the bytes actually look like -- and treats the
extension as a second, independent witness. When the two disagree the file is
refused rather than reconciled, because a disagreement is the shape of every
content-type confusion attack there is: a `.pdf` that is really a PNG, a
`.csv` that is really an image, a `.docx` that is really a ZIP of something
else entirely.

It sits *before* the extractors. `pdf-extraction-pymupdf` and
`pdf-extraction-ocr` will each happily try to parse whatever path they are
given; this is the component that decides whether they should be asked.

## Public API

| Symbol | Shape | Notes |
|---|---|---|
| `FileTypeValidator(*, max_bytes=50 MiB, allowed=None, use_magic=False)` | | the ceiling, the accepted types, and whether to delegate to libmagic |
| `.assess(path) -> FileAssessment` | | report only; raises just when the file cannot be reported on |
| `.validate(path) -> FileAssessment` | | assess, then refuse anything not accepted |
| `FileAssessment` | frozen dataclass: `path`, `size_bytes`, `extension`, `kind`, `evidence`, `extension_agrees`, `libmagic_type` | the facts, separate from any policy |
| `FileAssessment.is_inferred` | `-> bool` | true when the kind rests on absence of a signature |
| `FileKind` | `PDF DOCX XLSX PPTX ZIP PNG JPEG GIF TIFF GZIP BZ2 RTF TEXT MARKDOWN CSV JSON HTML UNKNOWN` | what the file is |
| `FileTypeError` | base exception | catches every refusal below |
| `FileMissing` / `UnreadableFile` | `FileTypeError` | could not be examined |
| `EmptyFile` / `FileTooLarge` | `FileTypeError` | examined, and not a document |
| `UnsupportedType` | `FileTypeError` | detected, and not accepted |
| `ExtensionContentMismatch` | `FileTypeError` | the two witnesses disagree |
| `MissingOptionalDependency` | `FileTypeError` | libmagic asked for and absent |

## Example

```python
from file_type_validator import ExtensionContentMismatch, FileTypeValidator

validator = FileTypeValidator(max_bytes=10 * 1024 * 1024)

assessment = validator.assess("uploads/report.pdf")
print(assessment.kind, assessment.evidence)   # FileKind.PDF content-signature
print(assessment.extension_agrees)           # True

try:
    validator.validate("uploads/report.pdf")
except ExtensionContentMismatch as exc:
    print(exc)   # refused rather than guessed at
```

## Dependencies

None required. `python-magic` (>=0.4.27) is optional and only consulted when
you construct `FileTypeValidator(use_magic=True)`; without it the component
runs on its own signature table and says so. Note that `python-magic` wraps
`libmagic`, which is a **system** library, not a wheel -- so installing the
package is not always sufficient.

## Trade-offs

**Use it when** a pipeline ingests files from somewhere you do not control, and
you would rather refuse a strange one at the door than find out what
`pymupdf` does with it.

**It costs** a signature table that is deliberately small. It recognises
thirteen binary formats and infers six text ones; it does not attempt the long
tail libmagic covers. If you need that tail, pass `use_magic=True` and accept
the system dependency. The table is not a bug -- a small list you can read is
easier to audit than a large one you cannot, and every entry here was chosen
because a document format this library actually handles.

**Alternatives:** libmagic outright; or trusting the extension, which is what
this component exists to not do.

**When not to use it:** as a security boundary on its own. It establishes what
a file looks like, not what it contains. An accepted `.pdf` can still be a
malicious PDF, which is a question for the extractor and for
`retrieved-content-safety`, not for this.

## Known limitations

- **Text is inferred, never asserted.** A file with no signature that decodes
  as UTF-8 is reported as text with `evidence == "inferred-text"`. This is
  stated rather than hidden because it is the one place the component is
  reasoning from absence. A caller who cannot accept an inference can check
  `assessment.is_inferred` and refuse; the component will not make that choice
  for them.
- **Office formats are identified by member path, not schema.** `word/`,
  `xl/` and `ppt/` in the archive separate docx from xlsx from pptx. A ZIP
  arranged to look like an Office document passes as one. That is an accepted
  limit: the alternative is parsing OOXML, which is a different component.
- **Encrypted and password-protected containers appear as plain ZIPs.** The
  central directory is readable; the content is not.
- **A ZIP with an unreadable central directory is reported UNKNOWN**, not as a
  healthy plain zip, so a truncated or corrupt archive is refused rather than
  accepted by its signature alone.

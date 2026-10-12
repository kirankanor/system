"""Runnable example for `pdf-extraction-ocr`.

    python example.py

Reads text out of a page IMAGE using Tesseract, and shows what this component
does when you hand it a PDF instead. `pytesseract` and `pillow` are OPTIONAL
dependencies, and the tesseract binary is a third thing no Python installer
provides -- so this example runs, and exits 0, on a machine with none of them,
demonstrating the behaviour through the injection seam the tests use.

The six guarantees this example demonstrates:

1. an image comes back as text, one page at a time;
2. a PDF is REFUSED with a typed error naming both remedies;
3. an encrypted PDF is refused BEFORE any library touches it, detected from
   the trailer alone -- with no PDF library involved;
4. `language` is validated as a command-line argument, not as data;
5. `extract_many` raises on the first failure rather than returning a partial
   document;
6. the dependency is two packages plus a binary, and the error says so.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ocr import (
    MANIFEST_MESSAGE,
    MissingDependency,
    OcrExtractor,
    OcrInputError,
    PdfEncryptedError,
    PdfExtractionError,
)

WORK = Path(tempfile.mkdtemp(prefix="ragkit-ocr-example-"))


def _png(name: str, text: str = "") -> Path:
    """A stand-in for a scan. Real bytes, so the PDF check reads them."""
    path = WORK / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return path


PAGES = {
    "page-001.png": "The json module serialises Python objects into JSON text.",
    "page-002.png": "Python raises TypeError when a function receives the\nwrong type of argument.",
    "page-003.png": "   \n",
}


class FakeTesseract:
    """Stands in for pytesseract: has an `image_to_string`."""

    def __init__(self, recognised: dict[str, str]) -> None:
        self.recognised = recognised
        self.calls: list[str] = []

    def image_to_string(self, image) -> str:
        key = Path(image.name).name
        self.calls.append(key)
        return self.recognised[key]


class FakeImage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


def main() -> None:
    for name, text in PAGES.items():
        _png(name)

    engine = FakeTesseract(PAGES)
    # Two seams, one per external call: opening an image, and turning one
    # into text. Neither library is imported when both are supplied.
    ocr = OcrExtractor(image_factory=FakeImage, engine_factory=lambda: engine)

    print(f"{ocr}\n")
    print("1. An image comes back as text, one page at a time:")
    result = ocr.extract(WORK / "page-001.png")
    print(f"   {result}")
    print(f"   text: {result.pages[0].text!r}")
    print(f"   metadata: {result.metadata}")
    print("   (metadata records WHAT was asked for -- engine, language, dpi.")
    print("    It is a declaration of settings, not a report from the engine.)")

    print("\n   A blank scan is recognised as blank, not as empty text:")
    blank = ocr.extract(WORK / "page-003.png").pages[0]
    print(f"     {blank!r}  is_blank={blank.is_blank}")

    print("\n2. A PDF is REFUSED, and the refusal names both remedies:")
    pdf = WORK / "scanned.pdf"
    pdf.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    for label, path in [
        ("a PDF named .pdf", pdf),
        ("a PDF named .png", _renamed_pdf()),
    ]:
        try:
            ocr.extract(path)
            print(f"   {label:<18} NO ERROR")
        except OcrInputError as exc:
            message = str(exc)
            print(f"   {label:<18} OcrInputError")
            print(f"      names the pymupdf option: "
                  f"{'pdf-extraction-pymupdf' in message}")
            print(f"      names rasterizing       : {'rasterize' in message}")

    print("\n   The check reads the first five BYTES, not the filename, so a")
    print("   PDF named page.png is caught too. Turning a PDF page into")
    print("   pixels needs a renderer, and a renderer is precisely the")
    print("   dependency this sub-component exists to avoid.")

    print("\n3. An encrypted PDF is refused BEFORE any library touches it:")
    locked = WORK / "locked.pdf"
    locked.write_bytes(b"%PDF-1.7\n/Encrypt 5 0 R\n" + b"\x00" * 64)
    try:
        ocr.extract(locked)
        print("   NO ERROR")
    except PdfEncryptedError as exc:
        print(f"   {type(exc).__name__}")
        print(f"     carries .path : {Path(exc.path).name}")
        print(f"     is a PdfExtractionError: "
              f"{isinstance(exc, PdfExtractionError)}")
    print("   Detected from the /Encrypt marker in the first 1024 bytes,")
    print("   which the spec requires of every encrypted PDF -- so no PDF")
    print("   parser is involved. That matters: the whole point of this")
    print("   sub-component is having no PDF dependency, and detecting")
    print("   encryption with one would reintroduce what it removed.")

    print("\n4. `language` is validated as a COMMAND-LINE ARGUMENT, not data:")
    print("   Tesseract is shell-invoked, so this is an allow-list, not an")
    print("   escape. An escaping scheme has edge cases; an allow-list does")
    print("   not.")
    for label, kwargs in [
        ("language='eng+deu'", {"language": "eng+deu"}),
        ("language='eng; rm -rf /'", {"language": "eng; rm -rf /"}),
        ("language='../../etc'", {"language": "../../etc"}),
        ("language=''", {"language": ""}),
        ("dpi=10", {"dpi": 10}),
        ("dpi=5000", {"dpi": 5000}),
        ("dpi=True", {"dpi": True}),
    ]:
        try:
            made = OcrExtractor(image_factory=FakeImage, **kwargs)
            print(f"   {label:<26} accepted -> {made!r}")
        except OcrInputError as exc:
            print(f"   {label:<26} {type(exc).__name__}: "
                  f"{str(exc)[:44]}...")

    print("\n5. extract_many raises on the FIRST failure rather than")
    print("   returning a partial document -- a half-extracted document is")
    print("   worse than none, because the caller cannot tell which page")
    print("   went missing:")
    ordered = [WORK / "page-001.png", WORK / "page-002.png", WORK / "page-003.png"]
    many = ocr.extract_many(ordered)
    print(f"   {many}")
    for number, page in enumerate(many.pages, start=1):
        preview = " / ".join(page.text.split())[:46]
        print(f"     page {number}: {preview!r}")

    print("\n   ...and it refuses the shapes that would silently misbehave:")
    for label, paths in [
        ("a bare string", "page-001.png"),
        ("an empty list", []),
        ("a page that is a PDF", [pdf]),
    ]:
        try:
            ocr.extract_many(paths)
            print(f"   {label:<18} NO ERROR")
        except OcrInputError as exc:
            print(f"   {label:<18} {type(exc).__name__}: "
                  f"{str(exc)[:44]}...")
    print("   A bare string is iterable, so passing one would OCR it")
    print("   CHARACTER BY CHARACTER.")

    print("\n6. The dependency is two packages AND a binary:")
    print("   The component does not fail at import -- that is the point of")
    print("   declaring it OPTIONAL. It fails at first use, and it says what")
    print("   is missing:")
    try:
        OcrExtractor().extract(WORK / "page-001.png")
        print("      the libraries happen to be installed here")
    except MissingDependency as exc:
        print(f"   {type(exc).__name__}")
        for line in str(exc).splitlines():
            print(f"     {line}")
    print(f"\n   Verbatim from the manifest: {MANIFEST_MESSAGE!r}")

    print("\nWhat this component will not do:")
    print("   no OCR confidence is surfaced. image_to_string does not report")
    print("   it, and getting at it means a second pass over every page -- on")
    print("   the slowest component in this library. It is left out rather")
    print("   than shipped as a field that is always None, which would read")
    print("   as 'the engine was certain of nothing'.")
    print("   and it has never been run against a real Tesseract binary, so")
    print("   every guarantee above was executed through the injection seam.")


def _renamed_pdf() -> Path:
    path = WORK / "actually-a-pdf.png"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    return path


if __name__ == "__main__":
    main()

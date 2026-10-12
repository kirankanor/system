"""Runnable example for `pdf-extraction-pymupdf`.

    python example.py

Extracts text from a born-digital PDF, and shows what this component does when
a document is encrypted. `pymupdf` is an OPTIONAL dependency: if it is not
installed, this example still runs, still exits 0, and shows you the exact
message the component raises plus the extraction logic through the injection
seam the tests use.

The six guarantees this example demonstrates:

1. text comes out of a real PDF, page by page;
2. an encrypted PDF raises a TYPED error carrying .path and .needs_pass;
3. page selection is zero-based, and an out-of-range selection raises
   rather than returning a short result;
4. an empty or negative selection is a caller bug, not a result;
5. pages are joined by a blank line, not a bare newline;
6. the dependency is imported lazily -- adding the file does not break the
   package in a project that has not installed pymupdf.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

from pymupdf import (
    MANIFEST_MESSAGE,
    ExtractedPdf,
    MissingDependency,
    PdfEncryptedError,
    PdfExtractionError,
    PdfExtractor,
    PdfInputError,
)

WORK = Path(tempfile.mkdtemp(prefix="ragkit-pymupdf-example-"))
HERE = Path(__file__).resolve().parent


def _real_pymupdf() -> bool:
    """Make ``import pymupdf`` find the library rather than this component.

    This example sits beside ``pymupdf.py``, so its directory is ``sys.path[0]``
    and the name resolves to the *component* itself. Both the path entry and
    the cached ``sys.modules`` entry have to go, permanently, or the
    component's own lazy ``__import__("pymupdf")`` grabs the component again
    -- the exact hazard ``PdfExtractor`` now reports explicitly instead of
    failing three frames later with "could not be opened as a PDF".

    Dropping the cache on *failure* too is what makes the fallback honest:
    with nothing left to resolve the name, the component's lazy import
    genuinely fails and raises the real "not installed" message rather than
    the "you shadowed yourself" one.

    Nothing here is restored because the script exits immediately afterwards.

    The same hazard is why the gate test loads the component by file
    location instead of by name.
    """
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != HERE]
    sys.modules.pop("pymupdf", None)
    try:
        importlib.import_module("pymupdf")
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# A stand-in for pymupdf, used only when the real one is missing. It is the
# same shape the tests use -- this is what the injection seam is for.
# --------------------------------------------------------------------------- #
class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakeDocument:
    def __init__(self, pages, *, needs_pass=False, metadata=None) -> None:
        self._pages = list(pages)
        self.needs_pass = needs_pass
        self.metadata = metadata or {}

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> _FakePage:
        return _FakePage(self._pages[index])

    def close(self) -> None:
        pass


class _FakeModule:
    def __init__(self, document: _FakeDocument) -> None:
        self._document = document

    def open(self, path):
        return self._document


PAGES = [
    "Quarterly report, page one.\nThe json module serialises Python objects.",
    "Page two.\nPython raises TypeError when a function receives the wrong type.",
    "Page three, which has no text layer at all.",
]


def main() -> None:
    real = None
    if _real_pymupdf():
        real = _build_with_real_pymupdf(importlib.import_module("pymupdf"))

    if real is None:
        # An existing file, so the failure below is the library being absent
        # -- not a missing path, which _resolve() would report first.
        stand_in = WORK / "report.pdf"
        stand_in.write_bytes(b"%PDF-1.7 (stand-in)")
        print("pymupdf is not installed here.")
        print("The component does not fail at import -- that is the point of")
        print("declaring it OPTIONAL. It fails at first use, and says exactly")
        print("what to do:\n")
        try:
            PdfExtractor().extract(stand_in)
        except MissingDependency as exc:
            print(f"  {type(exc).__name__}")
            for line in str(exc).splitlines():
                print(f"    {line}")
        print("\nSo `ragkit add pdf-extraction-pymupdf` into a project without")
        print("pymupdf still yields an importable module. Showing the rest of")
        print("the behaviour through the injection seam the tests use:\n")
        extractor = PdfExtractor(document_factory=_FakeModule(
            _FakeDocument(PAGES, metadata={"title": "Quarterly report",
                                           "author": "", "producer": None})
        ))
        _walk_through(extractor, stand_in)
        locked = PdfExtractor(document_factory=_FakeModule(
            _FakeDocument(PAGES, needs_pass=True)
        ))
        _encrypted_walk_through(locked, stand_in)
    else:
        _walk_through(PdfExtractor(), real)
        _encrypted_walk_through(PdfExtractor(), WORK / "locked.pdf")

    print("\nBefore you add it:")
    print("  `ragkit add pdf-extraction-pymupdf` copies the file. It does NOT")
    print("  install pymupdf and does NOT edit any dependency file. Install")
    print("  it yourself, in the environment you actually run in:")
    print(f"    {MANIFEST_MESSAGE.split('Install it with: ')[1]}")


def _walk_through(extractor: PdfExtractor, path: Path) -> None:
    print("1. Text comes out of a real PDF, page by page:")
    result = extractor.extract(path)
    print(f"   {result}")
    for number, page in enumerate(result.pages, start=1):
        preview = " / ".join(line for line in page.splitlines() if line)[:52]
        print(f"   page {number}: {preview!r}")

    print("\n   The document's metadata comes along, filtered to usable values:")
    print(f"   {result.metadata}")
    print("   (empty and non-string values are dropped, so this is always a")
    print("    usable dict[str, str] -- and it is UNTRUSTED data straight from")
    print("    a file you did not write; validate it before it reaches a log")
    print("    or a prompt)")

    print("\n3. Page selection is zero-based, and an out-of-range selection")
    print("   RAISES rather than returning a short result:")
    print(f"   pages=[0]    -> {extractor.extract(path, pages=[0]).pages}")
    print(f"   page_count() without extracting -> {extractor.page_count(path)}")
    for label, pages in [
        ("pages=[9]", [9]),
        ("pages=[]", []),
        ("pages=[-1]", [-1]),
        ("pages=['0']", ["0"]),
    ]:
        try:
            extractor.extract(path, pages=pages)
            print(f"   {label:<14} NO ERROR")
        except PdfInputError as exc:
            print(f"   {label:<14} {type(exc).__name__}: {str(exc)[:56]}...")

    print("   A silently truncated extraction would be indistinguishable")
    print("   from a document that simply ended early.")

    print("\n4. Bad paths are typed errors too:")
    for label, bad in [
        ("a directory", WORK),
        ("not a string", 42),
    ]:
        try:
            extractor.extract(bad)
        except PdfInputError as exc:
            print(f"   {label:<14} {type(exc).__name__}: {str(exc)[:52]}...")

    print("\n5. Pages are joined by a BLANK line, not a bare newline:")
    print(f"   ExtractedPdf(pages=('a', 'b')).text -> "
          f"{ExtractedPdf(pages=('a', 'b')).text!r}")
    print("   a page boundary is a real break; a bare newline would run the")
    print("   last line of one page into the first line of the next")


def _encrypted_walk_through(extractor: PdfExtractor, path: Path) -> None:
    print("\n2. An encrypted PDF raises a TYPED error:")
    try:
        extractor.extract(path)
        print("   NO ERROR -- the stand-in did not report needs_pass")
    except PdfEncryptedError as exc:
        print(f"   {type(exc).__name__}")
        print(f"     carries .path      : {Path(exc.path).name}")
        print(f"     carries .needs_pass: {exc.needs_pass}")
        print(f"     is a PdfExtractionError: "
              f"{isinstance(exc, PdfExtractionError)}")
        print(f"     message            : {str(exc)[:70]}...")
    print("   It is its OWN type, not a generic failure, because the remedies")
    print("   differ: encrypted means 'you have the password, decrypt and")
    print("   retry'; a corrupt PDF means the file is wrong and nothing you")
    print("   can supply will fix it. A caller should not have to string-")
    print("   match a message to tell those apart.")
    print("   The error never echoes a password.")


def _build_with_real_pymupdf(pymupdf) -> Path:
    """Build a real PDF, and a real encrypted one, using pymupdf itself."""
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Quarterly report, page one.")
    page_one.insert_text(
        (72, 96), "The json module serialises Python objects."
    )
    page_two = document.new_page()
    page_two.insert_text(
        (72, 72),
        "Page two. Python raises TypeError on a wrong argument type.",
    )
    plain = WORK / "report.pdf"
    document.save(str(plain))
    document.close()

    source = pymupdf.open(str(plain))
    source.save(
        str(WORK / "locked.pdf"),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="correct horse",
    )
    source.close()
    return plain


if __name__ == "__main__":
    main()

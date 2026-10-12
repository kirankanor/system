"""Component tests for `pdf-extraction-pymupdf`.

Runs standalone: ``python -m pytest components/ingestion/pdf_extraction_pymupdf/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the two the component is built around: an encrypted PDF
raises a typed error carrying the facts a caller needs, and an out-of-range
page selection raises rather than returning a short result.

Why this file does not simply ``import pymupdf``
------------------------------------------------
Every other component test in this repository imports its component by name
after inserting the component's directory into ``sys.path``. This one cannot:
its file is literally called ``pymupdf.py``, so that would put the *component*
into ``sys.modules["pymupdf"]`` for the rest of the pytest session, and the
genuine library would never again be importable by name -- including from
``tests/unit/test_ingestion.py``, which exercises the real library end to end.
So the component is loaded by file location under a private name, and its
directory never touches ``sys.path``. Same technique the gate file uses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

COMPONENT_DIR = Path(__file__).resolve().parents[1]
COMPONENT_FILE = COMPONENT_DIR / "pymupdf.py"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


component = _load_by_path("_component_pymupdf_under_test", COMPONENT_FILE)

MANIFEST_MESSAGE = component.MANIFEST_MESSAGE
ExtractedPdf = component.ExtractedPdf
MissingDependency = component.MissingDependency
PdfEncryptedError = component.PdfEncryptedError
PdfExtractionError = component.PdfExtractionError
PdfExtractor = component.PdfExtractor
PdfInputError = component.PdfInputError


# --------------------------------------------------------------------------- #
# Fakes. The optional library is absent by design on some hosts, so every
# behavioural test runs through the injection seam.
# --------------------------------------------------------------------------- #


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class FakeDocument:
    def __init__(self, pages, *, needs_pass=False, metadata=None) -> None:
        self._pages = list(pages)
        self.needs_pass = needs_pass
        self.metadata = metadata if metadata is not None else {}
        self.closed = False

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> FakePage:
        return FakePage(self._pages[index])

    def close(self) -> None:
        self.closed = True


class FakeModule:
    """Stands in for pymupdf itself: has an ``open``."""

    def __init__(self, document: FakeDocument) -> None:
        self.document = document
        self.opened: list[str] = []

    def open(self, path):
        self.opened.append(str(path))
        return self.document


PAGES = ["page one text", "page two text", "page three text"]


@pytest.fixture
def pdf(tmp_path) -> Path:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.7 stand-in")
    return path


def extractor_for(pages=PAGES, **kwargs) -> PdfExtractor:
    document = FakeDocument(pages, **kwargs)
    return PdfExtractor(document_factory=FakeModule(document)), document


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #


def test_a_pdf_comes_back_as_one_string_per_page(pdf):
    extractor, _ = extractor_for()
    result = extractor.extract(pdf)
    assert isinstance(result, ExtractedPdf)
    assert result.pages == ("page one text", "page two text", "page three text")
    assert result.page_count == 3


def test_the_result_is_library_neutral(pdf):
    # No fitz.Document, no page objects, no handles -- which is what lets
    # pdf-extraction-ocr return the identical shape (FR-6).
    extractor, document = extractor_for()
    result = extractor.extract(pdf)
    assert all(isinstance(page, str) for page in result.pages)
    assert isinstance(result.metadata, dict)
    assert document.closed is True


def test_pages_are_joined_by_a_blank_line_not_a_newline(pdf):
    extractor, _ = extractor_for()
    text = extractor.extract(pdf).text
    assert text == "page one text\n\npage two text\n\npage three text"


def test_page_count_does_not_extract_any_text(tmp_path):
    path = tmp_path / "count.pdf"
    path.write_bytes(b"%PDF-1.7")
    factory = FakeModule(FakeDocument(["a", "b"]))
    extractor = PdfExtractor(document_factory=factory)
    assert extractor.page_count(path) == 2


def test_metadata_survives(pdf):
    extractor, _ = extractor_for(metadata={"title": "Report", "author": "K"})
    assert extractor.extract(pdf).metadata == {"title": "Report", "author": "K"}


def test_metadata_scalars_are_stringified(pdf):
    extractor, _ = extractor_for(metadata={"title": "T", "pages": 12})
    assert extractor.extract(pdf).metadata == {"title": "T", "pages": "12"}


def test_empty_and_unusable_metadata_values_are_dropped(pdf):
    extractor, _ = extractor_for(
        metadata={"title": "T", "author": "", "producer": None, "tags": ["a"]}
    )
    # So metadata is always a usable dict[str, str] -- never a None or a list
    # that a caller would have to defend against downstream.
    assert extractor.extract(pdf).metadata == {"title": "T"}


def test_a_document_without_metadata_yields_an_empty_mapping(pdf):
    class Bare:
        page_count = 1

        def __getitem__(self, index):
            return FakePage("text")

    result = PdfExtractor(document_factory=FakeModule(Bare())).extract(pdf)
    assert result.metadata == {}


def test_metadata_that_is_not_a_mapping_is_ignored(pdf):
    extractor, _ = extractor_for()
    extractor._document_factory.document.metadata = "not a mapping"
    assert extractor.extract(pdf).metadata == {}


def test_an_open_empty_document_yields_an_empty_result(pdf):
    extractor, _ = extractor_for([])
    result = extractor.extract(pdf)
    assert result.pages == ()
    assert result.page_count == 0
    assert result.text == ""


def test_the_source_is_the_resolved_path(pdf):
    extractor, _ = extractor_for()
    assert extractor.extract(pdf).source == str(pdf)


def test_a_string_path_is_accepted(tmp_path):
    path = tmp_path / "string.pdf"
    path.write_bytes(b"%PDF-1.7")
    extractor, _ = extractor_for()
    assert extractor.extract(str(path)).page_count == 3


# --------------------------------------------------------------------------- #
# Page selection
# --------------------------------------------------------------------------- #


def test_selection_is_zero_based(pdf):
    extractor, _ = extractor_for()
    assert extractor.extract(pdf, pages=[0]).pages == ("page one text",)
    assert extractor.extract(pdf, pages=[1, 2]).pages == (
        "page two text",
        "page three text",
    )


def test_a_selection_may_repeat_a_page(pdf):
    extractor, _ = extractor_for()
    assert extractor.extract(pdf, pages=[2, 2]).pages == (
        "page three text",
        "page three text",
    )


def test_a_selection_preserves_the_order_asked_for(pdf):
    extractor, _ = extractor_for()
    assert extractor.extract(pdf, pages=[2, 0]).pages == (
        "page three text",
        "page one text",
    )


# --------------------------------------------------------------------------- #
# The typed failures -- encrypted above all
# --------------------------------------------------------------------------- #


def test_an_encrypted_pdf_raises_a_typed_error(pdf):
    extractor, document = extractor_for(needs_pass=True)
    with pytest.raises(PdfEncryptedError) as excinfo:
        extractor.extract(pdf)
    error = excinfo.value
    assert error.needs_pass is True
    assert str(pdf) in error.path


def test_the_encrypted_error_carries_what_a_caller_branches_on(pdf):
    extractor, _ = extractor_for(needs_pass=True)
    with pytest.raises(PdfEncryptedError) as excinfo:
        extractor.extract(pdf)
    error = excinfo.value
    # Not "is it a PdfExtractionError", which is trivially true of its own
    # type -- the point is that the FACTS travel with the exception.
    assert isinstance(error, PdfExtractionError)
    assert hasattr(error, "path")
    assert hasattr(error, "needs_pass")


def test_the_encrypted_error_explains_but_never_parrots_the_document(pdf):
    document = FakeDocument(
        PAGES,
        needs_pass=True,
        metadata={"password": "hunter2", "user_pw": "s3cret"},
    )
    with pytest.raises(PdfEncryptedError) as excinfo:
        PdfExtractor(document_factory=FakeModule(document)).extract(pdf)
    message = str(excinfo.value)
    # It names the situation and the path, so a stack trace is readable.
    assert "password" in message.lower()
    assert str(pdf) in message
    # ...and it parrots nothing it found in the document. The encryption
    # check runs in open(), before a single page or metadata field is read,
    # so the error can only carry the path and the fact. That ordering is the
    # guarantee: the component is handed a document it did not write, and
    # anything it echoed could be a credential sitting in an info dictionary.
    assert "hunter2" not in message
    assert "s3cret" not in message


def test_the_document_is_closed_before_the_encrypted_error_propagates(pdf):
    document = FakeDocument(PAGES, needs_pass=True)
    try:
        PdfExtractor(document_factory=FakeModule(document)).extract(pdf)
    except PdfEncryptedError:
        pass
    assert document.closed is True


def test_an_out_of_range_selection_raises_rather_than_truncating(pdf):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(pdf, pages=[0, 9])
    assert "out of range" in str(excinfo.value)


def test_an_empty_selection_is_a_caller_bug(pdf):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(pdf, pages=[])
    assert "empty sequence" in str(excinfo.value)


def test_a_negative_index_does_not_select_from_the_end(pdf):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(pdf, pages=[-1])
    assert "non-negative" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["0", True, 1.5, None])
def test_a_non_int_selection_is_refused(pdf, bad):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError):
        extractor.extract(pdf, pages=[bad])


def test_a_string_is_not_a_selection_of_pages(pdf):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError):
        extractor.extract(pdf, pages="0")


def test_a_missing_file_is_a_typed_error(tmp_path):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(tmp_path / "absent.pdf")
    assert "no such file" in str(excinfo.value)


def test_a_directory_is_a_typed_error(tmp_path):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(tmp_path)
    assert "directory" in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 42, b"bytes.pdf", ["a.pdf"]])
def test_a_non_path_is_a_typed_error(bad):
    extractor, _ = extractor_for()
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(bad)
    assert "string or Path" in str(excinfo.value)


def test_a_page_that_cannot_be_indexed_is_reported(pdf):
    class NoIndex:
        page_count = 1

    extractor = PdfExtractor(document_factory=FakeModule(NoIndex()))
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(pdf)
    assert "could not be loaded" in str(excinfo.value)


def test_a_page_whose_text_extraction_fails_names_the_alternative(pdf):
    class Hostile:
        def get_text(self):
            raise RuntimeError("glyph table missing")

    class OnePage:
        page_count = 1
        metadata = {}

        def __getitem__(self, index):
            return Hostile()

    extractor = PdfExtractor(document_factory=FakeModule(OnePage()))
    with pytest.raises(PdfExtractionError) as excinfo:
        extractor.extract(pdf)
    message = str(excinfo.value)
    assert "could not be read as text" in message
    assert "pdf-extraction-ocr" in message  # the documented remedy


def test_a_bad_name_is_refused():
    with pytest.raises(PdfInputError):
        PdfExtractor(name="")


# --------------------------------------------------------------------------- #
# The optional dependency
# --------------------------------------------------------------------------- #


def test_the_manifest_message_is_verbatim_in_the_module():
    manifest = json.loads(
        (COMPONENT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["missing_dependency_message"] == MANIFEST_MESSAGE


def test_the_manifest_declares_no_component_dependencies():
    manifest = json.loads(
        (COMPONENT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dependencies"] == []


def test_the_dependency_is_imported_lazily_not_at_module_scope():
    # The module imported cleanly above with no pymupdf installed -- that IS
    # the assertion. Confirm the import site is inside a function.
    source = COMPONENT_FILE.read_text(encoding="utf-8")
    assert "__import__(module_name" in source
    for line in source.splitlines():
        if line.startswith("import pymupdf") or line.startswith("import fitz"):
            raise AssertionError(f"eager import at module scope: {line!r}")


def test_missing_dependency_is_both_vocabulary():
    assert issubclass(MissingDependency, PdfExtractionError)
    assert issubclass(MissingDependency, ImportError)


def test_the_errors_form_one_hierarchy():
    assert issubclass(PdfInputError, PdfExtractionError)
    assert issubclass(PdfEncryptedError, PdfExtractionError)
    assert not issubclass(PdfExtractionError, ValueError)


def test_the_repr_reports_counts_not_the_text():
    result = ExtractedPdf(pages=("abc", "de"), source="s.pdf")
    # 3 + 2 + the two-char blank-line join = 7, not 6.
    assert repr(result) == (
        "ExtractedPdf(source='s.pdf', page_count=2, character_count=7)"
    )


def test_the_result_is_frozen():
    result = ExtractedPdf(pages=("a",))
    with pytest.raises(Exception):
        result.pages = ("b",)  # type: ignore[misc]


def test_the_real_pymupdf_end_to_end(tmp_path):
    """Against actual pymupdf, when the host has it -- never a fake.

    Skipped rather than installed: installing would modify dependency files,
    which this project forbids.
    """
    library = _real_library()
    if library is None:
        pytest.skip("pymupdf is not installed; the fakes above still cover it")

    document = library.open()
    document.new_page().insert_text((72, 72), "Hello from a real PDF")
    path = tmp_path / "real.pdf"
    document.save(str(path))
    document.close()

    result = PdfExtractor().extract(path)
    assert result.page_count == 1
    assert "Hello from a real PDF" in result.pages[0]

    locked = tmp_path / "real_locked.pdf"
    reopended = library.open(str(path))
    reopended.save(
        str(locked),
        encryption=library.PDF_ENCRYPT_AES_256,
        user_pw="correct horse",
    )
    reopended.close()

    with pytest.raises(PdfEncryptedError) as excinfo:
        PdfExtractor().extract(locked)
    assert excinfo.value.needs_pass is True


def _real_library():
    """Import the genuine pymupdf, if this host happens to have it."""
    import importlib

    saved_path = list(sys.path)
    cached = sys.modules.pop("pymupdf", None)
    try:
        sys.path[:] = [p for p in saved_path if Path(p).resolve() != COMPONENT_DIR]
        return importlib.import_module("pymupdf")
    except ImportError:
        return None
    finally:
        if cached is not None:
            sys.modules["pymupdf"] = cached
        sys.path[:] = saved_path


# --------------------------------------------------------------------------- #
# No network, no services
# --------------------------------------------------------------------------- #


def test_the_component_opens_no_sockets_and_calls_no_services():
    source = COMPONENT_FILE.read_text(encoding="utf-8")
    for banned in (
        "import socket",
        "import httpx",
        "import requests",
        "import urllib",
        "import openai",
        "import anthropic",
    ):
        assert banned not in source

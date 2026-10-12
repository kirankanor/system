"""Component tests for `document-loader`.

Runs standalone: ``python -m pytest components/ingestion/document_loader/tests``
from the repository root. Does not import `rag_code_library`.

Two properties carry the weight here. Detection must be based on bytes and
must *name its evidence*, because a loader that quietly trusts an extension is
how a PDF gets handed to a text parser. And the two PDF extractors, which
return genuinely different page shapes, must come out the other side
identical -- which is the reason this module imports neither of them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from document_loader import (
    AdapterError,
    Detection,
    DocumentLoader,
    EmptyDocument,
    Evidence,
    HtmlAdapter,
    LoaderError,
    LoadedDocument,
    MarkdownAdapter,
    MissingAdapter,
    Page,
    PdfAdapter,
    TextAdapter,
    UnreadableDocument,
    UnsupportedFormat,
    detect_format,
)

PDF_BYTES = b"%PDF-1.7\n1 0 obj\ntrailer\n"
BINARY_BYTES = bytes(range(256)) * 8


# ---------------------------------------------------------------------------
# Fixtures and stand-ins
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "notes.txt").write_text("alpha\n\f\nbeta\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\n\nBody.\n", encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "<!doctype html><html><head><title>Demo</title></head>"
        "<body><h1>H</h1><p>One.</p><p>Two.</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "report.pdf").write_bytes(PDF_BYTES)
    (tmp_path / "blob.bin").write_bytes(BINARY_BYTES)
    (tmp_path / "empty.txt").write_bytes(b"")
    return tmp_path


class StringPagesExtractor:
    """pdf-extraction-pymupdf's shape: pages are plain strings."""

    def extract(self, path):
        return SimpleNamespace(pages=("one", "two"), metadata={}, source=str(path))


class ObjectPagesExtractor:
    """pdf-extraction-ocr's shape: pages carry a .text attribute."""

    def extract(self, path):
        pages = (SimpleNamespace(text="one"), SimpleNamespace(text="two"))
        return SimpleNamespace(pages=pages, metadata={}, source=str(path))


def _pdf_loader(extractor) -> DocumentLoader:
    return DocumentLoader(adapters={"pdf": PdfAdapter(extractor)})


# ---------------------------------------------------------------------------
# Detection: bytes decide, and the evidence is named
# ---------------------------------------------------------------------------


def test_a_pdf_signature_beats_a_text_extension(tmp_path):
    """The whole point: a .txt file containing PDF bytes is a PDF."""
    lying = tmp_path / "actually-a.pdf.txt"
    lying.write_bytes(PDF_BYTES)
    detection = detect_format(lying)
    assert detection.format == "pdf"
    assert detection.evidence is Evidence.CONTENT_SIGNATURE
    assert "PDF" in detection.detail


def test_an_html_signature_beats_a_markdown_extension(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("<!doctype html><html><body>hi</body></html>", encoding="utf-8")
    detection = detect_format(page)
    assert detection.format == "html"
    assert detection.evidence is Evidence.CONTENT_SIGNATURE


def test_markdown_is_detected_by_extension_and_says_so(root):
    detection = detect_format(root / "guide.md")
    assert detection.format == "markdown"
    assert detection.evidence is Evidence.DECLARED_EXTENSION
    assert "no byte signature" in detection.detail


def test_plain_text_is_a_default_and_says_so(root):
    detection = detect_format(root / "notes.txt")
    assert detection.format == "text"
    assert detection.evidence is Evidence.DEFAULTED_TO_TEXT


def test_binary_with_no_signature_is_reported_as_unknown(root):
    detection = detect_format(root / "blob.bin")
    assert detection.format == "unknown"
    assert detection.evidence is Evidence.NO_MATCH
    assert "not valid UTF-8" in detection.detail


def test_detection_reads_only_a_bounded_prefix(tmp_path):
    big = tmp_path / "big.pdf"
    big.write_bytes(PDF_BYTES + b"\x00" * 200_000)
    assert detect_format(big).format == "pdf"


def test_a_missing_path_is_refused_at_detection(tmp_path):
    with pytest.raises(UnreadableDocument) as excinfo:
        detect_format(tmp_path / "nope.md")
    assert "nope.md" in str(excinfo.value)


def test_a_directory_is_refused_at_detection(root):
    directory = root / "sub"
    directory.mkdir()
    with pytest.raises(UnreadableDocument):
        detect_format(directory)


# ---------------------------------------------------------------------------
# Loading: one shape out
# ---------------------------------------------------------------------------


def test_text_splits_on_form_feed_into_locatable_pages(root):
    doc = DocumentLoader().load(root / "notes.txt")
    assert doc.format == "text"
    assert doc.page_count == 2
    assert [p.number for p in doc.pages] == [1, 2]
    assert [p.source_locator for p in doc.pages] == ["page 1", "page 2"]
    assert "alpha" in doc.pages[0].text
    assert "beta" in doc.pages[1].text


def test_a_document_without_form_feeds_is_one_page(root):
    (root / "single.txt").write_bytes(b"just one page\n")
    doc = DocumentLoader().load(root / "single.txt")
    assert doc.page_count == 1
    assert doc.text == "just one page\n"


def test_line_endings_are_preserved_not_normalized(root):
    """A boundary worth pinning: loading must not transform.

    ``text-normalizer`` owns whitespace and line endings. If the loader
    silently rewrote them here, provenance back to the original bytes would be
    lost before anyone had a chance to keep it.
    """
    (root / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    doc = DocumentLoader().load(root / "crlf.txt")
    assert doc.text == "a\r\nb\r\n"
    assert doc.pages[0].text == "a\r\nb\r\n"


def test_markdown_loads_with_a_warning_that_it_was_inferred(root):
    doc = DocumentLoader().load(root / "guide.md")
    assert doc.format == "markdown"
    assert doc.loader == "markdown"
    assert any("declared-extension" in warning for warning in doc.warnings)


def test_html_is_reduced_to_text_and_keeps_its_title(root):
    doc = DocumentLoader().load(root / "page.html")
    assert doc.format == "html"
    assert doc.metadata["title"] == "Demo"
    assert "One." in doc.text
    assert "Two." in doc.text
    assert "<p>" not in doc.text


def test_html_block_tags_do_not_run_paragraphs_together(root):
    doc = DocumentLoader().load(root / "page.html")
    assert "One.\nTwo." in doc.text or "One.\n\nTwo." in doc.text


def test_page_count_and_character_count_agree_with_the_pages(root):
    doc = DocumentLoader().load(root / "guide.md")
    assert doc.page_count == len(doc.pages)
    assert doc.character_count == len(doc.text)


def test_the_detection_is_carried_onto_the_document(root):
    doc = DocumentLoader().load(root / "notes.txt")
    assert isinstance(doc.detection, Detection)
    assert doc.detection.evidence is Evidence.DEFAULTED_TO_TEXT


# ---------------------------------------------------------------------------
# The two PDF extractor shapes must come out identical
# ---------------------------------------------------------------------------


def test_both_pdf_extractor_shapes_normalise_identically(root):
    from_string = _pdf_loader(StringPagesExtractor()).load(root / "report.pdf")
    from_object = _pdf_loader(ObjectPagesExtractor()).load(root / "report.pdf")

    assert [p.text for p in from_string.pages] == ["one", "two"]
    assert [p.text for p in from_object.pages] == ["one", "two"]
    assert [p.source_locator for p in from_string.pages] == [
        p.source_locator for p in from_object.pages
    ]
    assert from_string.page_count == from_object.page_count


def test_a_pdf_page_never_loses_its_locator(root):
    doc = _pdf_loader(StringPagesExtractor()).load(root / "report.pdf")
    assert doc.pages[1].source_locator == "page 2"


def test_a_pdf_result_metadata_is_preserved(root):
    class WithMetadata:
        def extract(self, path):
            return SimpleNamespace(pages=("x",), metadata={"title": "T"}, source=str(path))

    doc = _pdf_loader(WithMetadata()).load(root / "report.pdf")
    assert doc.metadata["title"] == "T"


def test_a_page_that_is_neither_a_string_nor_text_bearing_is_rejected(root):
    class BadPages:
        def extract(self, path):
            return SimpleNamespace(pages=(object(),), metadata={}, source=str(path))

    with pytest.raises(AdapterError) as excinfo:
        _pdf_loader(BadPages()).load(root / "report.pdf")
    assert "neither a str nor an object with a .text" in str(excinfo.value)


def test_an_extractor_without_a_pages_attribute_is_rejected(root):
    class NoPages:
        def extract(self, path):
            return SimpleNamespace(metadata={}, source=str(path))

    with pytest.raises(AdapterError) as excinfo:
        _pdf_loader(NoPages()).load(root / "report.pdf")
    assert "no .pages attribute" in str(excinfo.value)


def test_an_extractor_that_raises_is_reported_with_its_type(root):
    class Exploding:
        def extract(self, path):
            raise ValueError("damaged xref")

    with pytest.raises(AdapterError) as excinfo:
        _pdf_loader(Exploding()).load(root / "report.pdf")
    assert "ValueError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_an_extractor_result_is_not_swallowed_into_a_false_success(root):
    class WrongReturn:
        def extract(self, path):
            return "a bare string"

    with pytest.raises(AdapterError):
        _pdf_loader(WrongReturn()).load(root / "report.pdf")


def test_wrapping_something_that_cannot_extract_is_refused_immediately():
    with pytest.raises(AdapterError) as excinfo:
        PdfAdapter(object())
    assert "callable .extract(path)" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OCR is a fallback, and only ever a fallback
# ---------------------------------------------------------------------------


def test_ocr_runs_only_on_the_pages_that_came_back_blank(root):
    class BlankMiddle:
        def extract(self, path):
            return SimpleNamespace(pages=("real text", "", "   "), metadata={}, source=str(path))

    calls: list[int] = []

    def fallback(path, number):
        calls.append(number)
        return f"ocr page {number}"

    loader = DocumentLoader(adapters={"pdf": PdfAdapter(BlankMiddle())}, ocr_fallback=fallback)
    doc = loader.load(root / "report.pdf")

    assert calls == [2, 3]
    assert [p.text for p in doc.pages] == ["real text", "ocr page 2", "ocr page 3"]
    assert doc.pages[1].source_locator == "page 2"
    assert any("OCR fallback used" in warning for warning in doc.warnings)


def test_no_fallback_is_called_when_every_page_has_text(root):
    class AllText:
        def extract(self, path):
            return SimpleNamespace(pages=("a", "b"), metadata={}, source=str(path))

    def fallback(path, number):
        raise AssertionError("must not be called")

    doc = DocumentLoader(adapters={"pdf": PdfAdapter(AllText())}, ocr_fallback=fallback).load(
        root / "report.pdf"
    )
    assert doc.warnings == ()


def test_a_fallback_that_raises_keeps_the_blank_page_and_warns(root):
    class BlankMiddle:
        def extract(self, path):
            return SimpleNamespace(pages=("real", ""), metadata={}, source=str(path))

    def fallback(path, number):
        raise RuntimeError("tesseract not installed")

    loader = DocumentLoader(adapters={"pdf": PdfAdapter(BlankMiddle())}, ocr_fallback=fallback)
    doc = loader.load(root / "report.pdf")

    assert doc.pages[1].text == ""
    assert any("OCR fallback raised RuntimeError" in warning for warning in doc.warnings)
    assert doc.pages[0].text == "real"


def test_blank_pages_are_listed_by_number(root):
    class BlankMiddle:
        def extract(self, path):
            return SimpleNamespace(pages=("a", "", "c"), metadata={}, source=str(path))

    doc = _pdf_loader(BlankMiddle()).load(root / "report.pdf")
    assert doc.blank_pages == (2,)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_an_empty_file_is_refused_rather_than_parsed_as_nothing(root):
    with pytest.raises(EmptyDocument) as excinfo:
        DocumentLoader().load(root / "empty.txt")
    assert "zero bytes" in str(excinfo.value)


def test_a_missing_file_names_the_path(root, tmp_path):
    with pytest.raises(UnreadableDocument) as excinfo:
        DocumentLoader().load(tmp_path / "ghost.md")
    assert "ghost.md" in str(excinfo.value)


def test_a_directory_is_not_a_document(root):
    directory = root / "sub"
    directory.mkdir()
    with pytest.raises(UnreadableDocument):
        DocumentLoader().load(directory)


def test_binary_content_is_refused_and_names_the_adapters(root):
    with pytest.raises(UnsupportedFormat) as excinfo:
        DocumentLoader().load(root / "blob.bin")
    message = str(excinfo.value)
    assert "blob.bin" in message
    assert "html" in message and "markdown" in message and "text" in message


def test_a_pdf_with_no_adapter_is_refused_not_silently_parsed(root):
    with pytest.raises(MissingAdapter) as excinfo:
        DocumentLoader().load(root / "report.pdf")
    assert "pdf" in str(excinfo.value)
    assert "Available:" in str(excinfo.value)


def test_the_built_in_formats_are_always_available(root):
    """Adding adapters extends the set; it cannot accidentally remove one."""
    loader = DocumentLoader(adapters={})
    assert loader.formats == ("html", "markdown", "text")
    assert loader.load(root / "guide.md").format == "markdown"


def test_the_error_hierarchy_has_one_root():
    for exc in (UnreadableDocument, EmptyDocument, UnsupportedFormat, MissingAdapter, AdapterError):
        assert issubclass(exc, LoaderError)


def test_an_adapter_that_is_not_callable_is_refused():
    with pytest.raises(AdapterError):
        DocumentLoader(adapters={"pdf": object()})


@pytest.mark.parametrize("bad", [0, -1, True, "8192"])
def test_a_non_positive_prefix_is_refused(bad):
    with pytest.raises(AdapterError):
        DocumentLoader(prefix_bytes=bad)


# ---------------------------------------------------------------------------
# Results are values
# ---------------------------------------------------------------------------


def test_a_loaded_document_is_immutable(root):
    doc = DocumentLoader().load(root / "guide.md")
    with pytest.raises(AttributeError):
        doc.format = "text"  # type: ignore[misc]


def test_a_page_is_immutable():
    with pytest.raises(AttributeError):
        Page(number=1, text="x", source_locator="page 1").text = "y"  # type: ignore[misc]


def test_a_blank_page_reports_itself():
    assert Page(number=1, text="   ", source_locator="page 1").is_blank
    assert not Page(number=1, text="x", source_locator="page 1").is_blank


def test_the_builtin_adapters_agree_on_their_format_name():
    assert (TextAdapter().format, MarkdownAdapter().format, HtmlAdapter().format) == (
        "text",
        "markdown",
        "html",
    )


def test_the_component_imports_with_no_third_party_packages():
    import document_loader

    source = Path(document_loader.__file__).read_text(encoding="utf-8")
    for forbidden in ("pydantic", "numpy", "requests", "bs4", "lxml", "fitz"):
        assert f"import {forbidden}" not in source


def test_the_document_carries_the_loader_that_produced_it(root):
    assert DocumentLoader().load(root / "guide.md").loader == "markdown"


def test_the_repr_does_not_dump_the_whole_document(root):
    doc = DocumentLoader().load(root / "guide.md")
    assert "guide.md" in repr(doc)
    assert "Body." not in repr(doc)


def test_a_loaded_document_reports_the_source_path_it_came_from(root):
    doc = DocumentLoader().load(root / "notes.txt")
    assert doc.source_path == root / "notes.txt"
    assert isinstance(doc, LoadedDocument)

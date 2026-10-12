"""Component tests for `pdf-extraction-ocr`.

Runs standalone: ``python -m pytest components/ingestion/pdf_extraction_ocr/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the two the component is built around: a PDF is refused with
a typed error naming what it would need, and an encrypted PDF is refused
before any library touches it, detected from the trailer alone.

No Tesseract binary has ever been run against this component, so every
behavioural test goes through the two public injection seams: ``image_factory``
for opening an image, ``engine_factory`` for recognising one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from ocr import (
    MANIFEST_MESSAGE,
    ExtractedPage,
    MissingDependency,
    OcrExtractor,
    OcrInputError,
    OcrResult,
    PdfEncryptedError,
    PdfExtractionError,
)

COMPONENT_DIR = Path(__file__).resolve().parents[1]
COMPONENT_FILE = COMPONENT_DIR / "ocr.py"


# --------------------------------------------------------------------------- #
# Fakes -- the two seams, one per external call.
# --------------------------------------------------------------------------- #


class FakeImage:
    def __init__(self, path) -> None:
        self.path = path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEngine:
    """Stands in for pytesseract."""

    def __init__(self, recognised: dict[str, str] | str = "") -> None:
        self.recognised = recognised
        self.calls: list[Path] = []

    def image_to_string(self, image) -> str:
        self.calls.append(image.path)
        if isinstance(self.recognised, dict):
            return self.recognised[Path(image.path).name]
        return self.recognised


@pytest.fixture
def image(tmp_path) -> Path:
    path = tmp_path / "page-001.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return path


@pytest.fixture
def encrypted(tmp_path) -> Path:
    path = tmp_path / "locked.pdf"
    path.write_bytes(b"%PDF-1.7\n/Encrypt 5 0 R\n" + b"\x00" * 64)
    return path


def ocr_with(recognised="", **kwargs) -> tuple[OcrExtractor, FakeEngine]:
    engine = FakeEngine(recognised)
    return (
        OcrExtractor(
            image_factory=FakeImage, engine_factory=lambda: engine, **kwargs
        ),
        engine,
    )


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #


def test_an_image_comes_back_as_text(image):
    extractor, _ = ocr_with("The json module serialises Python objects.\n")
    result = extractor.extract(image)
    assert isinstance(result, OcrResult)
    assert result.pages[0].text == "The json module serialises Python objects.\n"
    assert result.page_count == 1


def test_the_result_is_the_same_shape_as_the_pymupdf_ones(image):
    extractor, _ = ocr_with("text")
    result = extractor.extract(image)
    # The FR-6 promise: whichever sub-component you chose is interchangeable
    # downstream of the extraction call.
    assert isinstance(result.pages, tuple)
    assert isinstance(result.metadata, dict)
    assert isinstance(result.text, str)
    assert isinstance(result.source, str)


def test_metadata_records_what_was_asked_for(image):
    extractor, _ = ocr_with("x", language="eng+deu", dpi=400)
    result = extractor.extract(image)
    assert result.metadata == {
        "engine": "tesseract",
        "language": "eng+deu",
        "dpi": "400",
    }
    assert result.source == str(image)


def test_many_images_become_consecutive_pages(image, tmp_path):
    second = tmp_path / "page-002.png"
    second.write_bytes(b"\x89PNG\r\n\x1a\n")
    extractor, _ = ocr_with({image.name: "one", second.name: "two"})
    result = extractor.extract_many([image, second])
    assert result.page_count == 2
    assert result.text == "one\n\ntwo"
    assert result.source == str(image)


def test_pages_are_joined_by_a_blank_line(image):
    extractor, _ = ocr_with("a")
    assert extractor.extract_many([image, image]).text == "a\n\na"


def test_a_blank_page_is_recognised():
    assert ExtractedPage(text="   \n").is_blank is True
    assert ExtractedPage(text="").is_blank is True
    assert ExtractedPage(text="text").is_blank is False


def test_the_image_is_closed_after_ocr(image):
    opened = []

    def factory(path):
        handle = FakeImage(path)
        opened.append(handle)
        return handle

    extractor = OcrExtractor(
        image_factory=factory, engine_factory=lambda: FakeEngine("x")
    )
    extractor.extract(image)
    assert opened[0].closed is True


def test_the_image_is_closed_even_when_ocr_fails(image):
    opened = []

    def factory(path):
        handle = FakeImage(path)
        opened.append(handle)
        return handle

    class Exploding:
        def image_to_string(self, image):
            raise RuntimeError("tesseract is not on PATH")

    extractor = OcrExtractor(
        image_factory=factory, engine_factory=lambda: Exploding()
    )
    with pytest.raises(PdfExtractionError) as excinfo:
        extractor.extract(image)
    assert "tesseract binary is on PATH" in str(excinfo.value)
    assert opened[0].closed is True


def test_extract_one_returns_a_single_page(image):
    extractor, _ = ocr_with("just this")
    assert extractor.extract_one(image).text == "just this"


# --------------------------------------------------------------------------- #
# Refusing a PDF -- the component's defining behaviour
# --------------------------------------------------------------------------- #


def test_a_pdf_is_refused_with_a_typed_error(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    extractor, _ = ocr_with("never reached")
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract(path)
    assert isinstance(excinfo.value, PdfExtractionError)


def test_the_refusal_names_both_remedies(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract(path)
    message = str(excinfo.value)
    assert "pdf-extraction-pymupdf" in message
    assert "rasterize" in message


def test_the_check_reads_bytes_not_the_filename(tmp_path):
    # A PDF named .png is still a PDF.
    path = tmp_path / "actually-a-pdf.png"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError):
        extractor.extract(path)


def test_a_png_named_pdf_is_not_refused(tmp_path):
    # ...and the converse: a real image is not refused for its extension.
    path = tmp_path / "actually-an-image.pdf"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    extractor, _ = ocr_with("text")
    assert extractor.extract(path).pages[0].text == "text"


def test_an_encrypted_pdf_is_refused_as_its_own_type(encrypted):
    extractor, _ = ocr_with()
    with pytest.raises(PdfEncryptedError) as excinfo:
        extractor.extract(encrypted)
    assert isinstance(excinfo.value, PdfExtractionError)
    assert excinfo.value.path == str(encrypted)
    assert "encrypted" in str(excinfo.value)


def test_the_encrypted_check_runs_before_any_library_is_touched(encrypted):
    # No engine, no image factory: if the check needed a library, this would
    # raise MissingDependency instead. The /Encrypt marker is read straight
    # from the first 1024 bytes of the trailer.
    with pytest.raises(PdfEncryptedError):
        OcrExtractor().extract(encrypted)
    assert OcrExtractor._looks_like_pdf(encrypted) is True
    assert OcrExtractor._is_encrypted_pdf(encrypted) is True


def test_an_unencrypted_pdf_is_rejected_but_not_as_encrypted(tmp_path):
    path = tmp_path / "plain.pdf"
    path.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    with pytest.raises(OcrInputError) as excinfo:
        OcrExtractor().extract(path)
    assert not isinstance(excinfo.value, PdfEncryptedError)


def test_a_pdf_inside_extract_many_also_raises(image, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    extractor, _ = ocr_with("x")
    with pytest.raises(OcrInputError):
        extractor.extract_many([image, pdf])


# --------------------------------------------------------------------------- #
# extract_many's contract
# --------------------------------------------------------------------------- #


def test_a_bare_string_is_refused(image):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract_many("page.png")
    assert "iterable" in str(excinfo.value)


def test_an_empty_selection_is_a_caller_bug(image):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract_many([])
    assert "empty" in str(excinfo.value)


def test_many_raises_on_the_first_failure(image, tmp_path):
    # A half-extracted document is worse than none: the caller would have no
    # way to tell which page went missing.
    second = tmp_path / "missing.png"
    extractor, engine = ocr_with({image.name: "one"})
    with pytest.raises(OcrInputError):
        extractor.extract_many([image, second])
    assert engine.calls == [image]  # it stopped rather than carrying on


def test_non_sequence_paths_are_refused(image):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError):
        extractor.extract_many(42)


# --------------------------------------------------------------------------- #
# Configuration is validated before any library is imported
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("label", "kwargs", "needle"), [
    ("language blank", {"language": "  "}, "non-empty"),
    ("language injection", {"language": "eng; rm -rf /"}, "not letters"),
    ("language traversal", {"language": "../../etc/passwd"}, "not letters"),
    ("language a glob", {"language": "eng*"}, "not letters"),
    ("language a quote", {"language": "eng'"}, "not letters"),
    ("dpi too low", {"dpi": 10}, "between 72 and 1200"),
    ("dpi too high", {"dpi": 5000}, "between 72 and 1200"),
    ("dpi a bool", {"dpi": True}, "must be an int"),
    ("dpi a float", {"dpi": 300.5}, "must be an int"),
    ("dpi a string", {"dpi": "300"}, "must be an int"),
    ("dpi None", {"dpi": None}, "must be an int"),
    ("name empty", {"name": ""}, "non-empty"),
    ("name None", {"name": None}, "non-empty"),
])
def test_a_nonsense_configuration_is_refused_at_construction(label, kwargs, needle):
    with pytest.raises(OcrInputError) as excinfo:
        OcrExtractor(**kwargs)
    assert needle in str(excinfo.value), (label, excinfo.value)


def test_the_language_allow_list_is_documented_not_escaped():
    # An escaping scheme has edge cases; an allow-list does not. Tesseract is
    # shell-invoked, so this value becomes a command-line ARGUMENT.
    for accepted in ("eng", "eng+deu", "osd", "eng+fra+deu", "chi_sim"):
        assert OcrExtractor(language=accepted).language == accepted


def test_the_dpi_bounds_are_inclusive():
    assert OcrExtractor(dpi=72).dpi == 72
    assert OcrExtractor(dpi=1200).dpi == 1200


def test_the_repr_reports_the_settings():
    assert repr(OcrExtractor()) == (
        "OcrExtractor(language='eng', dpi=300, name='pdf-extraction-ocr')"
    )
    assert repr(OcrExtractor(language="eng+deu", dpi=400)) == (
        "OcrExtractor(language='eng+deu', dpi=400, name='pdf-extraction-ocr')"
    )


# --------------------------------------------------------------------------- #
# Path handling
# --------------------------------------------------------------------------- #


def test_a_missing_file_is_a_typed_error(tmp_path):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract(tmp_path / "absent.png")
    assert "no such file" in str(excinfo.value)


def test_a_directory_is_a_typed_error(tmp_path):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract(tmp_path)
    assert "directory" in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["a.png"]])
def test_a_non_path_is_a_typed_error(bad):
    extractor, _ = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        extractor.extract(bad)
    assert "string or Path" in str(excinfo.value)


def test_a_string_path_is_accepted(tmp_path):
    path = tmp_path / "s.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    extractor, _ = ocr_with("text")
    assert extractor.extract(str(path)).pages[0].text == "text"


# --------------------------------------------------------------------------- #
# The seams, and the optional dependency
# --------------------------------------------------------------------------- #


def test_the_engine_seam_is_public_constructor_state(image):
    # No poking at _libs. If a caller cannot reach it, it is not a seam.
    extractor, engine = ocr_with("x")
    assert extractor._engine_factory is not None
    extractor.extract(image)
    assert engine.calls == [image]


def test_the_image_seam_alone_still_requires_the_engine(image, monkeypatch):
    # Standing in for half the boundary must not report success. Opening an
    # image without an engine cannot OCR anything, so the real import still
    # happens and still raises when absent.
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    extractor = OcrExtractor(image_factory=FakeImage)
    with pytest.raises(MissingDependency):
        extractor.extract(image)


def test_the_engine_seam_alone_still_requires_pillow(tmp_path, monkeypatch):
    path = tmp_path / "p.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setitem(sys.modules, "PIL", None)
    extractor = OcrExtractor(engine_factory=lambda: FakeEngine("x"))
    with pytest.raises(MissingDependency):
        extractor.extract(path)


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
    assert [p["name"] for p in manifest["optional_packages"]] == [
        "pytesseract",
        "pillow",
    ]


def test_the_dependency_is_imported_lazily_not_at_module_scope():
    source = COMPONENT_FILE.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import pytesseract") or stripped.startswith(
            "from pytesseract"
        ):
            raise AssertionError(f"eager import at module scope: {stripped!r}")
        if stripped.startswith("import PIL") or stripped.startswith("from PIL"):
            raise AssertionError(f"eager import at module scope: {stripped!r}")


def test_the_missing_dependency_error_names_the_binary(monkeypatch):
    # Installing the package alone is not enough, and the error must say so
    # rather than letting the caller discover it from TesseractNotFoundError.
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(MissingDependency) as excinfo:
        OcrExtractor().extract(_any_png())
    message = str(excinfo.value)
    assert "BINARY on PATH" in message
    assert "installing the package alone will still fail" in message


def test_missing_dependency_is_both_vocabulary():
    assert issubclass(MissingDependency, PdfExtractionError)
    assert issubclass(MissingDependency, ImportError)


def test_the_errors_form_one_hierarchy():
    assert issubclass(OcrInputError, PdfExtractionError)
    assert issubclass(PdfEncryptedError, PdfExtractionError)
    assert not issubclass(PdfExtractionError, ValueError)


def _any_png() -> Path:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "any.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def test_the_result_is_frozen():
    result = OcrResult(pages=(ExtractedPage(text="a"),))
    with pytest.raises(Exception):
        result.pages = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        ExtractedPage(text="a").text = "b"  # type: ignore[misc]


def test_the_repr_reports_counts_not_the_text():
    result = OcrResult(
        pages=(ExtractedPage(text="abc"), ExtractedPage(text="de")),
        source="s.png",
    )
    # 3 + 2 + the two-char blank-line join = 7.
    assert repr(result) == (
        "OcrResult(source='s.png', page_count=2, character_count=7)"
    )


def test_confidence_is_deliberately_not_surfaced():
    # image_to_string does not report confidence, and no Tesseract binary has
    # ever been run against this file -- so a number here would be a field
    # nothing had ever verified. Left out rather than shipped as None, which
    # would read as "the engine was certain of nothing".
    extractor, _ = ocr_with("x")
    page = extractor.extract(_any_png()).pages[0]
    assert not hasattr(page, "confidence")


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

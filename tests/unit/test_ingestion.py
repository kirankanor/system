"""Gate tests for the ingestion area (T-09f).

Three components:

* ``text-normalizer``            -- normalize extracted text, idempotently;
* ``pdf-extraction-pymupdf``     -- born-digital PDFs, via an optional library;
* ``pdf-extraction-ocr``         -- page images, via Tesseract, no PDF library.

The four properties the task names as observable are asserted here by name:

1. unicode normalization is idempotent,
2. control characters are stripped,
3. an encrypted PDF raises a typed error,
4. each sub-component is independently selectable.

On selectability (FR-6)
-----------------------
The two PDF extractors differ by library, so the human picks one and takes
only that one's dependency. That is a real, testable property, and it is
asserted below rather than assumed:

* neither module imports the other -- AST-checked, so a stray import cannot
  hide behind a lazy call;
* each declares no component dependencies, so ``ragkit add`` of one copies
  one file, not a graph;
* each raises its own missing-dependency error naming **only its own**
  packages, so installing one is never implied by the other;
* both return the same result shape, so whichever was chosen is interchangeable
  downstream of the extraction call.

None of this needs the optional libraries installed. ``pymupdf`` and
``pytesseract``/``pillow`` are absent by design: installing them would modify
dependency files, which this project forbids. The dependency paths are
exercised through the real ``ImportError`` guard, and the extraction logic
through the components' injection seams.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INGESTION = ROOT / "components" / "ingestion"

for directory in (INGESTION / "text_normalizer", INGESTION / "pdf_extraction_ocr"):
    sys.path.insert(0, str(directory))

# The pymupdf component's file is named ``pymupdf.py`` -- the SPEC design table
# names it that, and the destination keeps the name. Adding its directory to
# ``sys.path`` would put the *component* into ``sys.modules["pymupdf"]`` for the
# rest of the session, so the real library could never be imported by name and
# the end-to-end test below would silently exercise itself. It is therefore
# loaded by file location under a private name, which never claims that name.
import importlib.util  # noqa: E402


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_by_path(
    "_ingestion_pymupdf_component",
    INGESTION / "pdf_extraction_pymupdf" / "pymupdf.py",
)

from _ingestion_pymupdf_component import (  # noqa: E402
    MANIFEST_MESSAGE as PYMUPDF_MANIFEST_MESSAGE,
    ExtractedPdf,
    MissingDependency as PyMissingDependency,
    PdfEncryptedError,
    PdfExtractionError,
    PdfExtractor,
    PdfInputError,
)
from ocr import (  # noqa: E402
    MANIFEST_MESSAGE as OCR_MANIFEST_MESSAGE,
    ExtractedPage,
    MissingDependency as OcrMissingDependency,
    OcrExtractor,
    OcrInputError,
    OcrResult,
    PdfEncryptedError as OcrPdfEncryptedError,
    PdfExtractionError as OcrPdfExtractionError,
)
from text_normalizer import (  # noqa: E402
    FORMS,
    NormalizationResult,
    NormalizerError,
    NormalizerInputError,
    TextNormalizer,
    normalize_text,
)

# --------------------------------------------------------------------------- #
# Fakes. The optional libraries are absent by design, so extraction runs
# through the injection seams and the dependency paths run through the real
# ImportError guard.
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
        self.metadata = metadata or {}
        self.closed = False

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> FakePage:
        if index >= len(self._pages):
            raise IndexError(index)
        return FakePage(self._pages[index])

    def close(self) -> None:
        self.closed = True


class _FakeModule:
    """A stand-in for ``pymupdf``/``fitz``.

    Dispatches on the **file name**, never the whole path: pytest names the
    temporary directory after the test, so a substring check over the full
    path silently misfires the moment a test is called ``..._corrupt_...`` --
    which is exactly how the corrupt-PDF case came to be reported against
    ``locked.pdf``.
    """

    def __init__(self, document: FakeDocument) -> None:
        self._document = document

    def open(self, path):
        name = Path(str(path)).name
        if "corrupt" in name:
            raise RuntimeError("cannot find xref table")
        if "missing" in name:
            raise FileNotFoundError(str(path))
        return self._document


def fake_fitz(document: FakeDocument) -> _FakeModule:
    return _FakeModule(document)


def pdf(tmp_path: Path, name: str = "doc.pdf", payload: bytes = b"%PDF-1.7 x") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


@pytest.fixture
def born_digital(tmp_path: Path) -> Path:
    return pdf(tmp_path)


@pytest.fixture
def encrypted(tmp_path: Path) -> Path:
    # /Encrypt must appear in the first 1024 bytes, per the PDF spec.
    return pdf(tmp_path, "locked.pdf", b"%PDF-1.7\n1 0 obj\n<< /Encrypt 2 0 R >>\n")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "page1.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


@pytest.fixture
def normalizer() -> TextNormalizer:
    return TextNormalizer()


# =========================================================================== #
# 1. Unicode normalization is idempotent
# =========================================================================== #

NASTY = [
    ("crlf", "line one\r\nline two\rline three"),
    ("a form feed", "page one\x0cpage two"),
    ("a vertical tab", "a\x0bb"),
    ("controls", "a\x00b\x1bc\x85d"),
    ("runs of spaces", "a    b\t\tc"),
    ("trailing whitespace", "text   \nmore\t"),
    ("a flood of blank lines", "a\n\n\n\n\n\n\nb"),
    ("preserved indentation", "def f():\n        return 1"),
    ("an fi ligature", "the \ufb01le"),
    ("full-width latin", "\uff50\uff59\uff54"),
    ("nothing to do", "already clean"),
    ("empty", ""),
    ("everything at once", "a\r\n\x00\x0c b    c\n\n\n\n  d \ufb01"),
]


@pytest.mark.parametrize(("label", "text"), NASTY)
def test_unicode_normalization_is_idempotent(normalizer, label, text):
    """A re-run must not change what a first run produced.

    Ingestion stages get retried and resumed. If normalization were not
    idempotent, a resumed job would index different text than the run it
    resumed, and two documents ingested at different times would not be
    comparable -- defeating the purpose of normalizing at all.
    """
    once = normalizer.clean(text)
    twice = normalizer.clean(once)
    thrice = normalizer.clean(twice)
    assert once == twice == thrice, label


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("label", "text"), NASTY)
def test_every_form_is_idempotent(label, text, form):
    normalizer = TextNormalizer(form=form)
    once = normalizer.clean(text)
    assert normalizer.clean(once) == once, (form, label)


def test_is_idempotent_exposes_the_property_to_callers(normalizer):
    # A pipeline needs to assert this about its own corpus, not just ours.
    assert normalizer.is_idempotent("a\r\n\x00 b    c") is True
    assert normalizer.is_idempotent("") is True


def test_idempotence_survives_a_fuzzed_corpus_in_every_form():
    """The curated cases above are chosen; this one is not.

    16,000 seeded trials of control/format/whitespace soup across all four
    Unicode forms. Idempotence is the property this component is built
    around, and a hand-picked corpus is exactly the thing that would hide a
    rule ordering bug -- so the ordering is probed over random combinations
    too. Fixed seed, so a failure is reproducible.
    """
    import random

    alphabet = (
        list("abcXYZ 0123")
        + [
            "\n", "\r", "\r\n", "\t", "\x0b", "\x0c", "\x00", "\x1b", "\x85",
            "\x7f",
        ]
        # ligature, full-width, composed and decomposed accents, zero-width
        # space, soft hyphen, no-break space, line and paragraph separators,
        # ideographic space, NEL
        + ["\ufb01", "\uff50", "\u00e9", "e\u0301", "\u200b", "\u00ad", "\u00a0"]
        + ["\u2028", "\u2029", "\u3000", "\u0085"]
    )
    rng = random.Random(20261011)
    for form in FORMS:
        nn = TextNormalizer(form=form)
        for _ in range(4000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            once = nn.clean(text)
            assert nn.clean(once) == once, (form, ascii(text))


def test_nfkc_folds_compatibility_forms_and_nfc_preserves_them():
    ligature = "\ufb01"
    assert TextNormalizer(form="NFC").clean(ligature) == ligature
    assert TextNormalizer(form="NFKC").clean(ligature) == "fi"
    # NFC is the default because it is the least destructive choice; NFKC is
    # what you want when indexing for matching.
    assert TextNormalizer().form == "NFC"


def test_a_bad_form_is_refused_with_the_trade_off_explained():
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(form="NFX")
    assert "NFKC" in str(excinfo.value)


def test_the_rules_are_reported_rather_than_applied_silently(normalizer):
    result = normalizer.normalize("a\r\n\x00b    c")
    assert isinstance(result, NormalizationResult)
    assert result.changed is True
    assert "line-endings" in result.rules
    assert "controls" in result.rules
    assert "horizontal-whitespace" in result.rules
    assert result.original == "a\r\n\x00b    c"


def test_a_clean_string_reports_no_rules(normalizer):
    result = normalizer.normalize("already clean")
    assert result.rules == ()
    assert result.changed is False
    assert "no rules applied" in result.report()


# =========================================================================== #
# 2. Control characters are stripped
# =========================================================================== #


@pytest.mark.parametrize(("label", "text", "expected"), [
    ("a NUL", "a\x00b", "ab"),
    ("an escape", "a\x1bb", "ab"),
    ("a C1 control", "a\x85b", "ab"),
    ("DEL", "a\x7fb", "ab"),
    ("every C0 control", "".join(chr(c) for c in range(0x00, 0x09)), ""),
])
def test_control_characters_are_stripped(normalizer, label, text, expected):
    assert normalizer.clean(text) == expected, label


def test_a_form_feed_becomes_a_visible_page_break(normalizer):
    # Stripping it outright would delete the only signal that a page ended.
    assert normalizer.clean("one\x0ctwo") == "one\n\ntwo"


def test_a_vertical_tab_becomes_a_newline(normalizer):
    assert normalizer.clean("a\x0bb") == "a\nb"


def test_crlf_and_lone_cr_both_become_lf(normalizer):
    assert normalizer.clean("a\r\nb") == "a\nb"
    assert normalizer.clean("a\rb") == "a\nb"
    # The \r\n pass must run first, or a lone-\r pass would double every
    # Windows newline into a blank line.
    assert normalizer.clean("a\r\n\r\nb") == "a\n\nb"


def test_a_newline_survives_but_a_tab_collapses_to_a_space(normalizer):
    # Newline is structure and survives. Tab is horizontal whitespace and does
    # NOT: mid-line runs of spaces/tabs collapse to a single space, which is
    # the documented rule. Only *leading* indentation is preserved, because
    # that is where indentation is meaning.
    assert normalizer.clean("a\tb\nc") == "a b\nc"
    assert normalizer.clean("a  b") == "a b"
    # Leading tabs are indentation, so they survive:
    assert normalizer.clean("\t\tindented") == "\t\tindented"


def test_leading_indentation_survives(normalizer):
    # In a code corpus it is meaning, not layout noise.
    assert normalizer.clean("def f():\n        return 1") == (
        "def f():\n        return 1"
    )


def test_trailing_whitespace_is_dropped_per_line(normalizer):
    assert normalizer.clean("text   \nmore\t") == "text\nmore"


def test_a_flood_of_blank_lines_collapses_to_two(normalizer):
    assert normalizer.clean("a\n\n\n\n\n\n\nb") == "a\n\n\nb"


def test_whitespace_rules_can_be_turned_off():
    uncollapsed = TextNormalizer(
        collapse_whitespace=False, collapse_blank_lines=False
    )
    assert uncollapsed.clean("a    b\n\n\n\nc") == "a    b\n\n\n\nc"


def test_the_rules_applied_in_order_give_one_stable_result(normalizer):
    # \r\n -> \n, \x0c -> \n\n (a page break stays visible), \x00 removed,
    # "b    c" -> "b c", and the blank-line floods collapse to two each.
    # Two blank lines means \n\n\n -- the rule caps the *run*, it does not
    # delete every blank line.
    assert normalizer.clean("a\r\n\x00\x0cb    c\n\n\n\n  d") == (
        "a\n\n\nb c\n\n\n  d"
    )


# =========================================================================== #
# 3. An encrypted PDF raises a typed error
# =========================================================================== #


def test_an_encrypted_pdf_raises_a_typed_error(encrypted):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["x"], needs_pass=True)))
    with pytest.raises(PdfEncryptedError) as excinfo:
        extractor.extract(encrypted)
    assert isinstance(excinfo.value, PdfExtractionError)
    assert excinfo.value.needs_pass is True
    assert excinfo.value.path == str(encrypted)


def test_an_unencrypted_pdf_does_not_raise(encrypted):
    document = FakeDocument(["Page one.", "Page two."],
                            metadata={"title": "Doc", "author": ""})
    extractor = PdfExtractor(document_factory=fake_fitz(document))
    result = extractor.extract(encrypted)
    assert result.page_count == 2
    assert result.pages == ("Page one.", "Page two.")
    assert result.text == "Page one.\n\nPage two."
    assert result.metadata == {"title": "Doc"}  # empty values dropped


def test_the_encrypted_document_is_closed_not_leaked(encrypted):
    document = FakeDocument(["x"], needs_pass=True)
    extractor = PdfExtractor(document_factory=fake_fitz(document))
    with pytest.raises(PdfEncryptedError):
        extractor.extract(encrypted)
    assert document.closed is True


def test_a_document_is_closed_after_a_successful_extraction(born_digital):
    document = FakeDocument(["text"])
    extractor = PdfExtractor(document_factory=fake_fitz(document))
    extractor.extract(born_digital)
    assert document.closed is True


def test_encrypted_and_corrupt_are_tellable_apart(tmp_path):
    # Different remedies, so different types -- and no string-matching needed
    # to tell them apart. The fake document reports needs_pass the way the
    # real library does; the fake `open` reports corruption the way it does.
    locked = pdf(tmp_path, "locked.pdf", b"%PDF-1.7\n/Encrypt\n")
    corrupt = pdf(tmp_path, "corrupt.pdf")
    extractor = PdfExtractor(
        document_factory=fake_fitz(FakeDocument(["x"], needs_pass=True))
    )
    with pytest.raises(PdfEncryptedError):
        extractor.extract(locked)

    corrupting = PdfExtractor(
        document_factory=fake_fitz(FakeDocument(["x"]))
    )
    with pytest.raises(PdfInputError) as excinfo:
        corrupting.extract(corrupt)
    assert "could not be opened as a PDF" in str(excinfo.value)
    assert "cannot find xref" in str(excinfo.value)  # cause preserved


def test_the_ocr_sub_component_also_raises_a_typed_error_on_an_encrypted_pdf(encrypted):
    # Same posture, own hierarchy: neither sub-component imports the other.
    ocr = OcrExtractor(image_factory=lambda p: p)
    with pytest.raises(OcrPdfEncryptedError) as excinfo:
        ocr.extract(encrypted)
    assert isinstance(excinfo.value, OcrPdfExtractionError)
    assert "encrypted" in str(excinfo.value)


def test_the_ocr_sub_component_detects_encryption_from_the_signature(encrypted):
    # No PDF library here, so /Encrypt is read from the trailer directly.
    assert OcrExtractor._looks_like_pdf(encrypted) is True
    assert OcrExtractor._is_encrypted_pdf(encrypted) is True


# --------------------------------------------------------------------------- #
# Extraction behaviour
# --------------------------------------------------------------------------- #


def test_page_selection_is_zero_based_and_explicit(born_digital):
    document = FakeDocument(["a", "b", "c"])
    extractor = PdfExtractor(document_factory=fake_fitz(document))
    assert extractor.extract(born_digital, pages=[0, 2]).pages == ("a", "c")
    assert extractor.page_count(born_digital) == 3  # without extracting text


def test_an_out_of_range_selection_raises_instead_of_truncating(born_digital):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["a", "b"])))
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(born_digital, pages=[0, 9])
    assert "out of range" in str(excinfo.value)


@pytest.mark.parametrize(("label", "pages", "needle"), [
    ("empty", [], "empty sequence"),
    ("negative", [-1], "non-negative"),
    ("a bool", True, "sequence of ints"),
    ("a string", "01", "sequence of ints"),
    ("a str element", ["0"], "must contain ints"),
])
def test_a_bad_page_selection_is_rejected(born_digital, label, pages, needle):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["a"])))
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(born_digital, pages=pages)
    assert needle in str(excinfo.value), (label, excinfo.value)


def test_a_missing_file_is_rejected(tmp_path):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["a"])))
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(tmp_path / "nope.pdf")
    assert "no such file" in str(excinfo.value)


def test_a_directory_is_rejected(tmp_path):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["a"])))
    with pytest.raises(PdfInputError) as excinfo:
        extractor.extract(tmp_path)
    assert "is a directory" in str(excinfo.value)


@pytest.mark.parametrize("bad", [None, 42, ["a"]])
def test_a_non_path_is_rejected(born_digital, bad):
    extractor = PdfExtractor(document_factory=fake_fitz(FakeDocument(["a"])))
    with pytest.raises(PdfInputError):
        extractor.extract(bad)


def test_pages_are_joined_by_a_blank_line_not_a_bare_newline():
    # A page boundary is a real break; a bare newline would run the last line
    # of one page into the first line of the next.
    assert ExtractedPdf(pages=("a", "b")).text == "a\n\nb"
    assert ExtractedPdf(pages=()).text == ""


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #


class FakeImage:
    def __init__(self, path) -> None:
        self.path = path
        self.closed = False

    def close(self) -> None:
        self.closed = True


def fake_tesseract(text: str):
    class Module:
        @staticmethod
        def image_to_string(image):
            return text

    return Module


def ocr_with(text: str = "Hello from OCR.\n") -> OcrExtractor:
    # Both seams are public constructor arguments. Reaching into `_libs` to
    # fake the engine would test a private arrangement no caller has.
    return OcrExtractor(
        image_factory=FakeImage, engine_factory=lambda: fake_tesseract(text)
    )


def test_ocr_extracts_text_from_an_image(image):
    result = ocr_with().extract(image)
    assert isinstance(result, OcrResult)
    assert result.pages[0].text == "Hello from OCR.\n"
    assert result.metadata["engine"] == "tesseract"
    assert result.metadata["language"] == "eng"
    assert result.page_count == 1


def test_many_images_become_consecutive_pages(image):
    result = ocr_with("text").extract_many([image, image, image])
    assert result.page_count == 3
    assert result.text == "text\n\ntext\n\ntext"
    assert result.source == str(image)


def test_the_image_is_closed_after_ocr(image):
    images = []

    def factory(path):
        handle = FakeImage(path)
        images.append(handle)
        return handle

    extractor = OcrExtractor(
        image_factory=factory, engine_factory=lambda: fake_tesseract("x")
    )
    extractor.extract(image)
    assert images[0].closed is True


def test_an_empty_selection_is_a_caller_bug_not_a_result(image):
    ocr = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        ocr.extract_many([])
    assert "empty" in str(excinfo.value)


def test_a_bare_string_is_refused_because_it_would_ocr_every_character(image):
    # A string is iterable, so passing one would walk it character by
    # character. Refusing is the only safe reading.
    with pytest.raises(OcrInputError) as excinfo:
        ocr_with().extract_many("page.png")
    assert "iterable" in str(excinfo.value)


def test_confidence_is_deliberately_not_surfaced(image):
    # image_to_string does not report confidence, and no Tesseract binary has
    # ever been run against this file -- so a number here would be a field
    # nothing had ever verified. It is left out rather than shipped as None,
    # which would read as "the engine was certain of nothing".
    page = ocr_with().extract(image).pages[0]
    assert not hasattr(page, "confidence")


def test_a_blank_page_is_recognised():
    assert ExtractedPage(text="   \n").is_blank is True
    assert ExtractedPage(text="text").is_blank is False


@pytest.mark.parametrize(("label", "kwargs", "needle"), [
    ("language blank", {"language": "  "}, "non-empty"),
    ("language injection", {"language": "eng; rm -rf /"}, "not letters"),
    ("dpi too low", {"dpi": 10}, "between 72 and 1200"),
    ("dpi too high", {"dpi": 5000}, "between 72 and 1200"),
    ("dpi a bool", {"dpi": True}, "got bool"),
    ("name blank", {"name": ""}, "non-empty"),
])
def test_a_nonsense_ocr_configuration_is_refused(label, kwargs, needle):
    with pytest.raises(OcrInputError) as excinfo:
        OcrExtractor(**kwargs)
    assert needle in str(excinfo.value), (label, excinfo.value)


def test_language_is_validated_because_it_reaches_a_shell_invoked_binary():
    # Tesseract is invoked as a subprocess, so the language string is a
    # command-line argument, not merely data.
    assert OcrExtractor(language="eng+fra").language == "eng+fra"
    with pytest.raises(OcrInputError):
        OcrExtractor(language="eng --oem 0")


# =========================================================================== #
# 4. Each sub-component is independently selectable (FR-6)
# =========================================================================== #

SUBCOMPONENTS = {
    "pdf-extraction-pymupdf": (
        INGESTION / "pdf_extraction_pymupdf" / "pymupdf.py",
        INGESTION / "pdf_extraction_pymupdf" / "manifest.json",
        {"pymupdf"},
    ),
    "pdf-extraction-ocr": (
        INGESTION / "pdf_extraction_ocr" / "ocr.py",
        INGESTION / "pdf_extraction_ocr" / "manifest.json",
        {"pytesseract", "pillow"},
    ),
}


def _imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_neither_sub_component_imports_the_other():
    # AST-checked, so a stray import cannot hide behind a lazy call. This is
    # the property that makes them selectable rather than merely separate.
    pymupdf_path, _, _ = SUBCOMPONENTS["pdf-extraction-pymupdf"]
    ocr_path, _, _ = SUBCOMPONENTS["pdf-extraction-ocr"]
    assert "ocr" not in _imported(pymupdf_path)
    assert "pymupdf" not in _imported(ocr_path)


@pytest.mark.parametrize("component", SUBCOMPONENTS)
def test_each_sub_component_declares_no_component_dependencies(component):
    # So `ragkit add` of one copies one file, not a graph.
    _, manifest_path, _ = SUBCOMPONENTS[component]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dependencies"] == [], component
    assert manifest["external_services"] == [], component
    assert manifest["required_packages"] == [], component


@pytest.mark.parametrize("component", SUBCOMPONENTS)
def test_each_sub_component_declares_its_own_optional_packages(component):
    _, manifest_path, expected = SUBCOMPONENTS[component]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = {entry["name"] for entry in manifest["optional_packages"]}
    assert declared == expected, component


def test_the_two_sub_components_have_distinct_ids():
    ids = {component for component in SUBCOMPONENTS}
    assert len(ids) == 2
    assert ids == {"pdf-extraction-pymupdf", "pdf-extraction-ocr"}


def _block_modules(monkeypatch, *names: str) -> None:
    """Make ``import <name>`` fail, whatever is actually installed.

    ``pymupdf`` and ``pillow`` may well be present in the interpreter running
    the suite, so absence cannot be assumed -- it has to be forced. Setting
    the entry to ``None`` is how CPython reports "found but not importable",
    which is exactly the ImportError the guard must handle.
    """
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def test_a_missing_pymupdf_names_only_its_own_packages(
    monkeypatch, born_digital, image
):
    # Installing one sub-component's packages is never implied by the other's.
    _block_modules(monkeypatch, "pymupdf", "fitz")
    with pytest.raises(PyMissingDependency) as py_error:
        PdfExtractor().extract(born_digital)
    assert "pymupdf" in str(py_error.value)
    assert "pytesseract" not in str(py_error.value)
    assert "pillow" not in str(py_error.value)


def test_a_missing_ocr_dependency_names_only_its_own_packages(
    monkeypatch, born_digital, image
):
    _block_modules(monkeypatch, "pytesseract", "PIL")
    with pytest.raises(OcrMissingDependency) as ocr_error:
        OcrExtractor().extract(image)
    assert "pytesseract" in str(ocr_error.value)
    assert "pillow" in str(ocr_error.value)
    assert "'pymupdf'" not in str(ocr_error.value)


def test_the_error_names_whichever_ocr_package_is_actually_missing(
    monkeypatch, image
):
    # pillow present, pytesseract not: only the missing one is named, so the
    # message tells the caller what to install rather than a shopping list.
    #
    # PIL is SUBSTITUTED rather than relied on. This host may or may not have
    # pillow installed -- and a test that passes only where it happens to be
    # installed is a test that has not been run everywhere it claims.
    monkeypatch.setitem(sys.modules, "PIL", types.ModuleType("PIL"))
    _block_modules(monkeypatch, "pytesseract")
    with pytest.raises(OcrMissingDependency) as excinfo:
        OcrExtractor().extract(image)
    assert "pytesseract" in str(excinfo.value)
    assert "Missing: pytesseract" in str(excinfo.value)


def test_the_error_says_the_tesseract_binary_is_a_separate_problem(
    monkeypatch, image
):
    # Installing the package does not install the binary, and a caller who
    # does not know that will debug the wrong thing.
    _block_modules(monkeypatch, "pytesseract")
    with pytest.raises(OcrMissingDependency) as excinfo:
        OcrExtractor().extract(image)
    assert "BINARY on PATH" in str(excinfo.value)


def test_the_pymupdf_error_names_both_import_names(monkeypatch, born_digital):
    # 'pymupdf' is current; 'fitz' is the deprecated pre-1.24 name. A caller
    # reading stack traces needs to know which is which.
    _block_modules(monkeypatch, "pymupdf", "fitz")
    with pytest.raises(PyMissingDependency) as excinfo:
        PdfExtractor().extract(born_digital)
    assert "'pymupdf'" in str(excinfo.value)
    assert "'fitz'" in str(excinfo.value)
    assert "deprecated" in str(excinfo.value)


def test_a_missing_dependency_message_matches_its_manifest_verbatim():
    # A manifest edit must not be able to drift away from the error a caller
    # actually sees.
    py_manifest = json.loads(
        (INGESTION / "pdf_extraction_pymupdf" / "manifest.json").read_text("utf-8")
    )
    ocr_manifest = json.loads(
        (INGESTION / "pdf_extraction_ocr" / "manifest.json").read_text("utf-8")
    )
    assert PYMUPDF_MANIFEST_MESSAGE == py_manifest["missing_dependency_message"]
    assert OCR_MANIFEST_MESSAGE == ocr_manifest["missing_dependency_message"]


def test_a_missing_dependency_is_also_an_importerror(monkeypatch, born_digital, image):
    # Catchable in either vocabulary: the libraries' or this component's.
    _block_modules(monkeypatch, "pymupdf", "fitz")
    with pytest.raises(ImportError):
        PdfExtractor().extract(born_digital)
    _block_modules(monkeypatch, "pytesseract", "PIL")
    with pytest.raises(ImportError):
        OcrExtractor().extract(image)


def test_the_dependency_is_imported_lazily_not_at_module_scope():
    # Importing eagerly would make the module unimportable in a project that
    # has not installed the optional package -- defeating the point of
    # declaring it optional.
    for path, _, _ in SUBCOMPONENTS.values():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not (top_level & {"fitz", "pymupdf", "pytesseract", "PIL"}), (
            path.name
        )


def test_both_sub_components_return_the_same_result_shape(born_digital, image):
    # Whichever was chosen is interchangeable downstream of the call.
    py_result = PdfExtractor(
        document_factory=fake_fitz(FakeDocument(["a", "b"]))
    ).extract(born_digital)
    ocr_result = ocr_with("a").extract_many([image, image])

    for attribute in ("pages", "page_count", "text", "character_count", "source"):
        assert hasattr(py_result, attribute), attribute
        assert hasattr(ocr_result, attribute), attribute
    assert py_result.text == "a\n\nb"
    assert ocr_result.text == "a\n\na"


def test_the_ocr_sub_component_rejects_a_pdf_and_says_why(tmp_path, born_digital):
    # It cannot rasterize without a PDF library, and claiming otherwise would
    # mean depending on one while saying it does not have one.
    ocr = ocr_with()
    with pytest.raises(OcrInputError) as excinfo:
        ocr.extract(born_digital)
    assert "PAGE IMAGES" in str(excinfo.value)
    assert "pdf-extraction-pymupdf" in str(excinfo.value)  # names the alternative


# --------------------------------------------------------------------------- #
# Against the real library, when it happens to be installed
# --------------------------------------------------------------------------- #


def _pymupdf_available() -> bool:
    import importlib.util

    return any(
        importlib.util.find_spec(name) is not None for name in ("pymupdf", "fitz")
    )


@pytest.mark.skipif(
    not _pymupdf_available(),
    reason="pymupdf is not installed; the fake-based tests above still cover "
           "the extraction logic",
)
def test_a_real_encrypted_pdf_raises_a_typed_error(tmp_path):
    """Against actual pymupdf, not a fake -- when the host has it.

    This is the one place the real library is exercised. It is skipped rather
    than installed, because installing it would modify dependency files, which
    this project forbids. Everywhere else the logic runs through the injection
    seam so the suite stays green on a host without pymupdf.
    """
    import pymupdf  # the real library -- see the loader note at the top of
    # this module for why the component does not shadow it.

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Hello from a real PDF")
    path = tmp_path / "real.pdf"
    document.save(str(path))
    document.close()

    result = PdfExtractor().extract(path)
    assert result.page_count == 1
    assert "Hello from a real PDF" in result.pages[0]
    assert result.metadata  # a real document carries an info dictionary

    # Now encrypt that same file and confirm the typed error still fires.
    locked = tmp_path / "real_locked.pdf"
    document = pymupdf.open(str(path))
    document.save(
        str(locked),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="secret",
    )
    document.close()

    with pytest.raises(PdfEncryptedError) as excinfo:
        PdfExtractor().extract(locked)
    assert excinfo.value.needs_pass is True
    assert "secret" not in str(excinfo.value)  # never echoes the password


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #


def test_no_ingestion_component_imports_rag_code_library():
    for path in INGESTION.rglob("*.py"):
        assert "rag_code_library" not in _imported(path), path.name


def test_no_ingestion_component_reaches_the_network():
    # The libraries involved are all local. Nothing here can call out.
    banned = {"openai", "anthropic", "httpx", "requests", "urllib", "socket", "aiohttp"}
    for path in INGESTION.rglob("*.py"):
        assert not (_imported(path) & banned), path.name


def test_every_ingestion_component_has_a_manifest_readme_example_and_tests():
    # FR-6: each sub-component carries its own, so any one can be taken alone.
    for directory in sorted(INGESTION.iterdir()):
        if not directory.is_dir():
            continue
        for required in ("manifest.json", "README.md", "example.py"):
            assert (directory / required).exists(), (directory.name, required)
        assert (directory / "tests" / "test_component.py").exists(), directory.name


def test_every_ingestion_component_has_the_same_error_hierarchy_shape():
    # Each defines its own; none reuses another's. That duplication is the
    # price of being copyable out alone.
    assert issubclass(PdfInputError, PdfExtractionError)
    assert issubclass(PdfEncryptedError, PdfExtractionError)
    assert issubclass(PyMissingDependency, PdfExtractionError)
    assert issubclass(OcrInputError, OcrPdfExtractionError)
    assert issubclass(OcrPdfEncryptedError, OcrPdfExtractionError)
    assert issubclass(NormalizerInputError, NormalizerError)
    assert not issubclass(PdfExtractionError, ValueError)
    assert not issubclass(NormalizerError, ValueError)


@pytest.mark.parametrize(("label", "call", "needle"), [
    ("a non-string", lambda n: n.normalize(None), "must be a string"),
    ("an int", lambda n: n.clean(42), "must be a string"),
    ("over the limit", lambda n: TextNormalizer(max_length=5).clean("x" * 10),
     "over the 5-character limit"),
])
def test_bad_normalizer_input_is_a_typed_error(label, call, needle):
    with pytest.raises(NormalizerInputError) as excinfo:
        call(TextNormalizer())
    assert needle in str(excinfo.value), (label, excinfo.value)


@pytest.mark.parametrize(("label", "kwargs", "needle"), [
    ("collapse_whitespace not bool", {"collapse_whitespace": 1}, "must be a bool"),
    ("collapse_blank_lines not bool", {"collapse_blank_lines": "x"}, "must be a bool"),
    ("max_length zero", {"max_length": 0}, "at least 1"),
    ("max_length a bool", {"max_length": True}, "must be an int"),
    ("name blank", {"name": ""}, "non-empty"),
])
def test_a_nonsense_normalizer_configuration_is_refused(label, kwargs, needle):
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(**kwargs)
    assert needle in str(excinfo.value), (label, excinfo.value)


def test_the_convenience_function_matches_the_class(normalizer):
    assert normalize_text("a\r\n\x00b") == normalizer.clean("a\r\n\x00b")
    assert normalize_text("ﬁ", form="NFKC") == "fi"


def test_the_normalizer_repr_is_informative():
    assert repr(TextNormalizer()) == (
        "TextNormalizer(form='NFC', collapse_whitespace=True, "
        "collapse_blank_lines=True)"
    )
    assert repr(PdfExtractor()) == "PdfExtractor(name='pdf-extraction-pymupdf')"
    assert repr(OcrExtractor()) == (
        "OcrExtractor(language='eng', dpi=300, "
        "name='pdf-extraction-ocr')"
    )

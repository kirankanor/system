"""Extract text from page images with Tesseract, with no PDF library.

What this sub-component is
--------------------------
One of two selectable PDF extractors (FR-6). The implementations differ by
technique, so the human picks which one to add and takes only that one's
dependencies.

This one reads **page images** -- PNG or JPEG -- with ``pytesseract`` and
``pillow``. It needs no PDF library, and that is the point: it is the path
for a corpus that is already images, or that someone else has already
rasterized.

The other is ``pdf-extraction-pymupdf``, which reads born-digital PDFs
directly. Neither imports the other, and neither needs the other's packages.

The boundary, stated plainly
----------------------------
This component does **not** open PDFs, and it says so with a typed error
rather than pretending. Here is the honest reason.

Tesseract reads *images*. A PDF is not an image -- it is a page description
that may or may not contain a raster. Turning a PDF page into pixels requires
a renderer, and a renderer is exactly the kind of dependency this sub-component
exists to avoid. So an image-only OCR path that claimed to accept PDFs would
be quietly depending on a PDF library while telling you it does not have one.

What that means in practice:

* **scanned PDF** -> rasterize it first (``pdftoppm``, or pymupdf), then hand
  the images here;
* **born-digital PDF** -> you do not want OCR at all. Use
  ``pdf-extraction-pymupdf``, which extracts the real text layer in
  milliseconds instead of guessing at glyphs.

OCR is the expensive, approximate path. It is for documents that have no text
layer, not a general-purpose alternative to one that does.

Encrypted PDFs
--------------
An encrypted PDF is rejected with :class:`PdfEncryptedError`, the same typed
error the pymupdf sub-component raises. It is caught on the file signature
before anything else, so the message is about encryption rather than about
this component's own limitations -- a caller who can decrypt should learn
that and retry, not be told OCR cannot read PDFs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MANIFEST_MESSAGE",
    "ExtractedPage",
    "MissingDependency",
    "OcrExtractor",
    "OcrInputError",
    "PdfEncryptedError",
    "PdfExtractionError",
]

#: Verbatim from this component's ``manifest.json``. A test asserts the two
#: agree, so a manifest edit cannot drift away from the error a caller sees.
MANIFEST_MESSAGE: Final[str] = (
    "pdf-extraction-ocr requires 'pytesseract' and 'pillow', which are not "
    "installed. Install them with: uv add pytesseract pillow"
)

#: The two imports, and the package each belongs to. Tesseract also needs the
#: *binary* on PATH, which no Python installer provides -- hence the separate
#: note in the error, because "installed the package" and "can run OCR" are
#: not the same state.
_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("PIL", "pillow"),
    ("pytesseract", "pytesseract"),
)

#: `%PDF-`. Checked as bytes, so a PDF cannot slip past by being named
#: `.png`, and an encrypted one is identified before any library touches it.
_PDF_MAGIC: Final[bytes] = b"%PDF-"

#: The standard encrypted-PDF markers, in the trailer. Present in the first
#: 1024 bytes of every encrypted PDF, per the spec.
_ENCRYPTED_MARKERS: Final[tuple[bytes, ...]] = (b"/Encrypt",)


class PdfExtractionError(Exception):
    """Base for every failure in this component."""


class OcrInputError(PdfExtractionError):
    """The caller passed something this component cannot read."""


class PdfEncryptedError(PdfExtractionError):
    """The file is an encrypted PDF, which this component will not open."""

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path


class MissingDependency(PdfExtractionError, ImportError):
    """``pytesseract`` or ``pillow`` is not installed.

    Subclasses ``ImportError`` as well as this component's own base, so a
    caller can catch either the libraries' vocabulary or this component's.
    """


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """One page's OCR result.

    Deliberately just the text. Tesseract's per-word confidence is *not*
    surfaced: ``image_to_string`` does not report it, and getting at it means
    calling ``image_to_data`` instead, which costs a second pass over every
    page -- on the slowest component in this library. Worse, no Tesseract
    binary has ever been run against this file, so a confidence number here
    would be a field nothing had ever verified. Both sub-components return the
    same shape, which is the promise of FR-6.
    """

    text: str

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class OcrResult:
    """The result of an extraction. Same shape as the pymupdf sub-component's."""

    pages: tuple[ExtractedPage, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def character_count(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return (
            f"OcrResult(source={self.source!r}, "
            f"page_count={self.page_count}, "
            f"character_count={self.character_count})"
        )


class OcrExtractor:
    """Read page images with Tesseract, imported on first use.

    ``image_factory`` and ``engine_factory`` are injection seams for tests
    and for callers who need to substitute a different image source or a
    different recognition engine. Each covers exactly one external call --
    opening an image, and turning one into text -- so a caller can replace
    either without touching the other, and neither is part of the normal API.
    """

    def __init__(
        self,
        *,
        language: str = "eng",
        dpi: int = 300,
        image_factory: Any = None,
        engine_factory: Any = None,
        name: str = "pdf-extraction-ocr",
    ) -> None:
        self.language = self._check_language(language)
        self.dpi = self._check_dpi(dpi)
        if not isinstance(name, str) or not name:
            raise OcrInputError(
                f"pdf-extraction-ocr: name must be a non-empty string, got "
                f"{name!r}"
            )
        self.name = name
        self._image_factory = image_factory
        self._engine_factory = engine_factory
        self._libs: dict[str, Any] = {}

    @staticmethod
    def _check_language(language: Any) -> str:
        if not isinstance(language, str) or not language.strip():
            raise OcrInputError(
                f"pdf-extraction-ocr: language must be a non-empty string, "
                f"got {language!r}"
            )
        # Tesseract passes this straight to a shell-invoked binary, so a
        # language string is a command-line argument, not just data.
        #
        # ``_`` is allowed as well as ``+``: real Tesseract codes use it --
        # ``chi_sim``, ``chi_tra``, ``jpn`` -- so an allow-list without it
        # would refuse languages the engine ships with.
        if not all(char.isalnum() or char in "+-_" for char in language):
            raise OcrInputError(
                f"pdf-extraction-ocr: language {language!r} contains "
                f"characters that are not letters, digits, '+', '-' or '_'. "
                f"It is passed to the tesseract binary as an argument, so "
                f"only language codes like 'eng', 'eng+fra' or 'chi_sim' "
                f"are accepted"
            )
        return language

    @staticmethod
    def _check_dpi(dpi: Any) -> int:
        if isinstance(dpi, bool) or not isinstance(dpi, int):
            raise OcrInputError(
                f"pdf-extraction-ocr: dpi must be an int, got "
                f"{type(dpi).__name__}"
            )
        if not 72 <= dpi <= 1200:
            raise OcrInputError(
                f"pdf-extraction-ocr: dpi must be between 72 and 1200, got "
                f"{dpi}. Below 72 there are too few pixels to recognise; "
                f"above 1200 the accuracy gain is negligible and the memory "
                f"cost is not"
            )
        return dpi

    # ------------------------------------------------------------------ #
    # Dependency handling
    # ------------------------------------------------------------------ #

    def _ensure(self) -> dict[str, Any]:
        """Import pillow and pytesseract on first use only.

        Lazy, for the same reason as the pymupdf sub-component: importing at
        module scope would make this file unimportable in a project that has
        not installed OCR -- defeating the point of declaring it optional.

        The two seams are **orthogonal**, and each skips only its own import.
        Substituting the image opener while leaving the engine alone still
        imports pytesseract and still raises when it is absent -- which is
        correct, because there is no OCR without an engine. It is not enough
        to stand in for half the boundary and then report success.
        """
        if self._libs:
            return self._libs
        loaded: dict[str, Any] = {}
        if self._engine_factory is not None:
            loaded["pytesseract"] = self._engine_factory()
        if self._image_factory is not None:
            loaded["PIL"] = None  # never touched; the factory stands in
        for module_name, package in _MODULES:
            if module_name in loaded:
                continue
            try:
                loaded[module_name] = __import__(module_name, fromlist=["*"])
            except ImportError as exc:
                missing = [pkg for mod, pkg in _MODULES if mod not in loaded]
                raise MissingDependency(
                    f"{MANIFEST_MESSAGE}\n\nMissing: "
                    f"{', '.join(missing)}. Note that pytesseract also needs "
                    f"the tesseract BINARY on PATH, which no Python installer "
                    f"provides -- installing the package alone will still "
                    f"fail. Underlying error: {exc}"
                ) from exc
        self._libs = loaded
        return loaded

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _looks_like_pdf(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                return handle.read(5) == _PDF_MAGIC
        except OSError:
            return False

    @staticmethod
    def _is_encrypted_pdf(path: Path) -> bool:
        """Detect encryption from the trailer, without a PDF library.

        The ``/Encrypt`` dictionary is required to appear within the first
        1024 bytes of an encrypted PDF, so a small read is enough and no
        parser is involved.
        """
        try:
            with path.open("rb") as handle:
                head = handle.read(1024)
        except OSError:
            return False
        return any(marker in head for marker in _ENCRYPTED_MARKERS)

    def _reject_pdf(self, path: Path) -> None:
        if not self._looks_like_pdf(path):
            return
        if self._is_encrypted_pdf(path):
            raise PdfEncryptedError(
                f"{self.name}: {path} is an encrypted PDF. It is rejected "
                f"before any library touches it. Decrypt it first, then "
                f"rasterize the pages and hand the images here",
                path=str(path),
            )
        raise OcrInputError(
            f"{self.name}: {path} is a PDF, and this sub-component reads "
            f"PAGE IMAGES, not PDFs. Turning a PDF page into pixels needs a "
            f"renderer, and a renderer is precisely the dependency this "
            f"sub-component exists to avoid -- so claiming to accept PDFs "
            f"would mean depending on a PDF library while saying it does not "
            f"have one. Either rasterize it first (pdftoppm, or pymupdf), or "
            f"use pdf-extraction-pymupdf, which reads born-digital PDFs "
            f"directly and in milliseconds"
        )

    def _resolve(self, path: str | Path) -> Path:
        if isinstance(path, Path):
            resolved = path
        elif isinstance(path, str):
            resolved = Path(path)
        else:
            raise OcrInputError(
                f"path must be a string or Path, got {type(path).__name__}"
            )
        if not resolved.exists():
            raise OcrInputError(
                f"no such file: {resolved}. Extraction reads from the "
                f"filesystem and will not invent a document"
            )
        if resolved.is_dir():
            raise OcrInputError(
                f"{resolved} is a directory, not an image. This component "
                f"reads one image at a time; iterate over a directory "
                f"yourself so a failure names the file that caused it"
            )
        self._reject_pdf(resolved)
        return resolved

    def extract_one(self, path: str | Path) -> ExtractedPage:
        """OCR a single image."""
        resolved = self._resolve(path)
        libs = self._ensure()
        Image, pytesseract = libs.get("PIL"), libs.get("pytesseract")

        if self._image_factory is not None:
            image = self._image_factory(resolved)
        else:
            try:
                # ``Image`` is the PIL module; .open is the call under test.
                image = Image.open(resolved)  # type: ignore[union-attr]
            except Exception as exc:
                raise OcrInputError(
                    f"{self.name}: {resolved} could not be opened as an "
                    f"image ({type(exc).__name__}: {exc})"
                ) from exc

        try:
            text = str(pytesseract.image_to_string(image))  # type: ignore[union-attr]
        except Exception as exc:
            raise PdfExtractionError(
                f"{self.name}: OCR failed on {resolved} "
                f"({type(exc).__name__}: {exc}). Check that the tesseract "
                f"binary is on PATH and that the {self.language!r} language "
                f"data is installed"
            ) from exc
        finally:
            close = getattr(image, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # pragma: no cover -- best effort
                    pass

        return ExtractedPage(text=text)

    def extract(self, path: str | Path) -> OcrResult:
        """OCR one image and return a result shaped like the pymupdf one."""
        resolved = self._resolve(path)
        return OcrResult(
            pages=(self.extract_one(resolved),),
            metadata={
                "engine": "tesseract",
                "language": self.language,
                "dpi": str(self.dpi),
            },
            source=str(resolved),
        )

    def extract_many(self, paths: Sequence[str | Path]) -> OcrResult:
        """OCR several images as consecutive pages.

        Raises on the first failure rather than returning what succeeded.
        A half-extracted document is worse than none: the caller would have
        no way to tell which page is missing.
        """
        if isinstance(paths, (str, bytes)):
            raise OcrInputError(
                f"paths must be a sequence, got {type(paths).__name__}. A "
                f"string is iterable, so passing one would OCR every "
                f"character"
            )
        try:
            chosen = list(paths)
        except TypeError as exc:
            raise OcrInputError(
                f"paths must be a sequence, got {type(paths).__name__}"
            ) from exc
        if not chosen:
            raise OcrInputError(
                "paths was empty. Producing an empty result would be "
                "indistinguishable from a caller bug"
            )
        pages = tuple(self.extract_one(item) for item in chosen)
        return OcrResult(
            pages=pages,
            metadata={
                "engine": "tesseract",
                "language": self.language,
                "dpi": str(self.dpi),
            },
            source=str(chosen[0]),
        )

    def __repr__(self) -> str:
        return (
            f"OcrExtractor(language={self.language!r}, dpi={self.dpi}, "
            f"name={self.name!r})"
        )

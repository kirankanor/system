"""Extract PDF text with pymupdf, declared as an optional dependency.

What this sub-component is
--------------------------
One of two selectable PDF extractors (FR-6). The implementations differ by
library, so the human picks which one to add and takes only that one's
dependency. This one uses ``pymupdf`` and gets real selectable text out of a
born-digital PDF.

The other is ``pdf-extraction-ocr``, which needs no PDF library at all and
reads page *images* instead. Neither imports the other, and neither is
required for the other to work -- that is what makes them selectable rather
than merely separate.

Why ``pymupdf`` and not a stdlib option
---------------------------------------
There is no PDF text extractor in the standard library, and writing one is
not a weekend project: a PDF's text is a stream of positioning operators that
have to be reconstructed into reading order, with ligatures, encodings and
embedded font tables in the way. So this is an optional dependency, and the
whole point of the manifest's ``optional_packages`` field.

Being optional is a real constraint on the code, not just a declaration: the
import happens **lazily, on first use**, never at module import. That means
``ragkit add pdf-extraction-pymupdf`` into a project that has not installed
``pymupdf`` still yields an importable module -- one that tells you exactly
what to install when you first call it, rather than breaking every import in
the package at load time.

Encrypted PDFs
--------------
An encrypted PDF raises :class:`PdfEncryptedError`, which carries the
``needs_pass`` flag and the file path. It is deliberately its own type rather
than a generic failure: "this document is encrypted" and "this document is
corrupt" have completely different remedies, and a caller that cannot decrypt
should be able to tell them apart without string-matching a message.

A PDF encrypted with an *empty* password is a real thing -- many tools produce
them -- and pymupdf opens those transparently. Only a document that actually
demands a password raises here, which is the honest boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MANIFEST_MESSAGE",
    "ExtractedPdf",
    "MissingDependency",
    "PdfEncryptedError",
    "PdfExtractionError",
    "PdfExtractor",
    "PdfInputError",
]

#: Verbatim from this component's ``manifest.json``. A test asserts the two
#: agree, so a manifest edit cannot drift away from the error a caller sees.
MANIFEST_MESSAGE: Final[str] = (
    "pdf-extraction-pymupdf requires 'pymupdf', which is not installed. "
    "Install it with: uv add pymupdf"
)

#: pymupdf's import name. It was ``fitz`` until 1.24, which deprecated that
#: name and emits a warning on every import. ``pymupdf`` is tried first so a
#: current install stays quiet; ``fitz`` remains as the fallback for older
#: installs rather than being dropped.
_MODULES_TO_TRY: Final[tuple[str, ...]] = ("pymupdf", "fitz")
_PACKAGE: Final[str] = "pymupdf"


class PdfExtractionError(Exception):
    """Base for every failure in this component."""


class PdfInputError(PdfExtractionError):
    """The caller passed a path this component cannot extract from."""


class PdfEncryptedError(PdfExtractionError):
    """The PDF is encrypted and demands a password.

    Carries ``path`` and ``needs_pass`` so a caller can branch on the fact
    rather than parsing a message.
    """

    def __init__(self, message: str, *, path: str, needs_pass: bool) -> None:
        super().__init__(message)
        self.path = path
        self.needs_pass = needs_pass


class MissingDependency(PdfExtractionError, ImportError):
    """``pymupdf`` is not installed.

    Subclasses ``ImportError`` as well as this component's own base, so a
    caller can catch either the specific library's vocabulary or this
    component's.
    """


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    """The result of an extraction, whatever produced it.

    Deliberately library-neutral: no ``fitz.Document``, no page objects, no
    handles. Just text per page plus the metadata worth keeping. That is what
    lets ``pdf-extraction-ocr`` return exactly the same shape, so the two
    sub-components are interchangeable downstream of the extraction call --
    which is the whole promise of FR-6.

    ``metadata`` is a plain mapping, not a dataclass, because PDF info
    dictionaries vary per document and inventing a fixed set of fields would
    either lose data or require a schema change per corpus.
    """

    pages: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        """Every page joined by a blank line.

        A blank line, not a bare newline: a page boundary is a real break in
        reading order, and joining with ``\\n`` would let the last line of one
        page run into the first line of the next.
        """
        return "\n\n".join(self.pages)

    @property
    def character_count(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return (
            f"ExtractedPdf(source={self.source!r}, "
            f"page_count={self.page_count}, "
            f"character_count={self.character_count})"
        )


class PdfExtractor:
    """Extract text from PDFs using pymupdf, imported on first use.

    ``document_factory`` is an injection seam for tests and for callers who
    need to substitute a different opener -- a file-like object, a virtual
    filesystem, a network handle. It is not part of the normal API; the
    default reaches for pymupdf.
    """

    def __init__(
        self,
        *,
        document_factory: Any = None,
        name: str = "pdf-extraction-pymupdf",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise PdfInputError(
                f"pdf-extraction-pymupdf: name must be a non-empty string, "
                f"got {name!r}"
            )
        self.name = name
        self._document_factory = document_factory
        self._fitz: Any = None

    # ------------------------------------------------------------------ #
    # Dependency handling
    # ------------------------------------------------------------------ #

    def _ensure_fitz(self) -> Any:
        """Return pymupdf, importing it on the first call only.

        Lazy on purpose. Importing at module scope would make this file
        unimportable in any project that has not installed pymupdf -- which
        would defeat the point of declaring it *optional*, and would break
        unrelated code that merely imports the package.

        ``pymupdf`` is tried before ``fitz``. The older name still works but
        emits a deprecation warning on every import since 1.24, and a library
        that warns on import turns a caller's clean stderr into noise they
        cannot silence from here.
        """
        if self._document_factory is not None:
            return self._document_factory
        if self._fitz is not None:
            return self._fitz
        failures: list[str] = []
        shadowed: list[str] = []
        for module_name in _MODULES_TO_TRY:
            try:
                module = __import__(module_name, fromlist=["*"])
            except ImportError as exc:
                failures.append(f"{module_name}: {exc}")
                continue
            if Path(str(getattr(module, "__file__", ""))).resolve() == Path(
                __file__
            ).resolve():
                # `import pymupdf` resolved to THIS file. That happens when a
                # test harness or an example puts the component's own
                # directory on sys.path. Left unchecked it surfaces frames
                # later as `module 'pymupdf' has no attribute 'open'`, wrapped
                # as "could not be opened as a PDF" -- a baffling complaint
                # about a document when the real problem is the import path.
                shadowed.append(module_name)
                continue
            self._fitz = module
            return module
        if shadowed and not failures:
            raise MissingDependency(
                f"{MANIFEST_MESSAGE}\n\n`import {shadowed[0]}` resolved to "
                f"THIS FILE rather than to the library, so the real pymupdf "
                f"is shadowed. That happens when a test harness or an "
                f"example puts the component's own directory on sys.path. In "
                f"a project the file lands at src/<package>/rag/ingestion/"
                f"pdf_extraction/pymupdf.py, inside a package namespace, "
                f"where it cannot shadow. Import the component as part of "
                f"its package rather than by adding its directory to sys.path."
            )
        raise MissingDependency(
            f"{MANIFEST_MESSAGE}\n\nTried the import names "
            f"{list(_MODULES_TO_TRY)!r} and neither was importable. "
            f"'pymupdf' is the current name; 'fitz' is the pre-1.24 name "
            f"and is deprecated. Underlying errors: " + "; ".join(failures)
        )

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #

    def open(self, path: str | Path) -> Any:
        """Open a PDF and verify it is neither missing nor encrypted.

        Returns an open pymupdf document. **The caller must close it.** This
        is exposed so a caller can inspect a document -- metadata, page count
        -- without paying for a full text extraction.
        """
        resolved = self._resolve(path)
        fitz = self._ensure_fitz()
        try:
            document = fitz.open(str(resolved))
        except FileNotFoundError:
            raise
        except Exception as exc:  # pymupdf raises many types for bad files
            raise PdfInputError(
                f"{self.name}: {resolved} could not be opened as a PDF "
                f"({type(exc).__name__}: {exc}). Either it is not a PDF, or "
                f"it is damaged"
            ) from exc

        needs_pass = bool(getattr(document, "needs_pass", False))
        if needs_pass:
            try:
                document.close()
            except Exception:  # pragma: no cover -- best effort cleanup
                pass
            raise PdfEncryptedError(
                f"{self.name}: {resolved} is encrypted and demands a "
                f"password, so no text can be extracted from it. Decrypt it "
                f"first, or extract it with a tool that holds the key. This "
                f"is a distinct failure from a corrupt PDF, because the "
                f"remedy is different",
                path=str(resolved),
                needs_pass=True,
            )
        return document

    def page_count(self, path: str | Path) -> int:
        """How many pages the PDF has, without extracting any text."""
        document = self.open(path)
        try:
            return int(document.page_count)
        finally:
            self._close(document)

    def extract(
        self,
        path: str | Path,
        *,
        pages: Sequence[int] | None = None,
    ) -> ExtractedPdf:
        """Extract text from ``path``, or from ``pages`` alone.

        ``pages`` are **zero-based** indices, matching pymupdf and every other
        PDF library. A caller holding a printed page number wants
        ``page_number - 1``; that conversion is left to them rather than
        guessed at here, because guessing wrong silently drops a page.

        Out-of-range selections raise rather than returning a short result.
        A silently truncated extraction is indistinguishable from a document
        that simply ended early.
        """
        resolved = self._resolve(path)
        selected = self._selection(pages)
        document = self.open(resolved)
        try:
            available = int(document.page_count)
            if selected is not None:
                unknown = [index for index in selected if index >= available]
                if unknown:
                    raise PdfInputError(
                        f"{self.name}: {resolved} has {available} page(s), "
                        f"so the zero-based selection {unknown!r} is "
                        f"out of range. Selecting silently short would be "
                        f"indistinguishable from a document that ended early"
                    )
            indices = list(range(available)) if selected is None else selected
            extracted = tuple(
                self._page_text(document, index) for index in indices
            )
            return ExtractedPdf(
                pages=extracted,
                metadata=self._metadata(document),
                source=str(resolved),
            )
        finally:
            self._close(document)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        if isinstance(path, Path):
            resolved = path
        elif isinstance(path, str):
            resolved = Path(path)
        else:
            raise PdfInputError(
                f"path must be a string or Path, got {type(path).__name__}"
            )
        if not resolved.exists():
            raise PdfInputError(
                f"no such file: {resolved}. Extraction reads from the "
                f"filesystem and will not invent a document"
            )
        if resolved.is_dir():
            raise PdfInputError(
                f"{resolved} is a directory, not a PDF. This component "
                f"extracts one document at a time; iterate over a directory "
                f"yourself so a failure names the file that caused it"
            )
        return resolved

    @staticmethod
    def _selection(pages: Sequence[int] | None) -> list[int] | None:
        if pages is None:
            return None
        if isinstance(pages, (str, bytes)):
            raise PdfInputError(
                f"pages must be a sequence of ints, got "
                f"{type(pages).__name__}"
            )
        try:
            chosen = list(pages)
        except TypeError as exc:
            raise PdfInputError(
                f"pages must be a sequence of ints, got "
                f"{type(pages).__name__}"
            ) from exc
        if not chosen:
            raise PdfInputError(
                "pages was an empty sequence. Selecting no pages would "
                "produce an empty result, which is a caller bug rather than "
                "a legitimate outcome"
            )
        for index in chosen:
            if isinstance(index, bool) or not isinstance(index, int):
                raise PdfInputError(
                    f"pages must contain ints, got {index!r} "
                    f"({type(index).__name__})"
                )
            if index < 0:
                raise PdfInputError(
                    f"pages must be zero-based and non-negative, got "
                    f"{index}. A negative index would silently select from "
                    f"the end, which is almost never what a caller means"
                )
        return chosen

    @staticmethod
    def _page_text(document: Any, index: int) -> str:
        try:
            page = document[index]
        except Exception as exc:
            raise PdfInputError(
                f"page {index} could not be loaded ({type(exc).__name__}: "
                f"{exc})"
            ) from exc
        try:
            return str(page.get_text())
        except Exception as exc:
            raise PdfExtractionError(
                f"page {index} could not be read as text "
                f"({type(exc).__name__}: {exc}). A scanned page has no text "
                f"layer; use pdf-extraction-ocr for those"
            ) from exc

    @staticmethod
    def _metadata(document: Any) -> dict[str, str]:
        raw = getattr(document, "metadata", None) or {}
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(key): str(value)
            for key, value in raw.items()
            if isinstance(value, (str, int, float, bool)) and value != ""
        }

    @staticmethod
    def _close(document: Any) -> None:
        close = getattr(document, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:  # pragma: no cover -- best effort cleanup
            pass

    def __repr__(self) -> str:
        return f"PdfExtractor(name={self.name!r})"

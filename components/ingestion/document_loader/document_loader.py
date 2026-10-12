"""Turn a file on disk into pages of text, and say which rule chose the parser.

The whole point
---------------
Loading is where a pipeline stops looking at files and starts looking at
*documents*. Two things go wrong here in practice, and this component is built
around both.

**Parsers get chosen by extension, and the extension lies.** A ``.txt`` file
full of PDF bytes is a PDF. So detection reads bytes: a signature is looked
for, and when no signature exists the loader says so rather than pretending it
decided from content. ``Detection.evidence`` is always populated, in the same
spirit as ``file-type-validator`` -- a caller that cannot accept an inference
can refuse it, but cannot be silently handed one.

**Parsers return different shapes.** ``pdf-extraction-pymupdf`` returns pages
as plain strings; ``pdf-extraction-ocr`` returns them as objects with a
``.text``. Both are reasonable and neither is wrong, but they are not
interchangeable, and a caller that has to write ``page if isinstance(page,
str) else page.text`` for every stage downstream has been handed a leak.
``PdfAdapter`` normalises either shape -- and does it by *duck typing the
result*, so this module imports neither extractor. That is what keeps both
sub-components selectable rather than merely separate.

OCR is a fallback, not an adapter
---------------------------------
An OCR engine reads page *images*. Turning a PDF page into pixels needs a
renderer, and a renderer is exactly the dependency ``pdf-extraction-ocr``
exists to avoid -- which is why it refuses PDFs outright. So the loader does
not pretend to rasterise anything. It accepts an optional ``ocr_fallback``
callable, and calls it *only* for pages that came back blank. Whoever supplies
that callable owns the rendering decision; the loader owns the policy of
"only where the text layer failed".

Detection, honestly
-------------------
* ``%PDF-`` in a bounded prefix means PDF. Signature, not extension.
* A block-level HTML tag in a bounded prefix means HTML.
* Markdown cannot be told from plain text by its bytes -- there is no
  signature, because Markdown *is* text. The extension is consulted and the
  evidence is recorded as ``declared-extension``.
* Otherwise the document is plain text, and the evidence says
  ``defaulted-to-text``.
* Bytes that are not valid UTF-8 and carry no known signature are not text and
  not anything else the library can open, so they are refused by name.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "AdapterError",
    "DEFAULT_READ_PREFIX",
    "DocumentLoader",
    "EmptyDocument",
    "Evidence",
    "HtmlAdapter",
    "LoaderError",
    "LoadedDocument",
    "MarkdownAdapter",
    "MissingAdapter",
    "OcrFallback",
    "Page",
    "PdfAdapter",
    "TextAdapter",
    "UnreadableDocument",
    "UnsupportedFormat",
    "detect_format",
]

#: Bounded prefix read for detection. Detection never loads a document whole.
DEFAULT_READ_PREFIX = 8192

_PAGE_JOINER = "\n\n"


class Evidence(str, Enum):
    """What the format decision was actually based on."""

    #: A byte signature matched. The strongest claim available.
    CONTENT_SIGNATURE = "content-signature"
    #: No signature exists for this format; the extension was consulted.
    DECLARED_EXTENSION = "declared-extension"
    #: Nothing matched; the document was treated as plain text.
    DEFAULTED_TO_TEXT = "defaulted-to-text"
    #: Nothing matched and the bytes are not text at all. This is what makes
    #: a detection *refusable* rather than merely weak.
    NO_MATCH = "no-match"


class LoaderError(Exception):
    """Base for every error this component raises by contract."""


class UnreadableDocument(LoaderError):
    """The path is missing, not a file, or could not be read."""


class EmptyDocument(LoaderError):
    """The file exists and is zero bytes -- nothing to parse."""


class UnsupportedFormat(LoaderError):
    """The bytes are neither text nor a format the configured adapters open."""


class MissingAdapter(LoaderError):
    """A format was detected but no adapter for it was configured."""


class AdapterError(LoaderError):
    """An adapter was called and raised; the original is chained, not hidden."""


@dataclass(frozen=True, slots=True)
class Detection:
    """A format decision plus the reason it was made."""

    format: str
    evidence: Evidence
    #: Human-readable detail, e.g. which signature matched.
    detail: str


@dataclass(frozen=True, slots=True)
class Page:
    """One page of extracted text, with a locator that survives downstream.

    ``source_locator`` is what a later citation points back at. It is recorded
    here, at extraction, because a chunking stage that has to guess which page
    a sentence came from will guess wrong.
    """

    #: 1-based, matching what a reader of the document would count.
    number: int
    text: str
    #: e.g. ``"page 3"``. Stable, human-readable, citation-ready.
    source_locator: str

    @property
    def is_blank(self) -> bool:
        """True when the page has no text at all, whitespace aside."""
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A file, parsed into pages, with everything a later stage needs to cite."""

    source_path: Path
    format: str
    detection: Detection
    pages: tuple[Page, ...]
    #: Non-fatal observations: a blank page, a fallback that ran, a parser
    #: that skipped something. Recorded rather than raised.
    warnings: tuple[str, ...]
    #: Which adapter produced this. Named so a run summary can attribute it.
    loader: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Every page joined with a blank line -- a page break is a real break."""
        return _PAGE_JOINER.join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def blank_pages(self) -> tuple[int, ...]:
        """Numbers of the pages that came back with no text."""
        return tuple(page.number for page in self.pages if page.is_blank)

    def __repr__(self) -> str:
        return (
            f"LoadedDocument(source_path={self.source_path.name!r}, "
            f"format={self.format!r}, page_count={self.page_count}, "
            f"character_count={self.character_count})"
        )


#: ``(path, page_number) -> text``. Supplying one is how a caller opts into OCR;
#: the loader never decides how to rasterise anything.
OcrFallback = Callable[[Path, int], str]


@runtime_checkable
class LoaderAdapter(Protocol):
    """Anything that can turn a path into pages."""

    #: Format name this adapter answers for, e.g. ``"pdf"``.
    format: str

    def extract(self, path: Path) -> tuple[tuple[Page, ...], Mapping[str, str]]:
        """Return ``(pages, metadata)``. Raises on a document it cannot parse."""
        ...


def _pages(texts: Sequence[str]) -> tuple[Page, ...]:
    """Wrap plain strings into numbered, located pages."""
    return tuple(
        Page(number=index, text=text, source_locator=f"page {index}")
        for index, text in enumerate(texts, start=1)
    )


def _read_prefix(path: Path, limit: int = DEFAULT_READ_PREFIX) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _decode(payload: bytes) -> str | None:
    """Decode as UTF-8, returning ``None`` when the bytes are not text."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A truncated multi-byte sequence at the very end of a bounded prefix
        # is an artefact of the bound, not evidence of binary content.
        if exc.start >= len(payload) - 3:
            try:
                return payload[: exc.start].decode("utf-8")
            except UnicodeDecodeError:
                return None
        return None


_HTML_SIGNAL = re.compile(
    r"<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>]|<title[\s>]",
    re.IGNORECASE,
)


def detect_format(path: Path | str, *, prefix_bytes: int = DEFAULT_READ_PREFIX) -> Detection:
    """Decide a format from bytes, and name the evidence behind the decision.

    Raises :class:`UnreadableDocument` for a path that cannot be read. Does not
    decide anything about a file being *parseable* -- only about what it is.
    """
    target = Path(path)
    if not target.exists():
        raise UnreadableDocument(f"no such file: {target}")
    if not target.is_file():
        raise UnreadableDocument(f"not a regular file: {target}")

    payload = _read_prefix(target, prefix_bytes)
    if payload.startswith(b"%PDF-"):
        return Detection("pdf", Evidence.CONTENT_SIGNATURE, "leading bytes are '%PDF-'")

    text = _decode(payload)
    if text is None:
        return Detection(
            "unknown",
            Evidence.NO_MATCH,
            "bytes are not valid UTF-8 and matched no known signature",
        )

    if _HTML_SIGNAL.search(text):
        return Detection("html", Evidence.CONTENT_SIGNATURE, "an HTML root tag appeared in the prefix")

    if target.suffix.lower() in {".md", ".markdown"}:
        return Detection(
            "markdown",
            Evidence.DECLARED_EXTENSION,
            "Markdown has no byte signature; the extension was consulted",
        )

    if not payload:
        return Detection("text", Evidence.DEFAULTED_TO_TEXT, "zero-byte file")

    return Detection("text", Evidence.DEFAULTED_TO_TEXT, "no signature matched; treated as plain text")


class TextAdapter:
    """Plain text, split on form feeds -- the one page break text actually has."""

    format = "text"

    def extract(self, path: Path) -> tuple[tuple[Page, ...], Mapping[str, str]]:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8", errors="replace")
        return _pages(decoded.split("\f")), {"bytes": str(len(raw))}

    def __repr__(self) -> str:
        return "TextAdapter()"


class MarkdownAdapter:
    """Markdown. Same parsing as text, deliberately -- splitting on headings is
    a *chunking* decision and does not belong at load time."""

    format = "markdown"

    def extract(self, path: Path) -> tuple[tuple[Page, ...], Mapping[str, str]]:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8", errors="replace")
        return _pages(decoded.split("\f")), {"bytes": str(len(raw))}

    def __repr__(self) -> str:
        return "MarkdownAdapter()"


class _TextExtractor(HTMLParser):
    """Strip markup to text, keeping block boundaries as line breaks."""

    _BLOCK = frozenset(
        {"p", "div", "section", "article", "li", "br", "tr", "blockquote",
         "h1", "h2", "h3", "h4", "h5", "h6", "pre", "ul", "ol"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title and self.title is None:
            self.title = data.strip()
        self._chunks.append(data)

    @property
    def text(self) -> str:
        joined = "".join(self._chunks)
        # Collapse the runs of blank lines block tags produce, but keep single
        # breaks: a paragraph boundary is real, four consecutive newlines are
        # an artefact of naive tag handling.
        collapsed = re.sub(r"\n{3,}", "\n\n", joined)
        return "\n".join(line.strip() for line in collapsed.splitlines()).strip()


class HtmlAdapter:
    """HTML, reduced to text with the standard library's own parser."""

    format = "html"

    def extract(self, path: Path) -> tuple[tuple[Page, ...], Mapping[str, str]]:
        payload = path.read_bytes()
        parser = _TextExtractor()
        parser.feed(payload.decode("utf-8", errors="replace"))
        metadata: dict[str, str] = {"bytes": str(len(payload))}
        if parser.title:
            metadata["title"] = parser.title
        return _pages([parser.text]), metadata

    def __repr__(self) -> str:
        return "HtmlAdapter()"


class PdfAdapter:
    """Normalise *either* PDF extractor's result into the common shape.

    The two sub-components differ only in what their ``pages`` sequence holds:
    plain strings, or objects exposing ``.text``. Both are accepted by duck
    typing, so neither extractor is imported here and neither can break the
    other by changing its internals.
    """

    format = "pdf"

    def __init__(self, extractor: object, *, name: str = "pdf") -> None:
        if not callable(getattr(extractor, "extract", None)):
            raise AdapterError(
                "PdfAdapter needs an object with a callable .extract(path); "
                f"got {type(extractor).__name__}"
            )
        self._extractor = extractor
        self._name = name

    @staticmethod
    def _as_text(page: object) -> str:
        """Accept a bare string or anything exposing ``.text`` -- nothing else."""
        if isinstance(page, str):
            return page
        text = getattr(page, "text", None)
        if isinstance(text, str):
            return text
        raise AdapterError(
            "a PDF extractor returned a page that is neither a str nor an "
            f"object with a .text attribute: {type(page).__name__}"
        )

    def extract(self, path: Path) -> tuple[tuple[Page, ...], Mapping[str, str]]:
        try:
            result = self._extractor.extract(path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise AdapterError(
                f"{self._name} failed on {path}: {type(exc).__name__}: {exc}"
            ) from exc

        raw_pages = getattr(result, "pages", None)
        if raw_pages is None:
            raise AdapterError(
                f"{self._name} returned {type(result).__name__}, which has no "
                ".pages attribute"
            )
        pages = _pages([self._as_text(page) for page in raw_pages])
        metadata = dict(getattr(result, "metadata", {}) or {})
        metadata.setdefault("source", str(getattr(result, "source", "") or path))
        return pages, metadata

    def __repr__(self) -> str:
        return f"PdfAdapter(name={self._name!r})"


class DocumentLoader:
    """Detect a format, run its adapter, and hand back one common shape.

        DocumentLoader().load(Path("guide.md"))
        DocumentLoader(adapters={"pdf": PdfAdapter(extractor)}).load(doc)
    """

    def __init__(
        self,
        *,
        adapters: Mapping[str, LoaderAdapter] | None = None,
        ocr_fallback: OcrFallback | None = None,
        prefix_bytes: int = DEFAULT_READ_PREFIX,
    ) -> None:
        if not isinstance(prefix_bytes, int) or isinstance(prefix_bytes, bool) or prefix_bytes <= 0:
            raise AdapterError(
                f"prefix_bytes must be a positive int, got {prefix_bytes!r}"
            )
        configured: dict[str, LoaderAdapter] = {
            "text": TextAdapter(),
            "markdown": MarkdownAdapter(),
            "html": HtmlAdapter(),
        }
        if adapters:
            for name, adapter in adapters.items():
                if not callable(getattr(adapter, "extract", None)):
                    raise AdapterError(
                        f"adapter for {name!r} has no callable .extract(path)"
                    )
                configured[name] = adapter
        self._adapters = configured
        self._ocr = ocr_fallback
        self._prefix_bytes = prefix_bytes

    @property
    def formats(self) -> tuple[str, ...]:
        """Every format this loader can currently open, sorted."""
        return tuple(sorted(self._adapters))

    def load(self, path: Path | str) -> LoadedDocument:
        """Parse one file into pages.

        Raises a typed error for a missing path, an empty file, unreadable
        bytes, or a format no configured adapter opens. Never silently
        substitutes a different parser for the one detection chose.
        """
        target = Path(path)
        if not target.exists():
            raise UnreadableDocument(f"no such file: {target}")
        if not target.is_file():
            raise UnreadableDocument(f"not a regular file: {target}")
        if target.stat().st_size == 0:
            raise EmptyDocument(
                f"{target.name} is zero bytes. An empty file is not an empty "
                "document -- it is a file that never arrived."
            )

        detection = detect_format(target, prefix_bytes=self._prefix_bytes)
        if detection.format == "unknown":
            raise UnsupportedFormat(
                f"{target.name}: {detection.detail}, so it is not a document "
                f"this loader can open. Adapters configured: {', '.join(self.formats)}"
            )

        adapter = self._adapters.get(detection.format)
        if adapter is None:
            raise MissingAdapter(
                f"{target.name} detected as {detection.format!r} "
                f"({detection.evidence.value}: {detection.detail}), but no "
                f"adapter is configured for it. Available: {', '.join(self.formats)}"
            )

        try:
            pages, metadata = adapter.extract(target)
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001 - attributed to the adapter
            raise AdapterError(
                f"the {detection.format!r} adapter failed on {target.name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        warnings: list[str] = []
        if detection.evidence is not Evidence.CONTENT_SIGNATURE:
            warnings.append(
                f"format {detection.format!r} chosen from "
                f"{detection.evidence.value}, not from a byte signature"
            )

        pages = self._apply_ocr_fallback(target, pages, warnings)

        return LoadedDocument(
            source_path=target,
            format=detection.format,
            detection=detection,
            pages=pages,
            warnings=tuple(warnings),
            loader=getattr(adapter, "format", detection.format),
            metadata=metadata,
        )

    def _apply_ocr_fallback(
        self, path: Path, pages: tuple[Page, ...], warnings: list[str]
    ) -> tuple[Page, ...]:
        """Re-run only the pages that came back blank, and only if opted in."""
        if self._ocr is None or not any(page.is_blank for page in pages):
            return pages

        rebuilt: list[Page] = []
        for page in pages:
            if not page.is_blank:
                rebuilt.append(page)
                continue
            try:
                text = self._ocr(path, page.number)
            except Exception as exc:  # noqa: BLE001 - a fallback must not kill a load
                warnings.append(
                    f"page {page.number}: OCR fallback raised "
                    f"{type(exc).__name__}; the blank page is kept as blank"
                )
                rebuilt.append(page)
                continue
            warnings.append(f"page {page.number}: text layer was empty; OCR fallback used")
            rebuilt.append(
                Page(number=page.number, text=text, source_locator=page.source_locator)
            )
        return tuple(rebuilt)

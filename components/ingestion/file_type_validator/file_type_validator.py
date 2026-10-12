"""Refuse a file by what it *is*, not by what it is called.

The whole point
---------------
An extension is a claim the sender makes about a file. This component answers a
different question -- what do the bytes actually look like -- and treats the
extension as a second, independent witness. When the two disagree, the file is
refused rather than reconciled, because a mismatch is the shape of every
content-type confusion attack there is.

Three refusal rules, and what each is for
-----------------------------------------
* **Nothing is guessed from silence.** A file with no recognised signature that
  decodes as UTF-8 is reported as inferred text, with that inference named in
  ``FileAssessment.evidence``. A caller who cannot accept an inference says so
  by refusing ``INFERRED`` evidence; this component does not silently upgrade
  "no signature found" into "this is text".
* **A read is bounded.** Signature sniffing reads a prefix and stops. The file
  is never loaded whole, never parsed, and never executed -- an HTML file
  containing a script is here exactly as much as a file containing the word
  "script" is, which is to say not at all.
* **A container is opened only as far as it must be.** Office documents are ZIP
  archives, so ``word/``, ``xl/`` and ``ppt/`` are what separates them; that is
  read from the archive's central directory, not by extracting anything.

Optional ``python-magic``
-------------------------
Content detection can delegate to libmagic, which knows thousands of formats
this table does not. It is optional and guarded by a specific ImportError
(FR-3): the component still imports and runs without it, and asks for it by
name only when a caller has opted in.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "DEFAULT_MAX_BYTES",
    "EVIDENCE_CONTENT",
    "EVIDENCE_INFERENCE",
    "EVIDENCE_MAGIC",
    "EVIDENCE_NONE",
    "EVIDENCE_ZIP",
    "EmptyFile",
    "ExtensionContentMismatch",
    "FileAssessment",
    "FileKind",
    "FileMissing",
    "FileTooLarge",
    "FileTypeError",
    "FileTypeValidator",
    "MissingOptionalDependency",
    "UnreadableFile",
    "UnsupportedType",
]


class FileKind(str, Enum):
    """What a file is, as far as this component is willing to claim."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    ZIP = "zip"
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    TIFF = "tiff"
    GZIP = "gzip"
    BZ2 = "bz2"
    RTF = "rtf"
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    UNKNOWN = "unknown"


#: How a kind was arrived at. Callers that need certainty reject inference.
EVIDENCE_CONTENT = "content-signature"
EVIDENCE_ZIP = "zip-container"
EVIDENCE_MAGIC = "libmagic"
EVIDENCE_INFERENCE = "inferred-text"
#: No signature matched and the bytes did not decode as text. Distinct from
#: :data:`EVIDENCE_INFERENCE` on purpose: "we concluded it was text from the
#: absence of a signature" and "we concluded nothing at all" are different
#: claims, and reporting the first for a binary would make ``evidence``
#: contradict the ``kind`` it is attached to.
EVIDENCE_NONE = "no-signature"

#: Longest prefix signature sniffing will read. Enough for every signature
#: below with room to spare; the point is that it is a constant, not the size
#: of the file.
_READ_PREFIX = 8192

#: Office formats are ZIP containers, told apart by a member path.
_ZIP_MARKERS: tuple[tuple[str, FileKind], ...] = (
    ("word/", FileKind.DOCX),
    ("xl/", FileKind.XLSX),
    ("ppt/", FileKind.PPTX),
)

#: Byte signatures, longest first so a prefix cannot shadow a longer one.
_SIGNATURES: tuple[tuple[bytes, FileKind], ...] = (
    (b"\x89PNG\r\n\x1a\n", FileKind.PNG),
    (b"\xff\xd8\xff", FileKind.JPEG),
    (b"GIF87a", FileKind.GIF),
    (b"GIF89a", FileKind.GIF),
    (b"II*\x00", FileKind.TIFF),
    (b"MM\x00*", FileKind.TIFF),
    (b"%PDF-", FileKind.PDF),
    (b"PK\x03\x04", FileKind.ZIP),
    (b"PK\x05\x06", FileKind.ZIP),  # an empty archive has no local entries
    (b"PK\x07\x08", FileKind.ZIP),
    (b"\x1f\x8b", FileKind.GZIP),
    (b"BZh", FileKind.BZ2),
    (b"{\\rtf", FileKind.RTF),
)

#: Extensions that name a text format. These carry no signature, so they can
#: only ever be inferred -- which is why they are listed separately from the
#: signature table rather than merged into it.
_EXTENSION_KINDS: dict[str, FileKind] = {
    ".txt": FileKind.TEXT,
    ".md": FileKind.MARKDOWN,
    ".markdown": FileKind.MARKDOWN,
    ".csv": FileKind.CSV,
    ".json": FileKind.JSON,
    ".html": FileKind.HTML,
    ".htm": FileKind.HTML,
    ".pdf": FileKind.PDF,
    ".docx": FileKind.DOCX,
    ".xlsx": FileKind.XLSX,
    ".pptx": FileKind.PPTX,
    ".zip": FileKind.ZIP,
    ".png": FileKind.PNG,
    ".jpg": FileKind.JPEG,
    ".jpeg": FileKind.JPEG,
    ".gif": FileKind.GIF,
    ".tif": FileKind.TIFF,
    ".tiff": FileKind.TIFF,
    ".gz": FileKind.GZIP,
    ".bz2": FileKind.BZ2,
    ".rtf": FileKind.RTF,
}

_DEFAULT_ALLOWED = frozenset(
    {
        FileKind.PDF,
        FileKind.DOCX,
        FileKind.XLSX,
        FileKind.PPTX,
        FileKind.ZIP,
        FileKind.PNG,
        FileKind.JPEG,
        FileKind.GIF,
        FileKind.TIFF,
        FileKind.RTF,
        FileKind.TEXT,
        FileKind.MARKDOWN,
        FileKind.CSV,
        FileKind.JSON,
        FileKind.HTML,
    }
)

DEFAULT_MAX_BYTES = 50 * 1024 * 1024


class FileTypeError(Exception):
    """Base for every refusal this component makes."""


class FileMissing(FileTypeError):
    """No such file."""


class UnreadableFile(FileTypeError):
    """Exists, but its bytes could not be read."""


class EmptyFile(FileTypeError):
    """Zero bytes. An empty file is never a document."""


class FileTooLarge(FileTypeError):
    """Over the configured ceiling."""


class UnsupportedType(FileTypeError):
    """Detected, and not among the types this validator accepts."""


class ExtensionContentMismatch(FileTypeError):
    """The extension and the bytes disagree."""


class MissingOptionalDependency(FileTypeError):
    """A caller asked for a capability this component ships as optional."""


@dataclass(frozen=True)
class FileAssessment:
    """The facts about one file, separate from any policy about it."""

    path: str
    size_bytes: int
    extension: str
    kind: FileKind
    evidence: str
    #: ``None`` when the extension names nothing this component knows, which is
    #: not the same as agreeing: an unknown extension is not a mismatch.
    extension_agrees: bool | None
    #: Set only when libmagic answered, so its verdict can be compared.
    libmagic_type: str | None = None

    @property
    def is_inferred(self) -> bool:
        return self.evidence == EVIDENCE_INFERENCE


def _libmagic_type(path: Path) -> str:
    """Ask libmagic, or explain precisely what is missing (FR-3)."""
    try:
        import magic
    except ImportError as exc:  # pragma: no cover - needs an env without it
        raise MissingOptionalDependency(
            "python-magic is optional and is not installed; the validator ran "
            "without it because it does so by default. Install it with "
            "`pip install python-magic` (and note that libmagic itself is a "
            "system library, not a Python wheel). To run without it, leave "
            "`use_magic` at its default."
        ) from exc
    try:
        return str(magic.from_file(str(path), mime=True))  # pragma: no cover
    except Exception as exc:  # pragma: no cover - libmagic's own failures
        raise UnreadableFile(f"libmagic could not read {path}: {exc}") from exc


def _zip_kind(path: Path) -> FileKind:
    """Tell the Office containers apart by their member paths.

    A ZIP whose central directory cannot be read is a corrupt archive, and is
    reported as unknown rather than assumed to be a plain zip -- the extension
    already claimed that, and claims are what this component is here to check.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return FileKind.UNKNOWN

    for marker, kind in _ZIP_MARKERS:
        if any(name.startswith(marker) for name in names):
            return kind
    return FileKind.ZIP


def _looks_like_text(sample: bytes) -> bool:
    """Decodes as UTF-8 with nothing in it that a text file should not have."""
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # A lone surrogate or an unassigned control character in the first few KB is
    # the signature of a binary that happens to decode.
    return not any(byte < 0x09 or 0x0E <= byte < 0x20 for byte in sample)


class FileTypeValidator:
    """Decide whether a file is a document this pipeline is willing to parse.

    ``assess`` reports; ``validate`` enforces. They are separate because the
    interesting failures are policy decisions a caller may want to make
    differently -- "reject the mismatch" versus "log it and continue" -- and a
    component that only ever raised would force the first choice on everyone.
    """

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        allowed: frozenset[FileKind] | None = None,
        use_magic: bool = False,
    ) -> None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            raise TypeError("max_bytes must be an int; a bool would defeat a bound")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive; 0 would reject everything")
        self.max_bytes = max_bytes
        self.allowed = _DEFAULT_ALLOWED if allowed is None else frozenset(allowed)
        self.use_magic = use_magic

    # -- reporting ---------------------------------------------------------
    def assess(self, path: str | Path) -> FileAssessment:
        """Report what the file is. Raises only when it cannot be reported on."""
        target = Path(path)

        try:
            size = target.stat().st_size
        except FileNotFoundError as exc:
            raise FileMissing(f"no such file: {target}") from exc
        except OSError as exc:
            raise UnreadableFile(f"could not stat {target}: {exc}") from exc

        if size == 0:
            raise EmptyFile(f"{target} is empty (0 bytes)")
        if size > self.max_bytes:
            raise FileTooLarge(
                f"{target} is {size} bytes, over the {self.max_bytes} byte limit"
            )

        try:
            with target.open("rb") as handle:
                prefix = handle.read(_READ_PREFIX)
        except OSError as exc:
            raise UnreadableFile(f"could not read {target}: {exc}") from exc

        magic_type = _libmagic_type(target) if self.use_magic else None
        kind, evidence = self._detect(target, prefix, magic_type)
        extension = target.suffix.lower()
        expected = _EXTENSION_KINDS.get(extension)
        if expected is None:
            agrees: bool | None = None
        elif expected is FileKind.ZIP and kind in {
            FileKind.DOCX,
            FileKind.XLSX,
            FileKind.PPTX,
        }:
            # .zip and an Office container are the same claim at two resolutions.
            agrees = True
        else:
            agrees = expected is kind

        return FileAssessment(
            path=str(target),
            size_bytes=size,
            extension=extension,
            kind=kind,
            evidence=evidence,
            extension_agrees=agrees,
            libmagic_type=magic_type,
        )

    def _detect(
        self, target: Path, prefix: bytes, magic_type: str | None
    ) -> tuple[FileKind, str]:
        """The detection ladder, most certain evidence first."""
        if magic_type is not None:
            kind = _KIND_FROM_MIME.get(magic_type)
            if kind is not None:
                return kind, EVIDENCE_MAGIC

        for signature, kind in _SIGNATURES:
            if prefix.startswith(signature):
                if kind is FileKind.ZIP:
                    return _zip_kind(target), EVIDENCE_ZIP
                return kind, EVIDENCE_CONTENT

        if _looks_like_text(prefix):
            # No signature, but the bytes behave like text. Stated as an
            # inference, because that is what it is.
            return _EXTENSION_KINDS.get(
                target.suffix.lower(), FileKind.TEXT
            ), EVIDENCE_INFERENCE

        return FileKind.UNKNOWN, EVIDENCE_NONE

    # -- policy ------------------------------------------------------------
    def validate(self, path: str | Path) -> FileAssessment:
        """Report, then refuse anything this validator will not accept."""
        assessment = self.assess(path)

        if assessment.kind is FileKind.UNKNOWN:
            raise UnsupportedType(
                f"{assessment.path}: content is not a recognised document type "
                f"(evidence: {assessment.evidence})"
            )
        if assessment.kind not in self.allowed:
            accepted = ", ".join(sorted(kind.value for kind in self.allowed))
            raise UnsupportedType(
                f"{assessment.path}: {assessment.kind.value} is not accepted; "
                f"accepted types are {accepted}"
            )
        if assessment.extension_agrees is False:
            expected = _EXTENSION_KINDS[assessment.extension].value
            raise ExtensionContentMismatch(
                f"{assessment.path}: extension claims {expected} but the "
                f"bytes are {assessment.kind.value} (evidence: "
                f"{assessment.evidence}). Refusing rather than guessing which "
                "one is telling the truth."
            )
        return assessment


#: libmagic reports a MIME type; map the ones this component can name.
_KIND_FROM_MIME: dict[str, FileKind] = {
    "application/pdf": FileKind.PDF,
    "image/png": FileKind.PNG,
    "image/jpeg": FileKind.JPEG,
    "image/gif": FileKind.GIF,
    "image/tiff": FileKind.TIFF,
    "application/zip": FileKind.ZIP,
    "application/gzip": FileKind.GZIP,
    "application/x-bzip2": FileKind.BZ2,
    "text/rtf": FileKind.RTF,
    "text/plain": FileKind.TEXT,
    "text/markdown": FileKind.MARKDOWN,
    "text/csv": FileKind.CSV,
    "application/json": FileKind.JSON,
    "text/html": FileKind.HTML,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileKind.DOCX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileKind.XLSX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileKind.PPTX,
}

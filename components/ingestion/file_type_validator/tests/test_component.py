"""Component tests for `file-type-validator`.

Runs standalone: ``python -m pytest components/ingestion/file_type_validator/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

The component exists to answer one question -- what are these bytes, really --
so the tests are built around the three refusals it makes: empty, oversized,
and unrecognised. The fourth, extension/content mismatch, is the one the
component is actually *for*, and it gets the most attention: a validator that
agrees with whatever the extension claims is not a validator.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from file_type_validator import (
    DEFAULT_MAX_BYTES,
    EVIDENCE_CONTENT,
    EVIDENCE_INFERENCE,
    EVIDENCE_NONE,
    EVIDENCE_ZIP,
    EmptyFile,
    ExtensionContentMismatch,
    FileAssessment,
    FileKind,
    FileMissing,
    FileTooLarge,
    FileTypeError,
    FileTypeValidator,
    MissingOptionalDependency,
    UnreadableFile,
    UnsupportedType,
)

# --- fixtures ---------------------------------------------------------------


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _zip_with(path: Path, member: str) -> Path:
    """A ZIP containing one member, i.e. a stand-in for an Office document."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "<root/>")
    return path


@pytest.fixture
def pdf(tmp_path):
    return _write(tmp_path / "report.pdf", b"%PDF-1.7\n%EOF\n")


@pytest.fixture
def docx(tmp_path):
    return _zip_with(tmp_path / "letter.docx", "word/document.xml")


@pytest.fixture
def plain_text(tmp_path):
    return _write(tmp_path / "notes.txt", b"just some notes\n")


@pytest.fixture
def validator():
    return FileTypeValidator()


# --- success paths ----------------------------------------------------------


def test_a_real_pdf_is_recognised_by_its_signature(validator, pdf):
    assessment = validator.assess(pdf)

    assert assessment.kind is FileKind.PDF
    assert assessment.evidence == EVIDENCE_CONTENT
    assert assessment.extension_agrees is True
    assert assessment.size_bytes == pdf.stat().st_size


def test_an_office_container_is_told_apart_from_a_plain_zip(validator, tmp_path):
    """docx, xlsx and pptx all begin with the same three bytes.

    Telling them apart is the difference between detecting a type and detecting
    a file signature, which is what the extension would have told anyone for
    free.
    """
    cases = {
        "letter.docx": ("word/document.xml", FileKind.DOCX),
        "sheet.xlsx": ("xl/workbook.xml", FileKind.XLSX),
        "deck.pptx": ("ppt/presentation.xml", FileKind.PPTX),
        "bundle.zip": ("payload.txt", FileKind.ZIP),
    }
    for name, (member, expected) in cases.items():
        path = _zip_with(tmp_path / name, member)
        assessment = validator.assess(path)
        assert assessment.kind is expected, name
        assert assessment.evidence == EVIDENCE_ZIP, name


def test_text_is_inferred_and_the_inference_is_named(validator, plain_text):
    """Absence of a signature is reported as an inference, not as a finding."""
    assessment = validator.assess(plain_text)

    assert assessment.kind is FileKind.TEXT
    assert assessment.evidence == EVIDENCE_INFERENCE
    assert assessment.is_inferred is True
    assert assessment.extension_agrees is True


def test_an_unknown_extension_is_not_a_mismatch(validator, tmp_path):
    """`None` and `False` are different answers and must stay different.

    An extension this component has never heard of says nothing, so it cannot
    disagree. Treating it as a mismatch would refuse every file a user renamed.
    """
    path = _write(tmp_path / "notes.dat", b"plain text despite the extension\n")

    assessment = validator.assess(path)

    assert assessment.extension == ".dat"
    assert assessment.extension_agrees is None
    validator.validate(path)  # must not raise


def test_validate_returns_the_same_assessment_it_reported(validator, pdf):
    assert validator.validate(pdf) == validator.assess(pdf)


def test_the_default_ceiling_is_a_real_size():
    assert DEFAULT_MAX_BYTES == 50 * 1024 * 1024


# --- failure paths ----------------------------------------------------------


def test_an_empty_file_is_refused_rather_than_treated_as_text(validator, tmp_path):
    """Zero bytes decodes as UTF-8, so only the size check catches this."""
    path = _write(tmp_path / "empty.txt", b"")

    with pytest.raises(EmptyFile):
        validator.validate(path)


def test_a_missing_file_names_the_path_it_could_not_find(validator, tmp_path):
    with pytest.raises(FileMissing) as excinfo:
        validator.validate(tmp_path / "nope.pdf")

    assert "nope.pdf" in str(excinfo.value)


def test_an_oversized_file_reports_both_sizes(validator, tmp_path):
    strict = FileTypeValidator(max_bytes=8)
    path = _write(tmp_path / "big.txt", b"x" * 64)

    with pytest.raises(FileTooLarge) as excinfo:
        strict.validate(path)

    message = str(excinfo.value)
    assert "64" in message and "8" in message


def test_an_oversized_file_is_refused_before_it_is_read(validator, tmp_path):
    """The ceiling is checked on stat, so a huge file is never opened.

    Reading first would make the limit a suggestion: the bytes would already be
    in memory by the time the size was rejected.
    """
    strict = FileTypeValidator(max_bytes=4)
    path = _write(tmp_path / "big.pdf", b"%PDF-" + b"x" * 100_000)

    with pytest.raises(FileTooLarge):
        strict.validate(path)


def test_a_pdf_that_is_actually_a_png_is_refused(validator, tmp_path):
    """The mismatch case the component exists for."""
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    path = _write(tmp_path / "invoice.pdf", png_bytes)

    with pytest.raises(ExtensionContentMismatch) as excinfo:
        validator.validate(path)

    message = str(excinfo.value)
    assert "pdf" in message and "png" in message


def test_refusing_a_mismatch_does_not_silently_pick_a_winner(validator, tmp_path):
    """The message must not claim to know which witness is lying."""
    path = _write(tmp_path / "data.csv", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    with pytest.raises(ExtensionContentMismatch) as excinfo:
        validator.validate(path)

    assert "Refusing rather than guessing" in str(excinfo.value)


def test_binary_that_decodes_as_nothing_is_unrecognised(validator, tmp_path):
    """NUL bytes rule out the text inference, and no signature matches."""
    path = _write(tmp_path / "blob.bin", bytes(range(256)) * 4)

    with pytest.raises(UnsupportedType):
        validator.validate(path)


def test_a_corrupt_zip_is_not_reported_as_a_healthy_one(validator, tmp_path):
    """A ZIP signature with an unreadable central directory is a corrupt file,
    and calling it a plain zip would let it through as an accepted type."""
    path = _write(tmp_path / "broken.docx", b"PK\x03\x04" + b"\x00" * 64)

    assessment = validator.assess(path)

    assert assessment.kind is FileKind.UNKNOWN
    with pytest.raises(UnsupportedType):
        validator.validate(path)


def test_a_type_outside_the_allowed_set_is_refused_even_when_genuine(validator, pdf):
    """Accepting a type and detecting a type are separate decisions."""
    images_only = FileTypeValidator(allowed=frozenset({FileKind.PNG}))

    with pytest.raises(UnsupportedType) as excinfo:
        images_only.validate(pdf)

    assert "pdf" in str(excinfo.value)


def test_an_unreadable_file_is_reported_as_such(tmp_path):
    """A path that is a directory is not a file and not a missing file."""
    directory = tmp_path / "as_folder.pdf"
    directory.mkdir()

    with pytest.raises((UnreadableFile, FileTypeError)):
        FileTypeValidator().assess(directory)


# --- construction -----------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_ceiling_is_rejected_at_construction(bad):
    """Zero would reject everything, and a bound that rejects everything is a
    caller's bug that should surface where they made it."""
    with pytest.raises(ValueError):
        FileTypeValidator(max_bytes=bad)


def test_a_boolean_ceiling_is_rejected_because_true_is_one():
    """`True` is an int equal to 1, which would silently mean a one-byte limit."""
    with pytest.raises(TypeError):
        FileTypeValidator(max_bytes=True)


def test_the_component_imports_without_libmagic_installed():
    """FR-3: python-magic is optional, so importing must never require it."""
    assert MissingOptionalDependency is not None


def test_opting_into_libmagic_without_it_names_the_package(tmp_path, monkeypatch):
    """The refusal must say what to install, not merely that something is absent."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "magic":
            raise ImportError("No module named 'magic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    path = _write(tmp_path / "notes.txt", b"hello\n")

    with pytest.raises(MissingOptionalDependency) as excinfo:
        FileTypeValidator(use_magic=True).assess(path)

    message = str(excinfo.value)
    assert "python-magic" in message
    assert "pip install python-magic" in message


def test_the_assessment_is_immutable(validator, pdf):
    """Frozen so a caller cannot quietly rewrite a verdict after the fact."""
    assessment = validator.assess(pdf)

    with pytest.raises(AttributeError):
        assessment.kind = FileKind.ZIP


def test_a_text_file_is_never_reported_as_certain(validator, tmp_path):
    """Pins the honesty rule: no signature means the kind is inferred."""
    for name in ("a.txt", "b.md", "c.csv", "d.json", "e.html"):
        path = _write(tmp_path / name, b'{"k": 1}\n')
        assessment = validator.assess(path)
        assert assessment.is_inferred, name
        assert isinstance(assessment, FileAssessment)


def test_evidence_never_contradicts_the_kind_it_belongs_to(validator, tmp_path):
    """"Inferred it was text" and "concluded nothing" are different claims.

    A binary blob once reported evidence='inferred-text' alongside
    kind=UNKNOWN, which made the field assert something about a kind it did not
    produce. The two are separate values so they cannot be confused again.
    """
    binary = _write(tmp_path / "blob.bin", bytes(range(256)) * 4)
    text = _write(tmp_path / "notes.txt", b"hello\n")

    assert validator.assess(binary).evidence == EVIDENCE_NONE
    assert validator.assess(text).evidence == EVIDENCE_INFERENCE
    assert validator.assess(binary).is_inferred is False

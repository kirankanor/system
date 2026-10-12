"""Runnable example for `file-type-validator`.

    python components/ingestion/file_type_validator/example.py

Builds a handful of files in a temporary directory and puts each one in front
of the validator, so every refusal below is produced by real bytes rather than
described. Nothing is parsed, executed or loaded whole -- the component reads a
bounded prefix and stops.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_type_validator import (
    EmptyFile,
    ExtensionContentMismatch,
    FileTooLarge,
    FileTypeValidator,
    UnsupportedType,
)

PDF = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def build(directory: Path) -> dict[str, Path]:
    """One file per case, each named for what it really is."""
    paths = {}

    paths["genuine pdf"] = directory / "report.pdf"
    paths["genuine pdf"].write_bytes(PDF)

    # Real bytes, dishonest extension.
    paths["png called a pdf"] = directory / "invoice.pdf"
    paths["png called a pdf"].write_bytes(PNG)

    paths["empty"] = directory / "empty.txt"
    paths["empty"].write_bytes(b"")

    # An Office document is a ZIP with a member path inside it.
    paths["docx"] = directory / "letter.docx"
    with zipfile.ZipFile(paths["docx"], "w") as archive:
        archive.writestr("word/document.xml", "<w:document/>")

    # Binary that no signature matches and that cannot decode as text.
    paths["unrecognised"] = directory / "blob.bin"
    paths["unrecognised"].write_bytes(bytes(range(256)) * 4)

    paths["oversized"] = directory / "huge.pdf"
    paths["oversized"].write_bytes(PDF + b"\x00" * 4096)
    return paths


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        paths = build(Path(tmp))

        validator = FileTypeValidator()
        print("== accepted ==")
        for label in ("genuine pdf", "docx"):
            assessment = validator.validate(paths[label])
            print(
                f"  {label:<18} -> {assessment.kind.value:<7} "
                f"({assessment.evidence}, agrees={assessment.extension_agrees})"
            )

        print("\n== refused ==")
        strict = FileTypeValidator(max_bytes=1024)
        cases = [
            ("png called a pdf", validator, ExtensionContentMismatch),
            ("empty", validator, EmptyFile),
            ("unrecognised", validator, UnsupportedType),
            ("oversized", strict, FileTooLarge),
        ]
        for label, active, expected in cases:
            try:
                active.validate(paths[label])
            except expected as exc:
                print(f"  {label:<18} -> {type(exc).__name__}")
                print(f"  {'':<18}    {exc}")
            else:  # pragma: no cover - the point of the example
                print(f"  {label:<18} -> NOT REFUSED (unexpected)")
                return 1

        print("\n== inference is stated, not assumed ==")
        text = Path(tmp) / "notes.txt"
        text.write_text("plain text\n", encoding="utf-8")
        assessment = validator.assess(text)
        print(
            f"  notes.txt          -> {assessment.kind.value} "
            f"(evidence={assessment.evidence}, is_inferred={assessment.is_inferred})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

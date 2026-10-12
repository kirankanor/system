"""Runnable example for `document-loader`.

    python example.py

Builds real files in a temporary directory and loads each one, then shows the
point of the component: two extractors with *different* result shapes are
normalised into exactly the same `LoadedDocument`. Nothing here is described
rather than run -- the unsupported and empty refusals are produced for real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_loader import (
    DocumentLoader,
    EmptyDocument,
    LoaderError,
    PdfAdapter,
    UnsupportedFormat,
    detect_format,
)


def write_fixtures(root: Path) -> None:
    (root / "notes.txt").write_text("line one\nline two\n\fpage two starts here\n", encoding="utf-8")
    (root / "guide.md").write_text("# Guide\n\nSome **bold** text.\n", encoding="utf-8")
    (root / "page.html").write_text(
        "<!doctype html><html><head><title>Demo</title></head>"
        "<body><h1>Heading</h1><p>First para.</p><p>Second para.</p></body></html>",
        encoding="utf-8",
    )
    (root / "report.pdf").write_bytes(b"%PDF-1.7\nnot really a pdf, but the signature is\n")
    (root / "blob.bin").write_bytes(bytes(range(256)) * 8)
    (root / "empty.txt").write_bytes(b"")


class StringPagesExtractor:
    """Stands in for pdf-extraction-pymupdf: pages are plain strings."""

    def extract(self, path):
        return SimpleNamespace(pages=("page one text", "page two text"), metadata={}, source=str(path))


class ObjectPagesExtractor:
    """Stands in for pdf-extraction-ocr: pages carry a .text attribute."""

    def extract(self, path):
        pages = (SimpleNamespace(text="page one text"), SimpleNamespace(text="page two text"))
        return SimpleNamespace(pages=pages, metadata={}, source=str(path))


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fixtures(root)
        loader = DocumentLoader()

        print("=== detection names its evidence ===")
        for name in ("notes.txt", "guide.md", "page.html", "report.pdf", "blob.bin"):
            detection = detect_format(root / name)
            print(f"  {name:<12} -> {detection.format:<9} [{detection.evidence.value}] {detection.detail}")

        print("\n=== one shape out, whatever went in ===")
        for name in ("notes.txt", "guide.md", "page.html"):
            doc = loader.load(root / name)
            print(f"  {name:<12} format={doc.format:<9} pages={doc.page_count} chars={doc.character_count}")
            print(f"               locators={[p.source_locator for p in doc.pages]}")
            if name == "page.html":
                print(f"               metadata={doc.metadata}")
            for warning in doc.warnings:
                print(f"               warning: {warning}")

        print("\n=== the two PDF extractor shapes normalise identically ===")
        for label, extractor in (
            ("pymupdf-style (str pages)", StringPagesExtractor()),
            ("ocr-style (.text pages)", ObjectPagesExtractor()),
        ):
            doc = DocumentLoader(adapters={"pdf": PdfAdapter(extractor)}).load(root / "report.pdf")
            print(f"  {label:<28} pages={[p.text for p in doc.pages]}")
            print(f"  {'':<28} locators={[p.source_locator for p in doc.pages]}")

        print("\n=== refusals, produced for real ===")
        try:
            loader.load(root / "blob.bin")
        except UnsupportedFormat as exc:
            print(f"  UnsupportedFormat: {exc}")
        try:
            loader.load(root / "empty.txt")
        except EmptyDocument as exc:
            print(f"  EmptyDocument:     {exc}")
        try:
            DocumentLoader().load(root / "report.pdf")
        except LoaderError as exc:
            print(f"  MissingAdapter:    {type(exc).__name__}: {str(exc)[:96]}...")
        try:
            loader.load(root / "nope.txt")
        except LoaderError as exc:
            print(f"  UnreadableDocument:{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

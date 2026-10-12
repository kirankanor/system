"""Runnable example for `metadata-enricher`.

    python components/ingestion/metadata_enricher/example.py

Builds a record the way an ingestion run would, then deliberately feeds the
enricher fields it cannot use. Every value printed is produced by real code;
no timestamp is read from a clock, because the whole point of the component is
that the one it uses was handed to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata_enricher import SYSTEM_PREFIX, MetadataEnricher  # noqa: E402

INGESTED_AT = "2026-10-11T09:30:00+00:00"


def show(label: str, bundle) -> None:
    print(f"\n== {label} ==")
    for name, bag in (
        ("extracted", bundle.extracted),
        ("system", bundle.system),
        ("source", bundle.source),
    ):
        print(f"  {name:<10} {bag}")
    print(f"  {'merged':<10} {bundle.merged}")
    print(f"  {'omitted':<10} {bundle.omitted or '(nothing dropped)'}")


def main() -> int:
    enricher = MetadataEnricher(ingested_at=INGESTED_AT)

    # 1. A well-formed record.
    good = enricher.enrich(
        source_name="uploads/report.pdf",
        source_metadata={"language": "en", "pages": 12},
        title="Quarterly Report",
        author="A. Author",
        page_number=3,
        content_sha256="a" * 64,
    )
    show("everything supplied is well-formed", good)

    # 2. Nothing supplied: the promise is what is *not* there.
    thin = enricher.enrich(source_name="uploads/scan.pdf")
    show("no title, author or page was available", thin)
    assert "title" not in thin.merged and "author" not in thin.merged
    print("  -> no title and no author were invented")

    # 3. Fields the enricher cannot use are dropped, and the rest survives.
    damaged = enricher.enrich(
        source_name="uploads/messy.pdf",
        title=42,  # wrong type
        author="   ",  # blank
        page_number=True,  # a bool is not a page number
        source_metadata={"language": "en", "authors": ["A", "B"]},  # nested
    )
    show("three malformed fields and one nested value", damaged)
    assert "language" in damaged.merged
    print("  -> the record survived; only the unusable parts were dropped")

    # 4. The document's own title is never overwritten by the caller's.
    contested = enricher.enrich(
        source_name="uploads/report.pdf",
        title="Title the pipeline guessed",
        source_metadata={"title": "Title written into the document"},
    )
    show("caller and document disagree about the title", contested)
    assert contested.merged["title"] == "Title written into the document"
    print("  -> source wins, so the document's own account is preserved")

    # 5. Determinism: the same inputs give the same output, every time.
    again = MetadataEnricher(ingested_at=INGESTED_AT).enrich(
        source_name="uploads/report.pdf",
        source_metadata={"language": "en", "pages": 12},
        title="Quarterly Report",
        author="A. Author",
        page_number=3,
        content_sha256="a" * 64,
    )
    assert again.merged == good.merged
    print("\n== replaying the same inputs ==")
    print(f"  -> identical output: {again.merged == good.merged}")
    print(f"  -> the injected time is stored verbatim: {again.system[f'{SYSTEM_PREFIX}ingested_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

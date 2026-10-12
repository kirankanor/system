"""Runnable example for `chunk-deduplicator`.

    python example.py

Builds a corpus containing the same paragraph several times -- inside one
document, across documents, and inside a shared template -- runs it under
each scope, and then shows the fingerprint gate accepting and refusing a
stored vector. Every outcome printed below comes from a real run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_deduplicator import (
    Deduplicator,
    EmbeddingReuseGate,
    FingerprintMismatch,
    Scope,
    fingerprint_of,
)


@dataclass(frozen=True)
class Chunk:
    """Mirrors the core-types contract."""

    text: str
    document_id: str | None = None
    source_name: str | None = None
    chunk_id: str | None = None
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)


SETUP = "Install the dependency with pip, then restart the kernel before importing."
API_KEY = "The API key is read from the environment at startup, never from source control."
TEMPLATE = "Copyright 2026 Example Corp. All rights reserved."
DIFFERENT = "Copyright 2026 Other Corp. All rights reserved."


def corpus() -> list[Chunk]:
    return [
        # install.md
        Chunk(SETUP, document_id="install", source_name="install.md", chunk_id="i-0", chunk_index=0),
        Chunk(API_KEY, document_id="install", source_name="install.md", chunk_id="i-1", chunk_index=1),
        # api.md -- repeats install.md's first paragraph verbatim
        Chunk(SETUP, document_id="api", source_name="api.md", chunk_id="a-0", chunk_index=0),
        Chunk(API_KEY, document_id="api", source_name="api.md", chunk_id="a-1", chunk_index=1),
        # The same template footer on both, plus a near-miss that is NOT a duplicate.
        Chunk(TEMPLATE, document_id="install", source_name="install.md", chunk_id="i-2", chunk_index=2),
        Chunk(TEMPLATE, document_id="api", source_name="api.md", chunk_id="a-2", chunk_index=2),
        Chunk(DIFFERENT, document_id="api", source_name="api.md", chunk_id="a-3", chunk_index=3),
        # An exact duplicate of i-1 inside install.md itself.
        Chunk(API_KEY, document_id="install", source_name="install.md", chunk_id="i-3", chunk_index=3),
        # Whitespace differs only in the middle; the declared normalization makes it equal.
        Chunk(SETUP.replace("before", "   before  "), document_id="docs", source_name="docs.md", chunk_id="d-0"),
    ]


def report(label: str, outcome) -> None:
    print(f"\n=== {label} ===")
    print(f"  retained {len(outcome.retained)} of 9, removed {outcome.removed} "
          f"across {outcome.duplicate_groups} group(s)")
    if outcome.is_clean:
        print("  (nothing was a duplicate)")
    for group in outcome.groups:
        print(f"  [{group.level}] {group.content_hash[:12]}...")
        print(f"      kept      {group.retained.where}")
        for appearance in group.collapsed:
            print(f"      removed   {appearance.where}")
        print(f"      documents involved: {', '.join(group.documents) or 'unknown'}")
    per_document = outcome.removals_by_document()
    if per_document:
        print("  removals by document: "
              + ", ".join(f"{doc}={count}" for doc, count in per_document.items()))


def main() -> None:
    chunks = corpus()

    for scope in Scope:
        report(f"scope={scope.value}", Deduplicator(scope=scope).dedupe(chunks))

    # What survives is the ORIGINAL object, not a wrapper.
    outcome = Deduplicator(scope=Scope.CORPUS).dedupe(chunks)
    print("\n  survivors are the same objects that went in: "
          f"{all(any(chunk is kept for kept in outcome.retained) for chunk in outcome.retained)}")
    print(f"  kept first occurrence: {outcome.retained[0].chunk_id!r}")

    # Case is preserved, so these two are NOT duplicates of each other.
    case_check = Deduplicator().dedupe([
        Chunk("Install the dependency now.", document_id="d", chunk_id="a"),
        Chunk("install the dependency now.", document_id="d", chunk_id="b"),
    ])
    print(f"  case preserved: {case_check.removed} removed "
          f"(near-identical differing only in case is out of scope)")

    # Near-misses are not duplicates either.
    near = Deduplicator().dedupe([
        Chunk(TEMPLATE, document_id="d", chunk_id="a"),
        Chunk(DIFFERENT, document_id="d", chunk_id="b"),
    ])
    print(f"  near-miss pair: {near.removed} removed (exact equality only)")

    # ---------------------------------------------------------------
    # The fingerprint half: when may a stored vector be reused?
    # ---------------------------------------------------------------
    print("\n=== fingerprint-gated embedding reuse ===")
    gate = EmbeddingReuseGate.configured(
        model="text-embedding-3-small",
        dimensions=1536,
        chunking_config={"size": 512, "overlap": 64},
    )
    print(f"  fingerprint to store: {gate.value}")

    stored = fingerprint_of(
        model="text-embedding-3-small",
        dimensions=1536,
        chunking_config={"overlap": 64, "size": 512},  # key order differs
    ).value
    print(f"  identical config, different key order -> may_reuse={gate.may_reuse(stored)}")

    swapped_model = fingerprint_of(
        model="text-embedding-ada-002",
        dimensions=1536,
        chunking_config={"size": 512, "overlap": 64},
    ).value
    print(f"  embedding model changed          -> may_reuse={gate.may_reuse(swapped_model)}")

    resized = fingerprint_of(
        model="text-embedding-3-small",
        dimensions=1536,
        chunking_config={"size": 1024, "overlap": 64},
    ).value
    print(f"  chunk size changed               -> may_reuse={gate.may_reuse(resized)}")
    print(f"  nothing stored yet               -> may_reuse={gate.may_reuse(None)}")

    # The error names the field, which is the whole point of parsing it.
    try:
        gate.check(swapped_model)
    except FingerprintMismatch as exc:
        print(f"\n  refused with a typed error naming the mismatch:\n    {exc}")


if __name__ == "__main__":
    main()

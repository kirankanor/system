"""Runnable example for `chunk-validator`.

    python example.py

Builds a batch of chunks that fails in every way the validator is designed to
catch, runs it three times under each fatal policy, and prints what each one
did with the same failures. Every outcome below is produced by a real run --
nothing is described that is not executed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_validator import (
    ChunkValidator,
    FatalPolicy,
    Rule,
    Severity,
)


@dataclass(frozen=True)
class Chunk:
    """Mirrors the core-types contract: identity fields may be None."""

    text: str
    document_id: str | None = None
    source_name: str | None = None
    chunk_id: str | None = None
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)


def batch() -> list[Chunk]:
    good = [
        Chunk(
            text="Install the dependency with pip, then restart the kernel.",
            document_id="doc-1",
            source_name="setup.md",
            chunk_id="c-001",
            chunk_index=0,
            page_start=1,
            page_end=1,
        ),
        Chunk(
            text="The API key is read from the environment at startup.",
            document_id="doc-1",
            source_name="setup.md",
            chunk_id="c-002",
            chunk_index=1,
            page_start=1,
            page_end=2,
        ),
    ]
    return [
        *good,
        # Whitespace only: fatal, and no repair can invent content that is not there.
        Chunk(text="   \n\t  ", document_id="doc-1", chunk_id="c-003", page_start=1, page_end=1),
        # Missing document_id: fatal, unrepairable -- nobody would author a source.
        Chunk(text="An orphaned fragment of useful prose.", chunk_id="c-004", page_start=1, page_end=1),
        # Missing chunk_id: fatal but derivable from document, index and text.
        Chunk(text="Identifiable by content alone.", document_id="doc-2", chunk_index=7, page_start=1, page_end=1),
        # Inverted page range: fatal, and the two values genuinely can be swapped.
        Chunk(text="A page range written back to front.", document_id="doc-2", chunk_id="c-005", page_start=4, page_end=1),
        # Missing location: a WARNING by default, so it is accepted regardless.
        Chunk(text="No page range, but perfectly retrievable.", document_id="doc-2", chunk_id="c-006"),
        # Oversized: fatal, because it will not fit the model.
        Chunk(text="x" * 1000, document_id="doc-3", chunk_id="c-007", page_start=1, page_end=1),
        # Extraction anomaly recorded in metadata: a warning, not a refusal.
        Chunk(
            text="Text recovered from a scan with a bad gutter.",
            document_id="doc-3",
            chunk_id="c-008",
            page_start=1,
            page_end=1,
            metadata={"extraction_warning": "two lines dropped at the gutter"},
        ),
    ]


def describe(label: str, outcome) -> None:
    print(f"\n=== {label} ===")
    print(f"  accepted   {len(outcome.accepted)}")
    print(f"  rejected   {len(outcome.rejected)}")
    print(f"  quarantined {len(outcome.quarantined)}")
    print(f"  fatal findings {outcome.fatal_count}, warning findings {outcome.warning_count}")
    for held in outcome.held():
        tag = "repair-failed" if held.repair_failed else "held"
        print(f"    [{tag}] chunk {held.index}:")
        for finding in held.findings:
            print(f"        {finding.rule.value} ({finding.severity.value}): {finding.message}")
    for repair in outcome.repairs:
        print(f"    [repaired] {repair.description}")
    if outcome.is_clean:
        print("    (every chunk was clean)")


def main() -> None:
    chunks = batch()

    describe(
        "QUARANTINE -- the default: hold it, keep the reason, lose nothing",
        ChunkValidator(min_chars=10, max_chars=500).validate(chunks),
    )

    describe(
        "REJECT -- the same failures, dropped from the accepted set",
        ChunkValidator(min_chars=10, max_chars=500, fatal_policy=FatalPolicy.REJECT).validate(chunks),
    )

    describe(
        "REPAIR -- derive what is derivable, hold the rest",
        ChunkValidator(min_chars=10, max_chars=500, fatal_policy=FatalPolicy.REPAIR).validate(chunks),
    )

    describe(
        "Severities overridden -- now a short chunk is fatal too",
        ChunkValidator(
            min_chars=10,
            severities={Rule.TEXT_TOO_SHORT: Severity.FATAL},
        ).validate([Chunk(text="tiny", document_id="d", chunk_id="c", page_start=1, page_end=1)]),
    )

    # The contract the orchestration layer relies on, checked for real:
    # re-inspect every accepted chunk and prove none carries a fatal finding.
    # The run's own `results` still show the pre-repair findings -- that is
    # what `repairs` records -- so only a fresh inspection can prove it.
    print()
    for policy in FatalPolicy:
        validator = ChunkValidator(min_chars=10, max_chars=500, fatal_policy=policy)
        outcome = validator.validate(chunks)
        fatal = [result.index for result in outcome.results if result.is_fatal]
        still_bad = [
            chunk for chunk in outcome.accepted if validator.inspect(chunk).is_fatal
        ]
        print(
            f"  [{policy.value:<10}] {len(fatal)} chunks had fatal findings, "
            f"{len(outcome.accepted)} reached the embedder, "
            f"{len(still_bad)} carried one through"
        )
        if still_bad:
            raise AssertionError("accepted set is not safe to embed")
    print("  contract holds: no unaddressed fatal finding reached the embedder.")


if __name__ == "__main__":
    main()

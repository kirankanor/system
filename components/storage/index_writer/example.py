"""Runnable example for `index-writer`.

    python example.py

Walks the four things this component refuses to do silently: an oversized
batch, a payload the index cannot hold, a batch that fails part way through,
and an index created over data that already exists. Every number printed
below comes from a real run.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parent

# This component imports core-types and vector-store-protocol, which live in
# their own component directories. The manifest declares both, so `ragkit add`
# copies them beside this file; here we reach across to siblings.
for sibling in ("core_types", "vector_store_protocol"):
    sys.path.insert(0, str(COMPONENT.parent / sibling))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
from index_writer import (  # noqa: E402
    BatchReport,
    DestructiveOperationRefused,
    IndexLifecycle,
    IndexWriter,
    PayloadMismatchError,
    bounded_batches,
    record_from,
)


class MemoryStore:
    """A tiny stand-in for a real vector backend: id -> row."""

    def __init__(self) -> None:
        self.rows: dict[str, core.EmbeddedChunk] = {}

    def add(self, items):
        for item in items:
            self.rows[item.chunk.chunk_id] = item
        return len(items)

    def clear(self) -> int:
        removed = len(self.rows)
        self.rows.clear()
        return removed


class FlakyStore(MemoryStore):
    """Rejects any batch containing the poisoned chunk, singly or in company."""

    def __init__(self, poison: str) -> None:
        super().__init__()
        self.poison = poison
        self.calls: list[int] = []

    def add(self, items):
        self.calls.append(len(items))
        if any(item.chunk.chunk_id == self.poison for item in items):
            raise RuntimeError(f"backend refused chunk {self.poison}")
        return super().add(items)


def chunks(count: int, *, width: int = 2) -> list[core.EmbeddedChunk]:
    return [
        core.EmbeddedChunk(
            chunk=core.Chunk(
                chunk_index=index,
                text=f"body of chunk {index}",
                chunk_id=f"c-{index}",
                document_id="doc-1",
                source_name="guide.md",
                page_start=1,
                page_end=1,
                metadata={
                    "start_offset": index * 100,
                    "end_offset": index * 100 + 80,
                    "section": "intro" if index % 2 == 0 else "body",
                },
            ),
            vector=tuple(float(index + n) for n in range(width)),
            index_fingerprint="model=m|dims=2|chunking=abc",
        )
        for index in range(count)
    ]


def main() -> None:
    print("=== 1. bounded batches ===")
    for size in (1, 4, 64):
        split = bounded_batches(list(range(10)), size)
        print(
            f"  batch_size={size:<3} -> {len(split)} batch(es), "
            f"largest {max(len(b) for b in split)} item(s)"
        )

    print("\n=== 2. a successful write ===")
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2, batch_size=3)
    report = writer.upsert(chunks(7))
    print(f"  {report.summary()}")
    print(f"  ok={report.ok}  written so far={writer.written}")
    print(f"  store now holds {len(store.rows)} row(s)")

    print("\n=== 3. provenance survives to the record (FR-48) ===")
    record = record_from(chunks(1)[0])
    for name, value in record.provenance.items():
        print(f"    {name:<14} = {value!r}")
    print(f"    index_fingerprint = {record.index_fingerprint!r}")
    print(f"    traceable back to source bytes: {record.is_traceable}")

    print("\n=== 4. a payload the index cannot hold, refused first (FR-47) ===")
    wide = chunks(2, width=3)
    before = len(store.rows)
    try:
        writer.upsert(wide)
    except PayloadMismatchError as exc:
        print(f"  refused before writing: {exc}")
        print(f"  rows before={before}, rows after={len(store.rows)} "
              f"(nothing landed)")

    print("\n=== 5. a schema the index cannot answer queries about ===")
    no_section = core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=0,
            text="a chunk that never recorded its section",
            chunk_id="c-bare",
            document_id="doc-1",
            source_name="guide.md",
        ),
        vector=(0.0, 1.0),
        index_fingerprint="model=m|dims=2|chunking=abc",
    )
    strict = IndexWriter(
        MemoryStore(),
        dimensions=2,
        required_fields=("document_id", "section"),
    )
    try:
        strict.upsert([no_section])
    except PayloadMismatchError as exc:
        print(f"  refused: {exc}")

    print("\n=== 6. a batch that fails part way through (FR-46) ===")
    flaky = FlakyStore(poison="c-4")
    flaky_writer = IndexWriter(flaky, dimensions=2, batch_size=3)
    partial: BatchReport = flaky_writer.upsert(chunks(7))
    print(f"  backend calls (batch attempts): {flaky.calls}")
    print(f"  {partial.summary()}")
    for result in partial.results:
        print(f"    {result.describe()}")
    print(f"  ok={partial.ok}; successes were NOT lost: "
          f"{len(flaky.rows)} row(s) present")
    try:
        partial.raise_if_failed()
    except Exception as exc:  # noqa: BLE001 - printed, not swallowed
        print(f"  raise_if_failed() -> {type(exc).__name__}: {exc}")

    print("\n=== 7. creation is not destruction (FR-47) ===")
    fresh = MemoryStore()
    lifecycle = IndexLifecycle(count=lambda name: len(fresh.rows))
    print(f"  create on an empty index -> {lifecycle.create('docs')} row(s)")
    writer2 = IndexWriter(fresh, dimensions=2, batch_size=4)
    writer2.upsert(chunks(5))
    print(f"  index now holds {len(fresh.rows)} row(s)")
    try:
        lifecycle.create("docs")
    except DestructiveOperationRefused as exc:
        print(f"  second create refused: {exc}")
    lost = lifecycle.replace(
        "docs", reason="switched to v3 schema", clear=lambda name: fresh.clear()
    )
    print(f"  explicit replace recorded: {lost.describe()}")
    print(f"  rows after replace: {len(fresh.rows)}; "
          f"destructions on record: {len(lifecycle.destructions)}")

    print("\n=== 8. a reason is required to destroy anything ===")
    try:
        IndexLifecycle(count=lambda name: 0).replace(
            "docs", reason="   ", clear=lambda name: 0
        )
    except Exception as exc:  # noqa: BLE001 - printed, not swallowed
        print(f"  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

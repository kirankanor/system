"""Runnable example for `index-verifier`.

    python example.py

Walks five situations an index can be in, and what verification says about
each: a correct index, a record removed behind the pipeline's back while the
count still matches, an index whose vectors are the wrong width, an index
that answers but degrades citations, and a write that is not visible yet.
Every line printed below comes from a real run.
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
from index_verifier import (  # noqa: E402
    ALL_CHECKS,
    IndexVerifier,
    VerificationError,
)

WIDTH = 4


class MemoryStore:
    """A small stand-in with the two methods verification needs.

    Like FaissVectorStore, it refuses a probe whose width it does not hold --
    which is what makes a wrong dimension visible as a typed failure rather
    than as an empty answer.
    """

    name = "memory-4d"

    def __init__(self, dimensions: int = WIDTH) -> None:
        self.rows: dict[str, core.EmbeddedChunk] = {}
        self.dimensions = dimensions

    # readback: optional, but it makes presence and dimension exact
    def get(self, chunk_id: str):
        item = self.rows.get(chunk_id)
        return None if item is None else item.vector

    def add(self, items) -> int:
        for item in items:
            self.rows[item.chunk.chunk_id] = item
        return len(items)

    def count(self) -> int:
        return len(self.rows)

    def query(self, vector, k: int, filter=None):
        if len(vector) != self.dimensions:
            raise ValueError(
                f"query vector has {len(vector)} dimensions but this store "
                f"holds {self.dimensions}-wide vectors"
            )
        hits = list(self.rows.values())
        if filter:
            hits = [
                i
                for i in hits
                if all(
                    getattr(i.chunk, key, None) == value
                    for key, value in filter.items()
                )
            ]
        hits.sort(key=lambda i: sum((a - b) ** 2 for a, b in zip(i.vector, vector)))
        return [
            core.ScoredDocument.from_chunk(i.chunk, score=0.0, source_name=self.name)
            for i in hits[:k]
        ]


def make(index: int, *, width: int = WIDTH, document: str = "doc-a",
         source: str | None = "guide.md", section: str | None = "intro"):
    metadata = {"section": section} if section else {}
    return core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=index,
            text=f"body of chunk {index}",
            chunk_id=f"c-{index}",
            document_id=document,
            source_name=source,
            page_start=1,
            page_end=1,
            metadata=metadata,
        ),
        vector=tuple(float(index + n) for n in range(width)),
        index_fingerprint=f"model=m|dims={width}|chunking=size512",
    )


def expectation(count: int = 4):
    return [make(i) for i in range(count)]


def show(label: str, store, expected, **kw) -> None:
    verifier = IndexVerifier(store, expected=expected, dimensions=WIDTH, **kw)
    verdict = verifier.verify()
    print(f"\n=== {label} ===")
    print(f"  {verdict.summary()}")
    for finding in verdict.findings:
        print(f"    {finding.describe()}")
    if verdict.ready:
        print("  -> READY: every check ran and passed")


def main() -> None:
    expected = expectation()

    # --------------------------------------------------------------- 1 --
    correct = MemoryStore()
    correct.add(expected)
    show("1. a correctly written index", correct, expected)

    # --------------------------------------------------------------- 2 --
    # A record is removed behind the pipeline's back. The count still matches
    # because a stale row took its place -- exactly why a count match alone
    # is not verification.
    removed = MemoryStore()
    removed.add(expected)
    del removed.rows["c-2"]
    removed.rows["c-99"] = make(99)
    print("\n=== 2. a record removed behind its back, count unchanged ===")
    print(f"  index reports {removed.count()} record(s); "
          f"expectation is {len(expected)} -- the count check passes")
    verdict = IndexVerifier(removed, expected=expected, dimensions=WIDTH).verify()
    for finding in verdict.fatal:
        print(f"    {finding.describe()}")
    print(f"  verdict.ok = {verdict.ok}, verdict.ready = {verdict.ready}")
    print(f"  named('c-2') -> {[f.describe() for f in verdict.named('c-2')]}")

    # --------------------------------------------------------------- 3 --
    # The index was rebuilt at a different width than the expectation says.
    wide = MemoryStore(dimensions=WIDTH + 1)
    wide.add([make(i, width=WIDTH + 1) for i in range(4)])
    print("\n=== 3. vectors whose dimension was altered ===")
    try:
        wide.query((0.0, 0.0, 0.0, 0.0), k=1)
    except ValueError as exc:
        print(f"  the store refuses the probe: {exc}")
    verdict = IndexVerifier(wide, expected=expected, dimensions=WIDTH).verify()
    for finding in verdict.fatal:
        print(f"    {finding.describe()}")

    # --------------------------------------------------------------- 4 --
    # The expectation itself cannot cite its records: no source file, no
    # section. It will be retrievable, just not quotable.
    degraded_expected = [make(i, source=None, section=None) for i in range(4)]
    degraded = MemoryStore()
    degraded.add(degraded_expected)
    show("4. an expectation whose records cannot be cited", degraded,
         degraded_expected)

    # --------------------------------------------------------------- 5 --
    # The write has happened but is not visible yet: eventual consistency.
    class LaggingStore(MemoryStore):
        def __init__(self) -> None:
            super().__init__()
            self.visible_after = 3
            self.reads = 0

        def count(self) -> int:
            self.reads += 1
            if self.reads < self.visible_after:
                return 0
            return super().count()

    lagging = LaggingStore()
    lagging.add(expected)
    sleeps: list[float] = []
    clock = {"t": 0.0}

    show(
        "5a. write not yet visible, single pass (wait_ms=0)",
        lagging,
        expected,
    )

    lagging.reads = 0
    print("\n=== 5b. same store, bounded wait for eventual consistency ===")
    verifier = IndexVerifier(
        lagging,
        expected=expected,
        dimensions=WIDTH,
        wait_ms=5_000,
        interval_ms=100,
        sleep=lambda seconds: (sleeps.append(seconds),
                               clock.__setitem__("t", clock["t"] + seconds)),
        now=lambda: clock["t"],
    )
    verdict = verifier.verify()
    print(f"  {verdict.summary()}")
    print(f"  slept {len(sleeps)} time(s), total "
          f"{sum(sleeps) * 1000:.0f}ms of a 5000ms budget")
    print(f"  -> {'READY' if verdict.ready else 'NOT READY'}")

    # --------------------------------------------------------------- 6 --
    # Readiness is withheld when a check could not run at all.
    print("\n=== 6. a verdict missing evidence is not a verdict that passed ===")
    from index_verifier import Finding, VerificationVerdict  # noqa: E402

    half = VerificationVerdict(
        findings=(),
        expected_records=4,
        actual_records=4,
        checks_run=("count", "presence"),
    )
    print(f"  ok={half.ok} (no fatal findings)")
    print(f"  ready={half.ready} -- withheld because "
          f"{', '.join(half.checks_not_run())} never ran")
    print(f"  all checks: {', '.join(ALL_CHECKS)}")

    # --------------------------------------------------------------- 7 --
    print("\n=== 7. construction mistakes are caught up front ===")
    for overrides in (
        {"dimensions": 0},
        {"dimensions": "4"},
        {"dimensions": WIDTH, "wait_ms": -1},
        {"dimensions": WIDTH, "probe_k": 0},
    ):
        try:
            IndexVerifier(correct, expected=expected, **overrides)
        except VerificationError as exc:
            print(f"  {overrides} -> {exc}")


if __name__ == "__main__":
    main()

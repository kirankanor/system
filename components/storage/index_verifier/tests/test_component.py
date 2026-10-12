"""Component tests for `index-verifier`.

Runs standalone: ``python -m pytest components/storage/index_verifier/tests``
from the repository root. Does not import `rag_code_library`.

Two contracts dominate, and neither is "the happy path returned":

* **FR-50** -- a successful write is never reported as verification. Every
  test that expects a green verdict asserts ``ready`` *and* that all seven
  checks ran, because ``ok`` alone can be produced by a verdict that looked
  at almost nothing.
* **AC-27** -- verification passes on a correctly written index and fails,
  naming the record and the reason, when a record is removed behind its back
  or a vector's dimension is altered. The removal test deliberately keeps the
  record count correct, so only ``presence`` can catch it.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types and vector-store-protocol, which live in
# their own component directories. The manifest declares both dependencies, so
# `ragkit add` copies them beside this file; here we reach across to siblings.
for sibling in ("core_types", "vector_store_protocol"):
    sys.path.insert(0, str(COMPONENT.parent / sibling))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
import pytest  # noqa: E402
from index_verifier import (  # noqa: E402
    ALL_CHECKS,
    COUNT,
    DIMENSION,
    EMBEDDING,
    FATAL,
    Finding,
    IndexVerifier,
    METADATA,
    PRESENCE,
    RELATIONSHIP,
    RETRIEVAL,
    WARNING,
    VerificationError,
    VerificationVerdict,
)

WIDTH = 4


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class MemoryStore:
    """Protocol-shaped store with a per-record readback.

    Like FaissVectorStore, it refuses a probe whose width it does not hold,
    which is what makes a wrong dimension a typed failure rather than an
    empty answer.
    """

    def __init__(self, dimensions: int = WIDTH) -> None:
        self.rows: dict[str, core.EmbeddedChunk] = {}
        self.dimensions = dimensions
        self.query_calls = 0

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
        self.query_calls += 1
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
        hits.sort(
            key=lambda i: sum((a - b) ** 2 for a, b in zip(i.vector, vector))
        )
        return [
            core.ScoredDocument.from_chunk(i.chunk, score=0.0, source_name="mem")
            for i in hits[:k]
        ]


class NoReadbackStore(MemoryStore):
    """Same store, but with no ``get`` -- the path a bare protocol offers."""

    get = None  # present as an attribute, deliberately not callable


class QueryOnlyStore:
    """No ``get`` attribute at all, and no ``add``/``delete`` either."""

    def __init__(self, rows) -> None:
        self.rows = list(rows)

    def count(self) -> int:
        return len(self.rows)

    def query(self, vector, k: int, filter=None):
        hits = list(self.rows)
        if filter:
            hits = [
                i
                for i in hits
                if all(
                    getattr(i.chunk, key, None) == value
                    for key, value in filter.items()
                )
            ]
        hits.sort(
            key=lambda i: sum((a - b) ** 2 for a, b in zip(i.vector, vector))
        )
        return [
            core.ScoredDocument.from_chunk(i.chunk, score=0.0, source_name="qo")
            for i in hits[:k]
        ]


def make(
    index: int = 0,
    *,
    width: int = WIDTH,
    chunk_id: str | None = None,
    document: str | None = "doc-a",
    source: str | None = "guide.md",
    page_start: int | None = 1,
    page_end: int | None = 1,
    metadata: dict | None = None,
    text: str | None = None,
) -> core.EmbeddedChunk:
    return core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=index,
            text=text if text is not None else f"body of chunk {index}",
            chunk_id=chunk_id if chunk_id is not None else f"c-{index}",
            document_id=document,
            source_name=source,
            page_start=page_start,
            page_end=page_end,
            metadata=metadata if metadata is not None else {"section": "intro"},
        ),
        vector=tuple(float(index + n) for n in range(width)),
        index_fingerprint=f"model=m|dims={width}|chunking=size512",
    )


def expectation(count: int = 4) -> list[core.EmbeddedChunk]:
    return [make(i) for i in range(count)]


def verify(store, expected=None, **kwargs):
    kwargs.setdefault("dimensions", WIDTH)
    return IndexVerifier(
        store, expected=expectation() if expected is None else expected, **kwargs
    ).verify()


class Clock:
    """A fake clock and sleep, so no test ever actually waits."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def frozen_now(self) -> float:
        """A clock that never advances -- the infinite-loop guard."""
        return 0.0


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_a_missing_store_is_refused():
    with pytest.raises(VerificationError) as exc:
        IndexVerifier(None, expected=expectation(), dimensions=WIDTH)
    assert "store" in str(exc.value)


def test_a_missing_expectation_is_refused():
    with pytest.raises(VerificationError) as exc:
        IndexVerifier(MemoryStore(), expected=None, dimensions=WIDTH)
    assert "expected" in str(exc.value)


@pytest.mark.parametrize("width", [0, -1, True, "4", 4.0, None])
def test_an_unusable_dimension_is_refused(width):
    with pytest.raises(VerificationError):
        IndexVerifier(MemoryStore(), expected=expectation(), dimensions=width)


@pytest.mark.parametrize("value", [-1, True, "1", 1.5])
def test_an_unusable_wait_is_refused(value):
    for field in ("wait_ms", "interval_ms"):
        with pytest.raises(VerificationError):
            IndexVerifier(
                MemoryStore(), expected=expectation(), dimensions=WIDTH,
                **{field: value},
            )


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_an_unusable_probe_k_is_refused(value):
    with pytest.raises(VerificationError):
        IndexVerifier(
            MemoryStore(), expected=expectation(), dimensions=WIDTH,
            probe_k=value,
        )


@pytest.mark.parametrize("missing", ["count", "query"])
def test_a_store_without_a_protocol_method_is_refused(missing):
    class Incomplete:
        def count(self):
            return 0

        def query(self, vector, k, filter=None):
            return []

    incomplete = Incomplete()
    delattr(type(incomplete), missing)
    with pytest.raises(VerificationError) as exc:
        IndexVerifier(incomplete, expected=expectation(), dimensions=WIDTH)
    assert missing in str(exc.value)


def test_verification_errors_are_their_own_type_not_a_value_error():
    assert issubclass(VerificationError, Exception)
    assert not issubclass(VerificationError, ValueError)
    assert not issubclass(VerificationError, KeyError)


# --------------------------------------------------------------------------- #
# AC-27 / FR-49 -- the seven comparisons
# --------------------------------------------------------------------------- #


def test_a_correctly_written_index_passes_and_is_ready():
    store = MemoryStore()
    store.add(expectation())
    verdict = verify(store)
    assert verdict.fatal == ()
    assert verdict.warnings == ()
    assert verdict.ok is True
    assert verdict.ready is True
    assert verdict.checks_not_run() == ()
    assert set(verdict.checks_run) == set(ALL_CHECKS)
    assert verdict.expected_records == 4
    assert verdict.actual_records == 4


def test_ready_reports_ready_and_the_counts():
    store = MemoryStore()
    store.add(expectation())
    summary = verify(store).summary()
    assert summary.startswith("READY")
    assert "4/4 records" in summary
    assert "checked 7/7" in summary
    assert "1 pass(es)" in summary


def test_a_short_count_is_a_fatal_count_finding():
    store = MemoryStore()
    store.add(expectation(3))
    verdict = verify(store)
    assert verdict.ok is False
    count_findings = [f for f in verdict.fatal if f.check == COUNT]
    assert len(count_findings) == 1
    assert "3" in count_findings[0].reason
    assert "4" in count_findings[0].reason
    assert count_findings[0].record is None
    assert count_findings[0].where == "<index>"


def test_ac27_a_removed_record_is_named_even_though_the_count_still_matches():
    """The sharpest form: a stale row took the removed record's place.

    Only ``presence`` can catch this -- the count is exactly right, which is
    precisely why a count match alone is not verification.
    """
    store = MemoryStore()
    store.add(expectation())
    del store.rows["c-2"]
    store.rows["c-99"] = make(99)

    assert store.count() == 4  # unchanged

    verdict = verify(store)
    assert not [f for f in verdict.fatal if f.check == COUNT]  # count is fine
    named = verdict.named("c-2")
    assert len(named) == 1
    assert named[0].check == PRESENCE
    assert named[0].fatal
    assert "removed behind the pipeline's back" in named[0].reason
    assert "c-2" in named[0].reason or named[0].record == "c-2"


def test_ac27_a_stale_row_that_was_never_expected_is_named_too():
    store = MemoryStore()
    store.add(expectation())
    store.rows["c-99"] = make(99)
    verdict = verify(store)
    named = verdict.named("c-99")
    assert named
    assert named[0].check == PRESENCE
    assert "stale row" in named[0].reason


def test_ac27_an_altered_vector_dimension_is_named():
    """A store rebuilt at another width, verified against the old one."""
    store = MemoryStore(dimensions=WIDTH + 1)
    store.add([make(i, width=WIDTH + 1) for i in range(4)])

    verdict = verify(store)
    assert verdict.ok is False
    dimension = [f for f in verdict.fatal if f.check == DIMENSION]
    assert dimension, "a width mismatch must be reported as a dimension finding"
    assert dimension[0].record is not None  # AC-27 names the record
    assert str(WIDTH + 1) in dimension[0].reason
    assert str(WIDTH) in dimension[0].reason


def test_a_per_record_width_change_is_seen_even_though_the_probe_is_accepted():
    """The index *claims* the right width; individual rows do not.

    Here the probe passes -- so the index-level check has nothing to say --
    and only a per-record readback can name the four rows that were altered.
    This is the case a smoke query alone would report as READY.
    """
    store = MemoryStore(dimensions=WIDTH)  # accepts a 4-wide probe...
    store.add([make(i, width=WIDTH + 1) for i in range(4)])  # ...holds 5-wide rows
    assert store.query((0.0,) * WIDTH, k=1)  # the probe is not refused

    verdict = verify(store)
    per_record = [
        f
        for f in verdict.fatal
        if f.check == DIMENSION and f.record in {f"c-{i}" for i in range(4)}
    ]
    assert len(per_record) == 4
    assert all("5-wide" in f.reason for f in per_record)
    assert verdict.ready is False


def test_a_missing_required_metadata_key_is_named():
    store = MemoryStore()
    store.add([make(i, metadata={}) for i in range(4)])
    verifier = IndexVerifier(
        store, expected=expectation(), dimensions=WIDTH,
        required_metadata=("section",),
    )
    verdict = verifier.verify()
    findings = [f for f in verdict.fatal if f.check == METADATA]
    assert findings
    assert findings[0].record is not None
    assert "'section'" in findings[0].reason


def test_a_reserved_key_leaking_into_visible_metadata_is_a_warning_only():
    class LeakyStore(MemoryStore):
        """An adapter that failed to strip its own prefix on the way out."""

        def query(self, vector, k: int, filter=None):
            hits = super().query(vector, k=k, filter=filter)
            for hit in hits:
                object.__setattr__(hit, "metadata", {"_rag_chunk_id": "c-0"})
            return hits

    leaky = LeakyStore()
    leaky.add(expectation())
    verdict = IndexVerifier(leaky, expected=expectation(), dimensions=WIDTH).verify()
    warnings = [f for f in verdict.warnings if f.check == METADATA]
    assert warnings
    assert "_rag_chunk_id" in warnings[0].reason
    assert verdict.ready is True  # a warning does not withhold readiness


def test_a_relationship_without_a_document_id_is_fatal():
    store = MemoryStore()
    items = [make(0, document=None)]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    findings = [f for f in verdict.fatal if f.check == RELATIONSHIP]
    assert findings
    assert findings[0].record == "c-0"
    assert "document_id" in findings[0].reason


def test_a_relationship_without_a_source_file_is_only_a_warning():
    store = MemoryStore()
    items = [make(0, source=None)]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    findings = [f for f in verdict.warnings if f.check == RELATIONSHIP]
    assert findings
    assert "no source file" in findings[0].reason
    assert verdict.ready is True


def test_a_record_with_neither_pages_nor_section_warns_about_its_citation():
    store = MemoryStore()
    items = [make(0, page_start=None, page_end=None, metadata={})]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    reasons = [f.reason for f in verdict.warnings if f.check == RELATIONSHIP]
    assert any("where in it" in r for r in reasons)


def test_a_section_alone_satisfies_the_citation_expectation():
    store = MemoryStore()
    items = [make(0, page_start=None, page_end=None, metadata={"section": "intro"})]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    assert not [f for f in verdict.warnings if f.check == RELATIONSHIP]


def test_duplicate_expected_chunk_ids_are_fatal():
    store = MemoryStore()
    items = [make(0, chunk_id="same"), make(1, chunk_id="same")]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    findings = [f for f in verdict.fatal if f.check == RELATIONSHIP]
    assert findings
    assert findings[0].record == "same"
    assert "upsert would collapse them" in findings[0].reason


def test_an_empty_index_returns_nothing_and_the_smoke_check_says_so():
    verdict = verify(MemoryStore())
    findings = [f for f in verdict.fatal if f.check == RETRIEVAL]
    assert findings
    assert "returned no rows" in findings[0].reason
    assert verdict.ready is False


def test_an_empty_expectation_without_a_probe_is_not_ready():
    verdict = IndexVerifier(MemoryStore(), expected=[], dimensions=WIDTH).verify()
    findings = [f for f in verdict.fatal if f.check == RETRIEVAL]
    assert findings
    assert "probe" in findings[0].reason
    assert "has not been verified" in findings[0].reason


def test_an_empty_expectation_can_still_be_smoke_checked_with_a_probe():
    store = MemoryStore()
    store.add(expectation())
    # Verifying "the index should be empty" against a probe we supply.
    empty = IndexVerifier(
        store, expected=[], dimensions=WIDTH, probe=make(0).vector
    ).verify()
    # It is not empty, so the count disagrees -- which is the right answer.
    assert empty.ok is False
    assert [f for f in empty.fatal if f.check == COUNT]


def test_a_record_from_an_unexpected_document_is_fatal():
    class ExtraDocumentStore(MemoryStore):
        def add(self, items):
            extra = make(7, chunk_id="c-7", document="doc-elsewhere")
            return super().add([*items, extra])

    store = ExtraDocumentStore()
    store.add(expectation())
    verdict = verify(store)
    findings = [f for f in verdict.fatal if f.check == RELATIONSHIP]
    assert findings
    assert findings[0].record == "c-7"
    assert "doc-elsewhere" in findings[0].reason


def test_the_probe_record_must_come_back_from_its_own_index():
    """The index answered, just not for the thing we asked about."""

    class HalfAnswering(MemoryStore):
        def query(self, vector, k: int, filter=None):
            if not filter:
                # The unfiltered smoke query returns only foreign rows.
                foreign = make(50, chunk_id="c-50", document="doc-a")
                return [
                    core.ScoredDocument.from_chunk(
                        foreign.chunk, score=0.0, source_name="mem"
                    )
                ]
            return super().query(vector, k=k, filter=filter)

    store = HalfAnswering()
    store.add(expectation())
    verdict = verify(store)
    findings = [f for f in verdict.fatal if f.check == RETRIEVAL]
    assert findings
    assert findings[0].record == "c-0"
    assert "did not come back" in findings[0].reason


# --------------------------------------------------------------------------- #
# FR-49 -- fatal versus warning, and what the verdict exposes
# --------------------------------------------------------------------------- #


def test_warnings_do_not_stop_the_index_being_ready():
    store = MemoryStore()
    items = [make(i, source=None) for i in range(4)]
    store.add(items)
    verdict = IndexVerifier(store, expected=items, dimensions=WIDTH).verify()
    assert verdict.ok is True
    assert verdict.ready is True
    assert verdict.warnings
    assert verdict.fatal == ()


def test_ok_is_not_ready_when_a_check_could_not_run():
    """The distinction FR-50 turns on, stated directly."""
    half = VerificationVerdict(
        findings=(),
        expected_records=4,
        actual_records=4,
        checks_run=(COUNT, PRESENCE),
    )
    assert half.ok is True
    assert half.ready is False
    assert set(half.checks_not_run()) == set(ALL_CHECKS) - {COUNT, PRESENCE}
    assert "ready=False" not in half.summary()
    assert half.summary().startswith("ok:")
    assert "not run" in half.report()


def test_ready_is_withheld_when_the_smoke_check_never_ran():
    no_smoke = VerificationVerdict(
        findings=(),
        expected_records=4,
        actual_records=4,
        checks_run=tuple(c for c in ALL_CHECKS if c != RETRIEVAL),
    )
    assert no_smoke.ok is True
    assert no_smoke.ready is False


def test_named_returns_only_findings_about_that_record():
    store = MemoryStore()
    store.add(expectation())
    store.rows["c-99"] = make(99)
    verdict = verify(store)
    assert verdict.named("c-99")
    assert verdict.named("c-0") == ()
    assert verdict.named("nothing") == ()


def test_findings_describe_themselves_with_their_severity_and_record():
    fatal = Finding(check=COUNT, severity=FATAL, reason="gone", record="c-1")
    warning = Finding(check=METADATA, severity=WARNING, reason="empty")
    assert fatal.describe() == "[FATAL] count c-1: gone"
    assert warning.describe() == "[warn] metadata <index>: empty"
    assert fatal.fatal is True
    assert warning.fatal is False
    assert warning.where == "<index>"


def test_raise_if_failed_is_silent_on_a_clean_verdict():
    store = MemoryStore()
    store.add(expectation())
    verify(store).raise_if_failed()  # must not raise


def test_raise_if_failed_names_the_first_finding_when_it_is_not_clean():
    verdict = verify(MemoryStore())
    with pytest.raises(VerificationError) as exc:
        verdict.raise_if_failed()
    assert "fatal finding" in str(exc.value)
    assert "NOT READY" in str(exc.value)


def test_a_verdict_carries_its_counts_and_run_check_tally():
    verdict = verify(MemoryStore())
    assert verdict.expected_records == 4
    assert verdict.actual_records == 0
    assert verdict.total_findings >= 1
    assert len(verdict.checks_run) == len(ALL_CHECKS)


def test_findings_are_immutable():
    with pytest.raises(AttributeError):
        Finding(check=COUNT, severity=FATAL, reason="x").reason = "y"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        VerificationVerdict(
            findings=(), expected_records=0, actual_records=0, checks_run=()
        ).attempts = 9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Presence evidence: readback versus query
# --------------------------------------------------------------------------- #


def test_a_store_without_readback_is_detected_and_still_verifies():
    store = NoReadbackStore()
    store.add(expectation())
    verifier = IndexVerifier(store, expected=expectation(), dimensions=WIDTH)
    assert verifier.has_readback is False
    verdict = verifier.verify()
    assert verdict.ready is True
    assert set(verdict.checks_run) == set(ALL_CHECKS)


def test_without_readback_a_removed_record_is_still_named():
    store = NoReadbackStore()
    store.add(expectation())
    del store.rows["c-2"]
    verdict = IndexVerifier(store, expected=expectation(), dimensions=WIDTH).verify()
    named = verdict.named("c-2")
    assert named
    assert named[0].check == PRESENCE
    assert named[0].fatal
    assert "filtered query" in named[0].reason


def test_readback_is_used_when_the_store_offers_it():
    verifier = IndexVerifier(MemoryStore(), expected=expectation(), dimensions=WIDTH)
    assert verifier.has_readback is True


def test_an_explicit_fetch_overrides_the_store_s_own_get():
    store = MemoryStore()
    store.add(expectation())
    seen: list[str] = []

    def fetch(chunk_id):
        seen.append(chunk_id)
        return store.get(chunk_id)

    verifier = IndexVerifier(
        store, expected=expectation(), dimensions=WIDTH, fetch=fetch
    )
    assert verifier.has_readback is True
    verifier.verify()
    assert seen  # the injected readback was consulted


def test_a_readback_that_raises_is_reported_as_a_reason_not_swallowed():
    store = MemoryStore()
    store.add(expectation())

    def broken(chunk_id):
        raise RuntimeError("readback exploded")

    verdict = IndexVerifier(
        store, expected=expectation(), dimensions=WIDTH, fetch=broken
    ).verify()
    named = verdict.named("c-0")
    assert named
    assert "readback exploded" in named[0].reason


def test_presence_is_established_by_query_when_nothing_can_be_read_back():
    store = QueryOnlyStore(expectation())
    verifier = IndexVerifier(store, expected=expectation(), dimensions=WIDTH)
    assert verifier.has_readback is False
    verdict = verifier.verify()
    assert verdict.ready is True


# --------------------------------------------------------------------------- #
# FR-50 -- bounded eventual consistency
# --------------------------------------------------------------------------- #


class LaggingStore(MemoryStore):
    """Reports no rows until it has been counted ``visible_after`` times."""

    def __init__(self, visible_after: int = 3) -> None:
        super().__init__()
        self.visible_after = visible_after
        self.reads = 0

    def count(self) -> int:
        self.reads += 1
        if self.reads < self.visible_after:
            return 0
        return super().count()


def test_zero_wait_means_exactly_one_pass():
    clock = Clock()
    store = LaggingStore()
    store.add(expectation())
    verdict = verify(
        store, wait_ms=0, interval_ms=100, sleep=clock.sleep, now=clock.now
    )
    assert verdict.attempts == 1
    assert clock.sleeps == []
    assert verdict.ok is False


def test_a_bounded_wait_retries_and_then_reports_ready():
    clock = Clock()
    store = LaggingStore(visible_after=3)
    store.add(expectation())
    verdict = verify(
        store, wait_ms=5_000, interval_ms=100, sleep=clock.sleep, now=clock.now
    )
    assert verdict.ready is True
    assert verdict.attempts == 3
    assert len(clock.sleeps) == 2
    assert verdict.waited_ms == 200


def test_the_wait_stops_as_soon_as_the_index_consistently_answers():
    clock = Clock()
    store = LaggingStore(visible_after=2)
    store.add(expectation())
    verdict = verify(
        store, wait_ms=5_000, interval_ms=100, sleep=clock.sleep, now=clock.now
    )
    assert verdict.attempts == 2
    assert verdict.ready is True


def test_the_wait_is_bounded_even_when_the_clock_never_advances():
    """The infinite-loop guard: attempts come from config, not the clock."""
    clock = Clock()
    store = LaggingStore(visible_after=10_000)  # never becomes visible
    store.add(expectation())
    verdict = IndexVerifier(
        store,
        expected=expectation(),
        dimensions=WIDTH,
        wait_ms=1_000,
        interval_ms=100,
        sleep=clock.sleep,
        now=clock.frozen_now,
    ).verify()
    assert verdict.attempts == 1 + math.ceil(1_000 / 100)
    assert len(clock.sleeps) == 10
    assert verdict.ok is False


def test_a_structural_finding_is_not_retried():
    """A dimension mismatch will not fix itself, so waiting only delays it."""
    clock = Clock()
    store = MemoryStore(dimensions=WIDTH + 1)
    store.add([make(i, width=WIDTH + 1) for i in range(4)])
    verdict = verify(
        store, wait_ms=5_000, interval_ms=100, sleep=clock.sleep, now=clock.now
    )
    assert verdict.attempts == 1
    assert clock.sleeps == []
    assert verdict.waited_ms == 0


def test_a_zero_interval_cannot_loop_forever():
    clock = Clock()
    store = LaggingStore(visible_after=10_000)
    store.add(expectation())
    verdict = IndexVerifier(
        store, expected=expectation(), dimensions=WIDTH,
        wait_ms=1_000, interval_ms=0,
        sleep=clock.sleep, now=clock.now,
    ).verify()
    assert verdict.attempts == 1
    assert clock.sleeps == []


def test_verify_never_raises_for_a_finding():
    """A failed verification is a result, not an exception."""
    verdict = verify(MemoryStore())
    assert isinstance(verdict, VerificationVerdict)
    assert verdict.ok is False  # it did fail -- as data


# --------------------------------------------------------------------------- #
# The component's own promises
# --------------------------------------------------------------------------- #


def test_the_verifier_exposes_no_way_to_write():
    """FR-50: a successful write is never reported as verification."""
    verifier = IndexVerifier(MemoryStore(), expected=expectation(), dimensions=WIDTH)
    for method in ("add", "put", "delete", "write", "upsert"):
        assert not hasattr(verifier, method), method
    assert hasattr(verifier, "verify")


def test_the_probe_defaults_to_the_first_expected_record():
    verifier = IndexVerifier(
        MemoryStore(), expected=expectation(), dimensions=WIDTH
    )
    assert verifier.probe == make(0).vector
    assert IndexVerifier(
        MemoryStore(), expected=[], dimensions=WIDTH
    ).probe is None


def test_the_component_imports_no_third_party_packages():
    """core-types and vector-store-protocol are declared; the rest is stdlib."""
    source = (COMPONENT / "index_verifier.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    stdlib = set(sys.stdlib_module_names)
    third_party = roots - stdlib - {
        "core_types",
        "vector_store_protocol",
        "index_verifier",
    }
    assert third_party == set(), third_party
    assert "rag_code_library" not in imported


def test_the_manifest_declares_what_this_component_imports():
    manifest = json.loads(
        (COMPONENT / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["dependencies"]) == {"core-types", "vector-store-protocol"}
    assert manifest["category"] == "storage"
    assert manifest["required_packages"] == []


def test_the_check_names_match_what_the_specification_calls_them():
    """A rename would silently orphan every assertion built on these."""
    assert ALL_CHECKS == (
        COUNT,
        PRESENCE,
        EMBEDDING,
        DIMENSION,
        METADATA,
        RELATIONSHIP,
        RETRIEVAL,
    )
    assert set(ALL_CHECKS) == {
        "count",
        "presence",
        "embedding",
        "dimension",
        "metadata",
        "relationship",
        "retrieval",
    }

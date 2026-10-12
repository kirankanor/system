"""Component tests for `index-writer`.

Runs standalone: ``python -m pytest components/storage/index_writer/tests``
from the repository root. Does not import `rag_code_library`.

Three contracts are under test, and none of them is "the happy path returns":

* **FR-47** -- a payload that does not fit the index is refused *before* the
  first row lands, and no operation discards stored rows without recording
  why. Every refusal test asserts the store is unchanged afterwards.
* **FR-46** -- a batch that fails part way through is reported **per item**,
  and the items that succeeded are still written. The test counts rows in the
  store, because "not lost" has to mean they are physically there.
* **FR-48** -- every record carries its provenance fields, present even when
  the value is ``None``, so there is always something to assert on.
"""

from __future__ import annotations

import ast
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
from index_writer import (  # noqa: E402
    PROVENANCE_FIELDS,
    BatchItemResult,
    BatchReport,
    BatchWriteError,
    DestructiveOperationRefused,
    IndexDestruction,
    IndexLifecycle,
    IndexWriter,
    IndexWriterError,
    PayloadMismatchError,
    StoredChunkRecord,
    bounded_batches,
    record_from,
    validate_payload,
)
from vector_store_protocol import StorageInputError  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures and test doubles
# --------------------------------------------------------------------------- #


class MemoryStore:
    """A tiny stand-in for a vector backend: id -> row."""

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


class PoisonStore(MemoryStore):
    """Rejects any batch containing ``poison``, whether alone or in company.

    Deliberately refuses single-item adds of the poison chunk too, so a
    per-item retry records a real failure rather than recovering by luck.
    """

    def __init__(self, poison: str) -> None:
        super().__init__()
        self.poison = poison
        self.calls: list[int] = []

    def add(self, items):
        self.calls.append(len(items))
        if any(item.chunk.chunk_id == self.poison for item in items):
            raise RuntimeError(f"backend refused chunk {self.poison}")
        return super().add(items)


class ExplodingStore(MemoryStore):
    """Accepts the first call, then fails forever."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def add(self, items):
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("backend went away")
        return super().add(items)


def make_chunk(
    index: int = 0,
    *,
    chunk_id: str | None = None,
    width: int = 2,
    metadata: dict | None = None,
    document_id: str | None = "doc-1",
    source_name: str | None = "guide.md",
    page_start: int | None = 1,
    page_end: int | None = 1,
    fingerprint: str = "model=m|dims=2|chunking=abc",
) -> core.EmbeddedChunk:
    base = {
        "start_offset": index * 100,
        "end_offset": index * 100 + 80,
        "section": "intro" if index % 2 == 0 else "body",
    }
    if metadata is not None:
        base = metadata
    return core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=index,
            text=f"body of chunk {index}",
            chunk_id=chunk_id or f"c-{index}",
            document_id=document_id,
            source_name=source_name,
            page_start=page_start,
            page_end=page_end,
            metadata=base,
        ),
        vector=tuple(float(index + n) for n in range(width)),
        index_fingerprint=fingerprint,
    )


def make_chunks(count: int, **kw) -> list[core.EmbeddedChunk]:
    return [make_chunk(index, **kw) for index in range(count)]


# --------------------------------------------------------------------------- #
# bounded_batches (FR-45)
# --------------------------------------------------------------------------- #


def test_items_are_split_into_consecutive_bounded_batches():
    batches = bounded_batches(list(range(10)), 4)
    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert [item for batch in batches for item in batch] == list(range(10))


def test_a_batch_size_larger_than_the_input_yields_one_batch():
    assert bounded_batches([1, 2, 3], 100) == [[1, 2, 3]]


def test_a_batch_size_of_one_writes_one_at_a_time():
    assert bounded_batches([1, 2, 3], 1) == [[1], [2], [3]]


def test_an_empty_input_produces_no_batches_not_one_empty_one():
    assert bounded_batches([], 10) == []


@pytest.mark.parametrize("size", [0, -1])
def test_a_non_positive_batch_size_is_refused(size):
    """A batch of zero would write nothing and report success."""
    with pytest.raises(IndexWriterError) as excinfo:
        bounded_batches([1, 2], size)
    assert "bounded batch" in str(excinfo.value)


@pytest.mark.parametrize("size", [True, "4", 4.0, None])
def test_a_non_integer_batch_size_is_refused(size):
    with pytest.raises(IndexWriterError):
        bounded_batches([1, 2], size)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# FR-48 -- the stored record
# --------------------------------------------------------------------------- #


def test_a_record_carries_every_provenance_field():
    record = record_from(make_chunk(0))
    assert set(record.provenance) == set(PROVENANCE_FIELDS)


def test_provenance_is_populated_from_the_chunk_it_came_from():
    record = record_from(make_chunk(3))
    assert record.document_id == "doc-1"
    assert record.chunk_id == "c-3"
    assert record.source_name == "guide.md"
    assert record.chunk_index == 3
    assert record.page_start == 1
    assert record.page_end == 1


def test_source_offsets_are_read_from_metadata():
    record = record_from(make_chunk(2))
    assert record.start_offset == 200
    assert record.end_offset == 280


def test_the_section_reference_is_read_from_metadata():
    assert record_from(make_chunk(0)).section == "intro"
    assert record_from(make_chunk(1)).section == "body"


def test_a_field_the_chunker_never_set_is_none_rather_than_absent():
    """AC-26 asserts on these fields, so they must always exist."""
    record = record_from(
        make_chunk(0, metadata={}, page_start=None, page_end=None)
    )
    for name in PROVENANCE_FIELDS:
        assert hasattr(record, name), name
    assert record.start_offset is None
    assert record.end_offset is None
    assert record.section is None
    assert record.page_start is None
    assert record.provenance["section"] is None


def test_the_vector_and_fingerprint_travel_with_the_record():
    record = record_from(make_chunk(1, fingerprint="model=x|dims=2|chunking=y"))
    assert record.vector == (1.0, 2.0)
    assert record.index_fingerprint == "model=x|dims=2|chunking=y"


def test_user_metadata_is_kept_alongside_the_derived_fields():
    record = record_from(make_chunk(0, metadata={"section": "intro", "lang": "en"}))
    assert record.metadata["lang"] == "en"
    assert record.section == "intro"  # also promoted to a first-class field


def test_a_record_with_pages_is_traceable():
    assert record_from(make_chunk(0)).is_traceable


def test_a_record_with_only_offsets_is_traceable():
    record = make_chunk(0, page_start=None, page_end=None)
    assert record_from(record).is_traceable


def test_a_record_with_only_a_section_is_traceable():
    record = make_chunk(
        0, page_start=None, page_end=None, metadata={"section": "intro"}
    )
    assert record_from(record).is_traceable


def test_a_record_pointing_at_a_document_but_nothing_in_it_is_not_traceable():
    """Saying *which* document without *where in it* is not a citation."""
    record = record_from(
        make_chunk(0, page_start=None, page_end=None, metadata={})
    )
    assert not record.is_traceable


def test_a_record_with_no_document_at_all_is_never_traceable():
    record = record_from(
        make_chunk(
            0,
            document_id=None,
            metadata={"section": "intro"},
        )
    )
    assert not record.is_traceable


def test_reading_an_unknown_field_by_name_says_so():
    with pytest.raises(PayloadMismatchError) as excinfo:
        record_from(make_chunk(0)).field("ttl")
    assert "ttl" in str(excinfo.value)
    assert "StoredChunkRecord" in str(excinfo.value)


def test_records_are_immutable():
    record = record_from(make_chunk(0))
    with pytest.raises(AttributeError):
        record.chunk_id = "other"  # type: ignore[misc]


def test_a_plain_chunk_without_a_vector_is_refused():
    with pytest.raises(PayloadMismatchError) as excinfo:
        record_from(object())
    assert "EmbeddedChunk" in str(excinfo.value)


def test_a_non_numeric_vector_is_refused():
    broken = core.EmbeddedChunk.model_construct(
        chunk=make_chunk(0).chunk, vector=("a", "b"), index_fingerprint="fp"
    )
    with pytest.raises(PayloadMismatchError) as excinfo:
        record_from(broken)
    assert "chunk 'c-0'" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# FR-47 -- refusing before writing
# --------------------------------------------------------------------------- #


def test_a_vector_of_the_wrong_width_is_refused_before_any_write():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2)
    with pytest.raises(PayloadMismatchError) as excinfo:
        writer.upsert(make_chunks(3, width=3))
    assert "3-dimensional" in str(excinfo.value)
    assert "2-dimensional" in str(excinfo.value)
    assert store.rows == {}  # nothing landed


def test_the_width_refusal_names_the_offending_chunk():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=4)
    with pytest.raises(PayloadMismatchError) as excinfo:
        writer.upsert([make_chunk(0, width=2)])
    assert excinfo.value.chunk_id == "c-0"
    assert excinfo.value.document_id == "doc-1"


def test_a_missing_required_field_is_refused_before_any_write():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2, required_fields=("section",))
    with pytest.raises(PayloadMismatchError) as excinfo:
        writer.upsert([make_chunk(0, metadata={})])
    assert "'section'" in str(excinfo.value)
    assert store.rows == {}


def test_a_blank_required_field_counts_as_missing():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2, required_fields=("section",))
    with pytest.raises(PayloadMismatchError):
        writer.upsert([make_chunk(0, metadata={"section": "   "})])
    assert store.rows == {}


def test_a_present_required_field_is_written_normally():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2, required_fields=("document_id",))
    report = writer.upsert(make_chunks(2))
    assert report.ok
    assert len(store.rows) == 2


def test_a_required_field_that_is_not_a_record_field_is_a_configuration_error():
    """Caught at construction, not halfway through a write."""
    with pytest.raises(IndexWriterError) as excinfo:
        IndexWriter(MemoryStore(), dimensions=2, required_fields=("ttl",))
    assert "ttl" in str(excinfo.value)
    assert "StoredChunkRecord" in str(excinfo.value)


def test_validate_payload_returns_records_without_writing():
    store = MemoryStore()
    records = validate_payload(make_chunks(3), dimensions=2)
    assert len(records) == 3
    assert store.rows == {}


@pytest.mark.parametrize("width", [0, -1, True, "2", 2.0])
def test_an_unusable_dimension_setting_is_refused(width):
    with pytest.raises(IndexWriterError):
        IndexWriter(MemoryStore(), dimensions=width)  # type: ignore[arg-type]


def test_a_store_with_no_add_method_is_refused():
    with pytest.raises(IndexWriterError) as excinfo:
        IndexWriter(object(), dimensions=2)
    assert "add" in str(excinfo.value)


def test_an_empty_payload_is_refused_by_the_shared_contract():
    with pytest.raises(StorageInputError):
        validate_payload([], dimensions=2)


def test_mixed_vector_widths_are_refused_by_the_shared_contract():
    with pytest.raises(StorageInputError):
        validate_payload([make_chunk(0, width=2), make_chunk(1, width=3)], dimensions=2)


# --------------------------------------------------------------------------- #
# Successful writes
# --------------------------------------------------------------------------- #


def test_a_clean_payload_is_written_in_bounded_batches():
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2, batch_size=3)
    report = writer.upsert(make_chunks(7))
    assert report.batches == 3
    assert report.batch_size == 3
    assert report.dimensions == 2
    assert report.total_items == 7
    assert len(store.rows) == 7


def test_the_report_records_every_item_in_input_order():
    store = MemoryStore()
    report = IndexWriter(store, dimensions=2, batch_size=2).upsert(make_chunks(5))
    assert [r.chunk_id for r in report.results] == [
        "c-0",
        "c-1",
        "c-2",
        "c-3",
        "c-4",
    ]


def test_every_batch_size_respects_the_bound():
    store = MemoryStore()
    for size in (1, 2, 3, 100):
        report = IndexWriter(store, dimensions=2, batch_size=size).upsert(
            make_chunks(7)
        )
        assert report.batches == -(-7 // size)


def test_raise_if_failed_is_a_no_op_on_a_clean_run():
    report = IndexWriter(MemoryStore(), dimensions=2).upsert(make_chunks(3))
    assert report.ok
    report.raise_if_failed()  # must not raise


def test_the_writer_accumulates_its_written_count_across_runs():
    writer = IndexWriter(MemoryStore(), dimensions=2, batch_size=2)
    writer.upsert(make_chunks(3))
    writer.upsert(make_chunks(4))
    assert writer.written == 7
    assert len(writer.reports) == 2


def test_upserting_the_same_ids_twice_replaces_rather_than_duplicating():
    """FR-46: re-issuing an identical write creates no duplicate record."""
    store = MemoryStore()
    writer = IndexWriter(store, dimensions=2)
    writer.upsert(make_chunks(4))
    writer.upsert(make_chunks(4))
    assert len(store.rows) == 4
    assert writer.written == 8  # both runs reported their items as written


def test_a_summary_names_the_shape_of_the_run():
    report = IndexWriter(MemoryStore(), dimensions=2, batch_size=3).upsert(
        make_chunks(7)
    )
    assert "7/7" in report.summary()
    assert "3 batch(es)" in report.summary()
    assert "2 dimensions" in report.summary()


# --------------------------------------------------------------------------- #
# FR-46 -- partial failure reported per item
# --------------------------------------------------------------------------- #


def test_a_rejected_batch_is_retried_item_by_item():
    store = PoisonStore(poison="c-4")
    report = IndexWriter(store, dimensions=2, batch_size=3).upsert(make_chunks(7))
    # 3 batch attempts, then 3 single-item retries for the failed batch, then
    # the final 1-item batch: [3, 3, 1, 1, 1, 1]
    assert store.calls == [3, 3, 1, 1, 1, 1]


def test_the_successes_of_a_failed_batch_are_not_lost():
    store = PoisonStore(poison="c-4")
    report = IndexWriter(store, dimensions=2, batch_size=3).upsert(make_chunks(7))
    assert report.failed == 1
    assert report.succeeded == 6
    assert len(store.rows) == 6  # physically present, not merely claimed
    assert "c-4" not in store.rows


def test_the_failure_is_reported_for_that_item_only():
    store = PoisonStore(poison="c-4")
    report = IndexWriter(store, dimensions=2, batch_size=3).upsert(make_chunks(7))
    failures = report.failures
    assert len(failures) == 1
    assert failures[0].chunk_id == "c-4"
    assert "RuntimeError" in failures[0].error
    assert failures[0].batch == 1


def test_a_partial_failure_makes_the_whole_report_not_ok():
    store = PoisonStore(poison="c-4")
    report = IndexWriter(store, dimensions=2, batch_size=3).upsert(make_chunks(7))
    assert report.ok is False
    assert report.failed == 1


def test_raise_if_failed_carries_the_report_and_the_successes():
    store = PoisonStore(poison="c-4")
    report = IndexWriter(store, dimensions=2, batch_size=3).upsert(make_chunks(7))
    with pytest.raises(BatchWriteError) as excinfo:
        report.raise_if_failed()
    assert excinfo.value.report is report
    assert "6 succeeded and remain written" in str(excinfo.value)


def test_a_backend_that_fails_every_item_after_the_first_still_reports_each():
    store = ExplodingStore()
    report = IndexWriter(store, dimensions=2, batch_size=2).upsert(make_chunks(6))
    # First batch lands whole; every later batch fails and is retried, and
    # each single-item retry fails too.
    assert report.batches == 3
    assert report.succeeded == 2
    assert report.failed == 4
    assert len(store.rows) == 2
    assert {r.chunk_id for r in report.failures} == {"c-2", "c-3", "c-4", "c-5"}


def test_a_item_result_describes_where_it_is_and_what_happened():
    ok = BatchItemResult(chunk_id="c-1", document_id="d", batch=0, ok=True)
    bad = BatchItemResult(chunk_id=None, document_id=None, batch=2, ok=False,
                          error="boom")
    assert ok.describe() == "[batch 0] d/c-1 written"
    assert "FAILED: boom" in bad.describe()
    assert bad.where == "unknown-document/?"


# --------------------------------------------------------------------------- #
# FR-47 -- creation is not destruction
# --------------------------------------------------------------------------- #


def test_creating_an_empty_index_is_allowed():
    lifecycle = IndexLifecycle(count=lambda name: 0)
    assert lifecycle.create("docs") == 0


def test_creating_over_stored_rows_is_refused():
    rows = {"docs": 5}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    with pytest.raises(DestructiveOperationRefused) as excinfo:
        lifecycle.create("docs")
    assert "5" in str(excinfo.value)
    assert "replace(" in str(excinfo.value)


def test_a_refusal_runs_no_initialisation():
    called: list[str] = []
    rows = {"docs": 3}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    with pytest.raises(DestructiveOperationRefused):
        lifecycle.create("docs", init=lambda name: called.append(name))
    assert called == []


def test_the_initialiser_runs_when_the_index_is_genuinely_empty():
    called: list[str] = []
    lifecycle = IndexLifecycle(count=lambda name: 0)
    lifecycle.create("docs", init=lambda name: called.append(name))
    assert called == ["docs"]


def test_an_explicit_replace_records_what_it_removed():
    rows = {"docs": 7}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    record = lifecycle.replace(
        "docs",
        reason="switched to v3 schema",
        clear=lambda name: rows.__setitem__(name, 0) or 7,
    )
    assert isinstance(record, IndexDestruction)
    assert record.rows_removed == 7
    assert record.reason == "switched to v3 schema"
    assert rows["docs"] == 0
    assert lifecycle.destructions == (record,)
    assert lifecycle.last_destruction is record


def test_a_replace_without_a_reason_is_refused():
    """A destructive operation with no stated reason cannot be reviewed."""
    lifecycle = IndexLifecycle(count=lambda name: 0)
    with pytest.raises(IndexWriterError) as excinfo:
        lifecycle.replace("docs", reason="   ", clear=lambda name: 0)
    assert "reason" in str(excinfo.value)


def test_a_replace_without_a_callable_clear_is_refused():
    lifecycle = IndexLifecycle(count=lambda name: 0)
    with pytest.raises(IndexWriterError):
        lifecycle.replace("docs", reason="why", clear="wipe")  # type: ignore[arg-type]


def test_a_count_that_is_not_callable_is_refused_at_construction():
    with pytest.raises(IndexWriterError):
        IndexLifecycle(count=5)  # type: ignore[arg-type]


def test_no_destruction_has_been_recorded_on_a_fresh_lifecycle():
    lifecycle = IndexLifecycle(count=lambda name: 0)
    assert lifecycle.destructions == ()
    assert lifecycle.last_destruction is None


def test_row_counts_are_read_through_the_injected_callable():
    rows = {"a": 2, "b": 0}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    assert lifecycle.rows("a") == 2
    assert lifecycle.rows("b") == 0
    assert lifecycle.rows("absent") == 0


def test_destruction_records_describe_themselves():
    record = IndexDestruction(index="docs", rows_removed=4, reason="schema v3")
    assert "4 row(s)" in record.describe()
    assert "schema v3" in record.describe()


# --------------------------------------------------------------------------- #
# Values are values; dependencies stay declared
# --------------------------------------------------------------------------- #


def test_batch_reports_are_immutable():
    report = BatchReport(results=(), batch_size=1, batches=0, dimensions=2)
    with pytest.raises(AttributeError):
        report.batch_size = 9  # type: ignore[misc]


def test_the_component_imports_no_third_party_packages():
    """core-types is a declared dependency; everything else must be stdlib."""
    source = (COMPONENT / "index_writer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    stdlib = set(sys.stdlib_module_names)
    third_party = roots - stdlib - {"core_types", "vector_store_protocol",
                                    "index_writer"}
    assert third_party == set(), third_party
    assert "rag_code_library" not in imported


def test_the_manifest_declares_the_two_components_this_one_imports():
    import json

    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["dependencies"]) == {"core-types", "vector-store-protocol"}
    assert manifest["category"] == "storage"
    assert manifest["required_packages"] == []


def test_no_write_happens_when_pre_flight_fails():
    """The property FR-47 turns on, stated as row counts rather than a claim."""
    store = MemoryStore()
    writer = IndexWriter(
        store, dimensions=2, batch_size=2, required_fields=("section",)
    )

    # (a) a width the index cannot hold
    with pytest.raises(PayloadMismatchError):
        writer.upsert(make_chunks(3, width=5))
    assert store.rows == {}

    # (b) a required field the chunk never recorded
    with pytest.raises(PayloadMismatchError):
        writer.upsert([make_chunk(0, metadata={})])
    assert store.rows == {}

    # (c) an unusable batch size, caught at construction rather than mid-write
    with pytest.raises(IndexWriterError):
        IndexWriter(store, dimensions=2, batch_size=0)
    assert store.rows == {}

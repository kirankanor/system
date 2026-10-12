"""Gate tests for the storage area (T-08).

Covers the three remaining storage components: the vector-store contract, the
document-store contract, and two opt-in backend adapters. ``core-types`` is
covered separately by T-04 and is only exercised here as a dependency.

Neither ``chromadb``, ``faiss`` nor ``numpy`` is installed in this checkout,
and installing them would modify dependency files -- which this library
forbids (FR-2). Both adapters are therefore exercised against injected
backends that speak the same shape as the real ones, and the FR-3 guards are
proven by blocking the import so the assertion is about the guard rather than
about the machine running the tests.

The conformance point ``core-types`` calls out as unenforceable by the type --
that results must name their backend in ``source_name`` -- is asserted here,
for both adapters.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "components" / "storage"

# Every storage component directory, each imported the way a copy of it would
# be. ``adapters`` and ``index_writer`` last: they import the others.
for name in (
    "core_types",
    "vector_store_protocol",
    "document_store",
    "adapters",
    "index_writer",
):
    sys.path.insert(0, str(STORAGE / name))

# The AC-26 chain test runs chunking and deduplication as real stages, so
# those two components are reachable too. ``fixed_size`` defines a module
# named ``chunk``; importing it under an alias keeps it from colliding with
# anything local to this file.
CHUNKING = ROOT / "components" / "chunking"
for name in ("fixed_size", "chunk_deduplicator"):
    sys.path.insert(0, str(CHUNKING / name))

import core_types as core  # noqa: E402
import document_store as docs  # noqa: E402
import storage_adapters as adapters  # noqa: E402
import vector_store_protocol as protocol  # noqa: E402
from chunk_deduplicator import Deduplicator, Scope  # noqa: E402
from fixed_size_chunker import chunk as fixed_size_chunk  # noqa: E402
from index_writer import (  # noqa: E402
    PROVENANCE_FIELDS,
    DestructiveOperationRefused,
    IndexLifecycle,
    IndexWriter,
    IndexWriterError,
    PayloadMismatchError,
    record_from,
)

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures and test doubles
# --------------------------------------------------------------------------- #


@pytest.fixture
def embedded():
    """Two embedded chunks in one vector space, with identity fields set."""
    return [
        core.EmbeddedChunk(
            chunk=core.Chunk(
                chunk_index=0,
                text="alpha text",
                chunk_id="c0",
                document_id="d1",
                source_name="a.txt",
            ),
            vector=(1.0, 2.0),
            index_fingerprint="fp1",
        ),
        core.EmbeddedChunk(
            chunk=core.Chunk(
                chunk_index=1,
                text="beta text",
                chunk_id="c1",
                document_id="d1",
                source_name="a.txt",
            ),
            vector=(3.0, 4.0),
            index_fingerprint="fp1",
        ),
    ]


class FakeCollection:
    """Speaks the subset of chromadb's collection API this adapter uses."""

    def __init__(self):
        self.rows: dict[str, dict[str, Any]] = {}

    def add(self, ids, embeddings, documents, metadatas):
        for row_id, vector, document, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            self.rows[row_id] = {
                "vector": list(vector),
                "document": document,
                "metadata": dict(metadata),
            }

    def query(self, query_embeddings, n_results, where=None, include=None):
        needle = query_embeddings[0]
        scored = []
        for row_id, row in self.rows.items():
            if where and any(
                row["metadata"].get(k) != v for k, v in where.items()
            ):
                continue
            distance = sum((a - b) ** 2 for a, b in zip(needle, row["vector"]))
            scored.append((distance, row_id))
        scored.sort()
        top = scored[:n_results]
        return {
            "ids": [[row_id for _, row_id in top]],
            "documents": [[self.rows[r]["document"] for _, r in top]],
            "metadatas": [[self.rows[r]["metadata"] for _, r in top]],
            "distances": [[distance for distance, _ in top]],
        }

    def delete(self, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def count(self):
        return len(self.rows)


class FakeChromaClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name, metadata=None):
        return self.collections.setdefault(name, FakeCollection())


def _chroma(embedded_items=None, **kwargs):
    store = adapters.ChromaVectorStore("docs", client=FakeChromaClient(), **kwargs)
    if embedded_items is not None:
        store.add(embedded_items)
    return store


def _client_for(collection):
    """Wrap a collection so it can be injected where a client is expected."""

    class _Client:
        def get_or_create_collection(self, name, metadata=None):
            return collection

    return _Client()


class FakeFaissIndex:
    """A brute-force index with the same surface as faiss's IndexIDMap."""

    ntotal = 0

    def __init__(self):
        self.rows: dict[int, list[float]] = {}
        self._numpy = __import__("numpy")

    def add_with_ids(self, matrix, labels):
        for vector, label in zip(matrix, labels):
            self.rows[int(label)] = [float(x) for x in vector]
        self.ntotal = len(self.rows)

    def search(self, matrix, k):
        needle = list(matrix[0])
        scored = sorted(
            (sum((a - b) ** 2 for a, b in zip(needle, vector)), label)
            for label, vector in self.rows.items()
        )[:k]
        return (
            self._numpy.array([[d for d, _ in scored]]),
            self._numpy.array([[label for _, label in scored]]),
        )

    def remove_ids(self, labels):
        for label in labels:
            self.rows.pop(int(label), None)
        self.ntotal = len(self.rows)


def _faiss(embedded_items=None, *, dimensions=2):
    pytest.importorskip(
        "numpy",
        reason="the faiss adapter is exercised against a fake index, but that "
        "fake still needs numpy to shape its results",
    )
    store = adapters.FaissVectorStore(dimensions=dimensions, index=FakeFaissIndex())
    if embedded_items is not None:
        store.add(embedded_items)
    return store


# --------------------------------------------------------------------------- #
# Shared input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module,label", [(protocol, "vector-store"), (docs, "document-store")])
@pytest.mark.parametrize(
    "bad", [{"a": {"b": 1}}, {"a": [1, 2]}, {"a": (1,)}, {"a": {1, 2}}], ids=["map", "list", "tuple", "set"]
)
def test_nested_filter_values_are_rejected_not_dropped(module, label, bad):
    """Dropping a nested value narrows the result set invisibly."""
    with pytest.raises((protocol.StorageError, docs.DocumentStoreError)) as exc:
        module.validate_filter(bad)
    assert "flat" in str(exc.value)


@pytest.mark.parametrize("module", [protocol, docs])
@pytest.mark.parametrize(
    "bad", ["str", [1, 2], 3, object()], ids=["str", "list", "int", "obj"]
)
def test_a_non_mapping_filter_is_rejected(module, bad):
    with pytest.raises((protocol.StorageError, docs.DocumentStoreError)):
        module.validate_filter(bad)


def test_an_empty_filter_key_is_rejected():
    with pytest.raises(protocol.StorageInputError):
        protocol.validate_filter({"": 1})


def test_none_filter_is_accepted_by_both_contracts():
    assert protocol.validate_filter(None) is None
    assert docs.validate_filter(None) is None


@pytest.mark.parametrize("module", [protocol, docs])
def test_a_bare_string_of_ids_is_rejected(module):
    """A str *is* a sequence of characters, so 'abc' would mean ['a','b','c']."""
    with pytest.raises((protocol.StorageError, docs.DocumentStoreError)) as exc:
        module.validate_ids("abc")
    assert "character" in str(exc.value)


@pytest.mark.parametrize("module", [protocol, docs])
def test_an_empty_id_is_rejected(module):
    with pytest.raises((protocol.StorageError, docs.DocumentStoreError)) as exc:
        module.validate_ids(["ok", ""])
    assert "[1]" in str(exc.value)


@pytest.mark.parametrize("module", [protocol, docs])
def test_ids_come_back_as_a_list(module):
    assert module.validate_ids(("a", "b")) == ["a", "b"]


def test_a_vector_must_be_non_empty():
    with pytest.raises(protocol.StorageInputError) as exc:
        protocol.validate_vector([])
    assert "at least one" in str(exc.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_vector_value_is_rejected(value):
    """NaN and inf compare against nothing and poison a search silently."""
    with pytest.raises(protocol.StorageInputError) as exc:
        protocol.validate_vector([1.0, value])
    assert "NaN" in str(exc.value) or "infinity" in str(exc.value)


def test_a_numeric_vector_is_normalised_to_floats():
    assert protocol.validate_vector([1, 2.5]) == [1.0, 2.5]


@pytest.mark.parametrize("bad", ["abc", b"abc", [True, 1.0], ["a", "b"], {1, 2}])
def test_an_unusable_vector_is_rejected(bad):
    with pytest.raises(protocol.StorageInputError):
        protocol.validate_vector(bad)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_an_unusable_k_is_rejected(k):
    with pytest.raises(protocol.StorageInputError):
        protocol.validate_k(k)


def test_k_is_accepted_at_one():
    assert protocol.validate_k(1) == 1


def test_mixed_vector_widths_in_one_add_are_rejected(embedded):
    """One call must add one vector space, or the index has no geometry."""
    mixed = embedded + [
        core.EmbeddedChunk(
            chunk=core.Chunk(chunk_index=2, text="wide", chunk_id="c2"),
            vector=(1.0, 2.0, 3.0),
            index_fingerprint="fp1",
        )
    ]
    with pytest.raises(protocol.StorageInputError) as exc:
        protocol.validate_embedded_chunks(mixed)
    assert "3-wide" in str(exc.value) and "0 has 2" in str(exc.value)


def test_an_empty_add_is_rejected(embedded):
    with pytest.raises(protocol.StorageInputError):
        protocol.validate_embedded_chunks([])


# --------------------------------------------------------------------------- #
# FR-3: optional backends are guarded, with the manifest's message
# --------------------------------------------------------------------------- #


def test_the_adapters_guard_chromadb_with_the_manifest_message(monkeypatch):
    manifest = json.loads(
        (STORAGE / "adapters" / "manifest.json").read_text(encoding="utf-8")
    )
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError) as exc:
        adapters.ChromaVectorStore("docs").count()
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == adapters.MISSING_DEPENDENCY_MESSAGE


def test_the_adapters_guard_faiss_with_the_manifest_message(monkeypatch):
    manifest = json.loads(
        (STORAGE / "adapters" / "manifest.json").read_text(encoding="utf-8")
    )
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ImportError) as exc:
        adapters.FaissVectorStore(dimensions=2).count()
    assert str(exc.value) == manifest["missing_dependency_message"]


def test_the_missing_dependency_error_is_an_import_error():
    assert issubclass(adapters.MissingStorageDependency, ImportError)
    assert not issubclass(adapters.MissingStorageDependency, protocol.StorageError)


def test_importing_the_adapters_needs_no_backend():
    """The module must import on a machine that installed neither package."""
    assert hasattr(adapters, "ChromaVectorStore")
    assert hasattr(adapters, "FaissVectorStore")


def test_the_manifest_lists_both_backends_as_optional():
    manifest = json.loads(
        (STORAGE / "adapters" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["required_packages"] == []
    assert {entry["name"] for entry in manifest["optional_packages"]} == {
        "chromadb",
        "faiss",
    }
    assert set(manifest["dependencies"]) == {"core-types", "vector-store-protocol"}


# --------------------------------------------------------------------------- #
# Both adapters satisfy the protocol, and name themselves in results
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_both_adapters_satisfy_the_vector_store_protocol(factory):
    assert isinstance(factory(), protocol.VectorStoreProtocol)


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_query_results_name_the_backend_that_produced_them(factory, embedded):
    """The conformance point core-types calls unenforceable by the type."""
    store = factory(embedded)
    results = store.query([1.0, 2.0], k=2)
    assert results
    assert all(r.source_name == store.name for r in results)
    assert all(isinstance(r, core.ScoredDocument) for r in results)


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_query_returns_nearest_first(factory, embedded):
    store = factory(embedded)
    results = store.query([1.0, 2.0], k=2)
    assert [r.chunk_id for r in results] == ["c0", "c1"]
    assert results[0].score < results[1].score


# --------------------------------------------------------------------------- #
# ChromaVectorStore
# --------------------------------------------------------------------------- #


def test_chroma_add_then_count(embedded):
    store = _chroma()
    assert store.add(embedded) == 2
    assert store.count() == 2


def test_chroma_identity_fields_survive_the_round_trip(embedded):
    """Stored under reserved keys, restored as plain fields on the result."""
    store = _chroma(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.chunk_id == "c0"
    assert hit.document_id == "d1"
    assert hit.source_name == "docs"
    assert hit.text == "alpha text"


def test_chroma_user_metadata_is_passed_through_untouched(embedded):
    embedded[0].chunk.metadata["section"] = "intro"
    store = _chroma(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.metadata == {"section": "intro"}
    assert not any(k.startswith(adapters.RESERVED_PREFIX) for k in hit.metadata)


def test_filtering_on_an_identity_field_matches(embedded):
    """The caller sees `document_id`, so the adapter rewrites it internally."""
    store = _chroma(embedded)
    assert [r.chunk_id for r in store.query([1.0, 2.0], k=2, filter={"document_id": "d1"})] == ["c0", "c1"]
    assert store.query([1.0, 2.0], k=2, filter={"document_id": "other"}) == []


def test_filtering_on_user_metadata_matches(embedded):
    embedded[0].chunk.metadata["section"] = "intro"
    embedded[1].chunk.metadata["section"] = "body"
    store = _chroma(embedded)
    assert [r.chunk_id for r in store.query([1.0, 2.0], k=2, filter={"section": "body"})] == ["c1"]


def test_a_filter_match_is_not_invented(embedded):
    store = _chroma(embedded)
    assert store.query([1.0, 2.0], k=2, filter={"section": "absent"}) == []


def test_chroma_delete_then_count(embedded):
    store = _chroma(embedded)
    assert store.delete(["c0"]) == 1
    assert store.count() == 1
    assert [r.chunk_id for r in store.query([1.0, 2.0], k=2)] == ["c1"]


def test_deleting_an_unknown_id_is_a_no_op_not_an_error(embedded):
    """Safe to call after a failed re-ingest, which is when it is needed."""
    store = _chroma(embedded)
    assert store.delete(["never-existed"]) == 1
    assert store.count() == 2


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_collection_name_is_rejected(name):
    with pytest.raises(protocol.StorageInputError):
        adapters.ChromaVectorStore(name, client=FakeChromaClient())


def test_a_backend_failure_is_wrapped_and_chained(embedded):
    class Broken(FakeCollection):
        def add(self, **kwargs):
            raise RuntimeError("disk full")

    store = adapters.ChromaVectorStore("docs", client=_client_for(Broken()))
    with pytest.raises(protocol.StorageBackendError) as exc:
        store.add(embedded)
    assert "disk full" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_malformed_backend_response_is_rejected():
    class Nonsense(FakeCollection):
        def query(self, **kwargs):
            return "not a mapping"

    store = adapters.ChromaVectorStore("docs", client=_client_for(Nonsense()))
    with pytest.raises(protocol.StorageBackendError) as exc:
        store.query([1.0, 2.0], k=1)
    assert "mapping" in str(exc.value)


# --------------------------------------------------------------------------- #
# FaissVectorStore
# --------------------------------------------------------------------------- #


def test_faiss_add_then_count(embedded):
    store = _faiss()
    assert store.add(embedded) == 2
    assert store.count() == 2


def test_faiss_identity_fields_survive_the_round_trip(embedded):
    store = _faiss(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.chunk_id == "c0"
    assert hit.document_id == "d1"
    assert hit.source_name == store.name


def test_faiss_user_metadata_is_passed_through_untouched(embedded):
    embedded[0].chunk.metadata["section"] = "intro"
    store = _faiss(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.metadata == {"section": "intro"}
    assert not any(k.startswith(adapters.RESERVED_PREFIX) for k in hit.metadata)


def test_faiss_delete_keeps_the_id_map_consistent(embedded):
    """Deleting row 0 must not leave later ids addressing the wrong rows."""
    store = _faiss(embedded)
    assert store.delete(["c0"]) == 1
    assert store.count() == 1
    remaining = store.query([1.0, 2.0], k=2)
    assert [r.chunk_id for r in remaining] == ["c1"]
    assert remaining[0].text == "beta text"


def test_deleting_everything_leaves_an_empty_store(embedded):
    store = _faiss(embedded)
    store.delete(["c0", "c1"])
    assert store.count() == 0
    assert store.query([1.0, 2.0], k=2) == []


def test_the_faiss_adapter_refuses_a_filter_rather_than_ignoring_it():
    """A raw faiss index holds no metadata, so filtering cannot be honoured."""
    store = _faiss()
    with pytest.raises(protocol.StorageInputError) as exc:
        store.query([1.0, 2.0], k=1, filter={"document_id": "d1"})
    assert "does not support" in str(exc.value)


def test_faiss_rejects_a_vector_of_the_wrong_width(embedded):
    store = _faiss(dimensions=3)
    with pytest.raises(protocol.StorageInputError) as exc:
        store.add(embedded)
    assert "3-wide" in str(exc.value)


def test_faiss_rejects_a_query_vector_of_the_wrong_width():
    store = _faiss(dimensions=3)
    with pytest.raises(protocol.StorageInputError) as exc:
        store.query([1.0, 2.0], k=1)
    assert "2 wide" in str(exc.value)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5, "3"])
def test_an_unusable_dimensions_setting_is_rejected(dimensions):
    with pytest.raises(protocol.StorageInputError):
        adapters.FaissVectorStore(dimensions=dimensions)


# --------------------------------------------------------------------------- #
# Document store contract
# --------------------------------------------------------------------------- #


def test_a_valid_document_passes_validation():
    doc = core.SourceDocument(
        document_id="d1", source_name="a.txt", sha256="a" * 64, page_count=1
    )
    assert docs.validate_source_document(doc) is doc


def test_a_non_document_is_rejected_by_type():
    with pytest.raises(docs.DocumentStoreInputError) as exc:
        docs.validate_source_document(object())
    assert "SourceDocument" in str(exc.value)


def test_the_document_store_errors_form_a_hierarchy():
    assert issubclass(docs.DocumentStoreInputError, docs.DocumentStoreError)
    assert issubclass(docs.DocumentStoreBackendError, docs.DocumentStoreError)
    assert not issubclass(docs.DocumentStoreError, ValueError)


def test_the_storage_errors_form_a_hierarchy():
    assert issubclass(protocol.StorageInputError, protocol.StorageError)
    assert issubclass(protocol.StorageBackendError, protocol.StorageError)
    assert not issubclass(protocol.StorageError, ValueError)


def test_the_vector_store_protocol_is_re_exported_for_implementers():
    """A store can be written against this component alone."""
    assert protocol.VectorStoreProtocol is core.VectorStoreProtocol
    assert docs.DocumentStoreProtocol is core.DocumentStoreProtocol


# --------------------------------------------------------------------------- #
# Message provenance
# --------------------------------------------------------------------------- #


def test_validation_messages_name_their_component():
    with pytest.raises(protocol.StorageInputError) as exc:
        protocol.validate_vector([])
    assert "vector-store-protocol" in str(exc.value)

    with pytest.raises(docs.DocumentStoreInputError) as exc:
        docs.validate_ids("abc")
    assert "document-store" in str(exc.value)

    with pytest.raises(protocol.StorageInputError) as exc:
        adapters.FaissVectorStore(dimensions=0)
    assert "storage-adapters" in str(exc.value)


def test_non_finite_is_not_silently_accepted():
    """The property that matters: no NaN ever reaches an index."""
    poisoned = core.EmbeddedChunk(
        chunk=core.Chunk(chunk_index=0, text="bad", chunk_id="c0"),
        vector=(float("nan"), 1.0),
        index_fingerprint="fp1",
    )
    with pytest.raises(protocol.StorageInputError):
        _chroma([poisoned])


# =========================================================================== #
# V2 FR-45 -- a document store that actually implements its own contract
# =========================================================================== #


def _doc(
    document_id: str = "d1",
    *,
    source_name: str = "a.txt",
    page_count: int = 1,
    metadata: dict | None = None,
) -> core.SourceDocument:
    return core.SourceDocument(
        document_id=document_id,
        source_name=source_name,
        sha256="a" * 64,
        page_count=page_count,
        metadata=metadata or {},
    )


@pytest.fixture
def doc_store():
    return docs.InMemoryDocumentStore()


def test_a_stored_document_round_trips(doc_store):
    doc = _doc()
    assert doc_store.put(doc) == "d1"
    assert doc_store.get("d1") == doc
    assert doc_store.count() == 1
    assert doc_store.list() == [doc]


def test_getting_an_unknown_document_returns_none_rather_than_raising(
    doc_store,
):
    """A chunk referencing a document that was never ingested is expected."""
    assert doc_store.get("never-ingested") is None


def test_the_backend_satisfies_its_own_protocol():
    """Nothing satisfied this before FR-45; the claim is now checkable."""
    store = docs.InMemoryDocumentStore()
    assert isinstance(store, core.DocumentStoreProtocol)
    assert isinstance(store, docs.DocumentStoreProtocol)


def test_the_backend_implements_the_two_methods_the_protocol_lacks():
    """FR-45 names put, get, delete *and* count."""
    store = docs.InMemoryDocumentStore()
    for name in ("put", "get", "delete", "count", "list"):
        assert callable(getattr(store, name)), name


def test_delete_removes_only_what_was_actually_there(doc_store):
    doc_store.put(_doc("d1"))
    doc_store.put(_doc("d2"))
    assert doc_store.delete(["d1", "never-existed"]) == 1
    assert doc_store.count() == 1
    assert doc_store.get("d1") is None
    assert doc_store.get("d2") is not None


def test_rewriting_the_same_id_replaces_rather_than_duplicating(doc_store):
    """FR-46: re-issuing an identical write creates no duplicate record."""
    doc_store.put(_doc(page_count=1))
    doc_store.put(_doc(page_count=9))
    assert doc_store.count() == 1
    assert doc_store.get("d1").page_count == 9  # the new value, not the old
    assert doc_store.put_calls == 2
    assert doc_store.replacements == 1


def test_a_genuinely_new_document_is_counted_as_such(doc_store):
    doc_store.put(_doc("d1"))
    doc_store.put(_doc("d2"))
    assert doc_store.count() == 2
    assert doc_store.replacements == 0


def test_reingesting_a_whole_corpus_is_idempotent():
    store = docs.InMemoryDocumentStore()
    for _ in range(3):
        for index in range(5):
            store.put(_doc(f"doc-{index}"))
    assert store.count() == 5
    assert store.replacements == 10  # two rounds of five replaced


def test_listing_returns_every_document_in_insertion_order(doc_store):
    for index in range(3):
        doc_store.put(_doc(f"doc-{index}"))
    assert [d.document_id for d in doc_store.list()] == ["doc-0", "doc-1", "doc-2"]


def test_listing_can_filter_on_a_top_level_field(doc_store):
    doc_store.put(_doc("d1", source_name="a.txt"))
    doc_store.put(_doc("d2", source_name="b.txt"))
    assert [d.document_id for d in doc_store.list({"source_name": "b.txt"})] == ["d2"]
    assert [d.document_id for d in doc_store.list({"page_count": 1})] == ["d1", "d2"]


def test_listing_can_filter_on_metadata(doc_store):
    doc_store.put(_doc("d1", metadata={"lang": "en"}))
    doc_store.put(_doc("d2", metadata={"lang": "fr"}))
    assert [d.document_id for d in doc_store.list({"lang": "fr"})] == ["d2"]


def test_a_filter_matching_nothing_returns_empty_not_an_error(doc_store):
    doc_store.put(_doc("d1"))
    assert doc_store.list({"document_id": "other"}) == []


def test_a_nested_filter_value_is_refused_not_dropped(doc_store):
    with pytest.raises(docs.DocumentStoreInputError):
        doc_store.list({"a": {"b": 1}})


def test_writing_something_that_is_not_a_document_is_refused(doc_store):
    with pytest.raises(docs.DocumentStoreInputError) as exc:
        doc_store.put(object())
    assert "SourceDocument" in str(exc.value)
    assert doc_store.count() == 0


def test_reading_with_a_non_string_id_is_refused(doc_store):
    with pytest.raises(docs.DocumentStoreInputError):
        doc_store.get(None)  # type: ignore[arg-type]


def test_deleting_a_bare_string_is_refused(doc_store):
    """A str is itself a sequence of characters, which is the classic trap."""
    with pytest.raises(docs.DocumentStoreInputError) as exc:
        doc_store.delete("d1")
    assert "Wrap it" in str(exc.value)


# =========================================================================== #
# V2 FR-47 -- refuse before writing; creation is not destruction
# =========================================================================== #


def test_a_vector_the_index_cannot_hold_is_refused_before_the_first_row():
    store = _chroma()
    writer = IndexWriter(store, dimensions=2, batch_size=3)
    wide = core.EmbeddedChunk(
        chunk=core.Chunk(chunk_index=0, text="wide", chunk_id="c0"),
        vector=(1.0, 2.0, 3.0),
        index_fingerprint="fp",
    )
    with pytest.raises(PayloadMismatchError) as exc:
        writer.upsert([wide])
    assert "3-dimensional" in str(exc.value)
    assert store.count() == 0  # not one row landed


def test_a_required_field_the_index_declares_is_checked_before_writing(embedded):
    store = _chroma()
    writer = IndexWriter(store, dimensions=2, required_fields=("section",))
    with pytest.raises(PayloadMismatchError) as exc:
        writer.upsert(embedded[:1])
    assert "'section'" in str(exc.value)
    assert store.count() == 0


def test_a_batch_that_fails_part_way_through_keeps_its_successes(embedded):
    """FR-46, asserted on rows actually in the store."""
    rows: dict[str, Any] = {}

    class HalfBroken:
        def add(self, items):
            if any(item.chunk.chunk_id == "c1" for item in items):
                raise RuntimeError("backend refused this batch")
            for item in items:
                rows[item.chunk.chunk_id] = item
            return len(items)

    report = IndexWriter(HalfBroken(), dimensions=2, batch_size=2).upsert(embedded)
    assert report.total_items == 2
    assert report.succeeded == 1
    assert report.failed == 1
    assert report.ok is False
    assert set(rows) == {"c0"}  # the good one is physically there
    assert report.failures[0].chunk_id == "c1"
    assert report.failures[0].error.startswith("RuntimeError")


def test_reports_are_carried_by_the_writer_across_runs(embedded):
    writer = IndexWriter(_chroma(), dimensions=2, batch_size=1)
    writer.upsert(embedded)
    assert len(writer.reports) == 1
    assert writer.written == 2


def test_creating_an_index_over_stored_rows_is_refused():
    rows = {"docs": 4}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    with pytest.raises(DestructiveOperationRefused) as exc:
        lifecycle.create("docs")
    assert "4" in str(exc.value)
    assert "replace(" in str(exc.value)


def test_creating_an_empty_index_is_allowed_and_records_nothing():
    lifecycle = IndexLifecycle(count=lambda name: 0)
    assert lifecycle.create("docs") == 0
    assert lifecycle.destructions == ()


def test_the_only_destructive_path_demands_a_reason_and_records_the_loss():
    rows = {"docs": 6}
    lifecycle = IndexLifecycle(count=lambda name: rows.get(name, 0))
    with pytest.raises(IndexWriterError):
        lifecycle.replace("docs", reason="  ", clear=lambda name: 0)
    assert lifecycle.destructions == ()  # nothing was destroyed

    record = lifecycle.replace(
        "docs", reason="v2 schema", clear=lambda name: rows.__setitem__(name, 0) or 6
    )
    assert record.rows_removed == 6
    assert lifecycle.last_destruction is record
    assert rows["docs"] == 0


# =========================================================================== #
# V2 FR-48 / AC-26 -- stored records carry their provenance
# =========================================================================== #


def _cited_chunk(index: int = 0) -> core.EmbeddedChunk:
    """A chunk carrying every provenance field AC-26 names."""
    return core.EmbeddedChunk(
        chunk=core.Chunk(
            chunk_index=index,
            text="body of the chunk",
            chunk_id=f"c{index}",
            document_id="d1",
            source_name="guide.md",
            page_start=2,
            page_end=3,
            metadata={
                "start_offset": 100 + index,
                "end_offset": 180 + index,
                "section": "setup",
            },
        ),
        vector=(1.0, 2.0),
        index_fingerprint="model=m|dims=2|chunking=abc",
    )


def test_a_stored_row_carries_every_ac26_field():
    """AC-26, asserted against what is actually in the backend."""
    store = _chroma([_cited_chunk()])
    row = next(iter(store.collection.rows.values()))
    metadata = row["metadata"]
    reserved = adapters.RESERVED_PREFIX

    for field in (
        "document_id",
        "chunk_id",
        "source_name",
        "chunk_index",
        "page_start",
        "page_end",
    ):
        assert f"{reserved}{field}" in metadata, field
    for field in ("start_offset", "end_offset", "section"):
        assert field in metadata, field
    assert f"{reserved}{adapters.FINGERPRINT_FIELD}" in metadata


def test_the_stored_values_are_the_ones_the_chunk_came_in_with():
    store = _chroma([_cited_chunk()])
    metadata = next(iter(store.collection.rows.values()))["metadata"]
    reserved = adapters.RESERVED_PREFIX
    assert metadata[f"{reserved}document_id"] == "d1"
    assert metadata[f"{reserved}chunk_id"] == "c0"
    assert metadata[f"{reserved}source_name"] == "guide.md"
    assert metadata[f"{reserved}page_start"] == 2
    assert metadata[f"{reserved}page_end"] == 3
    assert metadata["start_offset"] == 100
    assert metadata["end_offset"] == 180
    assert metadata["section"] == "setup"
    assert metadata[f"{reserved}{adapters.FINGERPRINT_FIELD}"] == (
        "model=m|dims=2|chunking=abc"
    )


def test_the_page_range_comes_back_out_as_a_real_field():
    """A caller should not have to dig a citation location out of metadata."""
    store = _chroma([_cited_chunk()])
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.page_start == 2
    assert hit.page_end == 3
    assert hit.document_id == "d1"
    assert hit.chunk_id == "c0"
    assert hit.metadata["section"] == "setup"
    # reserved keys stay out of caller-visible metadata
    assert not any(k.startswith(adapters.RESERVED_PREFIX) for k in hit.metadata)


def test_results_still_name_the_backend_while_the_row_keeps_the_file():
    """Two different facts, deliberately kept in two different places."""
    store = _chroma([_cited_chunk()])
    row = next(iter(store.collection.rows.values()))
    assert row["metadata"][f"{adapters.RESERVED_PREFIX}source_name"] == "guide.md"
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.source_name == "docs"  # core-types' unenforceable conformance


def test_pages_absent_from_the_source_stay_absent_rather_than_invented():
    plain = core.EmbeddedChunk(
        chunk=core.Chunk(chunk_index=0, text="no pages here", chunk_id="c0"),
        vector=(1.0, 2.0),
        index_fingerprint="fp",
    )
    store = _chroma([plain])
    metadata = next(iter(store.collection.rows.values()))["metadata"]
    reserved = adapters.RESERVED_PREFIX
    assert f"{reserved}page_start" not in metadata
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.page_start is None
    assert hit.page_end is None


def test_a_citation_field_can_be_filtered_on():
    """Stored under a reserved key, so the filter must be rewritten too."""
    store = _chroma([_cited_chunk(), _cited_chunk(1)])
    hits = store.query([1.0, 2.0], k=2, filter={"page_start": 2})
    assert {hit.chunk_id for hit in hits} == {"c0", "c1"}
    assert store.query([1.0, 2.0], k=2, filter={"page_start": 99}) == []


def test_the_record_builder_agrees_with_what_the_adapter_stored():
    """Two components, one description of where a chunk came from."""
    item = _cited_chunk()
    record = record_from(item)
    reserved = adapters.RESERVED_PREFIX
    stored = next(iter(_chroma([item]).collection.rows.values()))["metadata"]
    assert record.document_id == stored[f"{reserved}document_id"]
    assert record.chunk_id == stored[f"{reserved}chunk_id"]
    assert record.source_name == stored[f"{reserved}source_name"]
    assert record.page_start == stored[f"{reserved}page_start"]
    assert record.page_end == stored[f"{reserved}page_end"]
    assert record.index_fingerprint == stored[f"{reserved}index_fingerprint"]
    assert record.section == stored["section"]
    assert set(record.provenance) == set(PROVENANCE_FIELDS)


def _embed(text: str) -> tuple[float, ...]:
    """A deterministic stand-in for an embedding model.

    No embedding model is installed, and installing one would modify
    dependency files (FR-2). The *stage* is real -- chunks become vectors and
    those vectors are what gets indexed -- only the learned weights are not.
    """
    return (
        float(len(text) % 97),
        float(sum(text.encode("utf-8")) % 89),
        float(text.count(" ") % 13),
        1.0,
    )


def test_ac26_every_field_survives_chunking_dedup_embedding_and_indexing():
    """AC-26's end-to-end half: each stage is a real component, in order."""
    body = (
        "Introduction. This guide explains the setup. "
        "Install the dependency with pip, then restart the kernel. " * 12
    )

    # 1. chunking -- the real fixed-size chunker sets the boundaries.
    produced = fixed_size_chunk(body, size=160, overlap=0)
    assert len(produced) > 1

    # The app layer then identifies each chunk, exactly as core-types says it
    # must: a chunker is called as chunk(text, size=...) and cannot know the id.
    def identify(source, document_id, prefix):
        return core.Chunk(
            chunk_index=source.chunk_index,
            text=source.text,
            chunk_id=f"{prefix}-{source.chunk_index}",
            document_id=document_id,
            source_name=f"{document_id}.md",
            page_start=1,
            page_end=1,
            metadata={
                "section": "setup",
                "start_offset": source.chunk_index * 160,
                "end_offset": (source.chunk_index + 1) * 160,
            },
        )

    first = [identify(c, "doc-a", "a") for c in produced]
    # The same template in a second document -- what dedup exists to catch.
    second = [identify(c, "doc-b", "b") for c in produced]
    assert [c.text for c in first] == [c.text for c in second]

    # 2. deduplication, corpus scope, so the shared copy collapses.
    outcome = Deduplicator(scope=Scope.CORPUS).dedupe([*first, *second])
    assert len(outcome.retained) + outcome.removed == len(first) + len(second)
    assert outcome.removed >= len(second)
    spanning = [
        group for group in outcome.groups if set(group.documents) == {"doc-a", "doc-b"}
    ]
    assert spanning, "the shared copy was not attributed to both documents"

    # 3. embedding.
    embedded_chunks = [
        core.EmbeddedChunk(
            chunk=chunk,
            vector=_embed(chunk.text),
            index_fingerprint="model=stub|dims=4|chunking=size160",
        )
        for chunk in outcome.retained
    ]

    # 4. indexing, through the real adapter.
    store = _chroma(embedded_chunks)
    assert store.count() == len(outcome.retained)
    assert len(store.collection.rows) == len(outcome.retained)

    # Every stored row carries every field AC-26 names...
    reserved = adapters.RESERVED_PREFIX
    for row in store.collection.rows.values():
        metadata = row["metadata"]
        for field in (
            "document_id",
            "chunk_id",
            "source_name",
            "chunk_index",
            "page_start",
            "page_end",
        ):
            assert f"{reserved}{field}" in metadata, field
        for field in ("start_offset", "end_offset", "section"):
            assert field in metadata, field
        assert f"{reserved}index_fingerprint" in metadata
        assert metadata[f"{reserved}index_fingerprint"].startswith("model=stub")

    # ...and the same fields are readable back as a record, all populated.
    for item in embedded_chunks:
        record = record_from(item)
        for field in PROVENANCE_FIELDS:
            assert record.provenance[field] is not None, field
        assert record.is_traceable

    # Retrieval still finds them, with the citation location intact.
    hits = store.query(_embed(outcome.retained[0].text), k=2)
    assert hits
    best = hits[0]
    assert best.document_id == "doc-a"
    assert best.page_start == 1 and best.page_end == 1
    assert best.metadata["section"] == "setup"

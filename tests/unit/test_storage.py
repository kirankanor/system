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
# be. ``adapters`` last: it imports the other two.
for name in ("core_types", "vector_store_protocol", "document_store", "adapters"):
    sys.path.insert(0, str(STORAGE / name))

import core_types as core  # noqa: E402
import document_store as docs  # noqa: E402
import storage_adapters as adapters  # noqa: E402
import vector_store_protocol as protocol  # noqa: E402

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

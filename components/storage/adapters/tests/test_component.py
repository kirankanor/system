"""Tests for the storage-adapters component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4).

Neither `chromadb` nor `faiss` is installed in this checkout, and installing
them would modify dependency files -- which this library forbids (FR-2). Both
adapters are therefore exercised against injected fakes that speak the same
API surface as the real backends. The real integrations are unexecuted; the
README says so plainly.

FR-3 is proven by blocking the import rather than by asking whether the
package happens to be absent, so the assertion is about the guard and not
about the machine running the tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types and vector-store-protocol, which live in
# their own component directories. The manifest declares both dependencies, so
# `ragkit add` copies them beside this file; here we reach across to siblings.
for sibling in ("core_types", "vector_store_protocol"):
    sys.path.insert(0, str(COMPONENT.parent / sibling))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
import storage_adapters as sa  # noqa: E402
from storage_adapters import (  # noqa: E402
    MISSING_DEPENDENCY_MESSAGE,
    ChromaVectorStore,
    FaissVectorStore,
)
from vector_store_protocol import (  # noqa: E402
    MissingStorageDependency,
    StorageBackendError,
    StorageError,
    StorageInputError,
    VectorStoreProtocol,
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
            if where and any(row["metadata"].get(k) != v for k, v in where.items()):
                continue
            scored.append(
                (sum((a - b) ** 2 for a, b in zip(needle, row["vector"])), row_id)
            )
        scored.sort()
        top = scored[:n_results]
        return {
            "ids": [[r for _, r in top]],
            "documents": [[self.rows[r]["document"] for _, r in top]],
            "metadatas": [[self.rows[r]["metadata"] for _, r in top]],
            "distances": [[d for d, _ in top]],
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


def _client_for(collection):
    """Wrap a collection so it can be injected where a client is expected."""

    class _Client:
        def get_or_create_collection(self, name, metadata=None):
            return collection

    return _Client()


def _chroma(embedded_items=None, **kwargs):
    store = ChromaVectorStore("docs", client=FakeChromaClient(), **kwargs)
    if embedded_items is not None:
        store.add(embedded_items)
    return store


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
    store = FaissVectorStore(dimensions=dimensions, index=FakeFaissIndex())
    if embedded_items is not None:
        store.add(embedded_items)
    return store


# --------------------------------------------------------------------------- #
# Success paths -- both adapters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_both_adapters_satisfy_the_vector_store_protocol(factory):
    assert isinstance(factory(), VectorStoreProtocol)


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_add_then_count(factory, embedded):
    store = factory()
    assert store.add(embedded) == 2
    assert store.count() == 2


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_query_returns_nearest_first(factory, embedded):
    store = factory(embedded)
    results = store.query([1.0, 2.0], k=2)
    assert [r.chunk_id for r in results] == ["c0", "c1"]
    assert results[0].score < results[1].score


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_results_name_the_backend_that_produced_them(factory, embedded):
    """The conformance point core-types calls unenforceable by the type."""
    store = factory(embedded)
    results = store.query([1.0, 2.0], k=2)
    assert results
    assert all(r.source_name == store.name for r in results)
    assert all(isinstance(r, core.ScoredDocument) for r in results)


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_identity_fields_survive_the_round_trip(factory, embedded):
    """Stored under reserved keys, restored as plain fields on the result."""
    store = factory(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.chunk_id == "c0"
    assert hit.document_id == "d1"
    assert hit.text == "alpha text"


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_user_metadata_is_passed_through_untouched(factory, embedded):
    embedded[0].chunk.metadata["section"] = "intro"
    store = factory(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.metadata == {"section": "intro"}
    assert not any(k.startswith(sa.RESERVED_PREFIX) for k in hit.metadata)


@pytest.mark.parametrize("factory", [_chroma, _faiss], ids=["chroma", "faiss"])
def test_delete_then_count(factory, embedded):
    store = factory(embedded)
    assert store.delete(["c0"]) == 1
    assert store.count() == 1
    assert [r.chunk_id for r in store.query([1.0, 2.0], k=2)] == ["c1"]


# --------------------------------------------------------------------------- #
# ChromaVectorStore
# --------------------------------------------------------------------------- #


def test_chroma_identity_fields_come_back_unprefixed(embedded):
    store = _chroma(embedded)
    hit = store.query([1.0, 2.0], k=1)[0]
    assert hit.source_name == "docs"


def test_filtering_on_an_identity_field_matches(embedded):
    """The caller sees `document_id`, so the adapter rewrites it internally."""
    store = _chroma(embedded)
    assert [
        r.chunk_id for r in store.query([1.0, 2.0], k=2, filter={"document_id": "d1"})
    ] == ["c0", "c1"]
    assert store.query([1.0, 2.0], k=2, filter={"document_id": "other"}) == []


def test_filtering_on_user_metadata_matches(embedded):
    embedded[0].chunk.metadata["section"] = "intro"
    embedded[1].chunk.metadata["section"] = "body"
    store = _chroma(embedded)
    assert [
        r.chunk_id for r in store.query([1.0, 2.0], k=2, filter={"section": "body"})
    ] == ["c1"]


def test_a_filter_match_is_not_invented(embedded):
    store = _chroma(embedded)
    assert store.query([1.0, 2.0], k=2, filter={"section": "absent"}) == []


def test_the_identity_fields_the_adapter_reserves_are_documented():
    assert sa.IDENTITY_FIELDS == (
        "chunk_id",
        "document_id",
        "source_name",
        "chunk_index",
    )
    assert sa.RESERVED_PREFIX == "_rag_"


def test_deleting_an_unknown_id_is_a_no_op(embedded):
    """Safe to call after a failed re-ingest, which is when it is needed."""
    store = _chroma(embedded)
    assert store.delete(["never-existed"]) == 1
    assert store.count() == 2


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_collection_name_is_rejected(name):
    with pytest.raises(StorageInputError):
        ChromaVectorStore(name, client=FakeChromaClient())


def test_a_backend_failure_is_wrapped_and_chained(embedded):
    class Broken(FakeCollection):
        def add(self, **kwargs):
            raise RuntimeError("disk full")

    store = ChromaVectorStore("docs", client=_client_for(Broken()))
    with pytest.raises(StorageBackendError) as exc:
        store.add(embedded)
    assert "disk full" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_malformed_backend_response_is_rejected():
    class Nonsense(FakeCollection):
        def query(self, **kwargs):
            return "not a mapping"

    store = ChromaVectorStore("docs", client=_client_for(Nonsense()))
    with pytest.raises(StorageBackendError) as exc:
        store.query([1.0, 2.0], k=1)
    assert "mapping" in str(exc.value)


# --------------------------------------------------------------------------- #
# FaissVectorStore
# --------------------------------------------------------------------------- #


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
    with pytest.raises(StorageInputError) as exc:
        store.query([1.0, 2.0], k=1, filter={"document_id": "d1"})
    assert "does not support" in str(exc.value)


def test_faiss_rejects_a_vector_of_the_wrong_width(embedded):
    store = _faiss(dimensions=3)
    with pytest.raises(StorageInputError) as exc:
        store.add(embedded)
    assert "3-wide" in str(exc.value)


def test_faiss_rejects_a_query_vector_of_the_wrong_width():
    store = _faiss(dimensions=3)
    with pytest.raises(StorageInputError) as exc:
        store.query([1.0, 2.0], k=1)
    assert "2 wide" in str(exc.value)


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5, "3"])
def test_an_unusable_dimensions_setting_is_rejected(dimensions):
    with pytest.raises(StorageInputError):
        FaissVectorStore(dimensions=dimensions)


# --------------------------------------------------------------------------- #
# FR-3: both backends are guarded, with the manifest's message verbatim
# --------------------------------------------------------------------------- #


@pytest.fixture
def manifest():
    return json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))


def test_chromadb_is_guarded_with_the_manifest_message(manifest, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(ImportError) as exc:
        ChromaVectorStore("docs").count()
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == MISSING_DEPENDENCY_MESSAGE


def test_faiss_is_guarded_with_the_manifest_message(manifest, monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ImportError) as exc:
        FaissVectorStore(dimensions=2).count()
    assert str(exc.value) == manifest["missing_dependency_message"]


def test_the_missing_dependency_error_is_an_import_error():
    assert issubclass(MissingStorageDependency, ImportError)
    assert not issubclass(MissingStorageDependency, StorageError)


def test_importing_the_module_needs_no_backend():
    """The module must import on a machine that installed neither package."""
    assert hasattr(sa, "ChromaVectorStore")
    assert hasattr(sa, "FaissVectorStore")


def test_the_manifest_lists_both_backends_as_optional(manifest):
    assert manifest["required_packages"] == []
    assert {entry["name"] for entry in manifest["optional_packages"]} == {
        "chromadb",
        "faiss",
    }
    assert set(manifest["dependencies"]) == {"core-types", "vector-store-protocol"}


# --------------------------------------------------------------------------- #
# Message provenance
# --------------------------------------------------------------------------- #


def test_messages_name_the_component():
    """NFR-7: a caller can tell which component complained."""
    with pytest.raises(StorageInputError) as exc:
        FaissVectorStore(dimensions=0)
    assert "storage-adapters" in str(exc.value)


def test_the_guard_message_points_at_the_manifest_not_a_paraphrase(manifest):
    """A paraphrase drifts; this is copied verbatim from the manifest."""
    assert MISSING_DEPENDENCY_MESSAGE == manifest["missing_dependency_message"]
    assert "chromadb" in MISSING_DEPENDENCY_MESSAGE
    assert "faiss" in MISSING_DEPENDENCY_MESSAGE


def test_a_non_finite_vector_never_reaches_an_index():
    """The property that matters: no NaN is ever handed to a backend."""
    poisoned = core.EmbeddedChunk(
        chunk=core.Chunk(chunk_index=0, text="bad", chunk_id="c0"),
        vector=(float("nan"), 1.0),
        index_fingerprint="fp1",
    )
    with pytest.raises(StorageInputError):
        _chroma([poisoned])

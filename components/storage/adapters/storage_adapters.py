"""Opt-in backend adapters, with their optional dependencies isolated.

Two vector stores, each behind a lazy import:

* :class:`ChromaVectorStore` — over a ``chromadb`` collection.
* :class:`FaissVectorStore` — over a ``faiss`` index, with an id map so that
  deletes and re-adds behave the way a caller expects.

Both satisfy ``VectorStoreProtocol`` and both carry the same guarantees, so a
caller swaps one for the other without touching its own code (FR-5).

Isolation is the point
----------------------
Neither ``chromadb`` nor ``faiss`` is imported at module scope. Each is
imported inside the function that first needs it, and a missing package raises
:class:`MissingStorageDependency` carrying the manifest's
``missing_dependency_message`` verbatim — so the message a user sees is the
one the registry declares, rather than a paraphrase that drifts from it.
Importing this module therefore succeeds on a machine with neither installed,
which is what lets the tests run at all.

Results name their backend
--------------------------
Every :class:`ScoredDocument` returned from :meth:`query` carries the store's
name in ``source_name``. ``core-types`` calls this out as a conformance
requirement the type cannot enforce, and it is the one that matters most: a
hybrid retriever merging a dense result list with a sparse one has nothing to
merge on if the results do not say where they came from.

Storage keys vs. chunk fields
----------------------------
A vector backend stores flat key/value pairs. The chunk's identity fields
(``chunk_id``, ``document_id``, ``source_name``, ``chunk_index``) therefore
travel in the backend's metadata under reserved ``_rag_``-prefixed keys, and
user metadata is passed through untouched. The prefix keeps the two from
colliding even if a caller's metadata happens to use the same words.

The consequence is handled rather than left to the caller: a filter naming
one of those identity fields is rewritten to its reserved key before it
reaches the backend, because the caller sees the *unprefixed* name in the
results. Without that rewrite, filtering on ``document_id`` would match
nothing and come back empty -- which reads as "no such document" rather than
"wrong field name", and is the sort of silence this library exists to avoid.

Self-contained on purpose
-------------------------
Depends on ``core-types`` and ``vector-store-protocol`` (declared in the
manifest, so ``ragkit add`` copies them along).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core_types import EmbeddedChunk, Metadata, ScoredDocument
from vector_store_protocol import (
    MissingStorageDependency,
    StorageBackendError,
    StorageInputError,
    validate_embedded_chunks,
    validate_filter,
    validate_ids,
    validate_k,
    validate_vector,
)

__all__ = [
    "ChromaVectorStore",
    "FaissVectorStore",
    "IDENTITY_FIELDS",
    "MISSING_DEPENDENCY_MESSAGE",
    "MissingStorageDependency",
    "StorageBackendError",
    "StorageError",
    "StorageInputError",
]

from vector_store_protocol import StorageError  # noqa: E402  (re-exported above)

#: Copied verbatim from the component manifest's ``missing_dependency_message``.
#: The manifest validator cannot read a string out of a .py file, so this is
#: asserted against the manifest in the component's own test.
MISSING_DEPENDENCY_MESSAGE = (
    "storage-adapters requires 'chromadb' and 'faiss', which are not "
    "installed. Install them with: uv add chromadb faiss"
)

#: Prefix for identity fields this adapter stores in the backend's metadata.
#: Reserved: anything the caller put under this prefix is overwritten.
RESERVED_PREFIX = "_rag_"

#: Identity fields kept under reserved keys. A caller filters on the name they
#: see in the *results* -- ``document_id``, not ``_rag_document_id`` -- so a
#: filter naming one of these is rewritten before it reaches the backend.
#: Without the rewrite the filter matches nothing and returns an empty list,
#: which reads as "no such document" rather than "wrong field name".
IDENTITY_FIELDS = ("chunk_id", "document_id", "source_name", "chunk_index")


def _translate_filter(flat: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Rewrite identity field names to the reserved keys the backend stores."""
    if flat is None:
        return None
    return {
        f"{RESERVED_PREFIX}{field}" if field in IDENTITY_FIELDS else field: value
        for field, value in flat.items()
    }


def _serialise(items: Sequence[EmbeddedChunk], name: str) -> list[dict[str, Any]]:
    """Flatten embedded chunks into a backend's flat key/value shape."""
    ids, vectors, documents, metadatas = [], [], [], []
    for item in items:
        ids.append(item.chunk.chunk_id or f"{name}-{item.chunk.chunk_index}")
        vectors.append(list(item.vector))
        documents.append(item.chunk.text)
        metadata = dict(item.chunk.metadata or {})
        for field in ("chunk_id", "document_id", "source_name", "chunk_index"):
            value = getattr(item.chunk, field, None)
            if value is not None:
                metadata[f"{RESERVED_PREFIX}{field}"] = value
        metadatas.append(metadata)
    return [
        {"id": i, "vector": v, "document": d, "metadata": m}
        for i, v, d, m in zip(ids, vectors, documents, metadatas)
    ]


def _deserialise(
    hit_id: str,
    text: str,
    metadata: Mapping[str, Any],
    vector: Sequence[float],
    score: float,
    name: str,
) -> ScoredDocument:
    """Rebuild a result, restoring the identity fields this adapter stored."""
    from core_types import Chunk

    user = {
        key: value
        for key, value in (metadata or {}).items()
        if not key.startswith(RESERVED_PREFIX)
    }
    chunk = Chunk(
        chunk_id=metadata.get(f"{RESERVED_PREFIX}chunk_id", hit_id),
        document_id=metadata.get(f"{RESERVED_PREFIX}document_id"),
        source_name=name,
        chunk_index=int(metadata.get(f"{RESERVED_PREFIX}chunk_index", 0)),
        text=text,
        metadata=user,
    )
    return ScoredDocument.from_chunk(chunk, score=score, source_name=name)


def _load_chroma_client() -> Any:
    """Import ``chromadb`` and build a client -- the only heavy import."""
    try:
        import chromadb
    except ImportError as exc:
        raise MissingStorageDependency(MISSING_DEPENDENCY_MESSAGE) from exc
    return chromadb.Client()


def _load_faiss_index(dimensions: int) -> Any:
    """Import ``faiss`` and build an index -- the only heavy import."""
    try:
        import faiss
    except ImportError as exc:
        raise MissingStorageDependency(MISSING_DEPENDENCY_MESSAGE) from exc
    try:
        # IndexIDMap lets ids be arbitrary strings rather than row numbers,
        # which is what makes delete-by-id possible at all.
        return faiss.IndexIDMap(faiss.IndexFlatL2(dimensions))
    except Exception as exc:  # noqa: BLE001 - backend failure, re-raised
        raise StorageBackendError(
            f"storage-adapters: could not build a faiss index of "
            f"{dimensions} dimensions: {type(exc).__name__}: {exc}"
        ) from exc


def _load_numpy() -> Any:
    """Import ``numpy``, which ``faiss`` itself requires.

    Not declared separately in the manifest: ``faiss`` cannot be installed
    without it, so it arrives with the declared dependency. It is imported
    lazily for the same reason ``faiss`` is -- this module must import on a
    machine that has neither.
    """
    try:
        import numpy
    except ImportError as exc:
        raise MissingStorageDependency(MISSING_DEPENDENCY_MESSAGE) from exc
    return numpy


class ChromaVectorStore:
    """A vector store over a ``chromadb`` collection.

    Satisfies ``VectorStoreProtocol``.

    Args:
        collection_name: The collection to use; created if absent.
        metadata: Optional chromadb collection metadata.
        client: An already-built client. When omitted one is built on first
            use, which is what raises the FR-3 guard if ``chromadb`` is
            missing.
    """

    def __init__(
        self,
        collection_name: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name:
            raise StorageInputError(
                "storage-adapters: collection_name must be a non-empty str, "
                f"got {collection_name!r}"
            )
        self.name = collection_name
        self._metadata = dict(metadata or {})
        self._client = client
        self._collection: Any = None

    @property
    def collection(self) -> Any:
        """The underlying collection, resolved on first use."""
        if self._collection is None:
            client = self._client if self._client is not None else _load_chroma_client()
            try:
                self._collection = client.get_or_create_collection(
                    self.name, metadata=self._metadata or None
                )
            except Exception as exc:  # noqa: BLE001 - backend failure
                raise StorageBackendError(
                    f"storage-adapters: could not open collection "
                    f"{self.name!r}: {type(exc).__name__}: {exc}"
                ) from exc
        return self._collection

    def add(self, items: Sequence[EmbeddedChunk]) -> int:
        """Write items, returning how many were added.

        Raises:
            StorageInputError: the items are unusable, or their vectors have
                differing widths.
            StorageBackendError: the backend rejected the write.
            MissingStorageDependency: ``chromadb`` is absent (FR-3).
        """
        checked = validate_embedded_chunks(items)
        rows = _serialise(checked, self.name)
        try:
            self.collection.add(
                ids=[row["id"] for row in rows],
                embeddings=[row["vector"] for row in rows],
                documents=[row["document"] for row in rows],
                metadatas=[row["metadata"] for row in rows],
            )
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: adding {len(rows)} item(s) to "
                f"{self.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return len(rows)

    def query(
        self,
        vector: Sequence[float],
        k: int,
        filter: Metadata | None = None,
    ) -> list[ScoredDocument]:
        """Return the ``k`` nearest items, nearest first.

        Raises:
            StorageInputError: the vector, ``k`` or ``filter`` is unusable.
            StorageBackendError: the backend failed.
            MissingStorageDependency: ``chromadb`` is absent (FR-3).
        """
        needle = validate_vector(vector)
        limit = validate_k(k)
        flat = validate_filter(filter)
        try:
            raw = self.collection.query(
                query_embeddings=[needle],
                n_results=limit,
                where=_translate_filter(flat),
                include=["documents", "metadatas", "distances"],
            )
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: querying {self.name!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        results: list[ScoredDocument] = []
        for group in _as_lists(raw):
            for position, hit_id in enumerate(group["ids"]):
                results.append(
                    _deserialise(
                        hit_id,
                        group["documents"][position],
                        group["metadatas"][position],
                        needle,
                        float(group["distances"][position]),
                        self.name,
                    )
                )
        return results

    def delete(self, ids: Sequence[str]) -> int:
        """Delete by id, returning how many were deleted.

        An unknown id is not an error: deleting what is not there is a no-op,
        which is what makes this safe to call after a failed re-ingest.
        """
        checked = validate_ids(ids)
        try:
            self.collection.delete(ids=checked)
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: deleting from {self.name!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return len(checked)

    def count(self) -> int:
        """How many items the store holds."""
        try:
            return int(self.collection.count())
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: counting {self.name!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _as_lists(raw: Any) -> list[dict[str, Any]]:
    """Normalise chromadb's nested-by-query-shape response into flat rows."""
    if not isinstance(raw, Mapping):
        raise StorageBackendError(
            f"storage-adapters: the backend returned {type(raw).__name__}, "
            "expected a mapping of parallel lists"
        )
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    distances = raw.get("distances") or []
    rows: list[dict[str, Any]] = []
    for index in range(len(ids)):
        rows.append(
            {
                "ids": ids[index],
                "documents": documents[index] if documents else [],
                "metadatas": metadatas[index] if metadatas else [],
                "distances": distances[index] if distances else [],
            }
        )
    return rows


class FaissVectorStore:
    """A vector store over a ``faiss`` index, with string ids.

    ``faiss`` indexes are addressed by row number, which cannot survive a
    delete: removing row 3 leaves every later row addressing the wrong id. So
    this adapter keeps its own ``id -> row`` mapping alongside the index, and
    rebuilds the mapping after a delete rather than leaving it stale.

    Args:
        dimensions: Width of the vectors this store will hold.
        index: An already-built index. When omitted one is built on first
            use, which is what raises the FR-3 guard if ``faiss`` is missing.
    """

    def __init__(self, *, dimensions: int, index: Any = None) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise StorageInputError(
                f"storage-adapters: dimensions must be an int, got "
                f"{type(dimensions).__name__} ({dimensions!r})"
            )
        if dimensions <= 0:
            raise StorageInputError(
                f"storage-adapters: dimensions must be > 0, got {dimensions}"
            )
        self.dimensions = dimensions
        self.name = f"faiss-{dimensions}d"
        self._index = index
        self._rows: dict[str, int] = {}
        self._records: dict[str, dict[str, Any]] = {}

    @property
    def index(self) -> Any:
        """The underlying index, built on first use."""
        if self._index is None:
            self._index = _load_faiss_index(self.dimensions)
        return self._index

    def add(self, items: Sequence[EmbeddedChunk]) -> int:
        """Write items, returning how many were added."""
        checked = validate_embedded_chunks(items)
        for item in checked:
            if len(item.vector) != self.dimensions:
                raise StorageInputError(
                    f"storage-adapters: a vector is {len(item.vector)} wide "
                    f"but this store holds {self.dimensions}-wide vectors"
                )
        rows = _serialise(checked, self.name)
        try:
            numpy = _load_numpy()

            matrix = numpy.array([row["vector"] for row in rows], dtype="float32")
            labels = list(range(len(self._rows), len(self._rows) + len(rows)))
            self.index.add_with_ids(
                matrix, numpy.array(labels, dtype="int64")
            )
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: adding {len(rows)} item(s) to the faiss "
                f"index failed: {type(exc).__name__}: {exc}"
            ) from exc

        for row, label in zip(rows, labels):
            self._rows[row["id"]] = label
            self._records[row["id"]] = row
        return len(rows)

    def query(
        self,
        vector: Sequence[float],
        k: int,
        filter: Metadata | None = None,
    ) -> list[ScoredDocument]:
        """Return the ``k`` nearest items, nearest first.

        Raises:
            StorageInputError: the vector, ``k`` or ``filter`` is unusable.
                ``filter`` is rejected because a raw faiss index cannot filter;
                use a metadata-capable backend when you need it.
            StorageBackendError: the backend failed.
        """
        needle = validate_vector(vector)
        limit = validate_k(k)
        if validate_filter(filter) is not None:
            raise StorageInputError(
                "storage-adapters: the faiss adapter does not support "
                "metadata filtering, because a raw faiss index holds no "
                "metadata to filter on. Use ChromaVectorStore, or filter the "
                "results after the query"
            )
        if len(needle) != self.dimensions:
            raise StorageInputError(
                f"storage-adapters: a query vector is {len(needle)} wide but "
                f"this store holds {self.dimensions}-wide vectors"
            )
        try:
            numpy = _load_numpy()
            distances, labels = self.index.search(
                numpy.array([needle], dtype="float32"),
                min(limit, len(self._rows) or 1),
            )
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: querying the faiss index failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        reverse = {label: id_ for id_, label in self._rows.items()}
        results: list[ScoredDocument] = []
        for distance, label in zip(distances[0], labels[0]):
            id_ = reverse.get(int(label))
            if id_ is None:
                continue
            # The stored row, not the original chunk: the row is what carries
            # the reserved identity keys back out, exactly as chroma's
            # metadata round trip does.
            row = self._records[id_]
            results.append(
                _deserialise(
                    id_,
                    row["document"],
                    row["metadata"],
                    needle,
                    float(distance),
                    self.name,
                )
            )
        return results

    def delete(self, ids: Sequence[str]) -> int:
        """Delete by id and rebuild the id map so later lookups stay correct."""
        checked = validate_ids(ids)
        removed = 0
        for id_ in checked:
            label = self._rows.pop(id_, None)
            if label is None:
                continue
            self._records.pop(id_, None)
            try:
                numpy = _load_numpy()
                self.index.remove_ids(numpy.array([label], dtype="int64"))
            except MissingStorageDependency:
                raise
            except Exception as exc:  # noqa: BLE001 - backend failure
                raise StorageBackendError(
                    f"storage-adapters: deleting {id_!r} from the faiss index "
                    f"failed: {type(exc).__name__}: {exc}"
                ) from exc
            removed += 1
        return removed

    def count(self) -> int:
        """How many items the store holds."""
        try:
            return int(self.index.ntotal)
        except MissingStorageDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - backend failure
            raise StorageBackendError(
                f"storage-adapters: counting the faiss index failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

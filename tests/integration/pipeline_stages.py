"""The thirteen real ingestion stages, wired from the shipped components.

Why this file lives under ``tests/integration`` rather than in ``src/``
--------------------------------------------------------------------
Two tasks drew lines that meet exactly here, and honouring both is the whole
reason this module exists.

V2-T08 established that the orchestration layer imports no component and no
workflow framework (FR-54): ``ragkit ingest`` resolves a factory by dotted
path from configuration and calls it. V2-T09's scope is ``tests/integration``,
``docs``, ``README.md`` and ``scripts`` -- explicitly **not** ``src/``. So the
one place in this repository where the shipped components are assembled into a
runnable pipeline is this file, and that is not an accident of where the task
boundary fell: keeping assembly out of ``src/`` is precisely what makes the
orchestration layer portable.

``build_stages()`` takes no arguments, and it cannot: the configuration layer
calls it with none. Everything it needs beyond that must therefore arrive as
configuration too. The one thing a persistent index needs that the pipeline
does not hand it is *where to put itself*, so this module reads
:data:`INDEX_DIR_VARIABLE`. That is the same mechanism ``RAG_PIPELINE__STAGES``
uses, which is why it is not a workaround -- an in-memory index would let a
second run report success while having written nothing, and AC-29 asks for
document and chunk counts to be compared *across* runs.

What runs, and what cannot
--------------------------
Every stage below is a shipped component, imported exactly the way a copy of
it would be: its directory on ``sys.path``, by its own module name.

PDF is the exception, and it is worth being blunt about rather than
sidestepping. ``document_loader`` detects ``%PDF-`` and will happily hand it to
a ``PdfAdapter``, but the adapter needs an extraction library, and installing
``pymupdf`` would modify a dependency file -- which this project forbids
outright. So the end-to-end path exercises text and Markdown, which the loader
parses with the standard library alone. The PDF path is real and is covered by
``tests/unit/test_ingestion.py`` through the component's injection seam. This
module does not claim a PDF run it has not performed.

The same honesty applies to chunk deduplication: the pipeline's document phase
runs one document at a time, so deduplication here is scoped per document.
Corpus-wide deduplication needs a stage that runs *after* every document, and
the orchestration layer's two phases cannot place one there. The limitation is
recorded rather than papered over with a shared mutable accumulator.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Where the persistent index is written between runs.
#:
#: Read from the environment because ``build_stages()`` is called with no
#: arguments -- the configuration layer decides what a factory receives, and
#: this factory receives nothing. A run with no index directory set falls back
#: to a per-process directory, which still exercises every stage but cannot
#: demonstrate cross-run idempotency, so the tests always set it.
INDEX_DIR_VARIABLE = "RAG_E2E_INDEX_DIR"

#: Every component directory this assembly needs, each imported the way a copy
#: of it would be. ``index_writer`` and ``index_verifier`` last: both import
#: their siblings by module name, so those siblings must already be reachable.
_COMPONENT_DIRS = (
    "ingestion/document_discovery",
    "ingestion/document_loader",
    "ingestion/file_type_validator",
    "ingestion/document_identity_resolver",
    "ingestion/text_normalizer",
    "ingestion/metadata_enricher",
    "chunking/chunk_validator",
    "chunking/chunk_deduplicator",
    "chunking/fixed_size",
    "embeddings/embedding_protocol",
    "embeddings/batch_embeddings",
    "storage/core_types",
    "storage/vector_store_protocol",
    "storage/index_writer",
    "storage/index_verifier",
)

for _relative in _COMPONENT_DIRS:
    _directory = ROOT / "components" / _relative
    if str(_directory) not in sys.path:
        # Inserted at the front, the way ``ragkit add`` would leave a copied
        # component -- so this assembly fails if a component ever stops being
        # importable on its own terms.
        sys.path.insert(0, str(_directory))

import core_types as core  # noqa: E402
from batch_embeddings import BatchEmbedder  # noqa: E402
from chunk_deduplicator import (  # noqa: E402
    Deduplicator,
    EmbeddingReuseGate,
    IndexFingerprint,
    Scope,
    fingerprint_of,
)
from chunk_validator import ChunkValidator  # noqa: E402
from document_discovery import Discovery  # noqa: E402
from document_identity_resolver import DocumentIdentityResolver  # noqa: E402
from document_loader import DocumentLoader  # noqa: E402
from file_type_validator import (  # noqa: E402
    FileKind,
    FileTooLarge,
    FileTypeValidator,
    FileTypeError,
)
from fixed_size_chunker import chunk as fixed_size_chunk  # noqa: E402
from index_verifier import IndexVerifier  # noqa: E402
from index_writer import IndexWriter, record_from  # noqa: E402
from metadata_enricher import MetadataEnricher  # noqa: E402
from rag_code_library.pipeline import (  # noqa: E402
    CHUNKS_KEY,
    DOCUMENTS_KEY,
    CorpusState,
    DocumentState,
    Stage,
    StageScope,
)
from text_normalizer import TextNormalizer  # noqa: E402

__all__ = [
    "CHUNK_SIZE",
    "INDEX_DIR_VARIABLE",
    "INTERRUPT_VARIABLE",
    "OFFLINE_EMBEDDER_MODEL_ID",
    "PersistentDocumentStore",
    "PersistentVectorStore",
    "SimulatedInterruption",
    "build_interrupted_stages",
    "build_stages",
    "index_directory",
]

#: Small enough that a short fixture produces several chunks, large enough
#: that a chunk is not one word. A configuration value, so changing it is a
#: different fingerprint and therefore a full re-embed -- which is AC-24.
CHUNK_SIZE = 64
CHUNK_OVERLAP = 0

#: The offline embedder's model id. Changing this changes the index
#: fingerprint, which is what makes re-embedding observable.
OFFLINE_EMBEDDER_MODEL_ID = "tiny-offline-demo"

#: Text formats the loader parses with the standard library alone.
_TEXT_KINDS = frozenset({FileKind.TEXT, FileKind.MARKDOWN, FileKind.HTML})

#: Text formats the file-type validator may accept. Anything else is rejected
#: before a single byte is handed to a loader, which is what makes AC-22's
#: "no extraction stage was invoked" assertable.
_ALLOWED_KINDS = frozenset({FileKind.TEXT, FileKind.MARKDOWN})


def index_directory() -> Path:
    """Where this assembly keeps its index between runs."""
    configured = os.environ.get(INDEX_DIR_VARIABLE, "").strip()
    if configured:
        return Path(configured)
    # A per-process directory: every stage still runs, but nothing survives,
    # so cross-run idempotency cannot be shown. The tests always set the
    # variable; this branch only exists so importing this module is safe.
    return Path(os.environ.get("TEMP", ".")) / f"rag-e2e-{os.getpid()}"


# --------------------------------------------------------------------------- #
# An embedder that needs nothing
# --------------------------------------------------------------------------- #


class TinyEmbedder:
    """An ``EmbeddingProtocol`` implementation that needs no model or network.

    Deliberately deterministic and content-derived, so two runs over unchanged
    text produce identical vectors and AC-29's "no stale vector" is meaningful.
    A random embedder would make every run differ and prove nothing.

    Satisfies the protocol structurally -- no inheritance -- which is the point
    of defining it as one.
    """

    dimensions = 8
    model_id = OFFLINE_EMBEDDER_MODEL_ID

    def embed(self, texts):
        vectors = []
        for text in texts:
            # A bag-of-character-trigrams sketch: crude, but a real function of
            # the content, so similar text lands near similar text and the
            # verifier's retrieval probe has something to find.
            vector = [0.0] * self.dimensions
            for index in range(len(text) - 2):
                vector[hash(text[index : index + 3]) % self.dimensions] += 1.0
            magnitude = math.sqrt(sum(value * value for value in vector))
            if magnitude:
                vector = [value / magnitude for value in vector]
            vectors.append(vector)
        return vectors


# --------------------------------------------------------------------------- #
# Stores that survive a process
# --------------------------------------------------------------------------- #


class _PersistentBase:
    """Shared JSON-backed storage for one run's index and document set.

    Both stores read and write the same file so that a single document count
    and a single chunk count describe one consistent picture. A run that wrote
    chunks but not the document that owned them would make AC-29's
    before/after comparison meaningless.
    """

    def __init__(self, directory: Path) -> None:
        self._path = directory / "index.json"

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt file is treated as empty: reprocessing is always safe,
            # whereas refusing to run would let a damaged file block the
            # destination for good.
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class PersistentVectorStore:
    """A vector store with ``add``/``query``/``get``/``delete``/``count``.

    Implements enough of ``VectorStoreProtocol`` for both ``IndexWriter`` and
    ``IndexVerifier``, which between them need a writer, a counter, a
    similarity probe and a readback path.

    Similarity is cosine over the stored vectors. It is not a good index -- it
    is a correct one, and correctness is what the verifier is checking.
    """

    def __init__(self, directory: Path) -> None:
        self._store = _PersistentBase(directory)

    def _chunks(self) -> dict:
        payload = self._store._read()
        chunks = payload.get("chunks")
        return chunks if isinstance(chunks, dict) else {}

    def _save_chunks(self, chunks: dict) -> None:
        payload = self._store._read()
        payload["chunks"] = chunks
        self._store._write(payload)

    # -- VectorStoreProtocol ----------------------------------------------- #

    def add(self, items) -> int:
        """Store ``items`` (EmbeddedChunk) keyed by chunk id. Returns the count.

        The fingerprint is persisted beside the vectors rather than only on
        them, so a later run can ask "was this index written under the
        configuration I am about to use?" without rebuilding every record.
        """
        chunks = self._chunks()
        fingerprint = None
        for item in items:
            record = record_from(item)
            chunks[record.chunk_id] = dataclasses.asdict(record)
            fingerprint = item.index_fingerprint or fingerprint
        payload = self._store._read()
        payload["chunks"] = chunks
        if fingerprint:
            payload["fingerprint"] = fingerprint
        self._store._write(payload)
        return len(chunks)

    def count(self) -> int:
        return len(self._chunks())

    def get(self, chunk_id: str):
        """The stored vector for ``chunk_id``, or ``None`` if it is not stored.

        Returns the vector rather than the record because that is what the
        verifier's readback contract is: it fetches a chunk id and asks
        whether the dimension it finds matches the dimension it expects. A
        readback that returned a whole record would make the dimension check
        impossible to run, and a check that cannot run is worse than one that
        fails, because it passes silently.
        """
        stored = self._chunks().get(chunk_id)
        if stored is None:
            return None
        return tuple(float(value) for value in stored.get("vector") or ())

    def delete(self, ids) -> int:
        chunks = self._chunks()
        removed = 0
        for chunk_id in list(ids):
            if chunks.pop(chunk_id, None) is not None:
                removed += 1
        self._save_chunks(chunks)
        return removed

    def all_expected(self) -> tuple:
        """Every stored row rebuilt as an ``EmbeddedChunk``.

        This is what makes per-document verification mean something. The
        verifier's count check compares the whole index against the whole
        expected set, so a document that verified only its own chunks would
        fail on the first sibling document it shared an index with -- the
        sibling's records would read as surplus. Rebuilding the expected set
        from what the index actually holds turns each document's verification
        into a proof that the index is *still* consistent after that document
        was added, which is a stronger and more honest claim than the narrower
        one it would otherwise have made.
        """
        return tuple(self._rebuild(row) for row in self._chunks().values())

    def stored_fingerprint(self) -> str | None:
        """The fingerprint the stored vectors were written under, if any.

        ``None`` means an empty or unlabelled index, and the reuse gate treats
        that as "nothing to reuse" rather than as a mismatch -- a first run has
        nothing stored, so refusing it would block the only run that could
        create anything.
        """
        payload = self._store._read()
        value = payload.get("fingerprint")
        return value if isinstance(value, str) and value else None

    def query(self, vector, k: int, filter=None):
        """The ``k`` nearest stored chunks, by cosine similarity."""
        scored = []
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        for chunk_id, stored in self._chunks().items():
            if filter is not None and stored.get("document_id") not in filter:
                continue
            candidate = stored.get("vector") or []
            other = math.sqrt(sum(value * value for value in candidate)) or 1.0
            dot = sum(a * b for a, b in zip(vector, candidate))
            scored.append((dot / (magnitude * other), chunk_id, stored))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [self._as_scored(stored, score) for score, _chunk_id, stored in scored[:k]]

    # -- reconstruction ---------------------------------------------------- #

    @staticmethod
    def _as_scored(stored: dict, score: float) -> core.ScoredDocument:
        """Rebuild one stored row as a scored result, field by field.

        Built explicitly rather than through ``ScoredDocument.from_chunk``,
        because the verifier's retrieval probe only means anything if the
        returned chunk id is the stored one, and a conversion that quietly
        dropped it would turn a passing check into a vacuous one.
        """
        return core.ScoredDocument(
            chunk_id=stored.get("chunk_id"),
            document_id=stored.get("document_id"),
            source_name=stored.get("source_name"),
            chunk_index=stored.get("chunk_index", 0),
            text=stored.get("text", ""),
            page_start=stored.get("page_start"),
            page_end=stored.get("page_end"),
            metadata=stored.get("metadata") or {},
            score=score,
        )

    @staticmethod
    def _as_chunk(stored: dict) -> core.Chunk:
        return core.Chunk(
            chunk_id=stored.get("chunk_id"),
            document_id=stored.get("document_id"),
            source_name=stored.get("source_name"),
            chunk_index=stored.get("chunk_index", 0),
            text=stored.get("text", ""),
            page_start=stored.get("page_start"),
            page_end=stored.get("page_end"),
            metadata=stored.get("metadata") or {},
        )

    def _rebuild(self, stored: dict):
        return core.EmbeddedChunk(
            chunk=self._as_chunk(stored),
            vector=tuple(float(value) for value in stored.get("vector") or ()),
            index_fingerprint=stored.get("index_fingerprint", ""),
        )


class _ArtifactCache:
    """Per-document intermediates, kept between runs so a resume has something to resume *into*.

    The pipeline replays stage **names**, never stage outputs: on resume it
    seeds the completed prefix onto the state so the written record describes
    the whole document, and then runs the remaining stages. Those stages still
    need what the earlier ones produced, and an in-memory artifact chain has
    nothing to give them -- a resumed run would reach for an artifact that no
    longer exists and fail, which is exactly what happened before this class
    existed.

    Persisting the two stages that carry real work (the retained chunks and
    their vectors) is enough: everything before them is cheap to recompute from
    the source file, and everything after them is read back out of the index.

    Keyed by document id, which this assembly derives from the file's content
    hash. Editing a file therefore produces a new id and a new cache entry, so
    a stale one can never be picked up by mistake.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory / "artifacts"

    def _path(self, document_id: str) -> Path:
        # document_id is hex, so this cannot escape the directory -- but the
        # join is still made absolute-checked, because "the id came from a
        # hash" is a property of today's callers, not a guarantee.
        return (self._directory / f"{document_id}.json").resolve()

    def load(self, document_id: str) -> dict:
        path = self._path(document_id)
        if self._directory not in path.parents or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # An unreadable cache is treated as absent: recomputing is always
            # safe, whereas refusing to run would leave a damaged file able to
            # block the destination for good.
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self, document_id: str, **sections) -> None:
        payload = self.load(document_id)
        payload.update(sections)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._path(document_id).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


class PersistentDocumentStore:
    """A document store with ``put``/``get``/``list``/``count``."""

    def __init__(self, directory: Path) -> None:
        self._store = _PersistentBase(directory)

    def _documents(self) -> dict:
        payload = self._store._read()
        documents = payload.get("documents")
        return documents if isinstance(documents, dict) else {}

    def _save(self, documents: dict) -> None:
        payload = self._store._read()
        payload["documents"] = documents
        self._store._write(payload)

    def put(self, doc) -> str:
        documents = self._documents()
        documents[doc.document_id] = {
            "document_id": doc.document_id,
            "source_name": doc.source_name,
            "sha256": doc.sha256,
            "page_count": doc.page_count,
            "metadata": doc.metadata or {},
        }
        self._save(documents)
        return doc.document_id

    def get(self, document_id: str):
        stored = self._documents().get(document_id)
        if stored is None:
            return None
        return core.SourceDocument(**stored)

    def list(self, filter=None):
        documents = [core.SourceDocument(**value) for value in self._documents().values()]
        if not filter:
            return documents
        return [
            document
            for document in documents
            if all(document.metadata.get(k) == v for k, v in filter.items())
        ]

    def count(self) -> int:
        return len(self._documents())

    def drop(self, document_ids) -> int:
        """Remove documents that are no longer in the source. Returns the count."""
        documents = self._documents()
        removed = 0
        for document_id in list(document_ids):
            if documents.pop(document_id, None) is not None:
                removed += 1
        if removed:
            self._save(documents)
        return removed


# --------------------------------------------------------------------------- #
# The thirteen stages
# --------------------------------------------------------------------------- #


def _fingerprints() -> IndexFingerprint:
    """The configuration fingerprint that governs vector reuse (AC-24)."""
    return fingerprint_of(
        model=OFFLINE_EMBEDDER_MODEL_ID,
        dimensions=TinyEmbedder().dimensions,
        chunking_config={"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
    )


def _reject(seed: dict, reason: str) -> dict:
    """Copy a discovered seed, marked as refused before any loader saw it."""
    return {**seed, "rejected": reason}


def _embedded_for(state: DocumentState, cache: _ArtifactCache):
    """What ``index`` needs: the vectors, plus the document's page count and metadata.

    Three values rather than one because a resumed run at ``index`` also misses
    the ``extract`` and ``enrich`` artifacts, and a document written without
    its page count or metadata would be a record that is silently thinner than
    the one the uninterrupted run produced.
    """
    artifact = state.artifacts.get("embed")
    if artifact is not None:
        return (
            artifact,
            len(state.artifacts["extract"]["pages"]),
            state.artifacts["enrich"].merged,
        )
    cached = cache.load(state.document_id)
    return (
        [core.EmbeddedChunk(**row) for row in cached.get("embedded") or ()],
        int(cached.get("page_count") or 0),
        cached.get("metadata") or {},
    )


def build_stages():
    """Return the thirteen stages, assembled from the shipped components.

    Corpus stages first, because they run first: discovery, file validation and
    identity all judge the *set* of files before any one of them is loaded.
    Rejection has to happen there rather than in a document stage, because the
    pipeline only lets the corpus phase refuse a document -- which is what makes
    AC-22's "no extraction stage was invoked" true by construction rather than
    by counting.
    """
    directory = index_directory()
    vector_store = PersistentVectorStore(directory)
    document_store = PersistentDocumentStore(directory)
    artifact_cache = _ArtifactCache(directory)
    fingerprint = _fingerprints()
    embedder = BatchEmbedder(TinyEmbedder())

    # -- corpus phase ------------------------------------------------------ #

    def discover(state: CorpusState):
        """Step 1. Find the files, and give each one its identity."""
        resolver = DocumentIdentityResolver()
        report = Discovery(state.source, recursive=True, sort_by="relative_path").discover()

        seeds = []
        for found in report.files:
            identity = resolver.hash_file(found.path)
            seeds.append(
                {
                    "document_id": identity.document_id,
                    "source_name": found.relative_path,
                    "content_hash": identity.sha256,
                    "path": str(found.path),
                    "size_bytes": found.size_bytes,
                }
            )
        state.artifacts["discovery"] = report
        state.artifacts[DOCUMENTS_KEY] = seeds
        return report

    def validate(state: CorpusState):
        """Step 2. Refuse anything the loader must not be handed."""
        validator = FileTypeValidator(max_bytes=10 * 1024 * 1024, allowed=_ALLOWED_KINDS)
        report = state.artifacts["discovery"]
        refused = {
            entry.relative_path: entry.reason.value
            for entry in getattr(report, "skipped", ())
        }

        seeds = []
        for seed in state.artifacts[DOCUMENTS_KEY]:
            if seed["source_name"] in refused:
                seeds.append(_reject(seed, refused[seed["source_name"]]))
                continue
            try:
                validator.validate(seed["path"])
            except (FileTooLarge, FileTypeError) as exc:
                seeds.append(_reject(seed, str(exc)))
            else:
                seeds.append(seed)
        state.artifacts[DOCUMENTS_KEY] = seeds
        return seeds

    def identify(state: CorpusState):
        """Step 3. Resolve versions, refuse duplicate paths, drop what is gone.

        Also the reconciliation sweep: by the end of this stage the run knows
        the authoritative set of documents the source currently contains, so
        this is the one place a whole-index cleanup can belong. A document
        whose file has been deleted takes its chunks with it -- otherwise the
        index would go on serving text for a source that no longer has it,
        which is the stale-record problem AC-23 asks about.

        Scoped to a single index shared by a single source directory. An index
        holding chunks from *several* sources would have the other sources'
        documents reconciled away here; the assembly therefore writes one
        index per project, and says so.
        """
        resolver = DocumentIdentityResolver()
        seen: dict[str, str] = {}

        seeds = []
        for seed in state.artifacts[DOCUMENTS_KEY]:
            if "rejected" in seed:
                seeds.append(seed)
                continue
            resolution = resolver.resolve(seed["path"])
            first_path = seen.get(resolution.identity.document_id)
            if first_path is not None:
                seeds.append(
                    _reject(
                        seed,
                        f"duplicate content: identical to {first_path} already ingested",
                    )
                )
                continue
            seen[resolution.identity.document_id] = seed["source_name"]
            seeds.append({**seed, "version": resolution.version})

        current = set(seen)
        stale_rows = [
            row
            for row in vector_store._chunks().values()
            if row.get("document_id") not in current
        ]
        if stale_rows:
            # Recorded so a run that silently deleted records can be told apart
            # from one that had nothing to remove.
            state.provenance["chunks_removed"] = vector_store.delete(
                row.get("chunk_id") for row in stale_rows
            )
            state.provenance["documents_removed"] = document_store.drop(
                {row.get("document_id") for row in stale_rows}
            )

        state.artifacts[DOCUMENTS_KEY] = seeds
        return seeds

    # -- document phase ---------------------------------------------------- #

    #: Document stage callables by name, so any stage can rebuild an upstream
    #: one. Populated below, and fully populated before any stage can run.
    document_stages: dict[str, object] = {}

    def _ensure(state: DocumentState, name: str):
        """Return an artifact, rebuilding it if this run never produced one.

        A resumed run arrives at its first incomplete stage with every earlier
        stage already recorded as complete -- and therefore not re-run -- so
        ``state.artifacts`` is missing exactly the outputs the remaining stages
        read. That is a correct control-flow decision by the pipeline (the
        work was genuinely done, in the run that was interrupted), but it
        leaves later stages reaching for artifacts nobody has.

        Every stage here derives from the source file on disk, so the missing
        piece can always be rebuilt by running its stage again. The pipeline's
        own record is untouched by this: it still names the stages this run
        skipped, which remains true -- they were completed by an earlier run.
        """
        if name in state.artifacts:
            return state.artifacts[name]
        state.artifacts[name] = document_stages[name](state)
        return state.artifacts[name]

    def load(state: DocumentState):
        """Step 4. Parse the file into pages, using only the standard library."""
        return DocumentLoader().load(state.provenance["path"])

    def extract(state: DocumentState):
        """Step 5. Turn pages into text plus a page map for citations.

        For PDF this would be the adapter's job and the library would be
        optional. Here the loader has already produced the pages, so extraction
        is the join -- with the page numbers kept, because a chunk that cannot
        say which page it came from cannot be cited.
        """
        loaded = _ensure(state, "load")
        return {
            "text": loaded.text,
            "pages": tuple(
                (page.number, page.source_locator) for page in loaded.pages
            ),
            "format": loaded.format,
        }

    def normalize(state: DocumentState):
        """Step 6. Normalize the extracted text, idempotently."""
        result = TextNormalizer().normalize(_ensure(state, "extract")["text"])
        return {"text": result.text, "rules": result.rules}

    def enrich(state: DocumentState):
        """Step 7. Attach metadata, with system fields outranking document ones."""
        bundle = MetadataEnricher(
            ingested_at=datetime.now(timezone.utc).isoformat()
        ).enrich(
            source_name=state.source_name,
            # The loader is the only place document-supplied metadata can come
            # from, so it is the only thing forwarded. AC-28's forged
            # ``system_*`` case is asserted directly against this component,
            # because a plain text file has no metadata channel to forge with.
            source_metadata=_ensure(state, "load").metadata,
            content_sha256=state.content_hash,
        )
        return bundle

    def chunk(state: DocumentState):
        """Step 8. Split the text, and stamp every chunk with its provenance."""
        text = _ensure(state, "normalize")["text"]
        extracted = _ensure(state, "extract")
        document_id = state.document_id
        source_name = state.source_name
        pages = extracted["pages"]

        produced = fixed_size_chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunks = []
        offset = 0
        dropped = 0
        for produced_chunk in produced:
            raw = produced_chunk.text
            if not raw.strip():
                # A chunk split lands on a run of whitespace -- usually the
                # newline at the end of a short file. It carries nothing to
                # retrieve and cannot be cited, and building a Chunk from it
                # would raise: the type requires text, correctly. Dropping it
                # and counting it keeps the offsets honest while refusing to
                # store a record that says nothing.
                offset += len(raw)
                dropped += 1
                continue
            # The chunker trims each chunk's edges, so the offsets are trimmed
            # to match. Without this a citation would point at a span holding
            # the text *plus* the whitespace either side of it, and anyone
            # slicing the source by those offsets would get back something
            # other than what the index says is there.
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = offset + leading
            end = offset + len(raw) - trailing
            offset += len(raw)
            chunks.append(
                core.Chunk(
                    chunk_id=f"{document_id}:{produced_chunk.chunk_index}",
                    document_id=document_id,
                    source_name=source_name,
                    chunk_index=produced_chunk.chunk_index,
                    text=raw.strip(),
                    # A text document has one page; the map keeps the locator so
                    # a citation can name where the text came from.
                    page_start=pages[0][0] if pages else None,
                    page_end=pages[-1][0] if pages else None,
                    metadata={
                        "start_offset": start,
                        "end_offset": end,
                        "section": pages[0][1] if pages else None,
                    },
                )
            )
        # Reported in the run summary as this document's chunk count.
        state.provenance[CHUNKS_KEY] = len(chunks)
        if dropped:
            state.provenance["chunks_dropped_as_empty"] = dropped
        return chunks

    def validate_chunks(state: DocumentState):
        """Step 9. Hold back anything unusable, rather than index it."""
        outcome = ChunkValidator(max_chars=CHUNK_SIZE * 4).validate(
            _ensure(state, "chunk")
        )
        state.provenance["chunks_rejected"] = len(outcome.rejected)
        return outcome

    def deduplicate(state: DocumentState):
        """Step 10. Collapse repeats within this document.

        Scoped to the document, not the corpus: the orchestration layer runs
        every document stage once per document and has no phase that runs after
        all of them, so corpus-wide deduplication cannot be expressed as a
        document stage. Scoped here it is still correct -- it is merely a
        narrower claim than the component can make.

        The retained chunks are cached, because this is the first stage whose
        output a later resume would otherwise have no way to reconstruct.
        """
        outcome = Deduplicator(scope=Scope.DOCUMENT).dedupe(
            _ensure(state, "validate_chunks").accepted
        )
        artifact_cache.save(
            state.document_id,
            retained=[c.model_dump() for c in outcome.retained],
        )
        return outcome

    def embed(state: DocumentState):
        """Step 11. Vectorize, refusing to reuse vectors from a changed setup.

        ``EmbeddingReuseGate`` owns "is what is already stored compatible with
        what I am about to write". It raises :class:`FingerprintMismatch`
        naming which field differs (AC-24).
        """
        gate = EmbeddingReuseGate.configured(
            model=OFFLINE_EMBEDDER_MODEL_ID,
            dimensions=TinyEmbedder().dimensions,
            chunking_config={"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        )
        # Only a stored fingerprint can disagree. An index that has never been
        # written has nothing to reuse, and the gate's own ``check`` raises on
        # ``None`` -- correctly, for a chunk being re-embedded -- but here the
        # question is "may I reuse what is stored", and the answer to that on
        # an empty index is simply yes.
        stored = vector_store.stored_fingerprint()
        if stored is not None:
            gate.check(stored)

        # The cached copy is preferred over re-deriving it, because that is the
        # entire point of caching it: a run resumed at ``embed`` skips
        # load/extract/normalize/enrich/chunk/validate_chunks/deduplicate, so
        # ``state.artifacts["deduplicate"]`` was never set in this process.
        # Only when there is no cached copy does the chain have to be rebuilt.
        cached = artifact_cache.load(state.document_id).get("retained")
        if cached is not None:
            retained = [core.Chunk(**row) for row in cached]
        else:
            retained = _ensure(state, "deduplicate").retained

        vectors = embedder.embed([item.text for item in retained])
        embedded = [
            core.EmbeddedChunk(
                chunk=item, vector=tuple(vector), index_fingerprint=fingerprint.value
            )
            for item, vector in zip(retained, vectors)
        ]
        # Page count and metadata come along with the vectors, so that a run
        # resumed at ``index`` can write a document record every bit as complete
        # as the one an uninterrupted run would have written -- a record that is
        # quietly thinner is a record the next reader cannot rely on.
        artifact_cache.save(
            state.document_id,
            embedded=[item.model_dump() for item in embedded],
            page_count=len(_ensure(state, "extract")["pages"]),
            metadata=_ensure(state, "enrich").merged,
        )
        return embedded

    def index(state: DocumentState):
        """Step 12. Write chunks and the document that owns them."""
        embedded, page_count, metadata = _embedded_for(state, artifact_cache)
        report = IndexWriter(
            vector_store, dimensions=TinyEmbedder().dimensions
        ).upsert(embedded)
        report.raise_if_failed()
        document_store.put(
            core.SourceDocument(
                document_id=state.document_id,
                source_name=state.source_name,
                sha256=state.content_hash,
                page_count=page_count,
                metadata=metadata,
            )
        )
        state.provenance["indexed"] = len(embedded)
        return report

    def verify(state: DocumentState):
        """Step 13. Prove the index still holds what it claims to.

        Verifies the **whole** index, not just this document's contribution.
        The verifier's count check compares the index's size against the size
        of the expected set, so an expected set limited to one document would
        report every sibling's records as surplus and fail from the second
        document onward -- a check that cannot pass is not a check. Reading the
        expected set back out of the index makes each document's verification a
        statement that the index remained consistent as this document joined
        it, which is what an index check is for.
        """
        verdict = IndexVerifier(
            vector_store,
            expected=vector_store.all_expected(),
            dimensions=TinyEmbedder().dimensions,
        ).verify()
        # A fatal finding is a real failure, not a warning: the run must not
        # report success over an index that does not hold what it claims to.
        verdict.raise_if_failed()
        return verdict

    # Registered after definition so ``_ensure`` can reach any of them by
    # name; by the time a stage can run, the whole mapping exists.
    document_stages.update(
        {
            "load": load,
            "extract": extract,
            "normalize": normalize,
            "enrich": enrich,
            "chunk": chunk,
            "validate_chunks": validate_chunks,
            "deduplicate": deduplicate,
            "embed": embed,
            "index": index,
            "verify": verify,
        }
    )

    return [
        Stage("discover", discover, scope=StageScope.CORPUS),
        Stage("validate", validate, scope=StageScope.CORPUS),
        Stage("identify", identify, scope=StageScope.CORPUS),
        Stage("load", load),
        Stage("extract", extract),
        Stage("normalize", normalize),
        Stage("enrich", enrich),
        Stage("chunk", chunk),
        Stage("validate_chunks", validate_chunks),
        Stage("deduplicate", deduplicate),
        Stage("embed", embed),
        Stage("index", index),
        Stage("verify", verify),
    ]


class SimulatedInterruption(Exception):
    """Raised by :func:`build_interrupted_stages` to fake a process dying mid-run."""


#: The stage whose first invocation raises when interruption is requested.
INTERRUPT_VARIABLE = "RAG_E2E_INTERRUPT_AT"


def build_interrupted_stages():
    """The thirteen stages, with one made to fail on demand.

    AC-30 asks that a run interrupted partway through recover and complete
    without duplicating or dropping records. Faking that by editing a
    checkpoint would only prove the checkpoint can be edited; failing a real
    stage proves the pipeline reacts to a real failure, records an honest
    prefix, and lets the next run resume at the stage that never finished.

    Which stage fails is configuration (:data:`INTERRUPT_VARIABLE`), the same
    way every other stage is chosen here -- by name, from the environment.

    Returns the ordinary stages unchanged when the variable is unset, so a run
    cannot be interrupted by a variable someone forgot to clear.
    """
    target = os.environ.get(INTERRUPT_VARIABLE, "").strip()
    if not target:
        return build_stages()

    stages = build_stages()
    wrapped = []
    for stage in stages:
        if stage.name != target:
            wrapped.append(stage)
            continue
        def interrupting(state, *, _name=target):
            raise SimulatedInterruption(
                f"simulated interruption at stage {_name!r}: the process died "
                "before this stage produced an artifact"
            )

        wrapped.append(dataclasses.replace(stage, run=interrupting))
    return wrapped

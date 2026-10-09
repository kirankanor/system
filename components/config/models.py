"""Shared data contracts passed between pipeline stages.

Design choices (the WHY):
- Pydantic v2, frozen, ``extra="forbid"``: a typo'd field fails loudly and a
  record cannot be mutated after validation.
- Metadata is FLAT scalars only (str/int/float/bool). Most vector databases
  cannot filter on nested objects or lists, so we refuse them up front.
- IDs are DETERMINISTIC (derived from content). Re-ingesting the same file
  produces the same ids, so a vector-store *upsert* replaces instead of
  duplicating. Trade-off: identical bytes under two filenames share one id.
- Chunks carry ``source_name`` and page range so a citation can be built from
  a retrieved chunk alone, without a second lookup.
"""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- id helpers --------------------------------------------------------------
def make_document_id(content_sha256: str) -> str:
    """Document id = first 32 hex chars of the file's SHA-256."""
    if not _SHA256_RE.match(content_sha256):
        raise ValueError("content_sha256 must be 64 lowercase hex characters")
    return content_sha256[:32]


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Chunk id from (document, position, text). Changes if the text or order changes."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{document_id}:{chunk_index}:{text_hash}".encode()).hexdigest()[:32]


# --- ingestion ---------------------------------------------------------------
class SourceDocument(_Frozen):
    document_id: _NonEmpty
    source_name: _NonEmpty
    sha256: Annotated[str, StringConstraints(pattern=_SHA256_RE.pattern)]
    page_count: Annotated[int, Field(ge=0)]
    metadata: Metadata = Field(default_factory=dict)


class PageText(_Frozen):
    """Neutral per-page text. Map your extractor's output into this
    (e.g. pymupdf_extractor.PageResult -> PageText(page_number=p.page_number, text=p.text))."""

    page_number: Annotated[int, Field(ge=1)]
    text: str


# --- chunking / indexing -----------------------------------------------------
class Chunk(_Frozen):
    chunk_id: _NonEmpty
    document_id: _NonEmpty
    source_name: _NonEmpty
    chunk_index: Annotated[int, Field(ge=0)]
    text: _NonEmpty
    page_start: Annotated[int, Field(ge=1)]
    page_end: Annotated[int, Field(ge=1)]
    metadata: Metadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _pages_ordered(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class EmbeddedChunk(_Frozen):
    """A chunk plus its vector. ``index_fingerprint`` records which embedding/chunking
    configuration produced it (see Settings.index_fingerprint) so stale data is detectable."""

    chunk: Chunk
    vector: Annotated[tuple[float, ...], Field(min_length=1)]
    index_fingerprint: _NonEmpty


# --- retrieval / answer ------------------------------------------------------
class Query(_Frozen):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    filters: Metadata = Field(default_factory=dict)


class RetrievedChunk(_Frozen):
    chunk: Chunk
    score: float  # meaning depends on the metric/retriever; do not compare across retrievers
    rank: Annotated[int, Field(ge=1)]
    retriever: _NonEmpty  # e.g. "dense", "bm25", "hybrid", "rerank"


class Citation(_Frozen):
    chunk_id: str
    document_id: str
    source_name: str
    page_start: int
    page_end: int

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Self:
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_name=chunk.source_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )


class RagAnswer(_Frozen):
    answer: str
    citations: tuple[Citation, ...] = ()
    retrieved: tuple[RetrievedChunk, ...] = ()
    model: str | None = None
    abstained: bool = False  # True when the system declined because context was insufficient

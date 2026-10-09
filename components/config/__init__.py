"""Shared configuration layer for the RAG pipeline: settings, errors, models, prompts.

Only your orchestration code should import this. Standalone library components
must NOT depend on it (they ship their own models/exceptions).
"""

from .enums import DistanceMetric, PipelineStage
from .errors import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    EvaluationError,
    ExternalServiceError,
    ExtractionError,
    GenerationError,
    IndexingError,
    IngestionError,
    RagError,
    RetrievalError,
    UnsupportedFormatError,
    is_retryable,
)
from .models import (
    Chunk,
    Citation,
    EmbeddedChunk,
    Metadata,
    MetadataValue,
    PageText,
    Query,
    RagAnswer,
    RetrievedChunk,
    SourceDocument,
    make_chunk_id,
    make_document_id,
)
from .prompts import GROUNDED_QA_SYSTEM, GROUNDED_QA_VERSION, build_user_prompt, format_context
from .settings import Settings, get_settings, load_settings

__all__ = [
    "Chunk",
    "ChunkingError",
    "Citation",
    "ConfigurationError",
    "DistanceMetric",
    "EmbeddedChunk",
    "EmbeddingError",
    "EvaluationError",
    "ExternalServiceError",
    "ExtractionError",
    "GROUNDED_QA_SYSTEM",
    "GROUNDED_QA_VERSION",
    "GenerationError",
    "IndexingError",
    "IngestionError",
    "Metadata",
    "MetadataValue",
    "PageText",
    "PipelineStage",
    "Query",
    "RagAnswer",
    "RagError",
    "RetrievalError",
    "RetrievedChunk",
    "Settings",
    "SourceDocument",
    "UnsupportedFormatError",
    "build_user_prompt",
    "format_context",
    "get_settings",
    "is_retryable",
    "load_settings",
    "make_chunk_id",
    "make_document_id",
]

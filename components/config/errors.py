"""One exception hierarchy for the whole pipeline.

Why: callers (API handlers, retry loops, logs) need to answer three questions
about any failure -- *where* did it happen, *can I retry*, and *what is a safe
thing to log?* Every RagError carries ``stage``, ``retryable``, ``code`` and
``details`` so you never parse message strings.

Rules:
- ``details`` must contain NON-SENSITIVE data only (ids, counts, limits).
  Never put API keys, document text or user queries in it.
- Wrap third-party exceptions with ``raise XError(...) from exc`` so the
  original cause is kept for debugging.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .enums import PipelineStage


class RagError(Exception):
    """Base class for every error raised by pipeline code."""

    stage: PipelineStage | None = None
    default_retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or type(self).__name__
        self.retryable = self.default_retryable if retryable is None else retryable
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Structured form for logs / API error bodies."""
        return {
            "code": self.code,
            "stage": self.stage.value if self.stage else None,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


# --- Configuration -----------------------------------------------------------
class ConfigurationError(RagError):
    """Invalid or missing settings. Fix the config; retrying will not help."""

    stage = PipelineStage.CONFIG


# --- Ingestion ---------------------------------------------------------------
class IngestionError(RagError):
    """Reading or extracting a source document failed."""

    stage = PipelineStage.INGESTION


class UnsupportedFormatError(IngestionError):
    """Source type is not supported by any extractor."""


class ExtractionError(IngestionError):
    """A supported file could not be parsed (corrupt, encrypted, over limits)."""


# --- Local processing --------------------------------------------------------
class ChunkingError(RagError):
    stage = PipelineStage.CHUNKING


# --- Calls to external services (network => transient failures are normal) ---
class ExternalServiceError(RagError):
    """Base for failures talking to an embedding / vector-store / LLM provider.

    Retryable by default (timeouts, 429, 5xx). Pass ``retryable=False`` for
    permanent failures such as invalid credentials or a rejected request.
    """

    default_retryable = True


class EmbeddingError(ExternalServiceError):
    stage = PipelineStage.EMBEDDING


class IndexingError(ExternalServiceError):
    stage = PipelineStage.INDEXING


class RetrievalError(ExternalServiceError):
    stage = PipelineStage.RETRIEVAL


class GenerationError(ExternalServiceError):
    stage = PipelineStage.GENERATION


# --- Evaluation --------------------------------------------------------------
class EvaluationError(RagError):
    stage = PipelineStage.EVALUATION


def is_retryable(exc: BaseException) -> bool:
    """True only for RagErrors flagged retryable. Unknown exceptions are NOT retried."""
    return isinstance(exc, RagError) and exc.retryable

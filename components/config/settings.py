"""Typed, validated, environment-driven settings.

Env vars use the prefix ``RAG_`` and ``__`` for nesting:
    RAG_CHUNKING__SIZE=800        ->  settings.chunking.size
    RAG_EMBEDDING__MODEL=...      ->  settings.embedding.model

Deliberate choices:
- Provider / model / dimensions have NO defaults. Your embedding model is part
  of your index's identity; silently defaulting one invites a mismatched index.
- Secrets are ``SecretStr``: they print as ``**********`` in repr and dumps.
- Unknown ``RAG_*`` keys in a .env file are an error (catches typos).
- No implicit .env loading: pass ``env_file=`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.exceptions import SettingsError

from .enums import DistanceMetric
from .errors import ConfigurationError

_Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Group(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IngestionSettings(_Group):
    """Resource limits for document ingestion (mirror the extractor's limits)."""

    max_file_size_bytes: Annotated[int, Field(gt=0)] = 100 * 1024 * 1024
    max_pages: Annotated[int, Field(gt=0)] = 2000
    max_total_chars: Annotated[int, Field(gt=0)] = 20_000_000


class ChunkingSettings(_Group):
    strategy: _Name = "recursive"  # name your chunker looks up; not validated against a list
    size: Annotated[int, Field(gt=0)] = 1000
    overlap: Annotated[int, Field(ge=0)] = 150
    unit: Literal["chars", "tokens"] = "chars"

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("chunking.overlap must be smaller than chunking.size")
        return self


class EmbeddingSettings(_Group):
    provider: _Name
    model: _Name
    dimensions: Annotated[int, Field(gt=0)]
    batch_size: Annotated[int, Field(gt=0, le=2048)] = 64
    timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    api_key: SecretStr | None = None


class VectorStoreSettings(_Group):
    provider: _Name
    collection: _Name = "rag_chunks"
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    url: str | None = None
    api_key: SecretStr | None = None


class RetrievalSettings(_Group):
    top_k: Annotated[int, Field(ge=1, le=100)] = 5
    score_threshold: float | None = None  # drop weaker hits; scale depends on your metric
    max_context_chars: Annotated[int, Field(gt=0)] = 12_000  # prompt-size guard


class LLMSettings(_Group):
    provider: _Name
    model: _Name
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    max_output_tokens: Annotated[int, Field(gt=0)] = 1024
    timeout_seconds: Annotated[float, Field(gt=0)] = 60.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    api_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        case_sensitive=False,
    )

    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings

    def index_fingerprint(self) -> str:
        """Short hash of every setting that changes what is stored in the index.

        Store it with your vectors. If the fingerprint of the running config
        differs from the stored one, the index is stale: re-embed / re-chunk
        instead of mixing incompatible vectors. Tuning-only settings (top_k,
        batch_size, timeouts, keys, LLM) are excluded on purpose.
        """
        payload: dict[str, Any] = {
            "embedding": [self.embedding.provider, self.embedding.model, self.embedding.dimensions],
            "chunking": [self.chunking.strategy, self.chunking.size, self.chunking.overlap, self.chunking.unit],
            "distance_metric": self.vector_store.distance_metric.value,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def safe_dump(self) -> dict[str, Any]:
        """JSON-safe dict with secrets masked. Use this for logging, never ``model_dump()``."""
        return self.model_dump(mode="json")


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load settings from the process environment (and ``env_file`` if given).

    Raises ConfigurationError listing the offending env var names. Messages never
    include the values you supplied, so secrets cannot leak through them.
    """
    try:
        return Settings(_env_file=env_file)  # type: ignore[call-arg]
    except ValidationError as exc:
        problems = []
        for err in exc.errors(include_input=False, include_url=False, include_context=False):
            loc = err["loc"]
            name = "RAG_" + "__".join(str(p).upper() for p in loc) if loc else "settings"
            problems.append({"setting": name, "problem": err["msg"]})
        text = "; ".join(f"{p['setting']}: {p['problem']}" for p in problems)
        raise ConfigurationError(
            f"Invalid configuration: {text}", code="invalid_config", details={"problems": problems}
        ) from None  # drop the pydantic context: it embeds input values
    except SettingsError as exc:
        raise ConfigurationError(f"Could not read settings: {exc}", code="unreadable_config") from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings from the environment. Call ``get_settings.cache_clear()`` in tests."""
    return load_settings()

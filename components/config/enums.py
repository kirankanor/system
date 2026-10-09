"""Small shared enums. Kept in their own file so errors.py and models.py can both import them."""

from __future__ import annotations

from enum import StrEnum


class PipelineStage(StrEnum):
    """Where in the RAG pipeline something happened. Stamped onto every error."""

    CONFIG = "config"
    INGESTION = "ingestion"      # file read, PDF/HTML/... extraction
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"        # vector store writes
    RETRIEVAL = "retrieval"      # vector store reads, rerank
    GENERATION = "generation"    # LLM call, prompt building
    EVALUATION = "evaluation"


class DistanceMetric(StrEnum):
    """Similarity metric of the vector index. Must match how your embeddings were trained."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"

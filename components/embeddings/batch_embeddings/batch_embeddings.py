"""Batch an embedder's calls, and fail loudly when a batch comes back wrong.

Most embedding APIs cap how many texts one request may carry, and a naive
``[embed([t]) for t in texts]`` turns one corpus into thousands of round
trips. This component wraps any object that satisfies
:class:`~embedding_protocol.EmbeddingProtocol` and splits the work into
batches, returning one vector per input text, in input order.

The real work here is not the slicing -- it is the checking. A provider that
returns a different number of vectors than it was asked for, or a vector of a
different width than its neighbours, produces an index that looks healthy and
silently misaligns every retrieval from then on. Both are detected at the
point they happen and raised as :class:`EmbeddingProviderError` naming the
batch, so the failure surfaces where it began instead of corrupting data
downstream.

A wrapped call that raises is re-raised as :class:`EmbeddingProviderError`
naming the batch index and how many texts it carried, with the original
exception chained as ``__cause__`` so nothing is lost.

Self-contained on purpose
-------------------------
The manifest declares no dependencies. This file imports the standard library
plus this area's own :mod:`embedding_protocol` contract, which sits beside it
in the same category -- nothing else. A copy of the two files runs alone.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Sequence

from embedding_protocol import (
    EmbeddingInputError,
    EmbeddingProtocol,
    EmbeddingProviderError,
    Vector,
    validate_texts,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "BatchEmbedder",
]

#: Batch size used when the caller does not name one. Deliberately modest: it
#: is a sensible default for most providers, and a caller who wants a
#: different trade-off says so.
DEFAULT_BATCH_SIZE = 32

#: Largest ``batch_size`` accepted. A larger value is almost always a caller's
#: arithmetic error rather than a real requirement, and it turns one bad call
#: into one enormous allocation. Rejected rather than capped, because a
#: silently reduced batch size is a silently changed performance profile.
MAX_BATCH_SIZE = 1024


class BatchEmbedder:
    """An embedder that splits calls into batches, and checks what comes back.

    Satisfies :class:`~embedding_protocol.EmbeddingProtocol` by delegation, so
    it composes: a batching wrapper around a remote embedder is still an
    embedder, and can be wrapped again.

    Args:
        embedder: Anything satisfying ``EmbeddingProtocol`` -- it must expose
            ``dimensions`` and ``model_id`` as attributes and an ``embed``
            method.
        batch_size: Texts per request; ``1 <= batch_size <=
            MAX_BATCH_SIZE``.

    Raises:
        EmbeddingInputError: ``embedder`` is not protocol-shaped, or
            ``batch_size`` is unusable. Raised at construction, so a bad
            configuration fails before any text is sent.
    """

    def __init__(self, embedder: Any, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        if not isinstance(embedder, EmbeddingProtocol):
            missing = [
                name
                for name in ("dimensions", "model_id", "embed")
                if not hasattr(embedder, name)
            ]
            raise EmbeddingInputError(
                "batch-embeddings: embedder does not satisfy the embedding "
                f"protocol; missing {', '.join(missing) if missing else 'its embed method'}. "
                "Wrap an object exposing dimensions, model_id and embed(texts)"
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise EmbeddingInputError(
                f"batch-embeddings: batch_size must be an int, got "
                f"{type(batch_size).__name__} ({batch_size!r})"
            )
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise EmbeddingInputError(
                f"batch-embeddings: batch_size must be between 1 and "
                f"{MAX_BATCH_SIZE}, got {batch_size}. A larger value is "
                "almost always an arithmetic error; raise MAX_BATCH_SIZE "
                "deliberately if it is genuinely needed"
            )
        self._embedder = embedder
        self.batch_size = batch_size

    @property
    def dimensions(self) -> int:
        """Delegated: the wrapped embedder owns the vector space."""
        return self._embedder.dimensions

    @property
    def model_id(self) -> str:
        """Delegated: the wrapped embedder owns the vector space."""
        return self._embedder.model_id

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed every text, in batches, returning one vector per text.

        Returns:
            ``len(result) == len(texts)``, in input order. An empty input
            returns ``[]`` without calling the wrapped embedder at all.

        Raises:
            EmbeddingInputError: ``texts`` is unusable, or ``batch_size`` is.
            EmbeddingProviderError: a batch raised, returned the wrong number
                of vectors, or returned vectors of differing widths.
        """
        validated = validate_texts(texts)
        if not validated:
            return []

        results: list[Vector] = []
        total = len(validated)
        for start in range(0, total, self.batch_size):
            batch = validated[start : start + self.batch_size]
            index = start // self.batch_size
            results.extend(self._call(batch, index, start))

        self._check_widths(results, total)
        return results

    def embed_one(self, text: str) -> Vector:
        """Embed a single text. Convenience over :meth:`embed`."""
        return self.embed([text])[0]

    def _call(self, batch: list[str], index: int, start: int) -> list[Vector]:
        """Make one provider call and sanity-check its shape."""
        label = (
            f"batch {index} ({len(batch)} text(s), inputs {start}.."
            f"{start + len(batch) - 1})"
        )
        try:
            raw = self._embedder.embed(batch)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise EmbeddingProviderError(
                f"batch-embeddings: {label} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(raw, (list, tuple)):
            raise EmbeddingProviderError(
                f"batch-embeddings: {label} returned "
                f"{type(raw).__name__}, expected a list of vectors"
            )
        if len(raw) != len(batch):
            raise EmbeddingProviderError(
                f"batch-embeddings: {label} returned {len(raw)} vector(s) for "
                f"{len(batch)} input(s). Every input must produce exactly one "
                "vector or the caller's texts and vectors fall out of step, "
                "and every retrieval from then on returns the wrong document"
            )
        return [self._as_vector(item, label) for item in raw]

    def _as_vector(self, item: Any, label: str) -> Vector:
        """Coerce one provider result to a list of numbers, or object."""
        if isinstance(item, (str, bytes)):
            raise EmbeddingProviderError(
                f"batch-embeddings: {label} returned a {type(item).__name__} "
                "where a vector was expected"
            )
        if isinstance(item, Sequence):
            for position, value in enumerate(item):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise EmbeddingProviderError(
                        f"batch-embeddings: {label} returned a non-numeric "
                        f"value at position {position}: {value!r}"
                    )
            return [float(value) for value in item]
        raise EmbeddingProviderError(
            f"batch-embeddings: {label} returned "
            f"{type(item).__name__}, expected a sequence of numbers"
        )

    def _check_widths(self, results: list[Vector], total: int) -> None:
        """Every vector must be the same width -- and the declared width."""
        if not results:
            return
        width = len(results[0])
        for position, vector in enumerate(results):
            if len(vector) != width:
                raise EmbeddingProviderError(
                    f"batch-embeddings: vector {position} is {len(vector)} "
                    f"wide but vector 0 is {width}. Vectors from different "
                    "calls, or from different models, cannot share an index"
                )
        declared = getattr(self._embedder, "dimensions", None)
        if isinstance(declared, int) and not isinstance(declared, bool):
            if width != declared:
                raise EmbeddingProviderError(
                    f"batch-embeddings: vectors came back {width} wide but "
                    f"{self.model_id!r} declares {declared} dimensions. The "
                    "provider is not serving the model the caller expects"
                )
        if len(results) != total:  # pragma: no cover - guarded above
            raise EmbeddingProviderError(
                f"batch-embeddings: produced {len(results)} vectors for "
                f"{total} input(s)"
            )

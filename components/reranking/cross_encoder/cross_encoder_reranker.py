"""Rerank candidates with a cross-encoder model, behind a guarded import.

What a cross-encoder is for
---------------------------
A bi-encoder (``embeddings``) embeds the query and each passage *separately*
and compares vectors. A cross-encoder reads the query and one passage
**together**, in a single forward pass, and scores how well they match. That
joint attention is why it is markedly more accurate -- and why it cannot
pre-compute anything: every query costs one pass per candidate, so it belongs
after retrieval, on a handful of candidates, not over a corpus.

That cost is the whole design of this component. It reranks a short list.

The dependency is genuinely optional, and genuinely guarded
----------------------------------------------------------
``sentence-transformers`` is heavy -- it drags in PyTorch, and a model
download on first use. This library's rule (DECISION-bf2566ea) is that such
libraries are never imported at module top level by shared code, so:

* importing this module never touches the wheel, and works without it;
* the import happens inside :meth:`CrossEncoderReranker.rerank`, on first use;
* a missing wheel raises ``MissingDependency`` -- an ``ImportError`` -- whose
  message is the manifest's ``missing_dependency_message`` **verbatim**, and
  names the package.

A caller can therefore list this reranker among several strategies, decide at
runtime whether the machine can run it, and fall back to
``relevance-ranking`` when it cannot.

Score direction
---------------
``CrossEncoder.predict`` returns a relevance logit per pair: **higher is more
relevant**. Unlike the raw output of some dense stores there is no direction
ambiguity here, and no normalisation is applied -- cross-encoder logits are
not comparable across models, so this component returns them untouched rather
than pretending otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core_types import ScoredDocument

__all__ = [
    "CrossEncoderReranker",
    "MissingDependency",
    "RerankingError",
    "RerankingInputError",
]

#: Read once at import; the wheel is never touched to produce this message.
MANIFEST_MESSAGE = json.loads(
    (Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8")
)["missing_dependency_message"]


class RerankingError(Exception):
    """Base for every failure in this component."""


class RerankingInputError(RerankingError):
    """The caller passed something this component cannot rerank."""


class MissingDependency(RerankingError, ImportError):
    """``sentence-transformers`` is not installed.

    Subclasses ``ImportError`` so a caller catching the standard exception
    works, and this component's own base so one ``except`` covers both.
    Deliberately *not* a subclass of ``ValueError`` or ``TypeError``: the
    machine is the problem, not the caller's arguments.
    """


def _load_sentence_transformers() -> Any:
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise MissingDependency(MANIFEST_MESSAGE) from exc
    return CrossEncoder


def _validate_documents(documents: object) -> list[ScoredDocument]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise RerankingInputError(
            f"cross-encoder-reranker: documents must be a sequence of "
            f"ScoredDocument, got {type(documents).__name__}"
        )
    items = list(documents)
    for position, item in enumerate(items):
        if not isinstance(item, ScoredDocument):
            raise RerankingInputError(
                f"cross-encoder-reranker: documents[{position}] is "
                f"{type(item).__name__}, expected ScoredDocument"
            )
    return items


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise RerankingInputError(
            f"cross-encoder-reranker: query must be a string, got "
            f"{type(query).__name__}"
        )
    if not query.strip():
        raise RerankingInputError(
            f"cross-encoder-reranker: query is empty or whitespace. Scoring "
            f"empty pairs would return real logits about nothing, which look "
            f"exactly like a good answer"
        )
    return query


def _validate_top_k(top_k: object, available: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RerankingInputError(
            f"cross-encoder-reranker: top_k must be an int, got "
            f"{type(top_k).__name__} ({top_k!r})"
        )
    if top_k < 1:
        raise RerankingInputError(
            f"cross-encoder-reranker: top_k must be at least 1 to return "
            f"anything, got {top_k}"
        )
    return min(top_k, available)


class CrossEncoderReranker:
    """Rerank a short candidate list with a cross-encoder.

    ``model_name`` is required rather than defaulted, because picking a model
    is a quality decision the caller should make deliberately; a hidden
    default would make results depend on which version of this file you have.

    Pass ``model=`` to supply an already-constructed cross-encoder. That is
    how callers who manage their own models (shared across rerankers, loaded
    at startup, pinned to a device) avoid a second load, and how these tests
    run without the wheel installed.

    The model is loaded lazily on first :meth:`rerank`, so constructing this
    object is free and never touches the dependency.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model: Any | None = None,
        device: str | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise RerankingInputError(
                f"cross-encoder-reranker: model_name must be a non-empty "
                f"string, got {model_name!r}"
            )
        if device is not None and not isinstance(device, str):
            raise RerankingInputError(
                f"cross-encoder-reranker: device must be a string such as "
                f"'cpu' or 'cuda', got {type(device).__name__}"
            )
        derived = name if name is not None else f"cross-encoder:{model_name}"
        if not isinstance(derived, str) or not derived:
            raise RerankingInputError(
                f"cross-encoder-reranker: name must be a non-empty string, "
                f"got {name!r}"
            )
        self.model_name = model_name
        self.device = device
        self.name = derived
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_sentence_transformers()(
                self.model_name, device=self.device
            )
        return self._model

    def is_loaded(self) -> bool:
        """Whether the model has been loaded, without triggering a load.

        Lets a caller inspect the cost before paying it.
        """
        return self._model is not None

    def rerank(
        self, query: str, documents: Sequence[ScoredDocument], top_k: int
    ) -> list[ScoredDocument]:
        """Score every candidate against ``query`` and return the best ``top_k``.

        Every candidate is scored, then truncated -- not the other way round.
        A cross-encoder's ranking of the top few is the point, and scoring a
        pre-truncated list would bake the first retriever's ordering into the
        answer.

        Ties resolve by ``chunk_index`` then ``chunk_id``, so two runs over the
        same candidates return the same order even when the model emits
        exactly equal logits, which it does.
        """
        text = _validate_query(query)
        items = _validate_documents(documents)
        limit = _validate_top_k(top_k, len(items))
        if not items:
            return []

        model = self._ensure_model()
        pairs = [[text, item.text] for item in items]
        try:
            raw = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - the model's own failure
            raise RerankingError(
                f"cross-encoder-reranker: scoring {len(pairs)} pair(s) with "
                f"{self.model_name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            scores = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise RerankingError(
                f"cross-encoder-reranker: the model returned "
                f"{type(raw).__name__}, which does not yield one number per "
                f"pair. Expected {len(pairs)} scores"
            ) from exc
        if len(scores) != len(items):
            raise RerankingError(
                f"cross-encoder-reranker: the model returned {len(scores)} "
                f"scores for {len(items)} document(s). One score per pair is "
                f"required, or the ranking would drop candidates silently"
            )

        ordered = sorted(
            zip(items, scores),
            key=lambda row: (-row[1], row[0].chunk_index, row[0].chunk_id or ""),
        )
        return [
            document.model_copy(update={"score": score})
            for document, score in ordered[:limit]
        ]

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker(model_name={self.model_name!r}, "
            f"name={self.name!r}, loaded={self.is_loaded()})"
        )

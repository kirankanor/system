"""Embed text with a locally-running model.

Runs the model in this process: no request leaves the machine, no API key is
needed, and the only network access that can happen at all is the *initial*
model download. Set ``TRANSFORMERS_OFFLINE=1`` to forbid even that, and the
component will fail rather than reach out -- which is the right behaviour on
an air-gapped host.

No default model
----------------
The manifest is explicit that the model id has no default, and this file
honours that: a model name must come from the caller or from
``RAG_LOCAL_EMBEDDINGS_MODEL``. A hidden default would mean the same call
produces different vectors on different machines, and vectors from different
models cannot share an index -- so the failure is raised at construction
instead, naming the variable to set.

Optional dependency (FR-3)
--------------------------
``sentence-transformers`` is declared *optional*, so it is imported lazily
inside the function that needs it. If it is missing the call raises
:class:`MissingEmbeddingDependency`, an ``ImportError`` carrying the
manifest's ``missing_dependency_message`` verbatim. Nothing here imports it at
module scope, which is what keeps this module importable on a machine that
never installed it.

Self-contained on purpose
-------------------------
Standard library only, plus this area's own :mod:`embedding_protocol`
contract, which sits beside it in the same category.
"""

from __future__ import annotations

import os
from numbers import Real
from typing import Any, Sequence

from embedding_protocol import (
    EmbeddingInputError,
    EmbeddingProviderError,
    EmbeddingProtocol,
    MissingEmbeddingDependency,
    Vector,
    validate_texts,
)

__all__ = [
    "MODEL_ENV_VAR",
    "MISSING_DEPENDENCY_MESSAGE",
    "LocalEmbedder",
]

#: Environment variable naming the model when the caller does not.
MODEL_ENV_VAR = "RAG_LOCAL_EMBEDDINGS_MODEL"

#: Copied verbatim from the component manifest's ``missing_dependency_message``.
#: The manifest validator cannot read a string out of a .py file, so this is
#: asserted against the manifest in the component's own test.
MISSING_DEPENDENCY_MESSAGE = (
    "local-embeddings requires 'sentence-transformers', which is not "
    "installed. Install it with: uv add sentence-transformers"
)


def _load_model(model_name: str) -> Any:
    """Import ``sentence-transformers`` and build the model -- the heavy import.

    Deliberately a function rather than a module-level import: FR-3 wants the
    optional dependency guarded at the point of use, and a fresh lookup is
    what lets a test prove the guard fires by removing the module from
    ``sys.modules``.

    Raises:
        MissingEmbeddingDependency: ``sentence-transformers`` is absent (FR-3).
        EmbeddingProviderError: the model could not be built -- typically an
            unknown model id, or a download refused by ``TRANSFORMERS_OFFLINE``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise MissingEmbeddingDependency(MISSING_DEPENDENCY_MESSAGE) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - provider failure, re-raised
        raise EmbeddingProviderError(
            f"local-embeddings: could not load model {model_name!r}: "
            f"{type(exc).__name__}: {exc}. Check the id, and that the model "
            "is cached locally if TRANSFORMERS_OFFLINE=1 is set"
        ) from exc


class LocalEmbedder:
    """An embedder backed by a model running in this process.

    Satisfies :class:`~embedding_protocol.EmbeddingProtocol`.

    Args:
        model_name: A model id ``sentence-transformers`` recognises. Defaults
            to the ``RAG_LOCAL_EMBEDDINGS_MODEL`` environment variable; there
            is no fallback beyond that.
        env: Mapping consulted for the default, so a test can supply one
            without touching the real environment.

    Raises:
        EmbeddingInputError: no model id from either source.
    """

    def __init__(self, model_name: str | None = None, *, env: Any = os.environ) -> None:
        resolved = model_name if model_name is not None else env.get(MODEL_ENV_VAR)
        if not isinstance(resolved, str) or not resolved:
            raise EmbeddingInputError(
                f"local-embeddings: no model id. Pass model_name, or set "
                f"${MODEL_ENV_VAR}. There is deliberately no default: a "
                "hidden model would mean the same call returns different "
                "vectors on different machines, and vectors from different "
                "models cannot share an index"
            )
        self.model_id = resolved
        self._env = env
        self._model: Any = None

    @property
    def model(self) -> Any:
        """The underlying model, built on first use."""
        if self._model is None:
            self._model = _load_model(self.model_id)
        return self._model

    @property
    def dimensions(self) -> int:
        """Width of this model's vectors, as the model itself declares it."""
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} does not report "
                "its embedding dimension"
            )
        value = getter()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} reported "
                f"{value!r} as its embedding dimension"
            )
        return value

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed every text, returning one vector per text, in input order.

        Raises:
            EmbeddingInputError: ``texts`` is unusable.
            EmbeddingProviderError: the model returned the wrong number of
                vectors, or a vector that is not a list of numbers.
            MissingEmbeddingDependency: ``sentence-transformers`` is absent.
        """
        validated = validate_texts(texts)
        if not validated:
            return []

        try:
            raw = self.model.encode(validated)
        except MissingEmbeddingDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - provider failure, re-raised
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} failed to embed "
                f"{len(validated)} text(s): {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(raw, (list, tuple)):
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} returned "
                f"{type(raw).__name__}, expected a list of vectors"
            )
        if len(raw) != len(validated):
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} returned "
                f"{len(raw)} vector(s) for {len(validated)} text(s). Every "
                "input must produce exactly one vector or the caller's texts "
                "and vectors fall out of step"
            )
        return [self._as_vector(item) for item in raw]

    def embed_one(self, text: str) -> Vector:
        """Embed a single text. Convenience over :meth:`embed`."""
        return self.embed([text])[0]

    def _as_vector(self, item: Any) -> Vector:
        if isinstance(item, (str, bytes)):
            raise EmbeddingProviderError(
                f"local-embeddings: model {self.model_id!r} returned a "
                f"{type(item).__name__} where a vector was expected"
            )
        if isinstance(item, Sequence):
            for position, value in enumerate(item):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise EmbeddingProviderError(
                        f"local-embeddings: model {self.model_id!r} returned "
                        f"a non-numeric value at position {position}: "
                        f"{value!r}"
                    )
            return [float(value) for value in item]
        raise EmbeddingProviderError(
            f"local-embeddings: model {self.model_id!r} returned "
            f"{type(item).__name__}, expected a sequence of numbers"
        )

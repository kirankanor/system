"""Framework-neutral embedding contracts shared by the whole area.

An embedder turns text into vectors in one fixed vector space. Everything
downstream -- indexing, retrieval, scoring -- is only meaningful when those
vectors come from the *same* space, which is why the space is described by two
attributes rather than left implicit: ``dimensions`` says how wide a vector is,
and ``model_id`` says which model produced it. Two embedders are comparable
only when both match.

Why a separate contract component
---------------------------------
:mod:`core_types` already declares an ``EmbeddingProtocol``. This file
declares one too, deliberately: the manifest for this component declares no
dependencies, so it must be copy-ready on its own. The two agree on the
attribute and method *names* -- ``dimensions``, ``model_id`` and
``embed(texts)`` -- so a caller written against either shape works with the
implementations here unchanged. That is the same deliberate, temporary
duplication :mod:`core_types` documents for its own ``Chunk``.

The error hierarchy lives here rather than in each adapter so that every
embedder in the area fails the same way. A caller can catch
:class:`EmbeddingError` once and handle the whole area, or narrow to
:class:`EmbeddingInputError` when it is at fault and
:class:`EmbeddingProviderError` when the provider is.

Self-contained on purpose
-------------------------
Standard library only: ``typing`` for the Protocol. No optional packages, no
network, no imports from the rest of the library.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = [
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingProtocol",
    "EmbeddingProviderError",
    "MissingEmbeddingDependency",
    "Vector",
    "validate_texts",
]

#: One embedding: a list of floats. Aliased rather than a ``NewType`` so it
#: stays interchangeable with whatever a vector store expects.
Vector = list[float]


class EmbeddingError(Exception):
    """Base class for every error raised in the embeddings area (FR-4).

    Catching this catches every failure below it. Nothing here is a bare
    ``ValueError`` or ``KeyError``, so a caller can tell an embedding
    component's failures apart from its own programming errors.
    """


class EmbeddingInputError(EmbeddingError):
    """The *caller* passed something unusable -- bad text, a missing setting.

    Fixing these means changing the call, not the provider.
    """


class EmbeddingProviderError(EmbeddingError):
    """The *provider* failed -- a rejected request, a bad response, a model
    that would not load.

    Fixing these means changing the provider, the credentials, or the
    environment.
    """


class MissingEmbeddingDependency(ImportError):
    """An optional dependency is not installed (FR-3).

    Deliberately an ``ImportError`` and deliberately *not* an
    :class:`EmbeddingError`: callers guard optional components with
    ``except ImportError``, and swallowing it under a domain error would hide
    the one failure that a ``ragkit add`` user most needs to see.
    """


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Turn text into vectors in one fixed vector space.

    ``dimensions`` and ``model_id`` describe the space: a vector store indexed
    with one space must not be queried with another, so implementations expose
    both as ordinary attributes rather than hiding them behind a method.

    ``embed`` returns one vector per input text, in input order. An empty
    input yields an empty result rather than an error -- there is nothing to
    do.
    """

    dimensions: int
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def validate_texts(texts: Any) -> list[str]:
    """Check embedding input before any provider is touched (FR-4).

    Rejects, in this order: anything that is not a sequence, a bare ``str``
    or ``bytes`` (both *are* sequences of characters, so accepting them would
    silently embed one vector per character), a non-string element, and an
    empty string. An empty *list* is not an error -- it is the caller saying
    there is nothing to embed, and the answer is ``[]``.

    Args:
        texts: The candidate input.

    Returns:
        The input as a ``list``, so callers can index and slice it safely.

    Raises:
        EmbeddingInputError: with a message naming what was wrong.
    """
    if isinstance(texts, (str, bytes, bytearray)):
        raise EmbeddingInputError(
            f"embedding-protocol: texts must be a sequence of strings, got a "
            f"{type(texts).__name__} ({texts!r:.60}); a str is itself a "
            "sequence of characters, so passing it would embed one vector per "
            "character. Wrap it: [texts]"
        )
    if not isinstance(texts, Sequence):
        raise EmbeddingInputError(
            f"embedding-protocol: texts must be a sequence of strings, got "
            f"{type(texts).__name__}"
        )

    validated: list[str] = []
    for position, item in enumerate(texts):
        if isinstance(item, (bytes, bytearray)):
            raise EmbeddingInputError(
                f"embedding-protocol: texts[{position}] is a "
                f"{type(item).__name__}, not a str"
            )
        if not isinstance(item, str):
            raise EmbeddingInputError(
                f"embedding-protocol: texts[{position}] must be a str, got "
                f"{type(item).__name__} ({item!r:.60})"
            )
        if not item:
            raise EmbeddingInputError(
                f"embedding-protocol: texts[{position}] is empty. An empty "
                "chunk means the pipeline upstream produced nothing; "
                "embedding it would add a vector with no meaning to the index"
            )
        validated.append(item)
    return validated

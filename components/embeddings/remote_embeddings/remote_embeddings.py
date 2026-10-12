"""Embed text through a remote provider's HTTP API.

Nothing about the provider is assumed. There is no default endpoint, no
default model and no default key: the manifest is explicit that all three
must be set, and this component raises at construction naming whichever is
missing rather than calling out to something the caller never chose. A hidden
default here would mean an accidental request to somebody else's service,
carrying the caller's text.

Configuration comes from the arguments or, failing that, the environment:
``RAG_EMBEDDINGS_URL``, ``RAG_EMBEDDINGS_API_KEY`` and
``RAG_EMBEDDINGS_MODEL``.

What it checks on the way back
------------------------------
A response that parses as JSON is not the same as a response that is usable.
This component verifies, in order, that the payload carries ``data``, that
every entry carries an ``index`` and an ``embedding``, that the indices are
exactly ``0..n-1`` with no gaps or repeats, and that the count matches the
number of texts sent. Providers do reorder entries, and a silently reordered
response attaches every vector to the wrong document -- so the indices are
honoured rather than assumed to arrive in order.

Optional dependency (FR-3)
--------------------------
``httpx`` is declared *optional*, imported lazily inside the function that
needs it. If missing, the call raises :class:`MissingEmbeddingDependency`, an
``ImportError`` carrying the manifest's ``missing_dependency_message``
verbatim. Nothing here imports it at module scope.

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
    "DEFAULT_TIMEOUT",
    "URL_ENV_VAR",
    "API_KEY_ENV_VAR",
    "MODEL_ENV_VAR",
    "MISSING_DEPENDENCY_MESSAGE",
    "RemoteEmbedder",
]

#: Seconds before an outstanding request is abandoned. Explicit rather than a
#: library default so a hung provider fails the pipeline instead of the
#: pipeline hanging.
DEFAULT_TIMEOUT = 30.0

URL_ENV_VAR = "RAG_EMBEDDINGS_URL"
API_KEY_ENV_VAR = "RAG_EMBEDDINGS_API_KEY"
MODEL_ENV_VAR = "RAG_EMBEDDINGS_MODEL"

#: Copied verbatim from the component manifest's ``missing_dependency_message``.
MISSING_DEPENDENCY_MESSAGE = (
    "remote-embeddings requires 'httpx', which is not installed. Install it "
    "with: uv add httpx"
)


def _load_client(timeout: float) -> Any:
    """Import ``httpx`` and build a client -- the only heavy import.

    Deliberately a function rather than a module-level import: FR-3 wants the
    optional dependency guarded at the point of use, and a fresh lookup is
    what lets a test prove the guard fires.

    Raises:
        MissingEmbeddingDependency: ``httpx`` is absent (FR-3).
    """
    try:
        import httpx
    except ImportError as exc:
        raise MissingEmbeddingDependency(MISSING_DEPENDENCY_MESSAGE) from exc
    return httpx.Client(timeout=timeout)


class RemoteEmbedder:
    """An embedder that calls a remote HTTP API.

    Satisfies :class:`~embedding_protocol.EmbeddingProtocol`.

    Args:
        url: Endpoint accepting a JSON POST. Defaults to ``$RAG_EMBEDDINGS_URL``.
        api_key: Sent as ``Authorization: Bearer ...``. Defaults to
            ``$RAG_EMBEDDINGS_API_KEY``.
        model: Provider model name sent in the request body. Defaults to
            ``$RAG_EMBEDDINGS_MODEL``.
        dimensions: Width of this model's vectors, if known in advance. When
            omitted it is measured from the first successful response and
            cached -- before then, reading ``dimensions`` raises rather than
            guessing.
        timeout: Seconds before a request is abandoned.
        env: Mapping consulted for the defaults, so a test can supply one
            without touching the real environment.

    Raises:
        EmbeddingInputError: any of the three settings is missing, or
            ``dimensions`` is not a positive int.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        env: Any = os.environ,
    ) -> None:
        self.url = self._setting(url, URL_ENV_VAR, env, "endpoint URL")
        self.api_key = self._setting(api_key, API_KEY_ENV_VAR, env, "API key")
        self.model_id = self._setting(model, MODEL_ENV_VAR, env, "model name")
        if dimensions is not None and (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise EmbeddingInputError(
                f"remote-embeddings: dimensions must be a positive int, got "
                f"{dimensions!r}"
            )
        self._dimensions = dimensions
        self.timeout = timeout
        self._env = env
        self._client: Any = None

    @staticmethod
    def _setting(value: str | None, var: str, env: Any, label: str) -> str:
        resolved = value if value is not None else env.get(var)
        if not isinstance(resolved, str) or not resolved:
            raise EmbeddingInputError(
                f"remote-embeddings: no {label}. Pass it, or set ${var}. "
                "There is deliberately no default provider, endpoint or "
                "model: a hidden one means the caller's text is sent "
                "somewhere the caller never chose"
            )
        return resolved

    @property
    def client(self) -> Any:
        """The HTTP client, built on first use."""
        if self._client is None:
            self._client = _load_client(self.timeout)
        return self._client

    @property
    def dimensions(self) -> int:
        """Vector width -- declared at construction, else measured once."""
        if self._dimensions is None:
            raise EmbeddingInputError(
                f"remote-embeddings: vector width for {self.model_id!r} is "
                "not known before the first request. Either pass "
                "dimensions= at construction, or call embed() once first"
            )
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed every text, returning one vector per text, in input order.

        Raises:
            EmbeddingInputError: ``texts`` is unusable.
            EmbeddingProviderError: the request failed, or the response was
                not a usable embedding payload.
            MissingEmbeddingDependency: ``httpx`` is absent (FR-3).
        """
        validated = validate_texts(texts)
        if not validated:
            return []

        payload = self._post(validated)
        vectors = self._interpret(payload, len(validated))
        if self._dimensions is None:
            self._dimensions = len(vectors[0])
        elif len(vectors[0]) != self._dimensions:
            raise EmbeddingProviderError(
                f"remote-embeddings: {self.model_id!r} returned "
                f"{len(vectors[0])}-wide vectors but {self._dimensions} were "
                "declared. The provider is not serving the model the caller "
                "expects"
            )
        return vectors

    def embed_one(self, text: str) -> Vector:
        """Embed a single text. Convenience over :meth:`embed`."""
        return self.embed([text])[0]

    def _post(self, validated: list[str]) -> Any:
        """Send one request; translate every failure into a domain error."""
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model_id, "input": validated},
            )
        except MissingEmbeddingDependency:
            raise
        except Exception as exc:  # noqa: BLE001 - transport failure
            raise EmbeddingProviderError(
                f"remote-embeddings: request to {self.url} failed after "
                f"{self.timeout}s budget: {type(exc).__name__}: {exc}"
            ) from exc

        status = getattr(response, "status_code", None)
        if status != 200:
            raise EmbeddingProviderError(
                f"remote-embeddings: {self.url} returned HTTP {status}. "
                f"Body: {self._describe(response)}"
            )
        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001 - malformed payload
            raise EmbeddingProviderError(
                f"remote-embeddings: {self.url} returned a non-JSON body "
                f"for an embedding request. Body: {self._describe(response)}"
            ) from exc

    @staticmethod
    def _describe(response: Any, limit: int = 200) -> str:
        """A short, redacted-ish description of a response body for a message.

        Truncated, because a provider's error body can be arbitrarily long,
        and prefixed with the first two hundred characters only.
        """
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - nothing useful to show
            return "<unreadable>"
        if not isinstance(text, str):
            return "<unreadable>"
        clipped = text[:limit]
        return f"{clipped!r}{'...' if len(text) > limit else ''}"

    def _interpret(self, payload: Any, expected: int) -> list[Vector]:
        """Turn a provider payload into ordered vectors, honouring indices."""
        if not isinstance(payload, dict):
            raise EmbeddingProviderError(
                f"remote-embeddings: expected an object from {self.url}, got "
                f"{type(payload).__name__}"
            )
        data = payload.get("data")
        if not isinstance(data, list):
            raise EmbeddingProviderError(
                f"remote-embeddings: response from {self.url} carries no "
                "'data' list; it is not an embedding payload"
            )
        if len(data) != expected:
            raise EmbeddingProviderError(
                f"remote-embeddings: {self.url} returned {len(data)} "
                f"embedding(s) for {expected} input(s). Every input must "
                "produce exactly one vector or the caller's texts and "
                "vectors fall out of step"
            )

        ordered: list[tuple[int, Any]] = []
        for position, entry in enumerate(data):
            if not isinstance(entry, dict) or "embedding" not in entry:
                raise EmbeddingProviderError(
                    f"remote-embeddings: entry {position} of the response "
                    "has no 'embedding'"
                )
            if "index" not in entry:
                raise EmbeddingProviderError(
                    f"remote-embeddings: entry {position} of the response "
                    "has no 'index'. Without it the order cannot be verified, "
                    "and a reordered response attaches every vector to the "
                    "wrong document"
                )
            index = entry["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise EmbeddingProviderError(
                    f"remote-embeddings: entry {position} has a non-integer "
                    f"'index': {index!r}"
                )
            ordered.append((index, entry["embedding"]))

        ordered.sort(key=lambda pair: pair[0])
        indices = [index for index, _ in ordered]
        if indices != list(range(expected)):
            raise EmbeddingProviderError(
                f"remote-embeddings: response indices are {indices}, expected "
                f"0..{expected - 1}. Duplicated or gapped indices mean the "
                "ordering is ambiguous, so no vector can be safely matched to "
                "its text"
            )
        return [self._as_vector(item) for _, item in ordered]

    def _as_vector(self, item: Any) -> Vector:
        if isinstance(item, (str, bytes)):
            raise EmbeddingProviderError(
                f"remote-embeddings: a {type(item).__name__} was returned "
                "where a vector was expected"
            )
        if isinstance(item, Sequence):
            for position, value in enumerate(item):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise EmbeddingProviderError(
                        f"remote-embeddings: a non-numeric value at position "
                        f"{position}: {value!r}"
                    )
            return [float(value) for value in item]
        raise EmbeddingProviderError(
            f"remote-embeddings: {type(item).__name__} was returned, "
            "expected a sequence of numbers"
        )

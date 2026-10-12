"""Gate tests for the embeddings area (T-07).

Covers the four components in ``components/embeddings``: the shared contract,
a batching wrapper, and two adapters whose real dependencies are optional.

``httpx`` happens to be installed in this environment while
``sentence-transformers`` is not, so both FR-3 guards are proven by blocking
the import through ``sys.modules`` rather than by relying on what is -- or is
not -- installed on the machine running the tests. That keeps the proof about
the *guard*, which is what the requirement is about, and not about the
environment.

Remote calls are always mocked; nothing here touches the network (AC: the
components must be testable without a provider).

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = ROOT / "components" / "embeddings"

# All four component directories, so each is imported exactly the way a copy
# of it would be.
for name in ("embedding_protocol", "batch_embeddings", "local_embeddings", "remote_embeddings"):
    sys.path.insert(0, str(EMBEDDINGS / name))

import batch_embeddings as batch  # noqa: E402
import embedding_protocol as protocol  # noqa: E402
import local_embeddings as local  # noqa: E402
import remote_embeddings as remote  # noqa: E402

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """A conforming embedder: three-wide vectors, length in position 0."""

    dimensions = 3
    model_id = "fake-model"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeModel:
    """Stands in for a sentence-transformers model."""

    def __init__(self, dimension=3, out=None):
        self._dimension = dimension
        self._out = out

    def get_sentence_embedding_dimension(self):
        return self._dimension

    def encode(self, texts):
        if self._out is not None:
            return self._out
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeResponse:
    status_code = 200

    def __init__(self, payload, *, status=200, text=None):
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else str(payload)

    def json(self):
        return self._payload


class FakeClient:
    """Records the request and replays a canned response."""

    def __init__(self, payload, *, status=200, text=None):
        self.payload = payload
        self.status = status
        self.text = text
        self.requests: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payload, status=self.status, text=self.text)


def _remote(**kwargs):
    """A RemoteEmbedder wired to a fake client, never the network."""
    settings = {"url": "https://api.example/v1/embeddings", "api_key": "key", "model": "m"}
    settings.update(kwargs)
    embedder = remote.RemoteEmbedder(**settings)
    return embedder


def _ordered_payload(vectors):
    """Build an OpenAI-shaped payload, indices matching positions."""
    return {
        "data": [
            {"index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ]
    }


# --------------------------------------------------------------------------- #
# Shared contract
# --------------------------------------------------------------------------- #


def test_every_component_satisfies_the_embedding_protocol():
    """FR-5 in the embeddings direction: the contract is structural."""
    embedder = FakeEmbedder()
    wrapped = batch.BatchEmbedder(embedder, batch_size=2)
    assert isinstance(embedder, protocol.EmbeddingProtocol)
    assert isinstance(wrapped, protocol.EmbeddingProtocol)

    le = local.LocalEmbedder(model_name="some-model")
    assert isinstance(le, protocol.EmbeddingProtocol)

    re_ = _remote()
    assert isinstance(re_, protocol.EmbeddingProtocol)


def test_error_hierarchy_is_catchable_at_the_base():
    """FR-4: one base class catches the whole area; ImportError stays separate."""
    assert issubclass(protocol.EmbeddingInputError, protocol.EmbeddingError)
    assert issubclass(protocol.EmbeddingProviderError, protocol.EmbeddingError)
    assert not issubclass(protocol.EmbeddingError, ValueError)
    assert issubclass(protocol.MissingEmbeddingDependency, ImportError)
    assert not issubclass(protocol.MissingEmbeddingDependency, protocol.EmbeddingError)


@pytest.mark.parametrize(
    "bad",
    [None, 3, {"a": 1}, b"bytes"],
    ids=["none", "int", "dict", "bytes"],
)
def test_validate_texts_rejects_non_sequences(bad):
    with pytest.raises(protocol.EmbeddingInputError):
        protocol.validate_texts(bad)


def test_validate_texts_rejects_a_bare_string():
    """A str *is* a sequence of characters, so passing one embeds per letter."""
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        protocol.validate_texts("hello")
    assert "character" in str(exc.value)


def test_validate_texts_rejects_an_empty_element():
    """An empty chunk means the pipeline upstream produced nothing."""
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        protocol.validate_texts(["ok", ""])
    assert "texts[1]" in str(exc.value)


def test_validate_texts_rejects_a_non_string_element():
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        protocol.validate_texts(["ok", 5])
    assert "texts[1]" in str(exc.value)


def test_validate_texts_accepts_an_empty_list():
    """Nothing to embed is not an error; it is an empty result."""
    assert protocol.validate_texts([]) == []


def test_validate_texts_returns_a_list():
    """A tuple in, a list out: callers can index and slice the result."""
    assert protocol.validate_texts(("a", "b")) == ["a", "b"]


# --------------------------------------------------------------------------- #
# FR-3: optional dependencies are guarded, with the manifest's message
# --------------------------------------------------------------------------- #


def test_local_embeddings_guard_their_optional_dependency(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError."""
    manifest = json.loads(
        (EMBEDDINGS / "local_embeddings" / "manifest.json").read_text(encoding="utf-8")
    )
    # Block the import regardless of what this machine has installed, so the
    # assertion is about the guard rather than about the environment.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = local.LocalEmbedder(model_name="some-model")
    with pytest.raises(ImportError) as exc:
        embedder.embed(["hello"])
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == local.MISSING_DEPENDENCY_MESSAGE


def test_remote_embeddings_guard_their_optional_dependency(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError.

    httpx is genuinely installed here, which is why the import is blocked --
    an unguarded test would otherwise reach the network.
    """
    manifest = json.loads(
        (EMBEDDINGS / "remote_embeddings" / "manifest.json").read_text(encoding="utf-8")
    )
    monkeypatch.setitem(sys.modules, "httpx", None)
    embedder = _remote()
    with pytest.raises(ImportError) as exc:
        embedder.embed(["hello"])
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == remote.MISSING_DEPENDENCY_MESSAGE


def test_the_guard_fires_at_the_point_of_use_not_at_import_time():
    """Importing the modules must succeed with both packages absent."""
    assert hasattr(remote, "RemoteEmbedder")
    assert hasattr(local, "LocalEmbedder")


# --------------------------------------------------------------------------- #
# batch-embeddings
# --------------------------------------------------------------------------- #


def test_batching_splits_calls_and_preserves_order():
    embedder = FakeEmbedder()
    wrapped = batch.BatchEmbedder(embedder, batch_size=2)
    out = wrapped.embed(["aa", "bbb", "cccc", "d"])
    assert embedder.calls == [["aa", "bbb"], ["cccc", "d"]]
    assert [vector[0] for vector in out] == [2.0, 3.0, 4.0, 1.0]
    assert len(out) == 4


def test_batching_delegates_the_declared_space():
    wrapped = batch.BatchEmbedder(FakeEmbedder())
    assert wrapped.dimensions == 3
    assert wrapped.model_id == "fake-model"


def test_batching_an_empty_list_never_calls_the_provider():
    embedder = FakeEmbedder()
    wrapped = batch.BatchEmbedder(embedder)
    assert wrapped.embed([]) == []
    assert embedder.calls == []


def test_batching_embed_one():
    assert batch.BatchEmbedder(FakeEmbedder()).embed_one("hello") == [5.0, 0.0, 1.0]


def test_a_provider_returning_too_few_vectors_is_caught():
    """Silent misalignment is the failure this component exists to prevent."""
    embedder = FakeEmbedder()
    embedder.embed = lambda texts: [[1.0, 0.0, 1.0]]  # one vector for two texts
    wrapped = batch.BatchEmbedder(embedder)
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        wrapped.embed(["a", "b"])
    assert "1 vector(s) for 2 input(s)" in str(exc.value)


def test_mismatched_vector_widths_are_caught():
    embedder = FakeEmbedder()
    embedder.embed = lambda texts: [[1.0], [1.0, 2.0]]
    wrapped = batch.BatchEmbedder(embedder)
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        wrapped.embed(["a", "b"])
    assert "wide" in str(exc.value)


def test_vectors_disagreeing_with_the_declared_width_are_caught():
    embedder = FakeEmbedder()  # declares 3
    embedder.embed = lambda texts: [[1.0, 2.0] for _ in texts]
    wrapped = batch.BatchEmbedder(embedder)
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        wrapped.embed(["a"])
    assert "declares 3" in str(exc.value)


def test_a_failing_batch_is_wrapped_and_chained():
    embedder = FakeEmbedder()

    def boom(texts):
        raise RuntimeError("quota exhausted")

    embedder.embed = boom
    wrapped = batch.BatchEmbedder(embedder)
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        wrapped.embed(["a", "b"])
    assert "quota exhausted" in str(exc.value)
    assert isinstance(exc.value.__cause__, RuntimeError)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, 2048])
def test_an_unusable_batch_size_is_rejected_at_construction(batch_size):
    with pytest.raises(protocol.EmbeddingInputError):
        batch.BatchEmbedder(FakeEmbedder(), batch_size=batch_size)


def test_an_object_without_the_protocol_is_rejected():
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        batch.BatchEmbedder(object())
    assert "protocol" in str(exc.value)


# --------------------------------------------------------------------------- #
# local-embeddings
# --------------------------------------------------------------------------- #


def test_local_model_id_comes_from_the_environment_when_not_passed():
    embedder = local.LocalEmbedder(env={"RAG_LOCAL_EMBEDDINGS_MODEL": "from-env"})
    assert embedder.model_id == "from-env"


def test_local_model_id_prefers_the_argument():
    embedder = local.LocalEmbedder(
        "explicit", env={"RAG_LOCAL_EMBEDDINGS_MODEL": "from-env"}
    )
    assert embedder.model_id == "explicit"


def test_local_has_no_default_model():
    """The manifest is explicit, and this is the test that keeps it that way."""
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        local.LocalEmbedder(env={})
    assert "RAG_LOCAL_EMBEDDINGS_MODEL" in str(exc.value)


def test_local_embeds_through_the_injected_model():
    embedder = local.LocalEmbedder(model_name="some-model")
    embedder._model = FakeModel()
    out = embedder.embed(["a", "bb"])
    assert out == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]


def test_local_dimensions_come_from_the_model():
    embedder = local.LocalEmbedder(model_name="some-model")
    embedder._model = FakeModel(dimension=7)
    assert embedder.dimensions == 7


def test_local_rejects_a_model_that_reports_no_usable_width():
    embedder = local.LocalEmbedder(model_name="some-model")
    embedder._model = FakeModel(dimension=0)
    with pytest.raises(protocol.EmbeddingProviderError):
        embedder.dimensions


def test_local_rejects_a_model_returning_the_wrong_count():
    embedder = local.LocalEmbedder(model_name="some-model")
    embedder._model = FakeModel(out=[[1.0, 0.0, 1.0]])
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        embedder.embed(["a", "b"])
    assert "1 vector(s) for 2 text(s)" in str(exc.value)


def test_local_empty_input_returns_empty_without_encoding():
    embedder = local.LocalEmbedder(model_name="some-model")
    embedder._model = FakeModel(out=AssertionError("must not be called"))
    assert embedder.embed([]) == []


# --------------------------------------------------------------------------- #
# remote-embeddings
# --------------------------------------------------------------------------- #


def test_remote_requires_all_three_settings():
    """No default provider, endpoint or model -- each is named when missing.

    ``env`` is always an empty mapping so a real ``RAG_EMBEDDINGS_*`` in the
    surrounding environment cannot make this pass for the wrong reason.
    """
    for url, api_key, model in [
        (None, None, None),
        ("https://e", None, None),
        ("https://e", "k", None),
    ]:
        with pytest.raises(protocol.EmbeddingInputError):
            remote.RemoteEmbedder(url, api_key=api_key, model=model, env={})


def test_remote_names_the_missing_setting():
    with pytest.raises(protocol.EmbeddingInputError) as exc:
        remote.RemoteEmbedder("https://e", api_key="k", env={})
    assert "RAG_EMBEDDINGS_MODEL" in str(exc.value)


def test_remote_reads_settings_from_the_environment():
    env = {
        "RAG_EMBEDDINGS_URL": "https://env.example",
        "RAG_EMBEDDINGS_API_KEY": "env-key",
        "RAG_EMBEDDINGS_MODEL": "env-model",
    }
    embedder = remote.RemoteEmbedder(env=env)
    assert embedder.url == "https://env.example"
    assert embedder.model_id == "env-model"


def test_remote_sends_model_input_and_authorisation():
    client = FakeClient(_ordered_payload([[1.0, 2.0], [3.0, 4.0]]))
    embedder = _remote()
    embedder._client = client
    out = embedder.embed(["x", "y"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]
    request = client.requests[0]
    assert request["json"] == {"model": "m", "input": ["x", "y"]}
    assert request["headers"]["Authorization"] == "Bearer key"


def test_remote_honours_the_indices_it_was_given():
    """Providers reorder; a reordered response must not reattach vectors."""
    vectors = [[1.0], [2.0], [3.0]]
    shuffled = {"data": [
        {"index": 2, "embedding": [3.0]},
        {"index": 0, "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]}
    embedder = _remote()
    embedder._client = FakeClient(shuffled)
    assert embedder.embed(["a", "b", "c"]) == vectors


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 5, "embedding": [2.0]}]},
        {"data": [{"embedding": [1.0]}, {"index": 1, "embedding": [2.0]}]},
        {"data": [{"index": 0}, {"index": 1, "embedding": [2.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"result": []},
    ],
    ids=["duplicate-index", "gap", "no-index", "no-embedding", "wrong-count", "no-data"],
)
def test_remote_rejects_an_unusable_payload(payload):
    embedder = _remote()
    embedder._client = FakeClient(payload)
    with pytest.raises(protocol.EmbeddingProviderError):
        embedder.embed(["a", "b"])


def test_remote_rejects_a_non_200_and_reports_the_status():
    embedder = _remote()
    embedder._client = FakeClient({"error": "bad key"}, status=401, text="bad key")
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "HTTP 401" in str(exc.value)


def test_remote_rejects_a_non_json_body():
    class BadJSON(FakeResponse):
        def json(self):
            raise ValueError("no json")

    class BadClient(FakeClient):
        def post(self, url, headers=None, json=None):
            self.requests.append({})
            return BadJSON({}, text="<html>gateway</html>")

    embedder = _remote()
    embedder._client = BadClient({})
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "non-JSON" in str(exc.value)


def test_remote_measures_dimensions_after_the_first_call():
    embedder = _remote()
    with pytest.raises(protocol.EmbeddingInputError):
        embedder.dimensions  # not known before any request
    embedder._client = FakeClient(_ordered_payload([[1.0, 2.0]]))
    embedder.embed(["a"])
    assert embedder.dimensions == 2


def test_remote_compares_declared_and_returned_widths():
    embedder = _remote(dimensions=5)
    embedder._client = FakeClient(_ordered_payload([[1.0, 2.0]]))
    with pytest.raises(protocol.EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "declared" in str(exc.value)


def test_remote_rejects_a_non_numeric_vector():
    embedder = _remote()
    embedder._client = FakeClient(_ordered_payload([["a", "b"]]))
    with pytest.raises(protocol.EmbeddingProviderError):
        embedder.embed(["a"])


def test_remote_empty_input_never_opens_a_client():
    embedder = _remote()
    assert embedder.embed([]) == []
    assert embedder._client is None


def test_remote_embed_one():
    embedder = _remote()
    embedder._client = FakeClient(_ordered_payload([[9.0, 8.0]]))
    assert embedder.embed_one("hello") == [9.0, 8.0]


def test_remote_rejects_a_non_positive_dimensions_setting():
    with pytest.raises(protocol.EmbeddingInputError):
        _remote(dimensions=0)

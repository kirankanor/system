"""Tests for the remote-embeddings component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

Every response here is mocked. The HTTP path is never executed, so the suite
depends on no provider being available, reachable or paid for -- and a machine
with `httpx` installed is not tempted to make a real request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports the embedding-protocol contract, which lives in its
# own component directory. The manifest declares it a dependency, so `ragkit
# add` copies it beside this file; here we reach across to the sibling.
sys.path.insert(0, str(COMPONENT.parent / "embedding_protocol"))
sys.path.insert(0, str(COMPONENT))

import remote_embeddings  # noqa: E402
from embedding_protocol import (  # noqa: E402
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProtocol,
    EmbeddingProviderError,
    MissingEmbeddingDependency,
)
from remote_embeddings import (  # noqa: E402
    API_KEY_ENV_VAR,
    MISSING_DEPENDENCY_MESSAGE,
    MODEL_ENV_VAR,
    URL_ENV_VAR,
    RemoteEmbedder,
)

import pytest  # noqa: E402

URL = "https://api.example/v1/embeddings"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


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
        self._payload = payload
        self._status = status
        self._text = text
        self.requests: list[dict] = []

    def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self._payload, status=self._status, text=self._text)


class NoJSON(FakeResponse):
    def json(self):
        raise ValueError("not json")


def _payload(vectors):
    """An OpenAI-shaped payload, indices matching positions."""
    return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


def _embedder(**kwargs):
    settings = {"url": URL, "api_key": "key", "model": "m"}
    settings.update(kwargs)
    return RemoteEmbedder(**settings)


def _with_client(payload, *, status=200, text=None, **kwargs):
    embedder = _embedder(**kwargs)
    embedder._client = FakeClient(payload, status=status, text=text)
    return embedder


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_embedder_satisfies_the_embedding_protocol():
    assert isinstance(_embedder(), EmbeddingProtocol)


@pytest.mark.parametrize(
    "url,api_key,model",
    [(None, None, None), (URL, None, None), (URL, "k", None)],
    ids=["nothing", "url-only", "url-and-key"],
)
def test_all_three_settings_are_required(url, api_key, model):
    """No default provider, endpoint or model. env is empty so a real
    RAG_EMBEDDINGS_* in the surrounding environment cannot make this pass."""
    with pytest.raises(EmbeddingInputError):
        RemoteEmbedder(url, api_key=api_key, model=model, env={})


def test_the_missing_setting_message_names_the_variable():
    with pytest.raises(EmbeddingInputError) as exc:
        RemoteEmbedder(URL, api_key="k", env={})
    assert MODEL_ENV_VAR in str(exc.value)


def test_settings_come_from_the_environment():
    env = {
        URL_ENV_VAR: "https://env.example",
        API_KEY_ENV_VAR: "env-key",
        MODEL_ENV_VAR: "env-model",
    }
    embedder = RemoteEmbedder(env=env)
    assert embedder.url == "https://env.example"
    assert embedder.model_id == "env-model"


def test_an_argument_beats_the_environment():
    env = {URL_ENV_VAR: "https://env.example", API_KEY_ENV_VAR: "e", MODEL_ENV_VAR: "e"}
    embedder = RemoteEmbedder("https://explicit", api_key="k", model="m", env=env)
    assert embedder.url == "https://explicit"
    assert embedder.model_id == "m"


def test_an_empty_setting_is_rejected():
    with pytest.raises(EmbeddingInputError):
        RemoteEmbedder("", api_key="k", model="m", env={})


def test_a_non_positive_dimensions_setting_is_rejected():
    with pytest.raises(EmbeddingInputError):
        _embedder(dimensions=0)


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_the_request_carries_the_model_the_texts_and_the_token():
    client = FakeClient(_payload([[1.0, 2.0], [3.0, 4.0]]))
    embedder = _with_client([[1.0, 2.0]], dimensions=2)
    embedder._client = client
    embedder.embed(["x", "y"])
    request = client.requests[0]
    assert request["url"] == URL
    assert request["json"] == {"model": "m", "input": ["x", "y"]}
    assert request["headers"]["Authorization"] == "Bearer key"


def test_vectors_come_back_in_index_order():
    embedder = _with_client(_payload([[1.0, 2.0], [3.0, 4.0]]), dimensions=2)
    assert embedder.embed(["x", "y"]) == [[1.0, 2.0], [3.0, 4.0]]


def test_a_reordered_response_is_corrected():
    """Providers do reorder; an unnoticed reorder reattaches every vector."""
    shuffled = {"data": [
        {"index": 2, "embedding": [3.0, 30.0]},
        {"index": 0, "embedding": [1.0, 10.0]},
        {"index": 1, "embedding": [2.0, 20.0]},
    ]}
    embedder = _with_client(shuffled, dimensions=2)
    assert embedder.embed(["a", "b", "c"]) == [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ]


def test_an_empty_input_never_opens_a_client():
    embedder = _embedder()
    assert embedder.embed([]) == []
    assert embedder._client is None


def test_embed_one():
    embedder = _with_client(_payload([[9.0, 8.0]]), dimensions=2)
    assert embedder.embed_one("hello") == [9.0, 8.0]


def test_a_tuple_of_texts_is_accepted():
    embedder = _with_client(_payload([[1.0, 2.0], [3.0, 4.0]]), dimensions=2)
    assert embedder.embed(("a", "b")) == [[1.0, 2.0], [3.0, 4.0]]


def test_dimensions_are_measured_after_the_first_call():
    embedder = _with_client(_payload([[1.0, 2.0, 3.0]]))
    assert embedder.embed(["a"]) == [[1.0, 2.0, 3.0]]
    assert embedder.dimensions == 3


def test_dimensions_are_measured_once_and_cached():
    embedder = _with_client(_payload([[1.0, 2.0]]))
    embedder.embed(["a"])
    assert embedder.dimensions == 2
    # A second call with the same width does not raise.
    embedder.embed(["b"])


def test_a_declared_dimensions_matching_the_response_is_accepted():
    embedder = _with_client(_payload([[1.0, 2.0]]), dimensions=2)
    assert embedder.embed(["a"]) == [[1.0, 2.0]]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_dimensions_before_any_request_raise_rather_than_guess():
    with pytest.raises(EmbeddingInputError) as exc:
        _embedder().dimensions
    assert "first" in str(exc.value)


def test_a_declared_dimensions_disagreeing_with_the_response_is_rejected():
    embedder = _with_client(_payload([[1.0, 2.0]]), dimensions=5)
    with pytest.raises(EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "declared" in str(exc.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 5, "embedding": [2.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": -1, "embedding": [2.0]}]},
        {"data": [{"embedding": [1.0]}, {"index": 1, "embedding": [2.0]}]},
        {"data": [{"index": 0}, {"index": 1, "embedding": [2.0]}]},
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"data": "not a list"},
        {"result": []},
        [],
        "just a string",
    ],
    ids=[
        "duplicate-index",
        "gap",
        "negative-index",
        "missing-index",
        "missing-embedding",
        "wrong-count",
        "data-not-a-list",
        "no-data",
        "payload-not-a-dict",
        "payload-a-string",
    ],
)
def test_an_unusable_payload_is_rejected(payload):
    embedder = _with_client(payload)
    with pytest.raises(EmbeddingProviderError):
        embedder.embed(["a", "b"])


def test_a_non_integer_index_is_rejected():
    payload = {"data": [
        {"index": "0", "embedding": [1.0]},
        {"index": 1, "embedding": [2.0]},
    ]}
    with pytest.raises(EmbeddingProviderError) as exc:
        _with_client(payload).embed(["a", "b"])
    assert "index" in str(exc.value)


@pytest.mark.parametrize(
    ("status", "needle"),
    [(400, "HTTP 400"), (401, "HTTP 401"), (429, "HTTP 429"), (500, "HTTP 500")],
)
def test_a_non_200_reports_the_status(status, needle):
    embedder = _with_client({"error": "nope"}, status=status, text="nope")
    with pytest.raises(EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert needle in str(exc.value)


def test_the_error_body_is_truncated_in_the_message():
    embedder = _with_client({}, status=500, text="x" * 5000)
    with pytest.raises(EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert len(str(exc.value)) < 600, "a provider's error body must not flood a log"


def test_a_non_json_body_is_rejected():
    class BadClient(FakeClient):
        def post(self, url, headers=None, json=None):
            self.requests.append({})
            return NoJSON({}, text="<html>gateway</html>")

    embedder = _embedder()
    embedder._client = BadClient({})
    with pytest.raises(EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "non-JSON" in str(exc.value)


def test_a_transport_failure_is_wrapped_and_chained():
    class Broken:
        def post(self, url, headers=None, json=None):
            raise OSError("connection reset")

    embedder = _embedder()
    embedder._client = Broken()
    with pytest.raises(EmbeddingProviderError) as exc:
        embedder.embed(["a"])
    assert "connection reset" in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


def test_a_non_numeric_vector_value_is_rejected():
    embedder = _with_client(_payload([["a", "b"]]))
    with pytest.raises(EmbeddingProviderError):
        embedder.embed(["a"])


def test_a_vector_that_is_not_a_sequence_is_rejected():
    embedder = _with_client(_payload([1.0]))
    with pytest.raises(EmbeddingProviderError):
        embedder.embed(["a"])


def test_bad_input_is_rejected_before_any_client_is_built():
    embedder = _embedder()
    with pytest.raises(EmbeddingInputError):
        embedder.embed(["ok", ""])
    assert embedder._client is None


def test_failures_are_catchable_under_the_area_base_class():
    with pytest.raises(EmbeddingError):
        RemoteEmbedder(env={})


def test_messages_name_the_component():
    with pytest.raises(EmbeddingInputError) as exc:
        RemoteEmbedder(env={})
    assert "remote-embeddings" in str(exc.value)


# --------------------------------------------------------------------------- #
# FR-3: the optional dependency is guarded, with the manifest's message
# --------------------------------------------------------------------------- #


def test_the_optional_dependency_is_guarded_with_the_manifest_message(monkeypatch):
    """FR-3: the manifest's message, verbatim, as an ImportError.

    httpx is genuinely installed in this environment, which is why the import
    is blocked -- an unguarded test would otherwise reach the network.
    """
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "httpx", None)
    embedder = _embedder()
    with pytest.raises(ImportError) as exc:
        embedder.embed(["hello"])
    assert str(exc.value) == manifest["missing_dependency_message"]
    assert str(exc.value) == MISSING_DEPENDENCY_MESSAGE


def test_the_missing_dependency_error_is_an_import_error():
    assert issubclass(MissingEmbeddingDependency, ImportError)
    assert not issubclass(MissingEmbeddingDependency, EmbeddingError)


def test_importing_the_module_does_not_require_the_dependency():
    """The module must import on a machine that never installed httpx."""
    assert hasattr(remote_embeddings, "RemoteEmbedder")
    assert remote_embeddings.MISSING_DEPENDENCY_MESSAGE == MISSING_DEPENDENCY_MESSAGE


def test_the_manifest_declares_the_dependency_optional():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["required_packages"] == []
    assert [entry["name"] for entry in manifest["optional_packages"]] == ["httpx"]
    assert "embedding-protocol" in manifest["dependencies"]

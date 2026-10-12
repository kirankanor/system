"""Runnable example for `remote-embeddings`.

    python example.py

Calls a provider's HTTP API. Runs standalone with `embedding_protocol.py`
beside it -- the manifest declares that component a dependency, so `ragkit
add` copies it along.

The four guarantees this example demonstrates:

1. the endpoint, key and model are all required, with no defaults, and a
   missing one is named in the failure;
2. the request carries the model, the texts, and the bearer token;
3. a reordered response is put back in order -- providers do reorder, and a
   response that is not noticed reattaches every vector to the wrong document;
4. unusable responses are rejected rather than indexed.

Every response below is mocked. Nothing reaches the network: this component's
tests and examples never depend on a live provider being available, reachable
or paid for.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The protocol sits in its own component directory. `ragkit add` copies it
# beside this file; here we reach across to the sibling directory so the
# example runs from the component tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embedding_protocol"))

from embedding_protocol import (  # noqa: E402
    EmbeddingInputError,
    EmbeddingProviderError,
    MissingEmbeddingDependency,
)

from remote_embeddings import URL_ENV_VAR, RemoteEmbedder  # noqa: E402

URL = "https://api.example/v1/embeddings"


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

    def __init__(self, payload, *, status=200):
        self._payload = payload
        self._status = status
        self.requests = []

    def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self._payload, status=self._status)


def main() -> None:
    # 1. No default endpoint, key or model.
    try:
        RemoteEmbedder(env={})
    except EmbeddingInputError as exc:
        print(f"no settings -> {exc}")

    embedder = RemoteEmbedder(
        URL, api_key="secret-key", model="text-embedding-3-small", dimensions=2
    )
    print(f"\nconfigured: {embedder.url} / model {embedder.model_id!r}")

    # 2 + 3. The provider returns its entries out of order; they are sorted
    #         back into position by the index each one declares.
    shuffled = {"data": [
        {"index": 2, "embedding": [3.0, 30.0]},
        {"index": 0, "embedding": [1.0, 10.0]},
        {"index": 1, "embedding": [2.0, 20.0]},
    ]}
    client = FakeClient(shuffled)
    embedder._client = client

    vectors = embedder.embed(["alpha", "beta", "gamma"])
    print(f"response arrived as indices [2, 0, 1]")
    print(f"returned in order -> {vectors}")
    request = client.requests[0]
    print(f"request body    -> {request['json']}")
    print(f"auth header     -> {request['headers']['Authorization'][:14]}...")

    # 4. Failure paths, shown rather than hidden.

    # Indices that do not form 0..n-1 leave no safe way to match vectors.
    bad = RemoteEmbedder(URL, api_key="k", model="m", dimensions=2)
    bad._client = FakeClient({"data": [
        {"index": 0, "embedding": [1.0]},
        {"index": 0, "embedding": [2.0]},
    ]})
    try:
        bad.embed(["a", "b"])
    except EmbeddingProviderError as exc:
        print(f"\nrejected as expected: {exc}")

    # A response with no data list is not an embedding payload at all.
    nope = RemoteEmbedder(URL, api_key="k", model="m", dimensions=2)
    nope._client = FakeClient({"result": []})
    try:
        nope.embed(["a"])
    except EmbeddingProviderError as exc:
        print(f"rejected as expected: {exc}")

    # An HTTP failure reports the status.
    denied = RemoteEmbedder(URL, api_key="k", model="m", dimensions=2)
    denied._client = FakeClient({"error": "bad key"}, status=401)
    try:
        denied.embed(["a"])
    except EmbeddingProviderError as exc:
        print(f"rejected as expected: {exc}")

    # 5. FR-3: the optional dependency is guarded at the point of use. It is
    #    present in this environment, so the import is blocked here to show
    #    the guard; the component itself is untouched by that.
    sys.modules["httpx"] = None
    try:
        RemoteEmbedder(URL, api_key="k", model="m").embed(["a"])
    except MissingEmbeddingDependency as exc:
        print(f"\nhttpx blocked, so the guard fired:")
        print(f"  {exc}")
    finally:
        del sys.modules["httpx"]

    print(f"\nendpoint came from ${URL_ENV_VAR}: not set here")


if __name__ == "__main__":
    main()

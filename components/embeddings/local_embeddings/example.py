"""Runnable example for `local-embeddings`.

    python example.py

Embeds text with a model running in this process. Runs standalone with
`embedding_protocol.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

The three guarantees this example demonstrates:

1. there is no default model: a missing id fails at construction, naming the
   variable to set, rather than silently choosing one;
2. `embed` returns one vector per text, in order, and an empty input costs
   nothing;
3. the optional `sentence-transformers` dependency is guarded at the point of
   use, and the message it raises is the manifest's, verbatim (FR-3).

`sentence-transformers` is not installed in this checkout -- installing it
would modify dependency files, which this library forbids. Part 3 shows the
guard firing; part 2 injects a small fake model so the encode path runs for
real without it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The protocol sits in its own component directory. `ragkit add` copies it
# beside this file; here we reach across to the sibling directory so the
# example runs from the component tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embedding_protocol"))

from embedding_protocol import EmbeddingInputError, MissingEmbeddingDependency  # noqa: E402

from local_embeddings import MODEL_ENV_VAR, LocalEmbedder  # noqa: E402


class FakeModel:
    """Stands in for a sentence-transformers model.

    One three-wide vector per text, with the text's length in position 0 so
    the output below is readable by hand.
    """

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def main() -> None:
    # 1. No default model. The failure names the variable, so the fix is in
    #    the message rather than in the docs.
    try:
        LocalEmbedder(env={})
    except EmbeddingInputError as exc:
        print(f"no model id -> {exc}")

    # The environment is the fallback, and it is read explicitly.
    from_env = LocalEmbedder(env={MODEL_ENV_VAR: "sentence-transformers/all-MiniLM-L6-v2"})
    print(f"\nfrom ${MODEL_ENV_VAR} -> model_id={from_env.model_id!r}")

    # 2. The encode path, through an injected model.
    embedder = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    embedder._model = FakeModel()
    vectors = embedder.embed(["alpha", "beta"])
    print(f"embed(['alpha', 'beta']) -> {vectors}")
    print(f"one vector per text, in order: "
          f"{[v[0] for v in vectors]} == {[5.0, 4.0]}")
    print(f"dimensions, as the model declares them: {embedder.dimensions}")
    print(f"embed_one('hello') -> {embedder.embed_one('hello')}")

    # 3. FR-3, on a machine without sentence-transformers: an ImportError
    #    carrying the manifest's message.
    real = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    try:
        real.embed(["hello"])
    except MissingEmbeddingDependency as exc:
        print(f"\nthe optional dependency is absent, so the guard fired:")
        print(f"  {exc}")
        print(f"  is an ImportError: {isinstance(exc, ImportError)}")

    print(f"\nmodel id came from ${MODEL_ENV_VAR}: {os.environ.get(MODEL_ENV_VAR)}")


if __name__ == "__main__":
    main()

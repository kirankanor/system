"""Runnable example for `cross-encoder-reranker`.

    python example.py

Shows the guarded-dependency contract working, and the ordering a
cross-encoder produces. Runs standalone with `core_types.py` beside it -- the
manifest declares that component a dependency, so `ragkit add` copies it
along.

Nothing is installed to make this run. `sentence-transformers` is optional and
is NOT installed in this repository, so:

* the real model is never exercised here;
* the model is supplied as a stand-in that speaks the same `.predict(pairs)`
  shape, which is exactly how a caller who manages their own model uses it;
* the missing-dependency path IS exercised for real, because the wheel is
  genuinely absent.

The five guarantees this example demonstrates:

1. the module imports and constructs with the wheel absent;
2. the model loads lazily, only when it is first needed;
3. the model's scores order the candidates, best first;
4. ties break on chunk_index, never on input order;
5. a missing wheel raises ImportError naming the package, with the manifest's
   message verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory under components/storage.
# `ragkit add` copies it beside this file; here we reach across to the sibling
# so the example runs from the component tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "storage" / "core_types")
)

import core_types as core  # noqa: E402

from cross_encoder_reranker import (  # noqa: E402
    CrossEncoderReranker,
    MissingDependency,
    RerankingError,
    RerankingInputError,
)

CANDIDATES = [
    core.ScoredDocument(chunk_id="c0", text="Python error handling with exceptions",
                        score=0.0, chunk_index=0, source_name="python/errors.md"),
    core.ScoredDocument(chunk_id="c1", text="Cooking pasta with salt",
                        score=0.0, chunk_index=1, source_name="cooking/pasta.md"),
    core.ScoredDocument(chunk_id="c2", text="Java exceptions and Python compared",
                        score=0.0, chunk_index=2, source_name="java/vs.md"),
]


class StandInCrossEncoder:
    """The slice of `sentence_transformers.CrossEncoder` this component calls.

    A real model returns a relevance logit per pair; so does this. Swapping
    the two is the only change a caller with a loaded model has to make.
    """

    def __init__(self, scores):
        self._scores = scores
        self.pairs_seen = 0

    def predict(self, pairs):
        self.pairs_seen = len(pairs)
        return self._scores


def main() -> None:
    print("1. The module imports and constructs with the wheel absent:")
    guard = CrossEncoderReranker("cross-encoder/ms-marco")
    print(f"   constructed -> {guard}")
    print(f"   is_loaded() -> {guard.is_loaded()}  (nothing has been imported yet)")

    print("\n2. The model loads lazily, on the first rerank -- not on construction:")
    lazy = CrossEncoderReranker("cross-encoder/ms-marco")
    print(f"   built with no model -> is_loaded() = {lazy.is_loaded()}")
    print("   (step 5 below is the proof: construction succeeded, and only the")
    print("    rerank raises -- so the import happens there, not here)")

    print("\n3. With a model in hand, its scores order the candidates:")
    model = CrossEncoderReranker(
        "cross-encoder/ms-marco", model=StandInCrossEncoder([0.9, 0.1, 0.5])
    )
    for document in model.rerank("python exception", CANDIDATES, 3):
        print(f"   {document.chunk_id}  score={document.score:.3f}  {document.text!r}")
    print(f"   (every candidate was scored: {model._model.pairs_seen} pairs sent, "
          f"then truncated to top_k)")

    print("\n   Higher logit means more relevant. They are NOT rescaled -- logits are")
    print("   not comparable across models, so pretending otherwise would mislead.")

    print("\n4. Ties break on chunk_index, never on input order:")
    tied = CrossEncoderReranker(
        "m", model=StandInCrossEncoder([0.5, 0.5, 0.5])
    )
    print(f"   original : {[d.chunk_id for d in tied.rerank('q', CANDIDATES, 3)]}")
    print(f"   reversed : {[d.chunk_id for d in tied.rerank('q', list(reversed(CANDIDATES)), 3)]}")

    print("\n5. A missing wheel raises ImportError naming the package:")
    try:
        guard.rerank("python exception", CANDIDATES, 2)
    except MissingDependency as exc:
        print(f"   caught   : {type(exc).__name__} (an ImportError: {isinstance(exc, ImportError)})")
        print(f"   message  : {exc}")
        print(f"   __cause__ is the original ImportError: {isinstance(exc.__cause__, ImportError)}")
    else:
        print("   sentence-transformers IS installed here, so the guard did not fire.")

    print("\nThe fallback this guard exists to make possible:")
    try:
        CrossEncoderReranker("m").rerank("q", CANDIDATES, 2)
    except ImportError:
        print("   cross-encoder unavailable -> fall back to `relevance-ranking`,")
        print("   which is dependency-free and returns a score in [0, 1]")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("an empty model_name", lambda: CrossEncoderReranker("")),
        ("a non-string device", lambda: CrossEncoderReranker("m", device=3)),
        ("a blank query", lambda: model.rerank("   ", CANDIDATES, 1)),
        ("a top_k of 0", lambda: model.rerank("q", CANDIDATES, 0)),
    ]:
        try:
            call()
        except RerankingInputError as exc:
            print(f"   {label:<22} -> {str(exc)[:56]}...")

    print("\nA model that misbehaves is wrapped, with its cause kept:")

    class Broken:
        def predict(self, pairs):
            raise RuntimeError("cuda out of memory")

    try:
        CrossEncoderReranker("ms-marco-MiniLM", model=Broken()).rerank("q", CANDIDATES, 2)
    except RerankingError as exc:
        print(f"   -> {str(exc)[:74]}...")
        print(f"   -> original cause preserved: {type(exc.__cause__).__name__}")

    print("\nIt satisfies RerankerProtocol, so it slots in beside a lexical reranker:")
    print(f"   isinstance(..., RerankerProtocol) -> "
          f"{isinstance(model, core.RerankerProtocol)}")


if __name__ == "__main__":
    main()

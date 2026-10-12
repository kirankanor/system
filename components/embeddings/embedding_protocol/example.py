"""Runnable example for `embedding-protocol`.

    python example.py

Shows the shared embedding contract and the input checks every adapter in the
area runs. Runs standalone: the only imports are this directory's
`embedding_protocol.py` and the standard library.

The three guarantees this example demonstrates:

1. the Protocol is satisfied structurally, so any duck-typed object conforms;
2. `validate_texts` rejects the inputs that would silently corrupt a caller's
   index, and accepts the empty list;
3. every failure in the area is catchable under one base class.
"""

from __future__ import annotations

from embedding_protocol import (
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProviderError,
    EmbeddingProtocol,
    MissingEmbeddingDependency,
    validate_texts,
)


class TinyEmbedder:
    """Nothing inherits from anything: the protocol is structural.

    One vector per text, with the text's length in position 0 so the output
    below is readable by hand.
    """

    dimensions = 3
    model_id = "tiny-demo"

    def embed(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def main() -> None:
    embedder = TinyEmbedder()

    # 1. No base class, no registration -- it just conforms.
    print(f"satisfies the protocol: {isinstance(embedder, EmbeddingProtocol)}")
    print(f"vector space: {embedder.dimensions} wide, model {embedder.model_id!r}")

    vectors = embedder.embed(["alpha", "beta"])
    print(f"embed(['alpha', 'beta']) -> {vectors}")

    # 2. Input validation, shown rather than described.
    print("\nvalidate_texts accepts:")
    print(f"  ['alpha', 'beta'] -> {validate_texts(['alpha', 'beta'])}")
    print(f"  ()                 -> {validate_texts(())}")
    print(f"  []                 -> {validate_texts([])}   (nothing to do, not an error)")

    print("\nvalidate_texts rejects:")
    for label, bad in [
        ("'alpha' (a str is a sequence of characters)", "alpha"),
        ("['alpha', ''] (an empty chunk)", ["alpha", ""]),
        ("['alpha', 5] (a non-string element)", ["alpha", 5]),
        ("None", None),
    ]:
        try:
            validate_texts(bad)
        except EmbeddingInputError as exc:
            print(f"  {label}\n    -> {str(exc)[:72]}...")

    # 3. One base class for the whole area, with a narrow alternative.
    print("\nevery failure in the area is catchable together:")
    for error in (EmbeddingInputError, EmbeddingProviderError):
        print(f"  {error.__name__} is an EmbeddingError: "
              f"{issubclass(error, EmbeddingError)}")
    print(f"  MissingEmbeddingDependency is an ImportError: "
          f"{issubclass(MissingEmbeddingDependency, ImportError)}")

    try:
        validate_texts(["ok", ""])
    except EmbeddingError as exc:
        print(f"\ncaught under the one base class: {type(exc).__name__}")


if __name__ == "__main__":
    main()

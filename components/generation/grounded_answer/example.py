"""Runnable example for `grounded-answer-contract`.

    python example.py

Shows an answer's claims being scored against the evidence, and a fabricated
claim being caught. Runs standalone with `core_types.py` beside it -- the
manifest declares that component a dependency, so `ragkit add` copies it
along.

Nothing is installed to make this run: there is no model here, no transport,
no network. The verdict is arithmetic over the evidence vocabulary, which is
why it can run offline and why it is a floor rather than a judge.

The six guarantees this example demonstrates:

1. an answer fully backed by the evidence is accepted;
2. a fabricated claim scores 0.0 and fails the whole answer;
3. one bad claim out of two is enough to fail it;
4. an invented number is weighted, because numbers are distinctive;
5. a wholly invented claim scores 0.00 and fails at any positive threshold;
6. an answer's support is bounded to [0, 1] at every setting.
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

from grounded_answer import (  # noqa: E402
    GroundednessContract,
    GroundingInputError,
    tokenize,
)

EVIDENCE = [
    core.ScoredDocument(
        chunk_id="c0", source_name="python/errors.md",
        text="Python raises TypeError when a function receives an argument of "
             "the wrong type.",
        score=0.9, chunk_index=0, metadata={},
    ),
    core.ScoredDocument(
        chunk_id="c1", source_name="json/module.md",
        text="The json module serialises Python objects into JSON text.",
        score=0.8, chunk_index=1, metadata={},
    ),
]

SUPPORTED = (
    "The json module serialises Python objects into JSON text. "
    "Python raises TypeError for a wrong argument type."
)

FABRICATED = (
    "Python raises TypeError for a wrong argument type. "
    "The Zephyr framework uses quantum entanglement for IPC."
)

MIXED = (
    "The json module serialises objects. "
    "GPT-9 runs at 400 teraflops on the second tuesday."
)


def context_of(chunks) -> core.Context:
    items = tuple(chunks)
    return core.Context(
        text="\n\n".join(c.text for c in items),
        chunks=items,
        token_count=5,
        budget=200,
    )


def show(label: str, contract, answer: str, context) -> None:
    result = contract.check(answer, context)
    print(f"\n{label}")
    print(f"   verdict: {'GROUNDED' if result.grounded else 'NOT GROUNDED'}")
    for claim in result.claims:
        flag = "ok  " if claim.supported else "FAIL"
        print(f"     {flag} {claim.support:.2f}  {claim.text}")
    if result.unsupported:
        print(f"   unsupported: {result.unsupported}")


def main() -> None:
    contract = GroundednessContract()
    context = context_of(EVIDENCE)
    print(f"{contract}\n")

    print("The evidence the answer must be backed by:")
    for chunk in EVIDENCE:
        print(f"   {chunk.source_name}: {chunk.text}")
    print(f"\n   tokenize drops stopwords and lowercases:")
    print(f"     'The JSON Module' -> {sorted(tokenize('The JSON Module'))}")

    show("1. An answer fully backed by the evidence is accepted:",
         contract, SUPPORTED, context)

    show("2. A fabricated claim scores 0.0 and fails the whole answer:",
         contract, FABRICATED, context)
    print("   -> 'Zephyr' and 'quantum entanglement' are in no passage, so each")
    print("      carries the MAXIMUM idf -- and lands in the denominator alone.")

    show("3. One bad claim out of two is enough to fail it:",
         contract, MIXED, context)
    print("   -> the first sentence is fine; the answer is still rejected.")

    show("4. An invented number is weighted, because numbers are distinctive:",
         contract, "Python raises TypeError 918273645.", context)

    print("\n5. A wholly invented claim scores 0.00 and fails at any positive threshold:")
    for threshold in (0.0, 0.25, 0.5, 1.0):
        loose = GroundednessContract(min_support=threshold)
        result = loose.check(FABRICATED, context)
        fabricated = [c for c in result.claims if "Zephyr" in c.text][0]
        note = "  <- check disabled" if threshold == 0.0 else ""
        print(f"     min_support={threshold:<4} -> that claim scores "
              f"{fabricated.support:.2f}, grounded={result.grounded}{note}")
    print("   -> min_support is a plain threshold: 0.0 is an opt-out, not a")
    print("      stricter bar. Use 0.01 to catch only total fabrication.")

    print("\n   A low threshold does admit a weak-but-sourced claim:")
    weak = "The json module does something with objects perhaps."
    for threshold in (0.0, 0.5, 1.0):
        print(f"     min_support={threshold:<4} -> "
              f"{GroundednessContract(min_support=threshold).check(weak, context).grounded}")

    print("\n6. Support stays in [0, 1] at every setting:")
    samples = [SUPPORTED, FABRICATED, MIXED, "The json module alone."]
    values = [
        c.support
        for threshold in (0.0, 0.5, 1.0)
        for answer in samples
        for c in GroundednessContract(min_support=threshold).check(answer, context).claims
    ]
    print(f"     {len(values)} claims across thresholds, all within bounds: "
          f"{all(0.0 <= v <= 1.0 for v in values)}")

    print("\nThe documented limit -- support is LEXICAL, so paraphrase costs:")
    near = contract.check("The json module serialises Python objects.", context)
    loose = contract.check("JSON conversion turns Python values into text.", context)
    print(f"     near-quote : {near.claims[0].support:.2f}")
    print(f"     paraphrase : {loose.claims[0].support:.2f}  "
          f"(faithful, but the words differ)")

    print("\nThe report is meant to be read by a person:")
    for line in contract.check(FABRICATED, context).report().splitlines():
        print(f"     {line}")

    print("\nWhat passing does and does not mean:")
    print("     PASSING = every sentence is backed by the supplied evidence.")
    print("     NOT passing = some sentence is not. Worth a look.")
    print("     Neither says whether the answer is correct or well written.")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("an empty answer", lambda: contract.check("  ", context)),
        ("a non-string answer", lambda: contract.check(None, context)),
        ("an empty context", lambda: contract.check("an answer", context_of([]))),
        ("a non-Context", lambda: contract.check("an answer", "not a context")),
        ("min_support=1.5", lambda: GroundednessContract(min_support=1.5)),
        ("min_support=True", lambda: GroundednessContract(min_support=True)),
        ("a non-frozenset stopwords", lambda: GroundednessContract(stopwords=set())),
    ]:
        try:
            call()
        except GroundingInputError as exc:
            print(f"     {label:<26} -> {str(exc)[:58]}...")

    print("\nTogether with prompt-builder, both without a provider:")
    print("     prompt-builder              numbers the evidence [1]..[n]")
    print("     grounded-answer-contract    scores claims against the same evidence")
    print("     citation-formatter          resolves the numbers in between")
    print("     (no module imports another, and none imports a client)")


if __name__ == "__main__":
    main()

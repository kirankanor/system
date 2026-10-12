"""Runnable example for `answer-metrics`.

    python example.py

Scores real answers against real evidence, and demonstrates the property the
component is built around: groundedness is computed ONLY from the text you
supply, and a term the evidence has never heard of takes the MAXIMUM weight
rather than zero -- so a fabricated answer scores low instead of scoring a
perfect 1.0. This component has NO dependencies, so it runs anywhere Python
3.12 does.

The five things this example demonstrates:

1. both endpoints, exactly -- 1.0 quoted, 0.0 invented;
2. the unseen-term rule, with the score that would have been wrong;
3. the score moves only when the evidence moves;
4. no stemming: `serialise` does not match `serialised`;
5. an empty answer scores 0.0 rather than crashing the evaluation.
"""

from __future__ import annotations

import math

from answer_metrics import AnswerMetrics, AnswerMetricsInputError

EVIDENCE = ["The json module serialises objects into JSON text."]
QUERY = "how does json serialise objects"


def main() -> None:
    metrics = AnswerMetrics()
    print(f"{metrics}\n")

    print("=" * 78)
    print("1. The two endpoints, exactly")
    print("=" * 78)
    cases = [
        ("a faithful paraphrase", "json serialises objects into JSON text."),
        ("quoting the evidence", "The json module serialises objects into JSON text."),
        ("invented entirely", "The quantum tunnelling manifold repolarises the substrate."),
        ("an empty answer", ""),
        ("stopwords only", "the and of"),
    ]
    for label, answer in cases:
        g = metrics.groundedness(answer, EVIDENCE)
        r = metrics.relevance(answer, QUERY)
        print(f"   {label:<24} groundedness={g:.6f}  relevance={r:.6f}")

    print("\n   The first two are 1.0 because every distinctive term is present.")
    print("   The empty answer is 0.0, NOT 1.0: vacuously grounded is not the")
    print("   same as grounded, and 1.0 would let an empty string top a")
    print("   leaderboard. It does not raise -- an evaluation has to be able")
    print("   to score a model that produced nothing.")

    print()
    print("=" * 78)
    print("2. The unseen-term rule, with the score that would have been wrong")
    print("=" * 78)
    fabricated = metrics.groundedness(
        "The quantum tunnelling manifold repolarises the substrate.", EVIDENCE
    )
    print(f"   fabricated answer vs evidence : {fabricated:.6f}")
    print("   Every distinctive term in that answer is absent from the")
    print("   evidence. Under the MAXIMUM weight each one drags the score to 0.")
    print()
    print("   Had unseen terms been given ZERO weight, each invented term")
    print("   would contribute 0/0, the sum would vanish, and the ratio would")
    print("   read 1.0 -- the most confabulated answer in the batch would win")
    print("   the comparison. That is why the rule is the component's central")
    print("   design decision rather than an implementation detail.")
    print()
    print("   Direction of the error matters: this fails SAFE. A fabricated")
    print("   claim scores low, which is the way round you want to be wrong.")

    print()
    print("=" * 78)
    print("3. The score depends ONLY on the supplied evidence")
    print("=" * 78)
    answer = "alpha gamma"
    for label, evidence in (
        ("evidence has alpha", ["alpha beta"]),
        ("evidence has gamma", ["gamma delta"]),
        ("evidence has both", ["alpha gamma"]),
        ("evidence has neither", ["beta delta"]),
    ):
        print(f"   {label:<22} -> {metrics.groundedness(answer, evidence):.6f}")

    print("\n   Same answer, four different numbers -- and no model, network")
    print("   or corpus behind any of them. Swap the evidence and the number")
    print("   moves; change nothing and it never does, so two runs a week")
    print("   apart agree. A groundedness score that consulted a model would")
    print("   be a second model's opinion about the first, and would drift")
    print("   when the vendor changed their weights -- useless as a baseline.")

    print("\n   With one passage the weights are closed form, so this is")
    print("   hand-checkable rather than merely plausible:")
    expected = math.log(4 / 3) / (math.log(4 / 3) + math.log(4))
    actual = metrics.groundedness("alpha gamma", ["alpha beta"])
    print(f"     idf present = log(4/3) = {math.log(4 / 3):.10f}")
    print(f"     idf absent  = log(4)   = {math.log(4):.10f}")
    print(f"     log(4/3) / (log(4/3)+log(4)) = {expected:.10f}")
    print(f"     component says              = {actual:.10f}")
    print(f"     match: {abs(expected - actual) < 1e-12}")

    print()
    print("=" * 78)
    print("4. No stemming -- `serialise` is not `serialised`")
    print("=" * 78)
    print(f"   query    : {QUERY!r}")
    print(f"   evidence : {EVIDENCE[0]!r}")
    answer = "Objects are serialised by the json module."
    print(f"   answer   : {answer!r}")
    print(f"   relevance: {metrics.relevance(answer, QUERY):.6f}")
    print("\n   Lower than it 'deserves': the query says serialise, the answer")
    print("   says serialised, and this component will not pretend they are")
    print("   one term. That is a deliberate gap -- it means the score never")
    print("   claims a match the text does not contain. Anyone who wants")
    print("   stemming can stem before calling.")
    print(f"\n   Same reason: 'exceptions' vs 'the exception was caught' -> "
          f"{metrics.groundedness('exceptions', ['the exception was caught']):.6f}")

    print()
    print("=" * 78)
    print("5. The whole score, and where it goes")
    print("=" * 78)
    score = metrics.score(
        answer="The json module serialises objects into JSON text.",
        query=QUERY,
        evidence=EVIDENCE,
    )
    print(f"   {score}")
    print(f"   as_metrics() -> {score.as_metrics()}")
    print("\n   Deliberately NOT a single averaged number: groundedness and")
    print("   relevance fail in different directions, and averaging them")
    print("   would let a faithful answer that ignores the question look")
    print("   identical to an irrelevant answer copied from the evidence.")

    print("\n   Bad input is a typed error, not a surprise:")
    for label, fn in [
        ("evidence='text'", lambda: metrics.groundedness("a", "text")),
        ("evidence=[]", lambda: metrics.groundedness("a", [])),
        ("evidence=['the and']", lambda: metrics.groundedness("a", ["the and"])),
        ("query=''", lambda: metrics.relevance("a", "")),
        ("evidence=[1]", lambda: metrics.groundedness("a", [1])),
        ("stopwords=[]", lambda: AnswerMetrics(stopwords=[])),
    ]:
        try:
            fn()
            print(f"   {label:<20} NO ERROR")
        except AnswerMetricsInputError as exc:
            print(f"   {label:<20} {type(exc).__name__}: {str(exc)[:54]}...")

    print("\nWhat this component will not do:")
    print("   it does not verify FACTS. It establishes that the answer's")
    print("   vocabulary is present in the evidence, not that the evidence is")
    print("   true or that the right inference was drawn. For a pass/fail gate")
    print("   before an answer reaches a user, see grounded-answer-contract --")
    print("   that one fails loudly, and this one scores softly.")


if __name__ == "__main__":
    main()

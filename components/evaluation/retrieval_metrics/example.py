"""Runnable example for `retrieval-metrics`.

    python example.py

Works one query through the three metrics, then shows the case the component
exists for: a change that moves one metric and not the others, so reporting a
single number would have called it an improvement. Ends with a degenerate
cutoff and a typed error. This component has NO dependencies -- no
core-types, no packages, nothing installed -- so it runs anywhere Python 3.12
does.

The five things this example demonstrates:

1. every metric matches a value worked out on paper;
2. hit rate and MRR disagree, and only one of them is lying;
3. a degenerate cutoff is refused or clamped, never silently zero;
4. duplicates cannot inflate recall;
5. `missed_queries` says which queries to open first.
"""

from __future__ import annotations

from retrieval_metrics import RetrievalMetrics, RetrievalMetricsInputError


def main() -> None:
    print("=" * 78)
    print("1. One query, three metrics, all hand-checkable")
    print("=" * 78)

    metrics = RetrievalMetrics(k=3)
    print(f"   {metrics}\n")

    worked = [
        ("the right chunk is second", ["c1", "c2", "c3"], ["c2"]),
        ("nothing landed", ["c5", "c6", "c7"], ["c2"]),
        ("two valid answers, one found", ["c3", "c2"], ["c1", "c2"]),
    ]
    for label, retrieved, relevant in worked:
        s = metrics.score(retrieved=retrieved, relevant=relevant)
        print(f"   {label:<30} {s}")

    print("\n   Worked out on paper first:")
    print("     query A  rank 2 -> rr = 1/2 = 0.5, hit = 1, recall = 1/1 = 1.0")
    print("     query B  no hit  -> rr = 0,        hit = 0, recall = 0      ")
    print("     query C  rank 2 -> rr = 0.5,        hit = 1, recall = 1/2 = 0.5")

    corpus = metrics.score_many([(r, rel) for _, r, rel in worked])
    print(f"\n   {corpus}")
    print("     hit_rate = (1 + 0 + 1) / 3     = 0.6666...")
    print("     mrr      = (0.5 + 0 + 0.5) / 3 = 0.3333...")
    print("     recall   = (1 + 0 + 0.5) / 3   = 0.5")
    print(f"     as_metrics() -> {corpus.as_metrics()}")

    print()
    print("=" * 78)
    print("2. Hit rate and MRR disagree -- and only one of them is lying")
    print("=" * 78)
    print("   The change below adds nine junk results ahead of the answer.")
    print("   Hit rate is unmoved. MRR collapses. A single reported number\n"
          "   would have called this a pass.\n")

    for label, retrieved in (
        ("before: answer first", ["target", "j1", "j2", "j3", "j4"]),
        (
            "after : answer tenth",
            ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "target"],
        ),
    ):
        k = RetrievalMetrics(k=10)
        s = k.score(retrieved=retrieved, relevant=["target"])
        print(f"   {label:<22} hit={s.hit}  rr={s.reciprocal_rank}  recall={s.recall}")

    print("\n   Both are 'correct'. That is why there are three numbers.")

    print()
    print("=" * 78)
    print("3. Degenerate cutoffs: refused or clamped, never silently zero")
    print("=" * 78)
    short = ["only", "two"]
    for k in (1, 2, 3, 100):
        s = RetrievalMetrics(k=k).score(retrieved=short, relevant=["two"])
        print(f"   k={k:<4} against a 2-result list -> rank={s.rank}, recall={s.recall}")

    print("\n   k past the end is clamped: recall@100 over a 2-result list")
    print("   asks a real question -- 'of everything you returned, how much")
    print("   was right' -- and answers it (1.0, because both results are on")
    print("   the label). It is not an error, and refusing it would make")
    print("   every caller pre-clamp by hand.\n")

    for k in (0, -1, 1.5, True, "10"):
        try:
            RetrievalMetrics(k=k)
            print(f"   k={k!r:<8} NO ERROR")
        except RetrievalMetricsInputError as exc:
            shown = str(exc).split(".")[0]
            print(f"   k={k!r:<8} {type(exc).__name__}: {shown}")

    print("\n   A cutoff of 0 would score every metric 0, and 0.0 would read\n"
          "   as a measurement of the retriever rather than as an absence of")
    print("   one. So it is refused.")

    print()
    print("=" * 78)
    print("4. Duplicates cannot inflate recall")
    print("=" * 78)
    honest = metrics.score(retrieved=["c1", "c2"], relevant=["c1"])
    padded = metrics.score(retrieved=["c1", "c1", "c1", "c2"], relevant=["c1"])
    print(f"   ['c1','c2']            -> {honest}")
    print(f"   ['c1','c1','c1','c2']  -> {padded}")
    print("   Same answer either way: the duplicate occupied one slot, so a")
    print("   retriever cannot game the metric by repeating its best result.")

    print("\n   A repeated LABEL is a set too, so recall cannot exceed 1.0:")
    twice = metrics.score(retrieved=["c1", "c2"], relevant=["c1", "c1"])
    print(f"   relevant=['c1','c1']   -> recall={twice.recall}")

    print()
    print("=" * 78)
    print("5. A mean of 0.67 tells you something is wrong; this says where")
    print("=" * 78)
    missed = corpus.missed_queries
    print(f"   queries in corpus : {corpus.examples}")
    print(f"   queries that missed: {len(missed)}")
    for query in missed:
        print(f"     {query}  <- open this one first")

    print("\n   The corpus mean is the same whether you diagnosed it or not.")
    print("   Diagnosing is the difference between knowing a number moved")
    print("   and knowing what to fix.")

    print()
    print("=" * 78)
    print("What this component will not do")
    print("=" * 78)
    print("   No graded relevance: a partly-right chunk scores like an exact")
    print("   one, because 'relevant' is a set of ids. No nDCG, no confidence")
    print("   intervals, no averaging over k -- each instance scores at ONE")
    print("   cutoff, because the mean over k hides where quality falls off.")
    print("   Rank comes from input position and is never recomputed: only")
    print("   the caller knows whether a higher score means better.")


if __name__ == "__main__":
    main()

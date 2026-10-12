"""Runnable example for `evaluation-dataset`.

    python example.py

Builds a labelled dataset, round-trips it through JSON, then runs the
assertion the component exists for: a baseline against a known-worse result,
which RAISES and names every metric that got worse. Also shows the case a
naive assertion would miss -- a baseline whose metric the new run stopped
reporting. This component has NO dependencies, so it runs anywhere Python
3.12 does.

The five things this example demonstrates:

1. a labelled dataset, round-tripped through JSON with no loss;
2. an improvement and an exact tie both pass;
3. a known-worse result fails, naming every offender;
4. `tolerance` absorbs noise but not damage -- and cannot swallow a metric;
5. why there is no `to_file`: this component writes nothing.
"""

from __future__ import annotations

from evaluation_dataset import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationInputError,
    RegressionError,
)


def build() -> EvaluationDataset:
    return EvaluationDataset(
        cases=(
            EvaluationCase(
                id="parse-json",
                query="how do I parse json",
                retrieved=("c1", "c2", "c3"),
                relevant=("c1",),
            ),
            EvaluationCase(
                id="write-json",
                query="how do I write json",
                retrieved=("c9", "c2"),
                relevant=("c2",),
            ),
            EvaluationCase(
                id="json-vs-yaml",
                query="json vs yaml",
                retrieved=("c7", "c8", "c1"),
                relevant=("c5", "c6"),
            ),
            EvaluationCase(
                id="empty-answer",
                query="what is json",
                retrieved=(),
                relevant=(),
                answer="",
                evidence=(),
            ),
        ),
        name="docs-search",
    )


def main() -> None:
    dataset = build()

    print("=" * 78)
    print("1. A labelled dataset, and JSON that round-trips exactly")
    print("=" * 78)
    print(f"   {dataset}\n")
    for case in dataset:
        kind = "retrieval" if case.has_retrieval else "answer"
        print(f"   {case.id:<14} [{kind}] query={case.query!r}")

    text = dataset.to_json()
    again = EvaluationDataset.from_json(text)
    print(f"\n   to_json()   : {len(text)} characters")
    print(f"   round-trip  : {again.to_dict() == dataset.to_dict()}")
    print(f"   repr        : {again!r}")

    print("\n   retrieval_examples() feeds retrieval-metrics directly:")
    print(f"   {dataset.retrieval_examples()}")

    print()
    print("=" * 78)
    print("2. An improvement and an exact tie both pass")
    print("=" * 78)
    baseline = {"hit_rate": 0.80, "mrr": 0.60, "recall": 0.70}
    runs = [
        ("exact tie", {"hit_rate": 0.80, "mrr": 0.60, "recall": 0.70}),
        ("an improvement", {"hit_rate": 0.85, "mrr": 0.65, "recall": 0.75}),
        ("a new metric appears", {"hit_rate": 0.80, "mrr": 0.60, "recall": 0.70, "ndcg": 0.5}),
    ]
    for label, current in runs:
        try:
            dataset.assert_no_regression(current=current, baseline=baseline)
            print(f"   {label:<22} PASSED")
        except RegressionError as exc:
            print(f"   {label:<22} FAILED: {exc}")

    print("\n   A metric the baseline never had has no history to regress")
    print("   from, so it is reported but never fails the check.")

    print()
    print("=" * 78)
    print("3. A known-worse result FAILS the assertion")
    print("=" * 78)
    worse = {"hit_rate": 0.70, "mrr": 0.60, "recall": 0.70}
    print(f"   baseline : {baseline}")
    print(f"   current  : {worse}")
    print("   only hit_rate moved, and only by 0.10.\n")
    try:
        dataset.assert_no_regression(current=worse, baseline=baseline)
        print("   PASSED -- THIS IS THE BUG")
    except RegressionError as exc:
        print(f"   {type(exc).__name__}")
        print(f"   {exc}")
        for item in exc.regressions:
            print(f"     -> {item!r}")
            print(f"        past tolerance by {item.worsening:g}")

    print("\n   Every offender is reported, not just the first -- one run")
    print("   tells you the whole story instead of a whack-a-mole sequence")
    print("   of fixes.\n")

    collapsed = {"hit_rate": 0.50, "mrr": 0.50, "recall": 0.50}
    try:
        dataset.assert_no_regression(current=collapsed, baseline=baseline)
    except RegressionError as exc:
        print(f"   all three at once: {exc}")

    print()
    print("=" * 78)
    print("4. Tolerance absorbs noise -- but cannot swallow a metric")
    print("=" * 78)
    noisy = {"hit_rate": 0.79, "mrr": 0.60, "recall": 0.70}
    for tolerance in (0.0, 0.05, 0.5):
        try:
            dataset.assert_no_regression(
                current=noisy, baseline=baseline, tolerance=tolerance
            )
            print(f"   tolerance={tolerance:<5} hit_rate 0.80 -> 0.79   PASSED")
        except RegressionError as exc:
            print(f"   tolerance={tolerance:<5} hit_rate 0.80 -> 0.79   {exc}")

    print("\n   The tolerance is absolute, in metric units. Choosing it is")
    print("   your judgement, not the component's: too loose and it hides a")
    print("   real regression, too tight and every run fails on one query.")
    print("   The default is 0.0, because there is no value that is right")
    print("   for every corpus.\n")

    print("   It cannot swallow a metric, either. A baseline of three")
    print("   metrics cannot be checked against a run that reports one:")
    try:
        dataset.assert_no_regression(current={"hit_rate": 0.95}, baseline=baseline)
        print("   PASSED -- THIS IS THE BUG")
    except EvaluationInputError as exc:
        print(f"   {type(exc).__name__}: {exc}")
    print("\n   That refusal is the important one. A subset comparison is")
    print("   precisely how a real regression hides: hit rate can hold")
    print("   steady while MRR collapses, and a lenient check calls it a pass.")

    print("\n   Lower-is-better metrics are opt-in, or the check would fail")
    print("   every improvement:")
    latency_baseline = {"latency_ms": 100.0}
    for label, current in (("faster", {"latency_ms": 90.0}), ("slower", {"latency_ms": 115.0})):
        try:
            dataset.assert_no_regression(
                current=current,
                baseline=latency_baseline,
                lower_is_better=["latency_ms"],
            )
            print(f"     {label:<8} 100 -> {current['latency_ms']}   PASSED")
        except RegressionError as exc:
            print(f"     {label:<8} 100 -> {current['latency_ms']}   {type(exc).__name__}: {str(exc)[:60]}...")

    print()
    print("=" * 78)
    print("5. Bad data is a typed error, and nothing is ever written")
    print("=" * 78)
    for label, fn in [
        ("empty dataset", lambda: EvaluationDataset(cases=())),
        ("duplicate ids", lambda: EvaluationDataset(cases=(
            EvaluationCase(id="a", query="q1"), EvaluationCase(id="a", query="q2")))),
        ("malformed json", lambda: EvaluationDataset.from_json("{")),
        ("unknown version", lambda: EvaluationDataset.from_dict(
            {"version": 99, "cases": [{"id": "a", "query": "q"}]})),
        ("case with no id", lambda: EvaluationDataset.from_dict(
            {"cases": [{"query": "q"}]})),
        ("missing case id", lambda: dataset.get("nope")),
        ("negative tolerance", lambda: dataset.assert_no_regression(
            current={"hit_rate": 1.0}, baseline={"hit_rate": 1.0}, tolerance=-1)),
    ]:
        try:
            fn()
            print(f"   {label:<20} NO ERROR")
        except EvaluationInputError as exc:
            print(f"   {label:<20} {type(exc).__name__}: {str(exc)[:52]}...")

    print("\n   There is no to_file()/from_file(). This component returns and")
    print("   accepts JSON STRINGS, and where they are written is the")
    print("   caller's decision -- a library that writes files on request has")
    print("   to answer where it is allowed to write, and that question")
    print("   belongs to the application, not to a metrics helper.")
    print("\n   Wiring it up elsewhere takes three lines:")
    print("     dataset = EvaluationDataset.from_json(Path('eval.json').read_text())")
    print("     baseline = RetrievalMetrics(k=10).score_many(dataset.retrieval_examples()).as_metrics()")
    print("     dataset.assert_no_regression(current=current, baseline=baseline)")


if __name__ == "__main__":
    main()

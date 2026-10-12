"""Gate tests for the evaluation area (T-09g).

Three components:

* ``retrieval-metrics``      -- hit rate, MRR and recall@k against labels;
* ``answer-metrics``         -- groundedness and relevance from supplied text;
* ``evaluation-dataset``     -- a labelled format with regression support.

The four properties the task names as observable are asserted here by name:

1. hit rate, MRR and recall@k match hand-computed values,
2. a degenerate k is handled,
3. groundedness is computed only from supplied evidence,
4. a regression assertion fails on a known-worse result.

FR-9 reads: *"Evaluation components provide retrieval metrics (at least hit
rate, MRR, recall@k), answer metrics (at least faithfulness/groundedness and
answer relevance computed from provided evidence), and a dataset format with
regression-test support."* Each clause is exercised below, and the last one
end to end: the dataset's cases are scored by the metrics components and fed
back into the dataset's own assertion.

Independence (AC-13, FR-4)
--------------------------
All three manifests declare ``dependencies: []``, so each component copies
out alone. That is asserted rather than assumed: an AST walk proves no
component imports a sibling evaluation component, none imports
`rag_code_library`, and none imports anything outside a pinned stdlib set --
which is also NFR-4's "no undeclared dependencies" for this area.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "components" / "evaluation"

for directory in ("retrieval_metrics", "answer_metrics", "evaluation_dataset"):
    sys.path.insert(0, str(EVALUATION / directory))

from answer_metrics import (  # noqa: E402
    AnswerMetrics,
    AnswerMetricsError,
    AnswerMetricsInputError,
)
from evaluation_dataset import (  # noqa: E402
    FORMAT_VERSION,
    EvaluationCase,
    EvaluationDataset,
    EvaluationDatasetError,
    EvaluationInputError,
    RegressionError,
)
from retrieval_metrics import (  # noqa: E402
    DEFAULT_K,
    RetrievalMetrics,
    RetrievalMetricsError,
    RetrievalMetricsInputError,
)

# The only modules these components are allowed to import. A pinned set is
# how "no undeclared dependencies" is checked without a runtime audit: a new
# import fails this test before it fails in someone else's environment.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "json",
        "math",
        "re",
        "typing",
    }
)

COMPONENTS = ("retrieval_metrics", "answer_metrics", "evaluation_dataset")


def _source(name: str) -> str:
    return (EVALUATION / name / f"{name}.py").read_text(encoding="utf-8")


def _imports(name: str) -> set[str]:
    """Module names the component imports, from its AST.

    ``from collections.abc import ...`` is kept as ``collections.abc`` rather
    than reduced to ``collections``, so the allowed set states the actual
    module and not just its top-level package.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(_source(name))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# --------------------------------------------------------------------------- #
# Observable 1 -- hit rate, MRR and recall@k match hand-computed values
# --------------------------------------------------------------------------- #

# Worked out on paper before the component was run:
#
#   query   retrieved        relevant   top-3   hit  rank  rr     recall
#   q1      c1 c2 c3         c2         c1c2c3   1    2    1/2    1/1 = 1.0
#   q2      c5 c6 c7         c2         c5c6c7   0    --   0      0/1 = 0.0
#   q3      c3 c2            c1 c2      c3c2     1    2    1/2    1/2 = 0.5
#
#   hit_rate = (1 + 0 + 1) / 3 = 2/3
#   mrr      = (0.5 + 0 + 0.5) / 3 = 1/3
#   recall   = (1 + 0 + 0.5) / 3 = 0.5
HAND_WORKED = [
    (["c1", "c2", "c3"], ["c2"]),
    (["c5", "c6", "c7"], ["c2"]),
    (["c3", "c2"], ["c1", "c2"]),
]


def test_hit_rate_mrr_and_recall_match_hand_computed_values():
    score = RetrievalMetrics(k=3).score_many(HAND_WORKED)
    assert score.hit_rate == pytest.approx(2 / 3, abs=1e-15)
    assert score.mrr == pytest.approx(1 / 3, abs=1e-15)
    assert score.recall == pytest.approx(0.5, abs=1e-15)
    assert score.examples == 3
    assert score.k == 3


def test_each_query_matches_its_own_hand_computed_row():
    metrics = RetrievalMetrics(k=3)
    rows = [metrics.score(retrieved=r, relevant=l) for r, l in HAND_WORKED]
    assert [(row.hit, row.reciprocal_rank, row.recall, row.rank) for row in rows] == [
        (1.0, 0.5, 1.0, 2),
        (0.0, 0.0, 0.0, None),
        (1.0, 0.5, 0.5, 2),
    ]


def test_mrr_of_a_first_rank_hit_is_exactly_one():
    assert RetrievalMetrics(k=10).score(retrieved=["right"], relevant=["right"]).reciprocal_rank == 1.0


def test_the_three_metrics_move_independently():
    # The reason there are three: a change that buries the answer still hits,
    # so hit rate is unmoved while MRR collapses.
    before = RetrievalMetrics(k=10).score(retrieved=["target"], relevant=["target"])
    after = RetrievalMetrics(k=10).score(
        retrieved=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "target"],
        relevant=["target"],
    )
    assert before.hit == after.hit == 1.0
    assert before.recall == after.recall == 1.0
    assert before.reciprocal_rank == 1.0
    assert after.reciprocal_rank == 0.1


# --------------------------------------------------------------------------- #
# Observable 2 -- a degenerate k is handled
# --------------------------------------------------------------------------- #


def test_a_k_past_the_end_of_the_results_is_clamped():
    score = RetrievalMetrics(k=100).score(retrieved=["only", "two"], relevant=["two"])
    assert score.rank == 2
    assert score.recall == 1.0


def test_a_k_of_zero_is_refused_rather_than_scoring_zero():
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=0)
    assert "at least 1" in str(excinfo.value)
    # And it says why, because a 0.0 would otherwise read as a measurement.
    assert "say nothing about the retriever" in str(excinfo.value)


@pytest.mark.parametrize("k", [-1, -100])
def test_a_negative_k_is_refused(k):
    with pytest.raises(RetrievalMetricsInputError):
        RetrievalMetrics(k=k)


@pytest.mark.parametrize(("k", "kind"), [(1.5, "float"), (True, "bool"), ("10", "str")])
def test_a_k_that_is_not_an_int_is_refused(k, kind):
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=k)
    assert "must be an int" in str(excinfo.value)
    assert kind in str(excinfo.value)


def test_a_bool_k_would_mean_top_one_so_it_is_refused():
    # True == 1, so accepting it would silently mean "the first result only".
    with pytest.raises(RetrievalMetricsInputError):
        RetrievalMetrics(k=True)


def test_the_default_k_is_conventional_not_hidden():
    assert DEFAULT_K == 10
    assert RetrievalMetrics().k == 10


def test_an_empty_retrieval_with_a_label_is_a_miss_not_an_error():
    score = RetrievalMetrics(k=5).score(retrieved=[], relevant=["c1"])
    assert (score.hit, score.reciprocal_rank, score.recall, score.rank) == (0.0, 0.0, 0.0, None)


def test_an_empty_label_is_refused():
    # Nothing can be correct, so every metric would be 0 and the result would
    # look like a broken retriever rather than an unlabelled query.
    with pytest.raises(RetrievalMetricsInputError) as excinfo:
        RetrievalMetrics(k=5).score(retrieved=["a"], relevant=[])
    assert "relevant was empty" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Observable 3 -- groundedness is computed only from supplied evidence
# --------------------------------------------------------------------------- #

EVIDENCE = ["The json module serialises objects into JSON text."]


def test_the_same_answer_scores_differently_against_different_evidence():
    am = AnswerMetrics()
    answer = "json serialises objects."
    assert am.groundedness(answer, EVIDENCE) == 1.0
    assert am.groundedness(answer, ["Rust ownership moves at compile time."]) == 0.0
    # Same three terms, supplied as a different passage: still 1.0, because
    # the score reads ONLY what it was handed.
    assert am.groundedness(answer, ["json serialises objects."]) == 1.0
    # A passage that shares one term keeps the score in between.
    partial = am.groundedness(answer, ["The json module writes to disk."])
    assert 0.0 < partial < 1.0


def test_an_answer_far_from_the_evidence_scores_zero_not_one():
    # Every distinctive term absent. Had unseen terms weighted 0, the sum
    # would vanish and this would read 1.0 -- the confabulation would win.
    am = AnswerMetrics()
    assert am.groundedness(
        "The quantum tunnelling manifold repolarises the substrate.", EVIDENCE
    ) == 0.0
    # An answer built entirely from the evidence's own terms is fully grounded.
    assert am.groundedness("json serialises objects into JSON text", EVIDENCE) == 1.0


def test_the_score_is_reproducible_across_runs():
    am = AnswerMetrics()
    answer = "json serialises objects into JSON text"
    values = {am.groundedness(answer, EVIDENCE) for _ in range(100)}
    assert len(values) == 1


#: With one passage of evidence the weights are closed form, so a partial
#: match is exactly this and not a number the component told us:
#: ``idf(present) = log(4/3)``, ``idf(absent) = log(4)``.
PARTIAL = math.log(4 / 3) / (math.log(4 / 3) + math.log(4))


def test_the_partial_case_matches_the_closed_form_idf_value():
    am = AnswerMetrics()
    assert am.groundedness("alpha gamma", ["alpha beta"]) == pytest.approx(PARTIAL, abs=1e-15)
    assert am.relevance("alpha gamma", "alpha beta") == pytest.approx(PARTIAL, abs=1e-15)


def test_relevance_is_computed_against_the_query_not_the_evidence():
    am = AnswerMetrics()
    assert am.relevance("alpha beta", "alpha beta") == 1.0
    assert am.relevance("gamma delta", "alpha beta") == 0.0
    # A query with no distinctive terms has nothing to measure against.
    with pytest.raises(AnswerMetricsInputError):
        am.relevance("alpha", "")


def test_no_stemming_is_claimed_or_performed():
    am = AnswerMetrics()
    assert am.groundedness("exceptions", ["the exception was caught"]) == 0.0
    assert am.groundedness("exception", ["the exception was caught"]) == 1.0
    # "run" is absent from "running fast", so it takes the maximum weight.
    assert am.relevance("running fast", "run fast") == pytest.approx(PARTIAL, abs=1e-15)


def test_both_answer_metrics_are_produced_and_stay_in_the_unit_interval():
    am = AnswerMetrics()
    score = am.score(answer="json serialises objects", query="json serialise", evidence=EVIDENCE)
    assert sorted(score.as_metrics()) == ["groundedness", "relevance"]
    assert 0.0 <= score.groundedness <= 1.0
    assert 0.0 <= score.relevance <= 1.0


def test_an_empty_answer_scores_zero_rather_than_failing_the_evaluation():
    am = AnswerMetrics()
    assert am.groundedness("", EVIDENCE) == 0.0
    assert am.relevance("", "json serialise") == 0.0


# --------------------------------------------------------------------------- #
# Observable 4 -- a regression assertion fails on a known-worse result
# --------------------------------------------------------------------------- #

BASELINE = {"hit_rate": 0.75, "mrr": 0.50, "recall": 0.75}


@pytest.fixture
def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        cases=(
            EvaluationCase(id="q1", query="a", retrieved=("a", "b"), relevant=("a",)),
            EvaluationCase(id="q2", query="b", retrieved=("c", "a"), relevant=("a",)),
            EvaluationCase(id="q3", query="c", retrieved=("x", "y"), relevant=("z",)),
            EvaluationCase(id="q4", query="d", retrieved=("b", "d"), relevant=("d",)),
        ),
        name="gate",
    )


def test_an_improvement_passes_the_assertion(dataset):
    assert dataset.assert_no_regression(
        current={"hit_rate": 0.80, "mrr": 0.55, "recall": 0.80}, baseline=BASELINE
    ) == ()


def test_a_known_worse_result_fails_the_assertion(dataset):
    worse = {"hit_rate": 0.50, "mrr": 0.25, "recall": 0.50}
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(current=worse, baseline=BASELINE)
    message = str(excinfo.value)
    assert "hit_rate" in message
    assert "regressed" in message
    # Every offender at once, not just the first.
    assert len(excinfo.value.regressions) == 3


def test_the_regression_says_exactly_what_got_worse(dataset):
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(
            current={"hit_rate": 0.50, "mrr": 0.50, "recall": 0.50},
            baseline=BASELINE,
            tolerance=0.1,
        )
    item = excinfo.value.regressions[0]
    assert item.metric == "hit_rate"
    assert item.baseline == 0.75
    assert item.current == 0.50
    assert item.delta == pytest.approx(-0.25)
    assert item.worsening == pytest.approx(0.15)  # 0.25 drop, 0.10 tolerated


def test_a_tie_is_not_a_regression(dataset):
    assert dataset.assert_no_regression(current=BASELINE, baseline=BASELINE) == ()


def test_tolerance_absorbs_noise_but_not_damage(dataset):
    noisy = {"hit_rate": 0.74, "mrr": 0.50, "recall": 0.75}
    assert dataset.assert_no_regression(current=noisy, baseline=BASELINE, tolerance=0.05) == ()
    with pytest.raises(RegressionError):
        dataset.assert_no_regression(
            current={"hit_rate": 0.50, "mrr": 0.50, "recall": 0.50},
            baseline=BASELINE,
            tolerance=0.05,
        )


def test_a_metric_the_run_stopped_reporting_is_refused(dataset):
    # The failure mode a lenient assertion would miss: compare only what
    # survived, and hit rate can hold steady while MRR collapses.
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(current={"hit_rate": 0.90}, baseline=BASELINE)
    assert "does not report" in str(excinfo.value)


def test_lower_is_better_metrics_must_be_opted_in(dataset):
    latency = {"latency_ms": 100.0}
    assert dataset.assert_no_regression(
        current={"latency_ms": 90.0}, baseline=latency, lower_is_better=["latency_ms"]
    ) == ()
    with pytest.raises(RegressionError):
        dataset.assert_no_regression(
            current={"latency_ms": 110.0}, baseline=latency, lower_is_better=["latency_ms"]
        )


# --------------------------------------------------------------------------- #
# End to end: the three components compose, with no component importing one
# another
# --------------------------------------------------------------------------- #


def test_the_dataset_scores_through_the_metrics_and_fails_the_assertion(dataset):
    """Observable 4, run through the whole pipeline."""
    k2 = RetrievalMetrics(k=2)
    good = k2.score_many(dataset.retrieval_examples())
    # Worked out on paper: q1 hit at rank 1, q2 hit at rank 2, q3 miss,
    # q4 hit at rank 2 -> hit_rate 3/4, mrr (1+0.5+0+0.5)/4, recall 3/4.
    assert good.hit_rate == pytest.approx(0.75, abs=1e-15)
    assert good.mrr == pytest.approx(0.50, abs=1e-15)
    assert good.recall == pytest.approx(0.75, abs=1e-15)

    baseline = good.as_metrics()

    # An identical re-run passes: the assertion is stable, not flaky.
    assert dataset.assert_no_regression(current=baseline, baseline=baseline) == ()

    # A worse retrieval run -- q1 loses its relevant chunk -- must fail.
    # Worked out on paper: hits at q2 and q4 only.
    #   hit_rate = (0+1+0+1)/4 = 0.50, mrr = (0+0.5+0+0.5)/4 = 0.25
    worse = k2.score_many(
        [
            (("b", "x"), ("a",)),  # q1 now misses
            (("c", "a"), ("a",)),
            (("x", "y"), ("z",)),
            (("b", "d"), ("d",)),
        ]
    ).as_metrics()
    assert worse["hit_rate"] == pytest.approx(0.50, abs=1e-15)
    assert worse["mrr"] == pytest.approx(0.25, abs=1e-15)
    assert worse["recall"] == pytest.approx(0.50, abs=1e-15)

    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(current=worse, baseline=baseline)
    assert len(excinfo.value.regressions) == 3


def test_the_metrics_bridge_is_exactly_the_three_expected_keys(dataset):
    score = RetrievalMetrics(k=2).score_many(dataset.retrieval_examples())
    assert set(score.as_metrics()) == {"hit_rate", "mrr", "recall"}
    # These names are the contract between the components: a rename would
    # silently stop the regression check from seeing that metric at all.
    assert set(score.as_metrics()) <= set(BASELINE)


def test_the_dataset_scores_answer_cases_through_the_answer_metrics(dataset):
    case = EvaluationCase(
        id="a", query="what is json", retrieved=(), relevant=(),
        answer="The json module serialises objects into JSON text.",
        evidence=EVIDENCE,
    )
    score = AnswerMetrics().score(
        answer=case.answer or "", query=case.query, evidence=case.evidence
    )
    assert score.groundedness == 1.0
    assert set(score.as_metrics()) == {"groundedness", "relevance"}


# --------------------------------------------------------------------------- #
# Independence: AC-13, FR-4, NFR-4
# --------------------------------------------------------------------------- #


def test_no_evaluation_component_imports_a_sibling():
    for name in COMPONENTS:
        imported = _imports(name)
        siblings = {other for other in COMPONENTS if other != name}
        assert not (imported & siblings), f"{name} imports {imported & siblings}"


def test_no_evaluation_component_imports_the_package():
    # AC-13: component tests must not import `rag_code_library`, and neither
    # may the component itself -- a component that reaches into the package
    # is not copy-out-alone.
    for name in COMPONENTS:
        assert "rag_code_library" not in _imports(name)


def test_every_import_is_stdlib_and_declared():
    # NFR-4: no undeclared dependencies, for this area.
    for name in COMPONENTS:
        imported = _imports(name)
        undeclared = imported - ALLOWED_IMPORTS
        assert not undeclared, f"{name} imports undeclared {undeclared}"


def test_no_evaluation_component_reaches_the_network_or_the_filesystem():
    for name in COMPONENTS:
        source = _source(name)
        for banned in (
            "import socket",
            "import httpx",
            "import requests",
            "urllib.request",
            "import openai",
            "open(",
            "write_text",
        ):
            assert banned not in source, f"{name} contains {banned!r}"


def test_each_component_defines_its_own_error_hierarchy():
    assert issubclass(RetrievalMetricsInputError, RetrievalMetricsError)
    assert issubclass(AnswerMetricsInputError, AnswerMetricsError)
    assert issubclass(EvaluationInputError, EvaluationDatasetError)
    assert issubclass(RegressionError, EvaluationDatasetError)
    # Three separate trees: installing one component never implies another.
    assert not issubclass(RetrievalMetricsError, EvaluationDatasetError)
    assert not issubclass(AnswerMetricsError, EvaluationDatasetError)
    assert not issubclass(EvaluationDatasetError, RetrievalMetricsError)


def test_no_error_leaks_a_standard_exception_type():
    for hierarchy in (RetrievalMetricsError, AnswerMetricsError, EvaluationDatasetError):
        assert not issubclass(hierarchy, ValueError)
        assert not issubclass(hierarchy, KeyError)
        assert not issubclass(hierarchy, AssertionError)


# --------------------------------------------------------------------------- #
# FR-4 / AC-4: real implementations, each with manifest, README, example, tests
# --------------------------------------------------------------------------- #

REQUIRED_FILES = ("manifest.json", "README.md", "example.py", "tests/test_component.py")


@pytest.mark.parametrize("name", COMPONENTS)
def test_each_component_ships_a_manifest_readme_example_and_tests(name):
    for filename in REQUIRED_FILES:
        path = EVALUATION / name / filename
        assert path.is_file(), f"{name} is missing {filename}"
        assert path.stat().st_size > 200, f"{name}/{filename} is a stub"


@pytest.mark.parametrize("name", COMPONENTS)
def test_each_component_has_a_real_implementation_not_a_stub(name):
    source = _source(name)
    # A stub is a single comment line; a real implementation has classes,
    # docstrings and typed errors.
    assert len(source) > 4000, f"{name} looks like a stub ({len(source)} bytes)"
    assert "class " in source
    assert '"""' in source


@pytest.mark.parametrize("name", COMPONENTS)
def test_each_component_test_has_both_a_success_and_a_failure_path(name):
    path = EVALUATION / name / "tests" / "test_component.py"
    source = path.read_text(encoding="utf-8")
    assert "pytest.raises" in source, f"{name} tests exercise no failure path"
    tests = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(tests) >= 5, f"{name} has only {len(tests)} tests"


@pytest.mark.parametrize("name", COMPONENTS)
def test_each_manifest_declares_no_dependencies_or_packages(name):
    manifest = json.loads((EVALUATION / name / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == []
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []
    assert manifest["external_services"] == []
    assert manifest["category"] == "evaluation"
    # FR-6: each sub-component names its own source, docs, tests and example.
    assert manifest["source"] == f"{name}.py"
    assert manifest["documentation"] == "README.md"
    assert manifest["tests"] == "tests/test_component.py"


@pytest.mark.parametrize("name", COMPONENTS)
def test_each_manifest_names_an_example(name):
    manifest = json.loads((EVALUATION / name / "manifest.json").read_text(encoding="utf-8"))
    # The install line is a template; `ragkit add` substitutes the id.
    assert manifest["install"].endswith("ragkit add <id>")
    assert "No additional packages required" in manifest["install"]


def test_the_three_component_ids_are_distinct_and_match_their_directories():
    ids = set()
    for name in COMPONENTS:
        manifest = json.loads((EVALUATION / name / "manifest.json").read_text(encoding="utf-8"))
        ids.add(manifest["id"])
    assert ids == {"retrieval-metrics", "answer-metrics", "evaluation-dataset"}


def test_the_format_version_is_pinned():
    # A reader that sees a version it does not know says so rather than
    # guessing at the fields.
    assert FORMAT_VERSION == 1


# --------------------------------------------------------------------------- #
# FR-9 coverage: all three clause groups exist and are callable
# --------------------------------------------------------------------------- #


def test_fr9_retrieval_metrics_provide_at_least_the_three_named():
    for attribute in ("hit_rate", "mrr", "recall"):
        assert hasattr(RetrievalMetrics(k=3).score_many(HAND_WORKED), attribute)


def test_fr9_answer_metrics_provide_groundedness_and_relevance():
    am = AnswerMetrics()
    assert callable(am.groundedness)
    assert callable(am.relevance)
    score = am.score(answer="json text", query="json", evidence=EVIDENCE)
    assert hasattr(score, "groundedness")
    assert hasattr(score, "relevance")


def test_fr9_provides_a_dataset_format_with_regression_support():
    dataset = EvaluationDataset(cases=(EvaluationCase(id="a", query="q"),))
    assert callable(dataset.to_json)
    assert callable(EvaluationDataset.from_json)
    assert callable(dataset.assert_no_regression)
    with pytest.raises(RegressionError):
        EvaluationDataset.from_json(dataset.to_json()).assert_no_regression(
            current={"m": 0.5}, baseline={"m": 1.0}
        )

"""Component tests for `evaluation-dataset`.

Runs standalone: ``python -m pytest components/evaluation/evaluation_dataset/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the one the component exists for: a regression assertion that
FAILS on a known-worse result, naming every metric that got worse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from evaluation_dataset import (
    FORMAT_VERSION,
    EvaluationCase,
    EvaluationDataset,
    EvaluationDatasetError,
    EvaluationInputError,
    Regression,
    RegressionError,
)

BASELINE = {"hit_rate": 0.80, "mrr": 0.60, "recall": 0.70}


@pytest.fixture
def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        cases=(
            EvaluationCase(id="q1", query="how to parse", retrieved=("c1", "c2"), relevant=("c1",)),
            EvaluationCase(id="q2", query="why json", retrieved=("c5",), relevant=("c2",)),
            EvaluationCase(
                id="q3", query="when", retrieved=(), relevant=(), answer="because",
                evidence=("because",),
            ),
        ),
        name="smoke",
    )


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #


def test_a_dataset_is_ordered_and_countable(dataset):
    assert len(dataset) == 3
    assert [case.id for case in dataset] == ["q1", "q2", "q3"]
    assert dataset.name == "smoke"


def test_membership_and_lookup(dataset):
    assert "q2" in dataset
    assert "nope" not in dataset
    assert dataset.get("q1").query == "how to parse"


def test_the_default_name_is_usable():
    assert EvaluationDataset(cases=(EvaluationCase(id="a", query="q"),)).name == "dataset"


def test_a_case_reports_which_metrics_it_can_feed(dataset):
    assert dataset.get("q1").has_retrieval is True
    assert dataset.get("q1").has_answer is False
    assert dataset.get("q3").has_retrieval is False
    assert dataset.get("q3").has_answer is True


def test_retrieval_examples_drops_the_answer_only_cases(dataset):
    # q3 carries no label, so there is nothing to retrieve against.
    assert dataset.retrieval_examples() == [(("c1", "c2"), ("c1",)), (("c5",), ("c2",))]


def test_the_format_version_is_pinned():
    assert FORMAT_VERSION == 1


def test_to_dict_carries_the_documented_shape(dataset):
    data = dataset.to_dict()
    assert data["name"] == "smoke"
    assert data["version"] == FORMAT_VERSION
    assert [case["id"] for case in data["cases"]] == ["q1", "q2", "q3"]
    # answer/evidence are omitted when there is no answer, so a retrieval-only
    # case does not carry two empty fields around.
    assert "answer" not in data["cases"][0]
    assert "answer" in data["cases"][2]


# --------------------------------------------------------------------------- #
# JSON round-trip
# --------------------------------------------------------------------------- #


def test_json_round_trips_exactly(dataset):
    assert EvaluationDataset.from_json(dataset.to_json()).to_dict() == dataset.to_dict()


def test_the_round_trip_preserves_every_field(dataset):
    again = EvaluationDataset.from_json(dataset.to_json())
    assert [case.id for case in again] == [case.id for case in dataset]
    assert again.get("q3").answer == "because"
    assert again.get("q3").evidence == ("because",)
    assert again.get("q1").retrieved == ("c1", "c2")
    assert again.get("q1").relevant == ("c1",)


def test_the_json_document_is_real_json(dataset):
    parsed = json.loads(dataset.to_json())
    assert parsed["version"] == FORMAT_VERSION
    assert len(parsed["cases"]) == 3


def test_to_json_accepts_a_tighter_indentation(dataset):
    assert "\n" not in dataset.to_json(indent=None)


def test_a_case_can_be_parsed_on_its_own():
    case = EvaluationCase.from_dict(
        {"id": "x", "query": "q", "retrieved": ["a"], "relevant": ["b"]}
    )
    assert case.id == "x"
    assert case.retrieved == ("a",)
    assert case.relevant == ("b",)


# --------------------------------------------------------------------------- #
# THE headline property: a regression fails on a known-worse result
# --------------------------------------------------------------------------- #


def test_an_improvement_passes(dataset):
    assert dataset.assert_no_regression(
        current={"hit_rate": 0.85, "mrr": 0.65, "recall": 0.75}, baseline=BASELINE
    ) == ()


def test_an_exact_tie_passes(dataset):
    assert dataset.assert_no_regression(current=BASELINE, baseline=BASELINE) == ()


def test_a_known_worse_result_fails_the_assertion(dataset):
    worse = {"hit_rate": 0.70, "mrr": 0.60, "recall": 0.70}
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(current=worse, baseline=BASELINE)
    message = str(excinfo.value)
    assert "hit_rate" in message
    assert "0.8" in message
    assert "0.7" in message


def test_the_failure_carries_every_offending_metric(dataset):
    collapsed = {"hit_rate": 0.50, "mrr": 0.50, "recall": 0.50}
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(current=collapsed, baseline=BASELINE)
    regressions = excinfo.value.regressions
    assert len(regressions) == 3
    assert {item.metric for item in regressions} == {"hit_rate", "mrr", "recall"}
    # Reported all at once rather than one at a time: one run tells the whole
    # story instead of a whack-a-mole sequence of fixes.
    assert all(isinstance(item, Regression) for item in regressions)


def test_the_regression_records_the_arithmetic(dataset):
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(
            current={"hit_rate": 0.70, "mrr": 0.60, "recall": 0.70}, baseline=BASELINE
        )
    item = excinfo.value.regressions[0]
    assert item.metric == "hit_rate"
    assert item.baseline == 0.80
    assert item.current == 0.70
    assert item.delta == pytest.approx(-0.10)
    assert item.worsening == pytest.approx(0.10)
    assert item.lower_is_better is False
    assert item.tolerance == 0.0


def test_the_message_says_how_far_past_the_tolerance_it_moved(dataset):
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(
            current={"hit_rate": 0.70, "mrr": 0.60, "recall": 0.70},
            baseline=BASELINE,
            tolerance=0.05,
        )
    assert "past it by 0.05" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Tolerance
# --------------------------------------------------------------------------- #


def test_tolerance_absorbs_a_drop_inside_it(dataset):
    noisy = {"hit_rate": 0.79, "mrr": 0.60, "recall": 0.70}
    assert dataset.assert_no_regression(current=noisy, baseline=BASELINE, tolerance=0.05) == ()


def test_tolerance_does_not_absorb_a_drop_outside_it(dataset):
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(
            current={"hit_rate": 0.70, "mrr": 0.60, "recall": 0.70},
            baseline=BASELINE,
            tolerance=0.05,
        )
    assert "past it by 0.05" in str(excinfo.value)


def test_the_default_tolerance_is_zero():
    assert EvaluationDataset(cases=(EvaluationCase(id="a", query="q"),)).assert_no_regression(
        current={"m": 1.0}, baseline={"m": 1.0}
    ) == ()


# --------------------------------------------------------------------------- #
# Comparing like with like
# --------------------------------------------------------------------------- #


def test_a_baseline_metric_the_run_stopped_reporting_is_refused(dataset):
    # The important refusal. A subset comparison is precisely how a real
    # regression hides: hit rate can hold steady while MRR collapses.
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(current={"hit_rate": 0.95}, baseline=BASELINE)
    assert "does not report" in str(excinfo.value)
    assert "mrr" in str(excinfo.value)
    assert "recall" in str(excinfo.value)


def test_a_metric_the_baseline_never_had_is_not_a_regression(dataset):
    assert dataset.assert_no_regression(
        current={**BASELINE, "ndcg": 0.5}, baseline=BASELINE
    ) == ()


def test_lower_is_better_metrics_are_opt_in(dataset):
    improving = {"latency_ms": 90.0}
    worsening = {"latency_ms": 110.0}
    baseline = {"latency_ms": 100.0}
    assert dataset.assert_no_regression(
        current=improving, baseline=baseline, lower_is_better=["latency_ms"]
    ) == ()
    with pytest.raises(RegressionError):
        dataset.assert_no_regression(
            current=worsening, baseline=baseline, lower_is_better=["latency_ms"]
        )


def test_a_higher_is_better_metric_fails_without_being_opted_in(dataset):
    with pytest.raises(RegressionError):
        dataset.assert_no_regression(current={"latency_ms": 90.0}, baseline={"latency_ms": 100.0})


def test_the_tolerance_applies_in_the_right_direction_for_lower_is_better(dataset):
    baseline = {"latency_ms": 100.0}
    assert dataset.assert_no_regression(
        current={"latency_ms": 104.0},
        baseline=baseline,
        lower_is_better=["latency_ms"],
        tolerance=5,
    ) == ()
    with pytest.raises(RegressionError):
        dataset.assert_no_regression(
            current={"latency_ms": 104.0},
            baseline=baseline,
            lower_is_better=["latency_ms"],
            tolerance=3,
        )


def test_the_regression_exposes_the_lower_is_better_flag(dataset):
    with pytest.raises(RegressionError) as excinfo:
        dataset.assert_no_regression(
            current={"latency_ms": 110.0},
            baseline={"latency_ms": 100.0},
            lower_is_better=["latency_ms"],
        )
    item = excinfo.value.regressions[0]
    assert item.lower_is_better is True
    assert item.worsening == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Invalid inputs -- failure paths
# --------------------------------------------------------------------------- #


def test_an_empty_dataset_is_refused():
    # A baseline over nothing scores 0.0, which would read as a measurement
    # rather than an absence of one.
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset(cases=())
    assert "the dataset is empty" in str(excinfo.value)


def test_duplicate_case_ids_are_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset(
            cases=(
                EvaluationCase(id="a", query="q1"),
                EvaluationCase(id="a", query="q2"),
            )
        )
    assert "duplicate case id" in str(excinfo.value)
    # And it explains why: a failure must name ONE example.
    assert "must name one example" in str(excinfo.value)


def test_a_non_case_entry_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset(cases=("not a case",))  # type: ignore[arg-type]
    assert "EvaluationCase" in str(excinfo.value)


def test_cases_that_are_not_a_sequence_are_refused():
    with pytest.raises(EvaluationInputError):
        EvaluationDataset(cases="q1")  # type: ignore[arg-type]


def test_an_empty_name_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset(cases=(EvaluationCase(id="a", query="q"),), name="")
    assert "non-empty" in str(excinfo.value)


def test_looking_up_a_missing_case_names_what_is_there(dataset):
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.get("nope")
    message = str(excinfo.value)
    assert "no case with id 'nope'" in message
    assert "3 cases" in message


def test_malformed_json_is_a_typed_error_not_a_traceback():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_json("{")
    assert "not valid JSON" in str(excinfo.value)


def test_from_json_rejects_a_non_string():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_json(5)  # type: ignore[arg-type]
    assert "expects a string" in str(excinfo.value)


def test_an_unknown_format_version_is_refused_rather_than_guessed_at():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_dict({"version": 99, "cases": [{"id": "a", "query": "q"}]})
    assert "version" in str(excinfo.value)


def test_a_dataset_without_cases_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_dict({"name": "x"})
    assert "no 'cases'" in str(excinfo.value)


def test_a_non_mapping_document_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_dict([1, 2, 3])  # type: ignore[arg-type]
    assert "must be a mapping" in str(excinfo.value)


def test_a_case_without_the_required_fields_names_them():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_dict({"cases": [{"query": "q"}]})
    assert "missing required" in str(excinfo.value)
    assert "'id'" in str(excinfo.value)


def test_an_empty_case_id_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationDataset.from_dict({"cases": [{"id": "", "query": "q"}]})
    assert "must not be empty" in str(excinfo.value)


def test_a_non_string_case_id_is_refused():
    with pytest.raises(EvaluationInputError):
        EvaluationDataset.from_dict({"cases": [{"id": 5, "query": "q"}]})


def test_a_non_string_answer_is_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationCase.from_dict({"id": "a", "query": "q", "answer": 5})
    assert "string or null" in str(excinfo.value)


def test_a_bare_string_for_ids_is_refused_because_it_is_a_list_of_characters():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationCase.from_dict({"id": "a", "query": "q", "retrieved": "abc"})
    assert "single characters" in str(excinfo.value)


def test_non_string_ids_are_refused():
    with pytest.raises(EvaluationInputError) as excinfo:
        EvaluationCase.from_dict({"id": "a", "query": "q", "relevant": [5]})
    assert "only id strings" in str(excinfo.value)


def test_a_negative_tolerance_is_refused(dataset):
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(current=BASELINE, baseline=BASELINE, tolerance=-0.1)
    assert "not be negative" in str(excinfo.value)


def test_a_non_numeric_tolerance_is_refused(dataset):
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(current=BASELINE, baseline=BASELINE, tolerance="x")
    assert "must be a number" in str(excinfo.value)


def test_a_bool_tolerance_is_refused(dataset):
    with pytest.raises(EvaluationInputError):
        dataset.assert_no_regression(current=BASELINE, baseline=BASELINE, tolerance=True)


@pytest.mark.parametrize("bad", [[("a", 1)], None, "x"])
def test_scores_that_are_not_mappings_are_refused(dataset, bad):
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(current=bad, baseline=BASELINE)  # type: ignore[arg-type]
    assert "must be a mapping" in str(excinfo.value)


def test_a_non_numeric_metric_value_is_refused(dataset):
    with pytest.raises(EvaluationInputError) as excinfo:
        dataset.assert_no_regression(
            current={"hit_rate": "0.9", "mrr": 0.6, "recall": 0.7}, baseline=BASELINE
        )
    assert "must be a number" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Documented guarantees
# --------------------------------------------------------------------------- #


def test_the_errors_are_the_components_own_hierarchy():
    assert issubclass(EvaluationInputError, EvaluationDatasetError)
    assert issubclass(RegressionError, EvaluationDatasetError)
    assert not issubclass(EvaluationDatasetError, ValueError)
    assert not issubclass(EvaluationDatasetError, AssertionError)


def test_the_dataset_is_frozen(dataset):
    with pytest.raises(Exception):
        dataset.name = "tampered"  # type: ignore[misc]
    with pytest.raises(Exception):
        dataset.cases = ()  # type: ignore[misc]


def test_a_case_is_frozen():
    case = EvaluationCase(id="a", query="q")
    with pytest.raises(Exception):
        case.id = "b"  # type: ignore[misc]


def test_the_repr_summarises_the_dataset(dataset):
    assert repr(dataset) == (
        "EvaluationDataset(name='smoke', cases=3, retrieval=2, answers=1)"
    )


def test_no_network_filesystem_or_optional_dependency_is_used():
    import evaluation_dataset as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for banned in (
        "open(",
        "Path(",
        "import openai",
        "import httpx",
        "import socket",
        "import rag_code_library",
    ):
        # There is deliberately no to_file/from_file: where a JSON string is
        # written is the caller's decision, and a library that writes files
        # has to answer where it is allowed to write.
        assert banned not in source

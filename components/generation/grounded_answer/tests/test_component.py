"""Tests for the grounded-answer-contract component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

No optional package, no model, no network: the verdict is arithmetic over the
evidence vocabulary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types, which lives in its own component
# directory under components/storage. The manifest declares it a dependency,
# so `ragkit add` copies it beside this file; here we reach across to the
# sibling so the test runs from the component tree.
sys.path.insert(0, str(COMPONENT.parent.parent / "storage" / "core_types"))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
from grounded_answer import (  # noqa: E402
    Claim,
    GroundednessContract,
    GroundednessResult,
    GroundingError,
    GroundingInputError,
    tokenize,
)

import pytest  # noqa: E402


def scored(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    return core.ScoredDocument(chunk_id=cid, text=text, score=1.0, **kwargs)


def ctx(chunks, *, budget: int = 100) -> core.Context:
    items = tuple(chunks)
    return core.Context(
        text="\n\n".join(c.text for c in items),
        chunks=items,
        token_count=5,
        budget=budget,
    )


EVIDENCE = [
    scored("c0", "Python raises TypeError when a function receives an argument "
                "of the wrong type.", chunk_index=0),
    scored("c1", "The json module serialises Python objects into JSON text.", chunk_index=1),
]

GROUNDABLE = ("The json module serialises Python objects into JSON text. "
              "Python raises TypeError for a wrong argument type.")
FABRICATED = ("Python raises TypeError for a wrong argument type. "
              "The Zephyr framework uses quantum entanglement for IPC.")


@pytest.fixture
def contract():
    return GroundednessContract()


@pytest.fixture
def context():
    return ctx(EVIDENCE)


# --- success paths ------------------------------------------------------- #


def test_a_fully_supported_answer_is_grounded(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert result.grounded is True
    assert result.unsupported == ()


def test_every_supported_claim_scores_one(contract, context):
    for claim in contract.check(GROUNDABLE, context).claims:
        assert claim.support == pytest.approx(1.0)


def test_is_grounded_is_the_boolean_form(contract, context):
    assert contract.is_grounded(GROUNDABLE, context) is True


def test_the_answer_is_split_into_one_claim_per_sentence(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert len(result.claims) == 2
    assert all(isinstance(c, Claim) for c in result.claims)


def test_a_answer_without_terminators_is_one_claim(contract, context):
    assert len(contract.check("a single fragment", context).claims) == 1


def test_support_is_bounded(contract, context):
    for answer in [GROUNDABLE, FABRICATED, "The json module alone."]:
        for claim in contract.check(answer, context).claims:
            assert 0.0 <= claim.support <= 1.0


# --- fabrication is what gets caught -------------------------------------- #


def test_a_fabricated_claim_fails_the_answer(contract, context):
    assert contract.check(FABRICATED, context).grounded is False


def test_the_fabricated_claim_is_named(contract, context):
    result = contract.check(FABRICATED, context)
    assert result.unsupported == (
        "The Zephyr framework uses quantum entanglement for IPC.",
    )


def test_the_fabricated_claim_scores_zero(contract, context):
    failed = [c for c in contract.check(FABRICATED, context).claims if not c.supported]
    assert len(failed) == 1
    assert failed[0].support == 0.0


def test_a_wholly_invented_answer_scores_zero(contract, context):
    result = contract.check("Zephyr quantum entanglement teleportation.", context)
    assert result.grounded is False
    assert result.claims[0].support == 0.0


def test_one_bad_claim_fails_the_whole_answer(contract, context):
    result = contract.check("The json module serialises objects. GPT-9 runs at 400 teraflops.",
                            context)
    assert result.grounded is False
    assert result.claims[0].supported is True
    assert result.claims[1].supported is False


def test_an_invented_number_is_weighted_heavily(contract, context):
    """Numbers are the most distinctive thing in a sentence."""
    result = contract.check("Python raises TypeError 918273645.", context)
    assert result.claims[0].support < 0.5
    assert result.grounded is False


def test_a_near_quote_beats_a_paraphrase(contract, context):
    """Documented limit: support is lexical, so paraphrase costs something."""
    near = contract.check("The json module serialises Python objects.", context)
    loose = contract.check("JSON conversion turns Python values into text.", context)
    assert near.claims[0].support > loose.claims[0].support


def test_a_stopwords_only_claim_passes(contract, context):
    """It makes no distinctive claim, so it cannot be an unsupported one."""
    assert contract.check("It is.", context).grounded is True


# --- the threshold -------------------------------------------------------- #


def test_a_low_threshold_admits_a_weak_claim(context):
    weak = "The json module does something with objects perhaps."
    assert GroundednessContract(min_support=0.0).check(weak, context).grounded is True
    assert GroundednessContract(min_support=1.0).check(weak, context).grounded is False


def test_a_zero_support_claim_fails_at_every_positive_threshold(context):
    """min_support is a plain threshold; 0.0 is an opt-out, not a stricter bar."""
    for threshold in (0.01, 0.25, 0.5, 1.0):
        result = GroundednessContract(min_support=threshold).check(FABRICATED, context)
        claim = [c for c in result.claims if "Zephyr" in c.text][0]
        assert claim.support == 0.0
        assert claim.supported is False, threshold
        assert result.grounded is False, threshold


def test_min_support_zero_disables_the_check_entirely(context):
    """Documented, not accidental."""
    result = GroundednessContract(min_support=0.0).check(
        "The Zephyr framework uses quantum entanglement for IPC.", context
    )
    assert result.claims[0].support == 0.0
    assert result.grounded is True


def test_the_threshold_is_recorded():
    assert GroundednessContract(min_support=0.25).min_support == 0.25


# --- reporting ------------------------------------------------------------ #


def test_the_report_names_each_claim(contract, context):
    report = contract.check(FABRICATED, context).report()
    assert "FAIL" in report
    assert "NOT GROUNDED" in report


def test_the_report_says_grounded_when_it_is(contract, context):
    assert "GROUNDED" in contract.check(GROUNDABLE, context).report()
    assert "NOT GROUNDED" not in contract.check(GROUNDABLE, context).report()


def test_the_result_is_a_typed_value(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert isinstance(result, GroundednessResult)
    assert isinstance(result.claims, tuple)


def test_the_repr_shows_the_threshold(contract):
    assert "min_support=0.5" in repr(contract)


# --- failure paths -------------------------------------------------------- #


@pytest.mark.parametrize("answer", ["", "   ", "\t\n"])
def test_an_empty_answer_is_rejected(contract, context, answer):
    with pytest.raises(GroundingInputError) as exc:
        contract.check(answer, context)
    assert "vacuously" in str(exc.value)


@pytest.mark.parametrize("answer", [None, 3, ["a"], {}])
def test_a_non_string_answer_is_rejected(contract, context, answer):
    with pytest.raises(GroundingInputError):
        contract.check(answer, context)


def test_a_non_context_is_rejected(contract):
    with pytest.raises(GroundingInputError) as exc:
        contract.check("an answer", "not a context")
    assert "must be a Context" in str(exc.value)


def test_an_empty_context_is_rejected(contract):
    with pytest.raises(GroundingInputError) as exc:
        contract.check("an answer", ctx([]))
    assert "no passages" in str(exc.value)


@pytest.mark.parametrize("value", [1.5, -0.1, 2, True, "0.5", None, float("nan")])
def test_an_unusable_min_support_is_rejected(value):
    with pytest.raises(GroundingInputError):
        GroundednessContract(min_support=value)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_a_valid_min_support_is_accepted(value):
    assert GroundednessContract(min_support=value).min_support == value


def test_a_non_frozenset_stopwords_is_rejected():
    with pytest.raises(GroundingInputError):
        GroundednessContract(stopwords=set())


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(GroundingInputError):
        GroundednessContract(name=name)


def test_tokenize_rejects_a_non_string():
    with pytest.raises(GroundingInputError):
        tokenize(3)


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The JSON Module") == frozenset({"json", "module"})


def test_the_errors_form_a_hierarchy():
    assert issubclass(GroundingInputError, GroundingError)
    assert not issubclass(GroundingError, ValueError)


def test_identical_passages_are_still_counted_twice(context):
    """IDF is over passages; collapsing duplicates would corrupt it."""
    duplicated = ctx([EVIDENCE[0], EVIDENCE[0], EVIDENCE[1]])
    result = GroundednessContract().check(GROUNDABLE, duplicated)
    assert result.grounded is True


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["category"] == "generation"
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["external_services"] == []

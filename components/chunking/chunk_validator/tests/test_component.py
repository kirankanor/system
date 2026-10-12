"""Component tests for `chunk-validator`.

Runs standalone: ``python -m pytest components/chunking/chunk_validator/tests``
from the repository root. Does not import `rag_code_library`.

The property under test is not "it finds bad chunks" -- that is easy. It is
the guarantee the orchestration layer leans on: **no chunk reaches `accepted`
carrying an unaddressed fatal finding**, and **no chunk ever disappears
without a finding saying why**. Every test that ends with an outcome checks
both halves of that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from chunk_validator import (
    DEFAULT_ANOMALY_MARKERS,
    ChunkValidation,
    ChunkValidator,
    FatalPolicy,
    Finding,
    Rule,
    Severity,
    UnrepairableChunk,
    ValidationError,
    ValidationOutcome,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Mirrors the core-types contract, including optional identity fields."""

    text: str
    document_id: str | None = None
    source_name: str | None = None
    chunk_id: str | None = None
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MinimalChunk:
    """Only the attributes the validator is required to read."""

    text: str
    metadata: dict = field(default_factory=dict)


def good(**overrides) -> Chunk:
    base = dict(
        text="A perfectly ordinary sentence of sufficient length.",
        document_id="doc-1",
        source_name="guide.md",
        chunk_id="c-1",
        chunk_index=0,
        page_start=1,
        page_end=1,
    )
    base.update(overrides)
    return Chunk(**base)


def assert_nothing_lost(outcome: ValidationOutcome, total: int) -> None:
    """The invariant: one result per chunk, and every chunk accounted for."""
    assert len(outcome.results) == total
    assert (
        len(outcome.accepted)
        + len(outcome.rejected)
        + len(outcome.quarantined)
        == total
    )


def assert_accepted_are_safe(validator: ChunkValidator, outcome: ValidationOutcome) -> None:
    """The property the orchestration layer depends on.

    Re-inspects every accepted chunk rather than trusting the run's own bookkeeping:
    a repaired chunk still carries its *original* fatal finding in ``results``
    (that is what ``repairs`` records), so only a fresh inspection can prove the
    accepted set is genuinely clean.
    """
    for chunk in outcome.accepted:
        fresh = validator.inspect(chunk)
        assert not fresh.is_fatal, (
            f"chunk reached accepted still carrying a fatal finding: "
            f"{[f.rule.value for f in fresh.findings if f.is_fatal]}"
        )


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_a_clean_chunk_produces_no_findings():
    result = ChunkValidator().inspect(good())
    assert result.is_valid
    assert result.findings == ()
    assert not result.is_fatal


def test_a_clean_batch_is_all_accepted_and_reports_itself():
    chunks = [good(chunk_index=i, chunk_id=f"c-{i}") for i in range(3)]
    outcome = ChunkValidator().validate(chunks)
    assert len(outcome.accepted) == 3
    assert outcome.held() == ()
    assert outcome.is_clean
    assert outcome.fatal_count == 0


def test_one_result_comes_back_per_chunk_in_order():
    chunks = [good(chunk_id=f"c-{i}", chunk_index=i) for i in range(5)]
    chunks.append(good(chunk_id="", chunk_index=5))  # fatal
    outcome = ChunkValidator().validate(chunks)
    assert [result.index for result in outcome.results] == list(range(6))
    assert outcome.results[5].chunk_id is None


def test_results_carry_the_chunk_id_so_a_finding_is_traceable():
    outcome = ChunkValidator().validate([good(chunk_id="c-42")])
    assert outcome.results[0].chunk_id == "c-42"


def test_warnings_are_recorded_without_removing_the_chunk():
    short = good(text="too short")
    outcome = ChunkValidator(min_chars=20).validate([short])
    assert len(outcome.accepted) == 1
    assert outcome.results[0].rules == (Rule.TEXT_TOO_SHORT,)
    assert outcome.results[0].is_fatal is False
    assert outcome.warning_count == 1


def test_a_missing_page_range_is_a_warning_not_a_refusal():
    outcome = ChunkValidator().validate([good(page_start=None, page_end=None)])
    assert len(outcome.accepted) == 1
    assert outcome.results[0].rules == (Rule.MISSING_SOURCE_LOCATION,)
    assert not outcome.results[0].is_fatal


def test_an_extraction_anomaly_is_a_warning_named_by_its_metadata_key():
    chunk = good(metadata={"extraction_warning": "two lines dropped"})
    outcome = ChunkValidator().validate([chunk])
    assert outcome.results[0].rules == (Rule.EXTRACTION_ANOMALY,)
    assert not outcome.results[0].is_fatal
    assert "two lines dropped" in outcome.results[0].findings[0].message


def test_a_truthy_marker_only_fires_when_present():
    outcome = ChunkValidator().validate([good(metadata={"extraction_warning": ""})])
    assert outcome.is_clean


def test_a_marker_can_be_supplied_by_the_caller():
    validator = ChunkValidator(anomaly_markers=("gutter_lost",))
    outcome = validator.validate([good(metadata={"gutter_lost": "3 lines"})])
    assert Rule.EXTRACTION_ANOMALY in outcome.results[0].rules
    # The default marker is no longer watched.
    other = validator.validate([good(metadata={"extraction_warning": "x"})])
    assert other.is_clean


def test_the_default_marker_set_is_the_documented_one():
    assert "extraction_warning" in DEFAULT_ANOMALY_MARKERS


# ---------------------------------------------------------------------------
# Failure paths -- each fatal rule, and what each policy does with it
# ---------------------------------------------------------------------------


def test_empty_text_is_fatal_and_nothing_can_repair_it():
    outcome = ChunkValidator().validate([good(text="")])
    assert outcome.results[0].rules == (Rule.EMPTY_TEXT,)
    assert outcome.results[0].is_fatal
    assert len(outcome.quarantined) == 1
    assert outcome.accepted == ()


def test_whitespace_only_text_is_fatal_because_it_retrieves_nothing():
    outcome = ChunkValidator().validate([good(text="   \n\t  \n")])
    assert outcome.results[0].rules == (Rule.WHITESPACE_ONLY_TEXT,)
    assert outcome.results[0].is_fatal
    assert not outcome.accepted


def test_non_string_text_is_fatal_and_says_what_it_got():
    outcome = ChunkValidator().validate([good(text=None)])  # type: ignore[arg-type]
    assert Rule.EMPTY_TEXT in outcome.results[0].rules
    assert "NoneType" in outcome.results[0].findings[0].message


def test_an_oversized_chunk_is_fatal_because_it_will_not_fit_the_model():
    outcome = ChunkValidator(max_chars=100).validate([good(text="x" * 101)])
    assert Rule.TEXT_TOO_LONG in outcome.results[0].rules
    assert outcome.results[0].is_fatal
    assert not outcome.accepted


def test_a_missing_document_id_is_fatal_because_the_chunk_cannot_be_cited():
    outcome = ChunkValidator().validate([good(document_id=None)])
    assert Rule.MISSING_DOCUMENT_ID in outcome.results[0].rules
    assert outcome.results[0].is_fatal


def test_a_blank_document_id_counts_as_missing():
    outcome = ChunkValidator().validate([good(document_id="   ")])
    assert Rule.MISSING_DOCUMENT_ID in outcome.results[0].rules


def test_a_missing_chunk_id_is_fatal_under_the_default_policy():
    outcome = ChunkValidator().validate([good(chunk_id=None)])
    assert Rule.MISSING_CHUNK_ID in outcome.results[0].rules
    assert len(outcome.quarantined) == 1
    assert outcome.repairs == ()


def test_a_non_string_chunk_id_is_fatal():
    outcome = ChunkValidator().validate([good(chunk_id=12345)])  # type: ignore[arg-type]
    assert Rule.INVALID_CHUNK_ID in outcome.results[0].rules
    assert outcome.results[0].is_fatal


def test_an_inverted_page_range_is_fatal():
    outcome = ChunkValidator().validate([good(page_start=5, page_end=2)])
    assert Rule.INVALID_PAGE_RANGE in outcome.results[0].rules
    assert outcome.results[0].is_fatal
    assert "inverted" in outcome.results[0].findings[0].message


def test_a_page_range_below_one_is_fatal():
    outcome = ChunkValidator().validate([good(page_start=0, page_end=3)])
    assert Rule.INVALID_PAGE_RANGE in outcome.results[0].rules
    assert "below 1" in outcome.results[0].findings[0].message


def test_a_non_integral_page_range_is_fatal():
    outcome = ChunkValidator().validate([good(page_start="1", page_end=2)])  # type: ignore[arg-type]
    assert Rule.INVALID_PAGE_RANGE in outcome.results[0].rules


def test_required_metadata_must_be_present_and_non_empty():
    validator = ChunkValidator(required_metadata=("language", "section"))
    chunk = good(metadata={"language": "en", "section": "  "})
    outcome = validator.validate([chunk])
    assert outcome.results[0].rules == (Rule.MISSING_REQUIRED_METADATA,)
    assert "section" in outcome.results[0].findings[0].message


def test_all_required_metadata_present_is_clean():
    validator = ChunkValidator(required_metadata=("language",))
    assert validator.validate([good(metadata={"language": "en"})]).is_clean


def test_every_finding_gets_a_human_readable_message():
    outcome = ChunkValidator(min_chars=50).validate(
        [good(text="x", document_id=None, chunk_id=None, page_start=3, page_end=1)]
    )
    for finding in outcome.results[0].findings:
        assert finding.message
        assert isinstance(finding.message, str)
        assert finding.rule in Rule


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_quarantine_holds_the_chunk_with_its_findings():
    outcome = ChunkValidator(fatal_policy=FatalPolicy.QUARANTINE).validate([good(text="")])
    assert len(outcome.quarantined) == 1
    assert outcome.rejected == ()
    assert outcome.quarantined[0].findings[0].rule is Rule.EMPTY_TEXT
    assert outcome.quarantined[0].repair_failed is False


def test_reject_drops_from_accepted_but_still_returns_the_chunk():
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REJECT).validate([good(text="")])
    assert len(outcome.rejected) == 1
    assert outcome.quarantined == ()
    assert outcome.rejected[0].chunk.text == ""


def test_held_combines_both_lists_regardless_of_policy():
    chunks = [good(text=""), good(page_start=None, page_end=None)]
    for policy in FatalPolicy:
        outcome = ChunkValidator(fatal_policy=policy).validate(chunks)
        if policy is FatalPolicy.REJECT:
            assert len(outcome.held()) == 1  # only the empty one is fatal
        assert_nothing_lost(outcome, 2)


def test_a_mixed_batch_lets_the_good_through_under_every_policy():
    chunks = [good(chunk_id="ok"), good(text=""), good(chunk_id="also-ok")]
    for policy in FatalPolicy:
        validator = ChunkValidator(fatal_policy=policy, min_chars=5)
        outcome = validator.validate(chunks)
        assert len(outcome.accepted) == 2
        assert_nothing_lost(outcome, 3)
        assert_accepted_are_safe(validator, outcome)


# ---------------------------------------------------------------------------
# Repairs
# ---------------------------------------------------------------------------


def test_a_missing_chunk_id_is_derived_not_invented():
    chunk = good(chunk_id=None, document_id="doc-9", chunk_index=4)
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([chunk])

    assert len(outcome.accepted) == 1
    assert len(outcome.repairs) == 1
    assert outcome.repairs[0].rule is Rule.MISSING_CHUNK_ID
    assert "derived" in outcome.repairs[0].description
    derived = outcome.accepted[0]
    assert derived.chunk_id.startswith("doc-9-4-")
    assert len(derived.chunk_id.rsplit("-", 1)[-1]) == 12


def test_a_derived_chunk_id_is_deterministic():
    validator = ChunkValidator(fatal_policy=FatalPolicy.REPAIR)
    a = validator.validate([good(chunk_id=None)]).accepted[0]
    b = validator.validate([good(chunk_id=None)]).accepted[0]
    assert a.chunk_id == b.chunk_id


def test_a_repair_never_mutates_a_frozen_chunk_the_caller_still_holds():
    original = good(chunk_id=None)
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([original])
    assert original.chunk_id is None  # untouched
    assert outcome.accepted[0] is not original
    assert outcome.accepted[0].chunk_id


def test_an_inverted_page_range_is_swapped():
    chunk = good(page_start=7, page_end=2)
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([chunk])
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].page_start == 2
    assert outcome.accepted[0].page_end == 7
    assert "swapped" in outcome.repairs[0].description


def test_a_page_range_below_one_is_held_because_swapping_would_not_fix_it():
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate(
        [good(page_start=0, page_end=3)]
    )
    assert outcome.repairs == ()
    assert len(outcome.quarantined) == 1
    assert outcome.quarantined[0].repair_failed is True


def test_unrepairable_fatal_findings_hold_the_chunk_and_are_all_reported():
    chunk = good(text="", document_id=None)
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([chunk])
    held = outcome.quarantined[0]
    assert held.repair_failed is True
    assert {Rule.EMPTY_TEXT, Rule.MISSING_DOCUMENT_ID} <= set(
        finding.rule for finding in held.findings
    )


def test_repair_also_covers_an_invalid_chunk_id():
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate(
        [good(chunk_id=99)]
    )
    assert len(outcome.accepted) == 1
    assert isinstance(outcome.accepted[0].chunk_id, str)
    assert outcome.repairs[0].rule is Rule.INVALID_CHUNK_ID


def test_a_non_dataclass_chunk_is_edited_in_place_because_there_is_no_copy():
    """A dataclass is always copied, frozen or not. A plain object cannot be."""

    class Mutable:
        def __init__(self) -> None:
            self.text = "mutable content here"
            self.document_id = "d"
            self.source_name = "s.md"
            self.chunk_id = None
            self.chunk_index = 0
            self.page_start = 1
            self.page_end = 1
            self.metadata = {}

    chunk = Mutable()
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([chunk])
    assert chunk.chunk_id is not None  # edited in place: no copy available
    assert outcome.accepted[0] is chunk
    assert len(outcome.accepted) == 1


def test_a_non_frozen_dataclass_is_still_copied_not_edited():
    """Uniformity over cleverness: every dataclass is copied, frozen or not."""

    @dataclass
    class Loose:
        text: str
        document_id: str | None = None
        chunk_id: str | None = None
        chunk_index: int = 0
        page_start: int | None = 1
        page_end: int | None = 1
        metadata: dict = field(default_factory=dict)

    chunk = Loose(text="loose content here", document_id="d")
    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate([chunk])
    assert chunk.chunk_id is None  # untouched
    assert outcome.accepted[0] is not chunk
    assert outcome.accepted[0].chunk_id


def test_a_chunk_that_cannot_be_repaired_at_all_is_held_not_guessed():
    """A frozen object that refuses both replace() and attribute writes."""

    class Inflexible:
        __slots__ = ("text", "metadata")

        def __init__(self) -> None:
            self.text = "some content"
            self.metadata = {}

    outcome = ChunkValidator(fatal_policy=FatalPolicy.REPAIR).validate(
        [Inflexible()]  # no document_id, no chunk_id, no page range
    )
    # Missing document_id is fatal and unrepairable, so it is held.
    assert len(outcome.quarantined) == 1
    assert outcome.quarantined[0].repair_failed is True


# ---------------------------------------------------------------------------
# One result per chunk; nothing vanishes
# ---------------------------------------------------------------------------


def test_no_chunk_ever_vanishes_from_the_outcome():
    chunks = [
        good(chunk_id="a"),
        good(text=""),
        good(chunk_id=None),
        good(text="   "),
        good(page_start=4, page_end=1),
        good(chunk_id="b"),
    ]
    for policy in FatalPolicy:
        validator = ChunkValidator(min_chars=5, fatal_policy=policy)
        outcome = validator.validate(chunks)
        assert_nothing_lost(outcome, 6)
        assert_accepted_are_safe(validator, outcome)


def test_the_accepted_set_contains_no_chunk_with_a_fatal_result():
    chunks = [good(text=""), good(chunk_id="fine")]
    outcome = ChunkValidator().validate(chunks)
    fatal_indices = {r.index for r in outcome.results if r.is_fatal}
    assert fatal_indices == {0}
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].chunk_id == "fine"


def test_results_are_ordered_exactly_as_the_input_was():
    chunks = [good(text="", chunk_id=f"c-{i}") for i in range(4)]
    outcome = ChunkValidator().validate(chunks)
    assert [r.index for r in outcome.results] == [0, 1, 2, 3]


def test_an_empty_batch_is_valid_and_empty():
    outcome = ChunkValidator().validate([])
    assert outcome.results == ()
    assert outcome.accepted == ()
    assert outcome.is_clean


# ---------------------------------------------------------------------------
# Duck typing: what the validator will and will not accept
# ---------------------------------------------------------------------------


def test_a_chunk_with_no_text_attribute_is_a_caller_error_not_a_finding():
    with pytest.raises(UnrepairableChunk) as excinfo:
        ChunkValidator().inspect(object(), index=3)
    assert "index 3" in str(excinfo.value)
    assert "'text'" in str(excinfo.value)


def test_a_chunk_with_no_metadata_attribute_is_a_caller_error():
    class TextOnly:
        """Has `text` but not `metadata` -- still a caller error."""

        text = "some text"

    with pytest.raises(UnrepairableChunk) as excinfo:
        ChunkValidator().inspect(TextOnly())
    assert "metadata" in str(excinfo.value)


def test_a_chunk_with_no_text_attribute_fails_first_and_names_the_type():
    with pytest.raises(UnrepairableChunk) as excinfo:
        ChunkValidator().inspect("a bare string")
    assert "'text'" in str(excinfo.value)
    assert "str" in str(excinfo.value)


def test_a_minimal_chunk_missing_all_identity_fields_reports_each_gap():
    outcome = ChunkValidator().validate([MinimalChunk(text="standalone text here")])
    rules = set(outcome.results[0].rules)
    assert Rule.MISSING_DOCUMENT_ID in rules
    assert Rule.MISSING_CHUNK_ID in rules
    assert Rule.MISSING_SOURCE_LOCATION in rules


def test_the_validator_imports_only_the_standard_library():
    """Structural typing is the point: no chunk class, no framework, is imported.

    Checked by walking the AST rather than by grepping for a phrase -- the
    module's own docstring *mentions* ``core_types``, and a substring check
    would have failed on prose rather than on a dependency.
    """
    import ast
    import sys as _sys

    source = Path(__file__).resolve().parents[1] / "chunk_validator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    stdlib = set(_sys.stdlib_module_names)
    external = roots - stdlib - {"chunk_validator"}

    assert external == set(), f"imports something beyond the stdlib: {external}"
    assert "rag_code_library" not in imported
    assert not any(name.startswith("core_types") for name in imported)


# ---------------------------------------------------------------------------
# Sizes and severities
# ---------------------------------------------------------------------------


def test_min_and_max_boundaries_are_inclusive():
    text = "x" * 10
    assert ChunkValidator(min_chars=10, max_chars=10).inspect(
        good(text=text, document_id="d", chunk_id="c", page_start=1, page_end=1)
    ).is_valid


def test_size_of_replaces_the_size_function_entirely():
    calls: list[int] = []

    def tokens(text: str) -> int:
        calls.append(len(text))
        return len(text.split())

    validator = ChunkValidator(min_chars=100, size_of=tokens)
    # Two whitespace-separated words, but 60+ characters.
    outcome = validator.validate(
        [good(text="word " * 20, document_id="d", chunk_id="c", page_start=1, page_end=1)]
    )
    # Measured in tokens: 20 words, well below the 100 minimum.
    assert Rule.TEXT_TOO_SHORT in outcome.results[0].rules
    assert calls  # the custom measurer, not len(), was used


def test_a_severity_can_be_overridden_per_rule():
    validator = ChunkValidator(
        min_chars=50,
        severities={Rule.TEXT_TOO_SHORT: Severity.FATAL},
    )
    outcome = validator.validate([good(text="short", document_id="d", chunk_id="c")])
    assert outcome.results[0].is_fatal
    assert not outcome.accepted
    assert validator.severities[Rule.TEXT_TOO_SHORT] is Severity.FATAL


def test_an_override_does_not_disturb_other_rules():
    validator = ChunkValidator(severities={Rule.TEXT_TOO_SHORT: Severity.FATAL})
    assert validator.severities[Rule.EMPTY_TEXT] is Severity.FATAL
    assert validator.severities[Rule.MISSING_SOURCE_LOCATION] is Severity.WARNING


def test_the_default_severity_of_a_warning_rule_is_warning():
    validator = ChunkValidator()
    assert validator.severities[Rule.MISSING_SOURCE_LOCATION] is Severity.WARNING
    assert validator.severities[Rule.EMPTY_TEXT] is Severity.FATAL


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("min_chars", -1),
        ("max_chars", -1),
        ("min_chars", True),
        ("max_chars", "10"),
    ],
)
def test_a_non_positive_size_bound_is_refused(keyword, value):
    with pytest.raises(ValidationError):
        ChunkValidator(**{keyword: value})


def test_min_cannot_exceed_max():
    with pytest.raises(ValidationError) as excinfo:
        ChunkValidator(min_chars=100, max_chars=10)
    assert "cannot exceed" in str(excinfo.value)


def test_an_unknown_policy_is_refused():
    with pytest.raises(ValidationError):
        ChunkValidator(fatal_policy="just-drop-them")  # type: ignore[arg-type]


def test_a_blank_required_metadata_key_is_refused():
    with pytest.raises(ValidationError):
        ChunkValidator(required_metadata=["", "language"])


def test_a_non_callable_size_of_is_refused():
    with pytest.raises(ValidationError):
        ChunkValidator(size_of=10)  # type: ignore[arg-type]


def test_an_unknown_rule_in_an_override_is_refused():
    with pytest.raises(ValueError):
        ChunkValidator(severities={"not-a-rule": Severity.FATAL})  # type: ignore[dict-item]


def test_a_non_severity_value_in_an_override_is_refused():
    with pytest.raises(ValidationError):
        ChunkValidator(severities={Rule.TEXT_TOO_SHORT: "fatal"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Values are values
# ---------------------------------------------------------------------------


def test_results_and_findings_are_immutable():
    result = ValidationResult(index=0, chunk_id="c", findings=())
    with pytest.raises(AttributeError):
        result.chunk_id = None  # type: ignore[misc]
    finding = Finding(rule=Rule.EMPTY_TEXT, severity=Severity.FATAL, message="x")
    with pytest.raises(AttributeError):
        finding.message = "y"  # type: ignore[misc]


def test_a_finding_reports_its_own_fatality():
    assert Finding(Rule.EMPTY_TEXT, Severity.FATAL, "x").is_fatal
    assert not Finding(Rule.TEXT_TOO_SHORT, Severity.WARNING, "x").is_fatal


def test_the_outcome_helpers_agree_with_the_results():
    # min_chars=6: "short" (5 chars) warns, while the default text passes.
    chunks = [good(text="short", document_id="d", chunk_id="c"), good()]
    outcome = ChunkValidator(min_chars=6).validate(chunks)
    assert outcome.warning_count == 1
    assert outcome.fatal_count == 0  # a warning rule stays a warning
    assert not outcome.is_clean
    assert len(outcome.accepted) == 2  # a warning never removes a chunk


def test_the_component_imports_with_no_third_party_packages():
    import chunk_validator

    source = Path(chunk_validator.__file__).read_text(encoding="utf-8")
    for forbidden in ("pydantic", "numpy", "pandas", "requests"):
        assert f"import {forbidden}" not in source


def test_the_fatal_policy_is_exposed_so_a_run_summary_can_record_it():
    validator = ChunkValidator(fatal_policy=FatalPolicy.REPAIR)
    assert validator.fatal_policy is FatalPolicy.REPAIR


def test_an_empty_metadata_mapping_is_not_an_error():
    outcome = ChunkValidator(required_metadata=()).validate([good(metadata={})])
    assert outcome.is_clean

"""Component tests for `chunk-deduplicator`.

Runs standalone: ``python -m pytest components/chunking/chunk_deduplicator/tests``
from the repository root. Does not import `rag_code_library`.

Two contracts are under test, and neither is "the function returns":

* **FR-43** -- a removal always leaves behind a record of what it removed and
  where it came from, and *only* exact content equality counts. Every
  near-miss, every case-only difference, every template variant must survive.
* **FR-44** -- nothing is reused unless the stored fingerprint provably matches,
  and when it does not, the refusal names the field that differs.

The second half is the one that fails silently in production, so it is tested
for *refusal* at least as hard as for acceptance.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from chunk_deduplicator import (
    Appearance,
    DedupError,
    DedupOutcome,
    Deduplicator,
    DuplicateGroup,
    EmbeddingReuseError,
    EmbeddingReuseGate,
    FingerprintMismatch,
    IndexFingerprint,
    MissingChunkText,
    Scope,
    default_normalize,
    fingerprint_of,
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


def chunk(text: str, document: str | None, chunk_id: str, index: int = 0, **kw) -> Chunk:
    return Chunk(
        text=text,
        document_id=document,
        source_name=f"{document}.md" if document else None,
        chunk_id=chunk_id,
        chunk_index=index,
        **kw,
    )


def assert_arithmetic_out_holds(outcome: DedupOutcome, total: int) -> None:
    """Nothing is destroyed: survivors + removals is always the input size."""
    assert len(outcome.retained) + outcome.removed == total


# ---------------------------------------------------------------------------
# Success paths -- what dedup does when it finds something
# ---------------------------------------------------------------------------


def test_a_clean_corpus_is_untouched():
    chunks = [chunk(f"distinct text {i}", "doc", f"c-{i}") for i in range(4)]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.is_clean
    assert outcome.removed == 0
    assert outcome.retained == tuple(chunks)


def test_an_empty_batch_is_clean_and_empty():
    outcome = Deduplicator().dedupe([])
    assert outcome.is_clean
    assert outcome.retained == ()
    assert outcome.groups == ()


def test_an_exact_duplicate_is_removed_and_the_survivor_is_the_first_one():
    chunks = [
        chunk("the same words", "a", "first", 0),
        chunk("other content", "a", "middle", 1),
        chunk("the same words", "a", "second", 2),
    ]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.removed == 1
    assert [c.chunk_id for c in outcome.retained] == ["first", "middle"]
    assert outcome.groups[0].retained.chunk_id == "first"


def test_output_preserves_input_order():
    chunks = [chunk(f"text {i}", "d", f"c-{i}") for i in range(6)]
    chunks[3] = chunk("text 0", "d", "dup", 3)
    outcome = Deduplicator().dedupe(chunks)
    assert [c.chunk_id for c in outcome.retained] == ["c-0", "c-1", "c-2", "c-4", "c-5"]


def test_retained_chunks_are_the_original_objects_not_wrappers():
    chunks = [chunk("keep me exactly as I was", "a", "c")]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.retained[0] is chunks[0]


def test_re_running_deduplication_on_the_result_is_a_no_op():
    chunks = [
        chunk("repeated", "a", "x"),
        chunk("repeated", "b", "y"),
        chunk("unique", "a", "z"),
    ]
    once = Deduplicator().dedupe(chunks)
    twice = Deduplicator().dedupe(once.retained)
    assert twice.is_clean
    assert twice.removed == 0
    assert twice.retained == once.retained


def test_the_same_input_produces_the_same_output_every_time():
    chunks = [chunk("same", "a", "1"), chunk("same", "a", "2"), chunk("other", "b", "3")]
    first = Deduplicator().dedupe(chunks)
    second = Deduplicator().dedupe(chunks)
    assert first.retained == second.retained
    assert first.groups == second.groups


# ---------------------------------------------------------------------------
# FR-43: scopes, and what each one promises not to remove
# ---------------------------------------------------------------------------


def test_document_scope_collapses_duplicates_within_one_document():
    chunks = [
        chunk("repeated content", "a", "a-0"),
        chunk("repeated content", "a", "a-1"),
    ]
    outcome = Deduplicator(scope=Scope.DOCUMENT).dedupe(chunks)
    assert outcome.removed == 1
    assert outcome.groups[0].level == "document"


def test_document_scope_keeps_identical_text_in_two_documents():
    """Removing it would leave one document silently incomplete."""
    chunks = [
        chunk("shared paragraph", "a", "a-0"),
        chunk("shared paragraph", "b", "b-0"),
    ]
    outcome = Deduplicator(scope=Scope.DOCUMENT).dedupe(chunks)
    assert outcome.is_clean
    assert outcome.removed == 0
    assert len(outcome.retained) == 2


def test_corpus_scope_collapses_across_documents():
    chunks = [
        chunk("shared paragraph", "a", "a-0"),
        chunk("shared paragraph", "b", "b-0"),
    ]
    outcome = Deduplicator(scope=Scope.CORPUS).dedupe(chunks)
    assert outcome.removed == 1
    assert outcome.groups[0].level == "corpus"


def test_corpus_scope_also_collapses_within_a_document():
    chunks = [
        chunk("repeated content", "a", "a-0"),
        chunk("repeated content", "a", "a-1"),
    ]
    outcome = Deduplicator(scope=Scope.CORPUS).dedupe(chunks)
    assert outcome.removed == 1


def test_both_scopes_collapse_everything_and_label_which_pass_did_it():
    chunks = [
        chunk("repeated", "a", "a-0"),
        chunk("repeated", "a", "a-1"),
        chunk("repeated", "b", "b-0"),
        chunk("other", "a", "a-2"),
    ]
    outcome = Deduplicator(scope=Scope.BOTH).dedupe(chunks)
    assert outcome.removed == 2
    assert {group.level for group in outcome.groups} == {"document", "corpus"}
    # The document pass ran first: it collapsed a-1 inside document a.
    document_groups = [g for g in outcome.groups if g.level == "document"]
    assert document_groups[0].retained.chunk_id == "a-0"
    assert document_groups[0].collapsed[0].chunk_id == "a-1"


def test_document_scope_is_the_strictest_and_removes_the_least():
    chunks = [
        chunk("same", "a", "a-0"),
        chunk("same", "a", "a-1"),
        chunk("same", "b", "b-0"),
    ]
    removed = {
        scope: Deduplicator(scope=scope).dedupe(chunks).removed
        for scope in Scope
    }
    assert removed[Scope.DOCUMENT] == 1
    assert removed[Scope.CORPUS] == 2
    assert removed[Scope.BOTH] == 2


def test_the_scope_is_reported_back_on_the_outcome():
    outcome = Deduplicator(scope=Scope.DOCUMENT).dedupe([])
    assert outcome.scope is Scope.DOCUMENT


# ---------------------------------------------------------------------------
# FR-43: what "exact" means
# ---------------------------------------------------------------------------


def test_whitespace_runs_are_normalized_before_hashing():
    chunks = [
        chunk("one two   three", "a", "a"),
        chunk("one two three", "a", "b"),
    ]
    assert Deduplicator().dedupe(chunks).removed == 1


def test_leading_and_trailing_whitespace_does_not_defeat_matching():
    chunks = [chunk("  padded  ", "a", "a"), chunk("padded", "a", "b")]
    assert Deduplicator().dedupe(chunks).removed == 1


def test_case_is_preserved_so_case_only_differences_are_not_duplicates():
    """Folding case would be similarity matching by another name (FR-43)."""
    chunks = [chunk("Install the dependency", "a", "a"), chunk("install the dependency", "a", "b")]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.removed == 0
    assert outcome.is_clean


def test_unicode_is_nfc_normalized_so_composed_and_decomposed_match():
    precomposed = "caf\u00e9 latte"
    decomposed = "cafe\u0301 latte"
    assert precomposed != decomposed  # genuinely different code points
    chunks = [chunk(precomposed, "a", "a"), chunk(decomposed, "a", "b")]
    assert Deduplicator().dedupe(chunks).removed == 1


def test_a_near_miss_is_not_a_duplicate():
    template = "Copyright 2026 Example Corp. All rights reserved."
    variant = "Copyright 2026 Other Corp. All rights reserved."
    chunks = [chunk(template, "a", "a"), chunk(variant, "b", "b")]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.is_clean
    assert outcome.removed == 0


def test_a_single_differing_word_is_enough_to_keep_both():
    chunks = [
        chunk("the quick brown fox jumps", "a", "a"),
        chunk("the quick brown fox leaps", "a", "b"),
    ]
    assert Deduplicator().dedupe(chunks).removed == 0


def test_the_normalizer_is_injectable():
    calls: list[str] = []

    def lowercase(text: str) -> str:
        calls.append(text)
        return text.lower().strip()

    chunks = [chunk("Hello World", "a", "a"), chunk("hello world", "a", "b")]
    outcome = Deduplicator(normalize=lowercase).dedupe(chunks)
    assert outcome.removed == 1
    assert calls  # the supplied normalizer, not the default, was used


def test_a_custom_normalizer_disables_the_default_rules():
    chunks = [chunk("one two   three", "a", "a"), chunk("one two three", "a", "b")]
    outcome = Deduplicator(normalize=lambda text: text).dedupe(chunks)
    assert outcome.removed == 0  # no whitespace collapsing now


def test_the_default_normalizer_is_the_declared_one():
    assert default_normalize("  a \n\t  b  ") == "a b"
    assert default_normalize("cafe\u0301") == "caf\u00e9"
    assert default_normalize("Case Preserved") == "Case Preserved"


def test_content_hash_is_deterministic_and_content_dependent():
    deduplicator = Deduplicator()
    assert deduplicator.content_hash("same") == deduplicator.content_hash("same")
    assert deduplicator.content_hash("same") != deduplicator.content_hash("different")
    assert len(deduplicator.content_hash("same")) == 64  # sha256


def test_a_different_hash_algorithm_produces_a_different_length():
    assert len(Deduplicator(algorithm="sha1").content_hash("x")) == 40


# ---------------------------------------------------------------------------
# FR-43: provenance -- every removal records where it went and where it came from
# ---------------------------------------------------------------------------


def test_every_removal_names_the_chunk_it_collapsed_into():
    chunks = [
        chunk("shared", "a", "retained", 0),
        chunk("shared", "b", "removed-from-b", 0),
        chunk("shared", "c", "removed-from-c", 0),
    ]
    group = Deduplicator().dedupe(chunks).groups[0]
    assert group.retained.chunk_id == "retained"
    assert [a.chunk_id for a in group.collapsed] == ["removed-from-b", "removed-from-c"]


def test_every_removal_records_the_document_it_came_from():
    chunks = [
        chunk("shared", "a", "a-0"),
        chunk("shared", "b", "b-0"),
        chunk("shared", "c", "c-0"),
    ]
    group = Deduplicator().dedupe(chunks).groups[0]
    assert group.documents == ("a", "b", "c")
    assert [a.document_id for a in group.collapsed] == ["b", "c"]


def test_an_appearance_carries_source_name_and_index_too():
    chunks = [chunk("shared", "a", "a-0", 0), chunk("shared", "b", "b-0", 7)]
    group = Deduplicator().dedupe(chunks).groups[0]
    removed = group.collapsed[0]
    assert removed.source_name == "b.md"
    assert removed.chunk_index == 7


def test_an_appearance_with_no_document_reports_it_honestly():
    appearance = Appearance(document_id=None, source_name=None, chunk_id="c", chunk_index=1)
    assert appearance.where == "unknown-document/c"


def test_collapsed_into_answers_which_survivor_absorbed_an_appearance():
    chunks = [chunk("shared", "a", "keep"), chunk("shared", "b", "drop")]
    outcome = Deduplicator().dedupe(chunks)
    dropped = Appearance(document_id="b", source_name="b.md", chunk_id="drop", chunk_index=0)
    kept = Appearance(document_id="a", source_name="a.md", chunk_id="keep", chunk_index=0)
    assert outcome.collapsed_into(dropped) == "a/keep"
    assert outcome.collapsed_into(kept) is None


def test_removals_are_summarised_per_document_for_a_run_summary():
    chunks = [
        chunk("shared", "a", "a-0"),
        chunk("shared", "b", "b-0"),
        chunk("shared", "b", "b-1"),
        chunk("shared", "b", "b-2"),
    ]
    outcome = Deduplicator().dedupe(chunks)
    # Only *removed* chunks are attributed, so `a` -- which retained its copy --
    # never appears in the summary rather than appearing with a count of zero.
    # All four chunks were identical, so three went and all three came from `b`.
    counts = outcome.removals_by_document()
    assert counts == {"b": 3}
    assert "a" not in counts
    assert outcome.removed == 3


def test_groups_are_reported_in_input_order():
    chunks = [
        chunk("first pair", "a", "a-0"),
        chunk("first pair", "a", "a-1"),
        chunk("second pair", "a", "b-0"),
        chunk("second pair", "a", "b-1"),
    ]
    outcome = Deduplicator().dedupe(chunks)
    assert [group.retained.chunk_index for group in outcome.groups] == [0, 0]
    assert outcome.groups[0].content_hash != outcome.groups[1].content_hash


def test_a_group_that_removed_nothing_is_not_reported():
    chunks = [chunk("unique", "a", "a-0")]
    outcome = Deduplicator().dedupe(chunks)
    assert outcome.groups == ()
    assert outcome.duplicate_groups == 0
    assert outcome.removed == 0


def test_provenance_survives_a_large_corpus():
    # 100 identical boilerplate chunks spread over doc-0..doc-3 (25 each),
    # plus one unique tail chunk -- 101 in, always 101 accounted for.
    chunks = [
        chunk("boilerplate", f"doc-{i // 25}", f"c-{i}", i % 25)
        for i in range(100)
    ] + [chunk("unique tail", "doc-9", "tail")]

    document_scoped = Deduplicator(scope=Scope.DOCUMENT).dedupe(chunks)
    # One survivor per document that held the boilerplate, plus the tail.
    assert len(document_scoped.retained) == 5
    assert document_scoped.removed == 96
    assert_outcome_adds_up(document_scoped, 101)
    assert {g.retained.document_id for g in document_scoped.groups} == {
        "doc-0",
        "doc-1",
        "doc-2",
        "doc-3",
    }
    # Each survivor's provenance still names where its 24 siblings went.
    for group in document_scoped.groups:
        assert len(group.collapsed) == 24
        assert group.documents == (group.retained.document_id,)

    corpus_scoped = Deduplicator().dedupe(chunks)
    # Corpus-wide, the boilerplate collapses to a single survivor...
    assert len(corpus_scoped.retained) == 2
    assert corpus_scoped.removed == 99
    assert_outcome_adds_up(corpus_scoped, 101)
    # ...but the record still says which documents it was found in.
    assert corpus_scoped.groups[0].documents == ("doc-0", "doc-1", "doc-2", "doc-3")


def assert_outcome_adds_up(outcome: DedupOutcome, total: int) -> None:
    assert_arithmetic_out_holds(outcome, total)
    for group in outcome.groups:
        assert group.removals == len(group.collapsed)
        assert group.documents  # never claim attribution you did not record


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_a_chunk_with_no_text_attribute_is_a_caller_error():
    with pytest.raises(MissingChunkText) as excinfo:
        Deduplicator().dedupe([object()])
    assert "index 0" in str(excinfo.value)
    assert "'text'" in str(excinfo.value)


def test_a_chunk_with_non_string_text_is_a_caller_error():
    with pytest.raises(MissingChunkText) as excinfo:
        Deduplicator().dedupe([Chunk(text=12345)])  # type: ignore[arg-type]
    assert "int" in str(excinfo.value)


def test_the_index_in_the_error_points_at_the_offending_chunk():
    with pytest.raises(MissingChunkText) as excinfo:
        Deduplicator().dedupe([chunk("fine", "a", "a"), Chunk(text=None)])  # type: ignore[arg-type]
    assert "index 1" in str(excinfo.value)


def test_an_unknown_scope_is_refused():
    with pytest.raises(DedupError):
        Deduplicator(scope="corpus")  # type: ignore[arg-type]


def test_a_non_callable_normalizer_is_refused():
    with pytest.raises(DedupError):
        Deduplicator(normalize="lower")  # type: ignore[arg-type]


def test_an_unknown_hash_algorithm_is_refused():
    with pytest.raises(DedupError) as excinfo:
        Deduplicator(algorithm="sha999")
    assert "sha999" in str(excinfo.value)


def test_content_hash_refuses_a_non_string():
    with pytest.raises(MissingChunkText):
        Deduplicator().content_hash(None)  # type: ignore[arg-type]


def test_missing_chunk_text_is_a_dedup_error():
    assert issubclass(MissingChunkText, DedupError)


# ---------------------------------------------------------------------------
# FR-44 -- the fingerprint gate
# ---------------------------------------------------------------------------


def test_the_fingerprint_round_trips_through_its_string_form():
    original = fingerprint_of(
        model="text-embedding-3-small",
        dimensions=1536,
        chunking_config={"size": 512, "overlap": 64},
    )
    parsed = IndexFingerprint.parse(original.value)
    assert parsed == original


def test_key_order_in_the_config_does_not_change_the_fingerprint():
    one = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 512, "overlap": 64})
    two = fingerprint_of(model="m", dimensions=8, chunking_config={"overlap": 64, "size": 512})
    assert one.value == two.value


def test_a_changed_chunking_config_changes_the_fingerprint():
    one = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 512})
    two = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 1024})
    assert one.value != two.value


def test_a_changed_embedding_model_changes_the_fingerprint():
    one = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 512})
    two = fingerprint_of(model="n", dimensions=8, chunking_config={"size": 512})
    assert one.value != two.value


def test_the_fingerprint_is_parseable_not_an_opaque_hash():
    """A bare hash could never name the field, which AC-24 requires."""
    value = fingerprint_of(model="m", dimensions=8, chunking_config="cfg").value
    assert value.startswith("model=m|dims=8|chunking=")
    assert "|dims=8|" in value


def test_a_malformed_stored_fingerprint_is_refused_not_guessed_at():
    with pytest.raises(FingerprintMismatch) as excinfo:
        IndexFingerprint.parse("not-a-fingerprint")
    assert "re-embedded" in str(excinfo.value)
    assert "model=...|dims=...|chunking=..." in str(excinfo.value)


def test_an_empty_stored_fingerprint_is_refused():
    with pytest.raises(FingerprintMismatch) as excinfo:
        IndexFingerprint.parse("")
    assert "no stored" in str(excinfo.value)


def test_a_fingerprint_missing_a_field_is_refused():
    with pytest.raises(FingerprintMismatch) as excinfo:
        IndexFingerprint.parse("model=m|dims=8")
    assert "chunking" in str(excinfo.value)


def test_a_non_integer_dimension_is_refused():
    with pytest.raises(FingerprintMismatch) as excinfo:
        IndexFingerprint.parse("model=m|dims=wide|chunking=c")
    assert "non-integer" in str(excinfo.value)


def test_differences_names_what_disagrees():
    one = fingerprint_of(model="m", dimensions=8, chunking_config="a")
    two = fingerprint_of(model="n", dimensions=8, chunking_config="a")
    assert one.differences(two) == ("model",)
    assert one.differences(one) == ()


def test_differences_names_every_field_that_disagrees():
    one = fingerprint_of(model="m", dimensions=8, chunking_config="a")
    two = fingerprint_of(model="n", dimensions=16, chunking_config="b")
    assert one.differences(two) == ("chunking", "dimensions", "model")


def test_a_gate_accepts_a_perfectly_matching_stored_value():
    gate = EmbeddingReuseGate.configured(
        model="m", dimensions=8, chunking_config={"size": 512}
    )
    stored = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 512}).value
    gate.check(stored)  # must not raise
    assert gate.may_reuse(stored) is True


def test_nothing_is_reused_when_nothing_is_stored():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=8, chunking_config="c")
    assert gate.may_reuse(None) is False
    with pytest.raises(FingerprintMismatch) as excinfo:
        gate.check(None)
    assert "no stored" in str(excinfo.value)


def test_changing_the_embedding_model_forces_re_embedding():
    """AC-24, first half."""
    gate = EmbeddingReuseGate.configured(
        model="text-embedding-3-small", dimensions=1536, chunking_config={"size": 512}
    )
    stored = fingerprint_of(
        model="text-embedding-ada-002", dimensions=1536, chunking_config={"size": 512}
    ).value
    assert gate.may_reuse(stored) is False


def test_changing_the_chunk_size_forces_re_embedding():
    """AC-24, second half."""
    gate = EmbeddingReuseGate.configured(
        model="m", dimensions=1536, chunking_config={"size": 512}
    )
    stored = fingerprint_of(model="m", dimensions=1536, chunking_config={"size": 256}).value
    assert gate.may_reuse(stored) is False


def test_the_refusal_is_typed_and_names_the_mismatched_field():
    gate = EmbeddingReuseGate.configured(
        model="text-embedding-3-small", dimensions=1536, chunking_config={"size": 512}
    )
    stored = fingerprint_of(
        model="text-embedding-ada-002", dimensions=1536, chunking_config={"size": 512}
    ).value
    with pytest.raises(FingerprintMismatch) as excinfo:
        gate.check(stored)
    message = str(excinfo.value)
    assert "model" in message
    assert "text-embedding-ada-002" in message          # stored side
    assert "text-embedding-3-small" in message          # current side
    assert "re-embedded" in message


def test_a_dimension_mismatch_is_named_as_a_dimension_mismatch():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=1536, chunking_config="c")
    stored = fingerprint_of(model="m", dimensions=3072, chunking_config="c").value
    with pytest.raises(FingerprintMismatch) as excinfo:
        gate.check(stored)
    assert "dimensions" in str(excinfo.value)
    assert "1536" in str(excinfo.value)
    assert "3072" in str(excinfo.value)


def test_a_chunking_mismatch_is_named_as_a_chunking_mismatch():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=8, chunking_config={"size": 512})
    stored = fingerprint_of(model="m", dimensions=8, chunking_config={"size": 128}).value
    with pytest.raises(FingerprintMismatch) as excinfo:
        gate.check(stored)
    assert "chunking" in str(excinfo.value)


def test_several_simultaneous_mismatches_are_all_reported():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=8, chunking_config="a")
    stored = fingerprint_of(model="n", dimensions=16, chunking_config="b").value
    with pytest.raises(FingerprintMismatch) as excinfo:
        gate.check(stored)
    message = str(excinfo.value)
    for name in ("model", "dimensions", "chunking"):
        assert name in message


def test_the_gate_exposes_the_value_it_wants_stored():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=8, chunking_config="c")
    assert gate.value == gate.fingerprint.value
    assert gate.may_reuse(gate.value) is True


def test_the_gate_does_not_silently_reuse_across_a_long_pipeline_run():
    """The property that fails silently in production, so it is stated directly."""
    gate = EmbeddingReuseGate.configured(
        model="m", dimensions=8, chunking_config={"size": 512, "overlap": 64}
    )
    stored = gate.value
    steps = [
        ("unchanged", stored),
        ("model swapped", fingerprint_of(model="x", dimensions=8, chunking_config={"size": 512, "overlap": 64}).value),
        ("dims changed", fingerprint_of(model="m", dimensions=16, chunking_config={"size": 512, "overlap": 64}).value),
        ("size changed", fingerprint_of(model="m", dimensions=8, chunking_config={"size": 256, "overlap": 64}).value),
        ("nothing stored", None),
    ]
    verdicts = {label: gate.may_reuse(value) for label, value in steps}
    assert verdicts["unchanged"] is True
    for label in ("model swapped", "dims changed", "size changed", "nothing stored"):
        assert verdicts[label] is False, f"{label} was silently reused"


def test_an_unparseable_stored_value_is_refused_by_the_gate_too():
    gate = EmbeddingReuseGate.configured(model="m", dimensions=8, chunking_config="c")
    assert gate.may_reuse("garbage") is False
    with pytest.raises(FingerprintMismatch):
        gate.check("garbage")


# ---------------------------------------------------------------------------
# Construction of the gate
# ---------------------------------------------------------------------------


def test_an_empty_model_is_refused():
    with pytest.raises(EmbeddingReuseError):
        fingerprint_of(model="", dimensions=8, chunking_config="c")


def test_a_non_string_model_is_refused():
    with pytest.raises(EmbeddingReuseError):
        fingerprint_of(model=None, dimensions=8, chunking_config="c")  # type: ignore[arg-type]


@pytest.mark.parametrize("dimensions", [0, -1, "8", True, None])
def test_a_non_positive_or_non_integer_dimension_is_refused(dimensions):
    with pytest.raises(EmbeddingReuseError):
        fingerprint_of(model="m", dimensions=dimensions, chunking_config="c")  # type: ignore[arg-type]


def test_the_gate_will_not_accept_a_bare_fingerprint_lookalike():
    with pytest.raises(EmbeddingReuseError):
        EmbeddingReuseGate("model=m|dims=8|chunking=c")  # type: ignore[arg-type]


def test_fingerprint_mismatch_is_an_embedding_reuse_error():
    assert issubclass(FingerprintMismatch, EmbeddingReuseError)


def test_chunking_config_accepts_a_string_a_list_and_a_mapping():
    for config in ("raw", [1, 2, 3], {"size": 512}, {"nested": {"a": 1}}):
        value = fingerprint_of(model="m", dimensions=8, chunking_config=config).value
        assert IndexFingerprint.parse(value).chunking_hash


def test_an_exotic_config_still_produces_a_stable_fingerprint():
    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    one = fingerprint_of(model="m", dimensions=8, chunking_config=Odd())
    two = fingerprint_of(model="m", dimensions=8, chunking_config=Odd())
    assert one.value == two.value


# ---------------------------------------------------------------------------
# Values are values; the component stays dependency-free
# ---------------------------------------------------------------------------


def test_outcomes_and_groups_are_immutable():
    outcome = DedupOutcome(retained=(), groups=(), scope=Scope.CORPUS)
    with pytest.raises(AttributeError):
        outcome.scope = Scope.DOCUMENT  # type: ignore[misc]
    group = DuplicateGroup(content_hash="h", level="corpus", retained=Appearance(None, None, None, None), collapsed=())
    with pytest.raises(AttributeError):
        group.level = "document"  # type: ignore[misc]


def test_appearances_are_immutable():
    appearance = Appearance(document_id="d", source_name="s", chunk_id="c", chunk_index=1)
    with pytest.raises(AttributeError):
        appearance.chunk_id = "other"  # type: ignore[misc]


def test_the_component_imports_only_the_standard_library():
    """Checked by walking the AST, not by grepping for a phrase."""
    import ast
    import sys as _sys

    source = Path(__file__).resolve().parents[1] / "chunk_deduplicator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    stdlib = set(_sys.stdlib_module_names)
    external = roots - stdlib - {"chunk_deduplicator"}

    assert external == set(), f"imports something beyond the stdlib: {external}"
    assert "rag_code_library" not in imported
    assert not any(name.startswith("core_types") for name in imported)


def test_the_component_declares_no_near_duplicate_matching():
    """FR-43 forbids claiming it. Assert the claim is absent from the code."""
    source = (
        Path(__file__).resolve().parents[1] / "chunk_deduplicator.py"
    ).read_text(encoding="utf-8").lower()
    for forbidden in ("cosine", "minhash", "tf-idf", "similarity_threshold"):
        assert forbidden not in source


def test_scope_is_a_string_enum_so_it_serialises_in_a_run_summary():
    assert Scope.CORPUS.value == "corpus"
    assert isinstance(Scope.CORPUS, str)

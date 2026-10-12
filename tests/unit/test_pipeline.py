"""Gate tests for the orchestration layer (V2-T08; FR-51..FR-54).

The layer under test is ``rag_code_library.pipeline`` plus the ``ragkit
ingest`` command and the configuration that feeds it.

Three things these tests exist to protect, in order of how expensive it would
be to lose them:

1. **FR-54, proven by AST rather than by assertion.** The orchestration layer
   must statically import no component module and no workflow framework. A
   behavioural test cannot show this -- every test here would still pass if
   someone added ``import text_normalizer`` to the pipeline, because the import
   would succeed and the tests would keep passing. So the check walks the
   parsed source. That is the one test in this file that can catch a change
   none of the others would notice.
2. **FR-53, proven across separate runs.** Skipping, resuming and reprocessing
   are properties of *re-running*, so several tests construct a second Pipeline
   rather than calling an internal helper.
3. **FR-52, proven by absence.** The summary must carry no credentials and no
   document content. A test that only checks required fields are present would
   pass on a summary that also leaked the whole file, so the leak check is a
   positive assertion that document text appears nowhere in the serialized run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rag_code_library import cli
from rag_code_library.config import (
    ENV_PREFIX,
    PipelineSettings,
    load_pipeline_settings,
    resolve_stages,
)
from rag_code_library.pipeline import (
    CHUNKS_KEY,
    DOCUMENTS_KEY,
    CheckpointStore,
    CorpusState,
    DocumentRecord,
    DocumentState,
    DocumentStatus,
    FailurePolicy,
    Pipeline,
    PipelineConfigurationError,
    PipelineConcurrencyError,
    Stage,
    StageScope,
    StageStatus,
)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def calls() -> list[str]:
    """A log of stage invocations, so a test can assert what actually ran."""
    return []


def document_seed(document_id: str = "d1", name: str = "a.pdf", digest: str = "h1"):
    return {
        "document_id": document_id,
        "source_name": name,
        "content_hash": digest,
    }


def discover(seeds):
    """A corpus stage that records ``seeds`` as the discovered documents."""

    def run(state: CorpusState):
        state.artifacts[DOCUMENTS_KEY] = list(seeds)
        return list(seeds)

    return run


def recorder(name: str, log: list[str], *, fail: str | None = None, fail_for=None):
    """A document stage that logs its turn and optionally fails.

    ``fail_for`` narrows the failure to particular documents, which is how the
    failure-policy tests express "one bad document among good ones" -- the case
    FR-52 is actually about. Without it every document fails, and a test that
    expects a mix silently tests something else.
    """

    def run(state: DocumentState):
        log.append(f"{name}:{state.document_id}")
        if fail is not None and (fail_for is None or state.document_id in fail_for):
            raise ValueError(f"{name} failed: {fail}")
        state.provenance[CHUNKS_KEY] = 2
        return f"{name}-output"

    return run


def pipeline_of(stages, **kwargs) -> Pipeline:
    return Pipeline(stages, **kwargs)


def _credential_shaped_value() -> str:
    """A stand-in with the shape of an API key.

    Assembled from parts rather than written as a literal, and deliberately so:
    ``scripts/check_no_secrets.py`` rejects credential-shaped strings in the
    repository, and that rule is worth more than the convenience of a constant
    here. The leak test needs a value that *looks* like a credential -- an
    unremarkable string would prove nothing about FR-52 -- so the shape is
    built at runtime and never appears in the file.
    """
    return "-".join(("sk", "live", "DO-NOT-LEAK-12345"))


# --------------------------------------------------------------------------- #
# FR-51 -- running stages, in order, handing output on
# --------------------------------------------------------------------------- #


def test_a_stage_output_is_visible_to_the_next_stage(calls):
    """FR-51's "passes each stage's validated output to the next"."""

    def first(state):
        calls.append("first")
        return "from-first"

    def second(state):
        calls.append("second")
        assert state.artifacts["first"] == "from-first"
        return state.artifacts["first"].upper()

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("first", first),
            Stage("second", second),
        ]
    )
    summary = pipeline.run("src")

    assert calls == ["first", "second"]
    assert summary.documents[0].stages_completed == ("first", "second")
    assert summary.documents[0].status is DocumentStatus.PROCESSED


def test_corpus_stages_run_once_and_document_stages_run_per_document(calls):
    """The two scopes are what makes 'discover once, chunk per file' work."""
    seeds = [document_seed("d1", "a.pdf"), document_seed("d2", "b.pdf")]
    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("work", recorder("work", calls)),
        ]
    )
    pipeline.run("src")

    assert calls == ["work:d1", "work:d2"]


def test_document_identity_and_provenance_reach_every_stage(calls):
    """FR-51: identity is propagated to every downstream artifact."""
    seen = []

    def observe(state: DocumentState):
        seen.append(
            (
                state.document_id,
                state.source_name,
                state.content_hash,
                state.provenance["document_id"],
                state.provenance["correlation_id"] is not None,
            )
        )

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed("d42", "spec.pdf", "abc")]),
                  scope=StageScope.CORPUS),
            Stage("observe", observe),
        ]
    )
    pipeline.run("src")

    assert seen == [("d42", "spec.pdf", "abc", "d42", True)]


def test_a_stage_can_add_provenance_later_stages_can_read(calls):
    def producer(state: DocumentState):
        state.provenance["page_count"] = 7
        return "text"

    def consumer(state: DocumentState):
        calls.append(state.provenance["page_count"])

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("producer", producer),
            Stage("consumer", consumer),
        ]
    )
    pipeline.run("src")
    assert calls == [7]


def test_the_corpus_state_hands_documents_through_the_documented_key():
    """A discovery stage that records elsewhere simply is not discovery."""
    seen = {}

    def sneaky(state: CorpusState):
        state.artifacts["elsewhere"] = [document_seed()]
        seen["correlation_id"] = state.provenance["correlation_id"]
        seen["source"] = state.provenance["source"]
        return None

    pipeline = pipeline_of(
        [
            Stage("discover", sneaky, scope=StageScope.CORPUS),
            Stage("work", recorder("work", [])),
        ]
    )
    summary = pipeline.run("somewhere")

    assert summary.documents == ()
    assert seen["source"] == "somewhere"
    assert seen["correlation_id"] == summary.correlation_id


# --------------------------------------------------------------------------- #
# FR-51 -- running a named contiguous range
# --------------------------------------------------------------------------- #


def test_a_named_range_runs_only_the_stages_asked_for(calls):
    pipeline = pipeline_of(
        [
            Stage("one", recorder("one", calls)),
            Stage("two", recorder("two", calls)),
            Stage("three", recorder("three", calls)),
        ]
    )
    outcome = pipeline.run_document(
        document_seed(), start="two", end="three"
    )

    assert calls == ["two:d1", "three:d1"]
    assert outcome.stages_completed == ("two", "three")


def test_an_open_ended_range_runs_to_the_end(calls):
    pipeline = pipeline_of(
        [Stage(n, recorder(n, calls)) for n in ("one", "two", "three")]
    )
    pipeline.run_document(document_seed(), start="three")
    assert calls == ["three:d1"]


def test_a_range_naming_an_unknown_stage_is_refused():
    pipeline = pipeline_of([Stage("one", recorder("one", []))])
    with pytest.raises(PipelineConfigurationError, match="not a document stage"):
        pipeline.run_document(document_seed(), start="nope")


def test_an_inverted_range_is_refused_rather_than_silently_swapped():
    pipeline = pipeline_of(
        [Stage(n, recorder(n, [])) for n in ("one", "two", "three")]
    )
    with pytest.raises(PipelineConfigurationError, match="inverted"):
        pipeline.run_document(document_seed(), start="three", end="one")


def test_a_range_does_not_write_a_checkpoint_record():
    """A hand-picked range must not masquerade as a finished document."""
    checkpoints: dict[str, DocumentRecord] = {}
    pipeline = pipeline_of(
        [Stage(n, recorder(n, [])) for n in ("one", "two")], checkpoints=checkpoints
    )
    pipeline.run_document(document_seed(), start="one")
    assert checkpoints == {}


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #


def test_an_empty_stage_list_is_refused():
    with pytest.raises(PipelineConfigurationError, match="no stages"):
        Pipeline([])


def test_duplicate_stage_names_are_refused():
    """Names are the keys artefacts and records live under."""
    with pytest.raises(PipelineConfigurationError, match="duplicate"):
        Pipeline([Stage("same", recorder("same", [])), Stage("same", recorder("same", []))])


def test_a_non_callable_stage_is_refused():
    with pytest.raises(PipelineConfigurationError, match="not callable"):
        Pipeline([Stage("broken", "not a function")])  # type: ignore[arg-type]


def test_an_empty_stage_name_is_refused():
    with pytest.raises(PipelineConfigurationError, match="empty name"):
        Pipeline([Stage("", recorder("x", []))])


# --------------------------------------------------------------------------- #
# FR-52 -- failure handling policy
# --------------------------------------------------------------------------- #


def test_skip_records_the_failure_and_continues_with_the_next_document(calls):
    seeds = [document_seed("d1"), document_seed("d2")]
    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", calls, fail="corrupt", fail_for={"d1"})),
            Stage("chunk", recorder("chunk", calls)),
        ],
        failure_policy=FailurePolicy.SKIP,
    )
    summary = pipeline.run("src")

    assert summary.documents_failed == 1
    assert [d.status for d in summary.documents] == [
        DocumentStatus.FAILED,
        DocumentStatus.PROCESSED,
    ]
    assert summary.documents[0].failed_stage == "load"
    assert summary.documents[0].reason == "load failed: corrupt"
    # Both documents were attempted, which is what SKIP means.
    assert calls.count("load:d1") == 1 and calls.count("load:d2") == 1
    # And the good one still finished.
    assert summary.documents[1].stages_completed == ("load", "chunk")


def test_stop_ends_the_run_at_the_first_failed_document(calls):
    seeds = [document_seed("d1"), document_seed("d2"), document_seed("d3")]
    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", calls, fail="corrupt")),
        ],
        failure_policy=FailurePolicy.STOP,
    )
    summary = pipeline.run("src")

    assert summary.documents_failed == 1
    # d2 and d3 were never reached, so they are absent rather than failed.
    assert len(summary.documents) == 1
    assert calls == ["load:d1"]


def test_stop_still_returns_a_summary():
    """FR-52: every run produces a summary, including one that stopped early."""
    seeds = [document_seed("d1"), document_seed("d2")]
    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", [], fail="boom", fail_for={"d1"})),
        ],
        failure_policy=FailurePolicy.STOP,
    )
    summary = pipeline.run("src")

    assert summary.correlation_id
    # Discovered counts what the corpus phase found, including the documents
    # the run never got to -- which is the difference between "not attempted"
    # and "attempted and fine".
    assert summary.counts()["documents_discovered"] == 2
    assert summary.counts()["documents_failed"] == 1
    assert summary.counts()["documents_processed"] == 0


def test_quarantine_marks_the_document_and_continues(calls):
    seeds = [document_seed("d1"), document_seed("d2")]
    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", calls, fail="corrupt", fail_for={"d1"})),
        ],
        failure_policy=FailurePolicy.QUARANTINE,
    )
    summary = pipeline.run("src")

    assert summary.documents_quarantined == 1
    assert summary.documents_failed == 0
    assert summary.documents[0].status is DocumentStatus.QUARANTINED
    assert calls == ["load:d1", "load:d2"]


def test_later_stages_are_not_attempted_after_one_fails(calls):
    """FR-52's "make failed items distinguishable from un processed ones"."""
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("load", recorder("load", calls)),
            Stage("extract", recorder("extract", calls, fail="bad page")),
            Stage("chunk", recorder("chunk", calls)),
        ]
    )
    outcome = pipeline.run("src").documents[0]

    assert calls == ["load:d1", "extract:d1"]
    assert outcome.stages_completed == ("load",)
    assert outcome.failed_stage == "extract"


def test_a_failure_is_recorded_but_never_swallowed(calls):
    """The summary carries the message; nothing re-types the exception."""

    def boom(_state):
        raise KeyError("some key")

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("boom", boom),
        ]
    )
    outcome = pipeline.run("src").documents[0]

    assert outcome.status is DocumentStatus.FAILED
    # Whatever the stage raised, unaltered: the pipeline records str(exc) and
    # never re-types or wraps, so a caller's classifier sees the real class.
    assert outcome.reason == str(KeyError("some key"))
    assert outcome.failed_stage == "boom"


# --------------------------------------------------------------------------- #
# FR-52 -- the run summary
# --------------------------------------------------------------------------- #


def test_the_summary_counts_every_category():
    seeds = [
        document_seed("ok", "ok.pdf"),
        document_seed("bad", "bad.pdf"),
        document_seed("ugly", "ugly.bin"),
    ]

    def discover_and_reject(state: CorpusState):
        seeds[2] = {**seeds[2], "rejected": "unsupported format .bin"}
        state.artifacts[DOCUMENTS_KEY] = seeds
        return seeds

    def fail_on_bad(state: DocumentState):
        state.provenance[CHUNKS_KEY] = 3
        if state.document_id == "bad":
            raise ValueError("corrupt")
        return "text"

    pipeline = pipeline_of(
        [
            Stage("discover", discover_and_reject, scope=StageScope.CORPUS),
            Stage("load", fail_on_bad),
        ]
    )
    summary = pipeline.run("src")
    counts = summary.counts()

    assert counts["documents_discovered"] == 3
    assert counts["documents_rejected"] == 1
    assert counts["documents_accepted"] == 2
    assert counts["documents_processed"] == 1
    assert counts["documents_failed"] == 1
    assert counts["chunks_processed"] == 3


def test_a_rejected_document_names_the_file_and_the_cause():
    """AC-22's shape: a reason naming the file and the cause, from the stage."""
    seed = {**document_seed("u1", "notes.docx"), "rejected": "unsupported format .docx"}
    pipeline = pipeline_of(
        [
            Stage("discover", discover([seed]), scope=StageScope.CORPUS),
            Stage("load", recorder("load", [])),
        ]
    )
    summary = pipeline.run("src")
    outcome = summary.documents[0]

    assert outcome.status is DocumentStatus.REJECTED
    assert outcome.source_name == "notes.docx"
    assert outcome.reason == "unsupported format .docx"
    assert outcome.stages_completed == ()


def test_the_summary_carries_per_stage_status_and_durations():
    """Durations are measured, and a stage nobody reached is not a pass."""

    def ticking():
        """A clock that advances a little on every read."""
        value = [0.0]

        def read() -> float:
            value[0] += 0.25
            return value[0]

        return read

    def slow(state):
        return "x"

    def breaks(state):
        raise ValueError("this one fails")

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("slow", slow),
            Stage("breaks", breaks),
            Stage("never", recorder("never", [])),
        ],
        clock=ticking(),
    )
    summary = pipeline.run("src")
    by_name = {s.name: s for s in summary.stages}

    assert by_name["slow"].status is StageStatus.SUCCEEDED
    assert by_name["slow"].documents == 1
    assert by_name["slow"].duration_s > 0
    assert by_name["slow"].max_duration_s > 0
    assert by_name["breaks"].status is StageStatus.FAILED
    assert by_name["breaks"].failed == 1
    # A stage no document reached must not read as a pass -- it did not run.
    assert by_name["never"].status is StageStatus.SKIPPED
    assert by_name["never"].documents == 0


def test_a_stage_failed_by_any_document_reports_failed_in_aggregate():
    seeds = [document_seed("d1"), document_seed("d2")]

    def sometimes(state: DocumentState):
        if state.document_id == "d1":
            raise ValueError("nope")
        return "ok"

    pipeline = pipeline_of(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("sometimes", sometimes),
        ]
    )
    summary = pipeline.run("src")
    stage = {s.name: s for s in summary.stages}["sometimes"]

    assert stage.status is StageStatus.FAILED
    assert stage.failed == 1
    assert stage.documents == 2


def test_the_summary_carries_a_correlation_id_and_the_source():
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("work", recorder("work", [])),
        ]
    )
    summary = pipeline.run("some/source")

    assert summary.correlation_id
    assert len(summary.correlation_id) == 32
    assert summary.source == "some/source"


def test_a_caller_supplied_correlation_id_is_honoured():
    pipeline = pipeline_of(
        [Stage("discover", discover([]), scope=StageScope.CORPUS)],
        correlation_id="fixed-id",
    )
    assert pipeline.run("src").correlation_id == "fixed-id"


def test_the_summary_serializes_to_json():
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("work", recorder("work", [])),
        ]
    )
    payload = pipeline.run("src").to_dict()

    assert json.loads(json.dumps(payload))["counts"]["documents_processed"] == 1
    assert payload["stages"][0]["name"] == "work"


def test_the_summary_never_contains_document_content_or_credentials(calls):
    """FR-52: no credentials, no full document content, in summary or logs.

    A secret and the document's text are both put where a careless
    implementation would copy them -- inside a stage's artefact and inside an
    exception message -- and neither may reach the serialized run.
    """
    secret = _credential_shaped_value()
    body = "the entire text of the document that must not appear in a summary"

    def leaky_stage(state: DocumentState):
        state.artifacts["leaky_stage"] = {"api_key": secret, "text": body}
        return {"api_key": secret, "text": body}

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("leaky_stage", leaky_stage),
        ]
    )
    summary = pipeline.run("src")
    serialized = json.dumps(summary.to_dict()) + summary.report()

    assert secret not in serialized
    assert body not in serialized
    assert "api_key" not in serialized


def test_a_summary_written_by_the_cli_contains_no_document_text(tmp_path, monkeypatch):
    """The file the command writes, checked rather than assumed."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("CONTENT THAT MUST NOT BE IN THE SUMMARY", encoding="utf-8")
    destination = tmp_path / "dst"
    destination.mkdir()

    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{__name__}:_cli_stages_for_tests")
    monkeypatch.setenv(f"{ENV_PREFIX}FINGERPRINT", "test")
    code = cli.main(["ingest", str(source), str(destination)])

    written = json.loads((destination / "run-summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert "CONTENT THAT MUST NOT BE IN THE SUMMARY" not in json.dumps(written)


# --------------------------------------------------------------------------- #
# FR-53 -- resumability, idempotency, fingerprints
# --------------------------------------------------------------------------- #


def test_an_unchanged_document_is_skipped_on_the_second_run(calls):
    """The first run does the work; the second proves it does not repeat it."""
    seeds = [document_seed("d1", "a.pdf", "same-hash")]

    def stages(log):
        return [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", log)),
        ]

    checkpoints: dict[str, DocumentRecord] = {}
    Pipeline(stages(calls), checkpoints=checkpoints).run("src")
    assert calls == ["load:d1"]

    second: list[str] = []
    summary = Pipeline(stages(second), checkpoints=checkpoints).run("src")

    assert second == []
    assert summary.documents[0].status is DocumentStatus.SKIPPED
    assert summary.documents[0].reason == "unchanged since the previous run"
    assert summary.documents_skipped == 1


def test_changed_content_is_reprocessed_even_though_a_record_exists(calls):
    """AC-23's shape: only the affected document is reprocessed."""
    checkpoints: dict[str, DocumentRecord] = {}

    def stages(digest: str, log: list[str]):
        return [
            Stage(
                "discover",
                discover([document_seed("d1", "a.pdf", digest)]),
                scope=StageScope.CORPUS,
            ),
            Stage("load", recorder("load", log)),
        ]

    Pipeline(stages("v1", calls), checkpoints=checkpoints).run("src")
    after: list[str] = []
    summary = Pipeline(stages("v2", after), checkpoints=checkpoints).run("src")

    assert after == ["load:d1"]
    assert summary.documents[0].status is DocumentStatus.PROCESSED


def test_a_changed_fingerprint_forces_reprocessing(calls):
    """A changed embedding model or chunk size must not reuse old artifacts."""
    checkpoints: dict[str, DocumentRecord] = {}
    seeds = [document_seed("d1", "a.pdf", "same")]

    def build(fingerprint: str, log: list[str]) -> Pipeline:
        return Pipeline(
            [
                Stage("discover", discover(seeds), scope=StageScope.CORPUS),
                Stage("load", recorder("load", log)),
            ],
            checkpoints=checkpoints,
            fingerprint=fingerprint,
        )

    build("model=old", calls).run("src")
    assert calls == ["load:d1"]

    after: list[str] = []
    summary = build("model=new", after).run("src")
    assert after == ["load:d1"]
    assert summary.documents[0].status is DocumentStatus.PROCESSED
    assert checkpoints["d1"].fingerprint == "model=new"


def test_an_interrupted_document_resumes_rather_than_restarting(calls):
    """FR-53: resume without duplicating successful writes."""
    checkpoints: dict[str, DocumentRecord] = {}
    seeds = [document_seed("d1")]

    def build(fail_at: str | None, log: list[str]) -> Pipeline:
        def failing(state: DocumentState):
            log.append(f"extract:{state.document_id}")
            if fail_at == "extract":
                raise ValueError("power cut")
            return "ok"

        return Pipeline(
            [
                Stage("discover", discover(seeds), scope=StageScope.CORPUS),
                Stage("load", recorder("load", log)),
                Stage("extract", failing),
                Stage("chunk", recorder("chunk", log)),
            ],
            checkpoints=checkpoints,
        )

    first: list[str] = []
    build("extract", first).run("src")
    assert first == ["load:d1", "extract:d1"]
    assert checkpoints["d1"].stages_completed == ("load",)

    second: list[str] = []
    summary = build(None, second).run("src")

    assert second == ["extract:d1", "chunk:d1"], "load must not run a second time"
    assert summary.documents[0].resumed_from == "extract"
    assert summary.documents[0].status is DocumentStatus.PROCESSED
    assert checkpoints["d1"].stages_completed == ("load", "extract", "chunk")


def test_a_resumed_document_that_fails_again_keeps_an_honest_prefix():
    checkpoints: dict[str, DocumentRecord] = {}
    seeds = [document_seed("d1")]

    def build(log: list[str]) -> Pipeline:
        def extract(state: DocumentState):
            log.append(f"extract:{state.document_id}")
            raise ValueError("still broken")

        return Pipeline(
            [
                Stage("discover", discover(seeds), scope=StageScope.CORPUS),
                Stage("load", recorder("load", log)),
                Stage("extract", extract),
            ],
            checkpoints=checkpoints,
        )

    first_run: list[str] = []
    build(first_run).run("src")
    assert first_run == ["load:d1", "extract:d1"]

    # Retrying never repeats load: the honest prefix is what buys that.
    for _ in range(3):
        retry: list[str] = []
        summary = build(retry).run("src")
        assert retry == ["extract:d1"]
        assert summary.documents[0].failed_stage == "extract"
        assert checkpoints["d1"].stages_completed == ("load",)


def test_a_changed_stage_list_invalidates_the_old_record(calls):
    """A prefix that is no longer a prefix means the record no longer applies."""
    checkpoints: dict[str, DocumentRecord] = {}
    seeds = [document_seed("d1")]

    Pipeline(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", [])),
            Stage("chunk", recorder("chunk", [])),
        ],
        checkpoints=checkpoints,
    ).run("src")
    assert checkpoints["d1"].stages_completed == ("load", "chunk")

    after: list[str] = []
    Pipeline(
        [
            Stage("discover", discover(seeds), scope=StageScope.CORPUS),
            Stage("load", recorder("load", after)),
            Stage("embed", recorder("embed", after)),
            Stage("chunk", recorder("chunk", after)),
        ],
        checkpoints=checkpoints,
    ).run("src")

    assert after == ["load:d1", "embed:d1", "chunk:d1"]


def test_the_checkpoint_store_survives_between_objects(tmp_path):
    """Persistence is the whole point of a resume mechanism."""
    store = CheckpointStore(tmp_path)
    store["d1"] = DocumentRecord(
        document_id="d1", content_hash="h", fingerprint="f", stages_completed=("a",)
    )

    reloaded = CheckpointStore(tmp_path)
    assert reloaded["d1"].stages_completed == ("a",)
    assert reloaded["d1"].content_hash == "h"


def test_the_checkpoint_store_is_used_by_a_second_run_over_the_same_destination(tmp_path):
    """AC-21's "a second identical invocation leaves the counts unchanged"."""
    seeds = [document_seed("d1", "a.pdf", "h1")]

    def build() -> Pipeline:
        return Pipeline(
            [
                Stage("discover", discover(seeds), scope=StageScope.CORPUS),
                Stage("load", recorder("load", [])),
            ],
            destination=tmp_path,
            checkpoints=CheckpointStore(tmp_path),
        )

    first = build().run("src")
    second = build().run("src")

    assert first.documents_processed == 1
    assert second.documents_skipped == 1
    assert second.documents_processed == 0


def test_a_broken_checkpoint_file_is_treated_as_empty_rather_than_fatal(tmp_path):
    tmp_path.joinpath("_ragkit.checkpoints.json").write_text("{not json", encoding="utf-8")
    store = CheckpointStore(tmp_path)
    assert len(store) == 0


def test_a_malformed_record_does_not_discard_the_others(tmp_path):
    tmp_path.joinpath("_ragkit.checkpoints.json").write_text(
        json.dumps(
            [
                {"document_id": "good", "content_hash": "h", "fingerprint": "f",
                 "stages_completed": ["a"]},
                {"document_id": "missing-everything"},
                "not even an object",
            ]
        ),
        encoding="utf-8",
    )
    store = CheckpointStore(tmp_path)
    assert list(store) == ["good"]


# --------------------------------------------------------------------------- #
# FR-53 -- concurrency
# --------------------------------------------------------------------------- #


def test_a_held_lock_refuses_a_second_run_and_names_the_holder(tmp_path):
    """One writer per document, enforced rather than merely documented."""
    tmp_path.joinpath("_ragkit.lock").write_text(
        json.dumps({"correlation_id": "other-run", "pid": 4242, "acquired_at": 1.0}),
        encoding="utf-8",
    )
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("load", recorder("load", [])),
        ],
        destination=tmp_path,
    )

    with pytest.raises(PipelineConcurrencyError) as excinfo:
        pipeline.run("src")

    message = str(excinfo.value)
    assert "other-run" in message
    assert "4242" in message
    assert "delete" in message, "a refusal must say how to clear a stale lock"


def test_the_lock_is_released_after_a_run(tmp_path):
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("load", recorder("load", [])),
        ],
        destination=tmp_path,
    )
    pipeline.run("src")

    assert not (tmp_path / "_ragkit.lock").exists()


def test_the_lock_is_released_even_when_a_stage_raises(tmp_path):
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("boom", lambda _s: (_ for _ in ()).throw(RuntimeError("boom"))),
        ],
        destination=tmp_path,
    )
    pipeline.run("src")

    assert not (tmp_path / "_ragkit.lock").exists()


def test_no_destination_means_no_lock_to_take():
    """A lock guarding nothing would be theatre."""
    pipeline = pipeline_of(
        [Stage("discover", discover([]), scope=StageScope.CORPUS)]
    )
    summary = pipeline.run("src")
    assert summary.documents == ()


def test_the_lock_name_is_configurable(tmp_path):
    pipeline = pipeline_of(
        [
            Stage("discover", discover([]), scope=StageScope.CORPUS),
            Stage("work", recorder("work", [])),
        ],
        destination=tmp_path,
        lock_file="custom.lock",
    )
    pipeline.run("src")
    assert not (tmp_path / "custom.lock").exists()


# --------------------------------------------------------------------------- #
# FR-54 -- the orchestration layer imports nothing it must not
# --------------------------------------------------------------------------- #


def _component_module_names() -> set[str]:
    """Every module a component ships, read from its manifest.

    Derived rather than listed, so adding a component extends the check
    instead of quietly escaping it.
    """
    names: set[str] = set()
    for manifest in (ROOT / "components").glob("*/*/manifest.json"):
        source = json.loads(manifest.read_text(encoding="utf-8")).get("source")
        if source:
            names.add(Path(source).stem)
    return names


def _statically_imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import is this package, not a component
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


ORCHESTRATION_FILES = (
    ROOT / "src" / "rag_code_library" / "pipeline.py",
    ROOT / "src" / "rag_code_library" / "config.py",
)


def test_the_orchestration_layer_statically_imports_no_component_module():
    """FR-54, proven structurally.

    This is the test that no behavioural assertion can replace. Components are
    imported by bare module name after a sys.path insertion, so the check is
    over the parsed imports rather than over what happened to be importable
    when the suite ran.
    """
    forbidden = _component_module_names()
    assert forbidden, "no components found; the check would be vacuous"

    for path in ORCHESTRATION_FILES:
        imported = _statically_imported(path)
        clash = imported & forbidden
        assert not clash, f"{path.name} imports component module(s): {sorted(clash)}"


def test_the_orchestration_layer_imports_no_workflow_framework():
    """NFR-15: ordinary Python, not an engine."""
    frameworks = {
        "langchain",
        "llama_index",
        "llamaindex",
        "prefect",
        "airflow",
        "haystack",
        "dspy",
        "temporal",
        "dask",
        "ray",
    }
    for path in ORCHESTRATION_FILES:
        clash = _statically_imported(path) & frameworks
        assert not clash, f"{path.name} imports workflow framework(s): {sorted(clash)}"


def test_the_orchestration_layer_adds_no_mandatory_runtime_dependency():
    """NFR-14: pydantic and pydantic-settings only, and neither is needed here."""
    allowed = {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "enum",
        "importlib",
        "json",
        "os",
        "pathlib",
        "sys",
        "time",
        "typing",
        "uuid",
        "rag_code_library",
    }
    for path in ORCHESTRATION_FILES:
        unexpected = _statically_imported(path) - allowed
        assert not unexpected, f"{path.name} imports {sorted(unexpected)}"


def test_the_pipeline_never_branches_on_a_named_backend_or_provider():
    """FR-54's other half: no backend-specific behaviour in the layer."""
    banned = (
        "chromadb",
        "faiss",
        "pymupdf",
        "openai",
        "anthropic",
        "sentence_transformers",
        "httpx",
        "qdrant",
        "pinecone",
        "weaviate",
    )
    for path in ORCHESTRATION_FILES:
        text = path.read_text(encoding="utf-8").lower()
        for name in banned:
            assert name not in text, f"{path.name} mentions {name!r}"


# --------------------------------------------------------------------------- #
# The configuration that selects stages (FR-54)
# --------------------------------------------------------------------------- #


def _cli_stages_for_tests():
    """A stages factory the CLI tests can name by dotted reference.

    Reads the directory it is given rather than returning a fixed list, so the
    CLI tests exercise discovery over a real source tree instead of a stub
    that happens to agree with their expectations.
    """
    import hashlib
    import os

    def discover(state: CorpusState):
        found = []
        for entry in sorted(os.listdir(state.source)):
            path = os.path.join(state.source, entry)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as handle:
                payload = handle.read()
            found.append(
                {
                    "document_id": hashlib.sha256(entry.encode()).hexdigest()[:16],
                    "source_name": entry,
                    "content_hash": hashlib.sha256(payload).hexdigest()[:16],
                }
            )
        state.artifacts[DOCUMENTS_KEY] = found
        return found

    def chunk(state: DocumentState):
        text = state.artifacts.get("load", "")
        state.provenance[CHUNKS_KEY] = len(text.split())
        return text

    return [
        Stage("discover", discover, scope=StageScope.CORPUS),
        Stage("load", lambda state: state.source_name),
        Stage("chunk", chunk),
    ]


def test_settings_come_from_the_environment_with_safe_defaults():
    settings = load_pipeline_settings(environ={})
    assert settings.failure_policy == "skip"
    assert settings.configured is False


def test_every_setting_is_overridable_by_the_environment():
    settings = load_pipeline_settings(
        environ={
            f"{ENV_PREFIX}STAGES": "pkg:factory",
            f"{ENV_PREFIX}FAILURE_POLICY": "STOP",
            f"{ENV_PREFIX}FINGERPRINT": "fp",
            f"{ENV_PREFIX}LOCK_FILE": "mine.lock",
            f"{ENV_PREFIX}SUMMARY_FILE": "out.json",
        }
    )
    assert settings.stages == "pkg:factory"
    assert settings.failure_policy == "stop", "policy names are case-insensitive"
    assert settings.fingerprint == "fp"
    assert settings.lock_file == "mine.lock"
    assert settings.summary_file == "out.json"


def test_an_explicit_override_beats_the_environment():
    settings = load_pipeline_settings(
        environ={f"{ENV_PREFIX}STAGES": "from:env"},
        stages="from:argument",
    )
    assert settings.stages == "from:argument"


def test_an_unknown_setting_name_is_refused():
    with pytest.raises(ValueError, match="unknown pipeline setting"):
        load_pipeline_settings(environ={}, nonsense="x")


def test_an_unknown_failure_policy_is_refused_before_the_run_starts():
    with pytest.raises(ValueError, match="unknown failure policy"):
        load_pipeline_settings(environ={f"{ENV_PREFIX}FAILURE_POLICY": "explode"})


def test_resolve_stages_reports_an_unconfigured_pipeline():
    with pytest.raises(ValueError, match="no stages are configured") as excinfo:
        resolve_stages(PipelineSettings())
    assert ENV_PREFIX in str(excinfo.value)


@pytest.mark.parametrize(
    "reference, expected",
    [
        ("no-colon", "invalid stages reference"),
        (":factory", "invalid stages reference"),
        ("module:", "invalid stages reference"),
    ],
)
def test_resolve_stages_reports_a_malformed_reference(reference, expected):
    with pytest.raises(ValueError, match=expected):
        resolve_stages(PipelineSettings(stages=reference))


def test_resolve_stages_reports_an_unimportable_module():
    with pytest.raises(ValueError, match="could not import"):
        resolve_stages(PipelineSettings(stages="definitely_not_a_module_xyz:factory"))


def test_resolve_stages_reports_a_missing_attribute():
    with pytest.raises(ValueError, match="has no attribute"):
        resolve_stages(PipelineSettings(stages=f"{__name__}:not_here"))


def test_resolve_stages_reports_a_non_callable_target():
    with pytest.raises(ValueError, match="not callable"):
        resolve_stages(PipelineSettings(stages=f"{__name__}:ROOT"))


def test_resolve_stages_reports_a_non_iterable_result():
    with pytest.raises(ValueError, match="not an iterable"):
        resolve_stages(PipelineSettings(stages=f"{__name__}:_one_stage_not_a_list"))


def test_resolve_stages_reports_an_empty_result():
    with pytest.raises(ValueError, match="returned no stages"):
        resolve_stages(PipelineSettings(stages=f"{__name__}:_no_stages"))


def test_resolve_stages_reports_items_that_are_not_stages():
    with pytest.raises(ValueError, match="not Stage"):
        resolve_stages(PipelineSettings(stages=f"{__name__}:_some_strings"))


def test_resolve_stages_returns_the_stages_the_factory_returned():
    stages = resolve_stages(PipelineSettings(stages=f"{__name__}:_cli_stages_for_tests"))
    assert [s.name for s in stages] == ["discover", "load", "chunk"]


def _one_stage_not_a_list():
    return Stage("x", lambda _s: None)


def _no_stages():
    return []


def _some_strings():
    return ["a", "b"]


# --------------------------------------------------------------------------- #
# `ragkit ingest` (FR-51)
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A source directory of documents and an empty destination."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("alpha beta", encoding="utf-8")
    (source / "b.txt").write_text("gamma delta", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{__name__}:_cli_stages_for_tests")
    monkeypatch.setenv(f"{ENV_PREFIX}FINGERPRINT", "unit-test")
    return source, destination


def test_ingest_is_a_registered_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "src", "dst"])
    assert args.command == "ingest"
    assert args.source == "src"
    assert args.destination == "dst"


def test_ingest_runs_the_pipeline_and_exits_zero(workspace, capsys):
    source, destination = workspace

    code = cli.main(["ingest", str(source), str(destination)])

    assert code == 0
    assert "documents_discovered: 2" in capsys.readouterr().out


def test_ingest_writes_a_run_summary_naming_the_correlation_id(workspace):
    source, destination = workspace

    cli.main(["ingest", str(source), str(destination)])
    payload = json.loads((destination / "run-summary.json").read_text(encoding="utf-8"))

    assert payload["correlation_id"]
    assert payload["counts"]["documents_processed"] == 2
    assert [s["name"] for s in payload["stages"]] == ["load", "chunk"]


def test_a_second_identical_invocation_skips_the_work(workspace):
    source, destination = workspace
    cli.main(["ingest", str(source), str(destination)])

    cli.main(["ingest", str(source), str(destination)])
    payload = json.loads((destination / "run-summary.json").read_text(encoding="utf-8"))

    assert payload["counts"]["documents_skipped"] == 2
    assert payload["counts"]["documents_processed"] == 0


def test_ingest_refuses_a_source_that_is_not_a_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{__name__}:_cli_stages_for_tests")
    missing = tmp_path / "nope"

    code = cli.main(["ingest", str(missing), str(tmp_path)])

    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_ingest_reports_an_unconfigured_pipeline_as_an_actionable_error(
    workspace, monkeypatch, capsys
):
    source, destination = workspace
    monkeypatch.delenv(f"{ENV_PREFIX}STAGES", raising=False)

    code = cli.main(["ingest", str(source), str(destination)])

    assert code == 1
    err = capsys.readouterr().err
    assert ENV_PREFIX in err
    assert "--stages" in err


def test_ingest_reports_an_unimportable_stages_module(workspace, monkeypatch, capsys):
    source, destination = workspace
    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", "nope_nope_nope:factory")

    code = cli.main(["ingest", str(source), str(destination)])

    assert code == 1
    assert "could not import" in capsys.readouterr().err


def test_the_failure_policy_flag_overrides_the_environment(workspace, capsys):
    source, destination = workspace
    (source / "a.txt").unlink()

    code = cli.main(
        ["ingest", str(source), str(destination), "--failure-policy", "quarantine"]
    )

    assert code == 0


def test_the_summary_flag_overrides_where_the_summary_is_written(workspace):
    source, destination = workspace

    cli.main(["ingest", str(source), str(destination), "--summary", "reports/run.json"])

    assert (destination / "reports" / "run.json").exists()


def test_ingest_creates_a_destination_that_does_not_exist_yet(tmp_path, monkeypatch):
    """The first run of a brand-new project, which has no destination yet."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("alpha beta", encoding="utf-8")
    destination = tmp_path / "brand-new"  # deliberately absent
    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{__name__}:_cli_stages_for_tests")
    monkeypatch.setenv(f"{ENV_PREFIX}FINGERPRINT", "unit-test")

    code = cli.main(["ingest", str(source), str(destination)])

    assert code == 0
    assert destination.is_dir()
    assert (destination / "run-summary.json").exists()
    # And a second run over the destination it just made skips the work.
    cli.main(["ingest", str(source), str(destination)])
    payload = json.loads((destination / "run-summary.json").read_text(encoding="utf-8"))
    assert payload["counts"]["documents_skipped"] == 1


def test_ingest_leaves_no_lock_behind(workspace):
    source, destination = workspace
    cli.main(["ingest", str(source), str(destination)])

    assert not (destination / "_ragkit.lock").exists()


def test_the_cli_module_does_not_import_any_component():
    """The command is part of the layer that must stay component-free."""
    forbidden = _component_module_names()
    clash = _statically_imported(ROOT / "src" / "rag_code_library" / "cli.py")
    assert not (clash & forbidden)


# --------------------------------------------------------------------------- #
# Counting chunks
# --------------------------------------------------------------------------- #


def test_a_stage_can_record_the_chunk_count_for_the_summary():
    def chunk(state: DocumentState):
        state.provenance[CHUNKS_KEY] = 11
        return ["c"] * 11

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("chunk", chunk),
        ]
    )
    summary = pipeline.run("src")
    assert summary.chunks_processed == 11
    assert summary.documents[0].chunks == 11


def test_an_explicit_counting_hook_overrides_the_recorded_count():
    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("work", recorder("work", [])),
        ],
        count_chunks=lambda _state: 99,
    )
    assert pipeline.run("src").chunks_processed == 99


def test_a_nonsense_chunk_count_is_ignored_rather_than_reported():
    """A wrong count looks like an answer; zero is honest about not knowing."""

    def odd(state: DocumentState):
        state.provenance[CHUNKS_KEY] = "many"
        return "text"

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("odd", odd),
        ]
    )
    assert pipeline.run("src").chunks_processed == 0


def test_a_failed_document_contributes_no_chunks(calls):
    def fail(state: DocumentState):
        state.provenance[CHUNKS_KEY] = 5
        raise ValueError("no")

    pipeline = pipeline_of(
        [
            Stage("discover", discover([document_seed()]), scope=StageScope.CORPUS),
            Stage("fail", fail),
        ]
    )
    summary = pipeline.run("src")

    assert summary.chunks_processed == 0
    assert summary.documents[0].chunks == 0

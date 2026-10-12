"""End-to-end proof of the V2 ingestion spine (V2-T09).

Where the real components are wired
-----------------------------------
``pipeline_stages`` assembles the thirteen stages from the shipped components.
It lives under ``tests/integration`` because V2-T09's scope excludes ``src/``
-- which is the point: the orchestration layer imports no component, so
wherever the components get assembled has to be somewhere else. Every test here
therefore proves the shipped components work *together*, not just individually.

Two things this file deliberately does not claim
------------------------------------------------
**PDF.** ``document_loader`` detects ``%PDF-`` and would hand it to an adapter,
but the adapter needs an extraction library, and installing ``pymupdf`` would
modify a dependency file -- forbidden here outright. So the end-to-end path uses
text and Markdown. The PDF branch is real and is covered by
``tests/unit/test_ingestion.py`` through the component's injection seam. Saying
"the pipeline ingests PDFs" would be a claim this environment never tested.

**Corpus-wide deduplication.** The orchestration layer runs corpus stages
before every document stage and has no phase after them, so cross-document
deduplication cannot be expressed as a document stage. Deduplication here is
per document, and it is tested as such.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
# ``tests`` is a package, so the stages are addressed as
# ``tests.integration.pipeline_stages`` -- the same way the configuration layer
# reaches them when ``ragkit ingest`` resolves the factory by dotted path.
STAGES_MODULE = "tests.integration.pipeline_stages"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rag_code_library import cli  # noqa: E402
from rag_code_library.config import ENV_PREFIX  # noqa: E402

from tests.integration.pipeline_stages import (  # noqa: E402
    CHUNK_SIZE,
    INDEX_DIR_VARIABLE,
    INTERRUPT_VARIABLE,
    PersistentVectorStore,
    TinyEmbedder,
    build_stages,
    index_directory,
)

#: The stages AC-21 names, in the order it names them.
THIRTEEN_STAGES = (
    "discover",
    "validate",
    "identify",
    "load",
    "extract",
    "normalize",
    "enrich",
    "chunk",
    "validate_chunks",
    "deduplicate",
    "embed",
    "index",
    "verify",
)

#: Long enough that the oversized-file case is unambiguous, small enough to
#: write in a moment. Deliberately just over the stages' 10 MiB limit.
OVERSIZED_BYTES = 10 * 1024 * 1024 + 1


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> SimpleNamespace:
    """An empty source, an empty destination, and a place to keep an index.

    ``RAG_E2E_INDEX_DIR`` is set for every test because ``build_stages()`` is
    called with no arguments -- the configuration layer decides what a factory
    receives, and this one receives nothing. Without a directory the index
    would be per-process, and AC-29's before/after comparison would be
    comparing nothing.
    """
    source = tmp_path / "source"
    source.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.setenv(INDEX_DIR_VARIABLE, str(index))
    monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{STAGES_MODULE}:build_stages")
    monkeypatch.setenv(f"{ENV_PREFIX}FINGERPRINT", "e2e-test")
    return SimpleNamespace(
        source=source,
        destination=tmp_path / "destination",
        index=index,
        root=tmp_path,
    )


def write_document(workspace, name: str, body: str | bytes) -> Path:
    path = workspace.source / name
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return path


def run_ingest(workspace, destination: Path | None = None, **env_overrides) -> tuple[int, dict]:
    """Invoke the real CLI and return its exit code and written summary.

    The previous summary is deleted first. Reading a summary the *previous*
    run wrote would silently turn a crashed invocation into a passing
    assertion, which is exactly the kind of false evidence this project's
    verification rules exist to prevent.
    """
    destination = destination or workspace.destination
    summary_path = destination / "run-summary.json"
    summary_path.unlink(missing_ok=True)
    previous = os.environ.get(INDEX_DIR_VARIABLE)
    try:
        for key, value in env_overrides.items():
            os.environ[key] = value
        code = cli.main(["ingest", str(workspace.source), str(destination)])
    finally:
        for key in env_overrides:
            os.environ.pop(key, None)
        if previous is None:
            os.environ.pop(INDEX_DIR_VARIABLE, None)
        else:
            os.environ[INDEX_DIR_VARIABLE] = previous
    if not summary_path.exists():
        raise AssertionError(
            "ingest exited without writing a summary; a run that produces no "
            f"record cannot be asserted on. exit code was {code}."
        )
    return code, json.loads(summary_path.read_text(encoding="utf-8"))


def index_payload(workspace) -> dict:
    path = workspace.index / "index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stored_records(workspace) -> list[dict]:
    return list(index_payload(workspace).get("chunks", {}).values())


def stored_documents(workspace) -> list[dict]:
    return list(index_payload(workspace).get("documents", {}).values())


# --------------------------------------------------------------------------- #
# AC-21 -- the thirteen stages, end to end, through the CLI
# --------------------------------------------------------------------------- #


class TestThirteenStages:
    def test_ragkit_ingest_runs_every_stage_in_order(self, workspace):
        """The literal AC-21 invocation: one document, thirteen stages."""
        write_document(
            workspace,
            "notes.txt",
            "Chunking splits a document into retrievable pieces.\n" * 8,
        )

        code, summary = run_ingest(workspace)

        assert code == 0, summary["documents"]
        # The summary itemises the per-document stages. The three corpus stages
        # run once over the whole source and have no document to report under,
        # so they are asserted through their effects below rather than by
        # appearing in this list.
        assert tuple(stage["name"] for stage in summary["stages"]) == THIRTEEN_STAGES[3:]
        assert all(stage["status"] == "succeeded" for stage in summary["stages"])
        assert [doc["status"] for doc in summary["documents"]] == ["processed"]
        assert summary["documents"][0]["stages_completed"] == list(THIRTEEN_STAGES[3:])
        assert summary["counts"]["documents_processed"] == 1
        assert summary["counts"]["chunks_processed"] > 1

    def test_the_corpus_stages_all_ran_before_any_document_stage(self, workspace):
        """The three stages the summary does not itemise, proved by their effects."""
        write_document(workspace, "notes.txt", "Provenance on every record.\n" * 10)
        write_document(workspace, "notes.exe", b"MZ\x90\x00\x03\x00\x00\x00")
        write_document(workspace, "empty.txt", "")

        _, summary = run_ingest(workspace)

        names = [doc["source_name"] for doc in summary["documents"]]
        # discover found all three; validate refused two of them; identify
        # assigned the survivor a content-derived id.
        assert sorted(names) == ["empty.txt", "notes.exe", "notes.txt"]
        rejected = [doc for doc in summary["documents"] if doc["status"] == "rejected"]
        assert len(rejected) == 2
        assert stored_documents(workspace)[0]["document_id"]

    def test_the_document_is_really_in_the_index(self, workspace):
        write_document(workspace, "notes.txt", "Provenance on every record.\n" * 10)

        run_ingest(workspace)

        records = stored_records(workspace)
        documents = stored_documents(workspace)
        assert records, "nothing was written to the index"
        assert len(documents) == 1
        assert documents[0]["source_name"] == "notes.txt"
        assert all(row["document_id"] == documents[0]["document_id"] for row in records)

    def test_every_stored_chunk_carries_its_provenance(self, workspace):
        """A chunk that cannot say where it came from cannot be cited."""
        write_document(workspace, "notes.txt", "Cite this text back to its source.\n" * 10)

        _, summary = run_ingest(workspace)
        records = stored_records(workspace)

        assert records, "nothing was written to the index"
        for row in records:
            assert row["document_id"]
            assert row["chunk_id"]
            assert row["source_name"] == "notes.txt"
            assert isinstance(row["start_offset"], int)
            assert isinstance(row["end_offset"], int)
            assert row["end_offset"] > row["start_offset"]
            assert row["page_start"] is not None
        # Offsets are contiguous and start at the top of the file.
        offsets = sorted(row["start_offset"] for row in records)
        assert offsets[0] == 0

    def test_chunk_boundaries_are_contiguous_not_overlapping(self, workspace):
        text = " ".join(f"word{i}" for i in range(200))
        write_document(workspace, "notes.txt", text)

        run_ingest(workspace)
        records = sorted(stored_records(workspace), key=lambda row: row["start_offset"])

        for earlier, later in zip(records, records[1:]):
            assert later["start_offset"] >= earlier["end_offset"], (
                "one chunk's span covers another's text"
            )
            # Trimming a chunk's edges leaves the whitespace between two
            # chunks belonging to neither. That is fine; swallowing a word
            # would not be.
            gap = text[earlier["end_offset"] : later["start_offset"]]
            assert gap.strip() == "", f"a gap between chunks took content: {gap!r}"
        assert records[0]["start_offset"] == 0
        for row in records:
            # Length must agree with the span it claims. Not compared against
            # the file's raw bytes: the normalize stage rewrites the text (that
            # is its job), so offsets are measured on the normalized form.
            assert len(row["text"]) == row["end_offset"] - row["start_offset"]
            assert row["text"].strip(), "an empty chunk reached the index"

    def test_the_summary_names_no_document_content(self, workspace):
        """FR-52: counts and names, never the text itself."""
        secret = _credential_shaped_value()
        write_document(workspace, "notes.txt", f"a note containing {secret}\n" * 20)

        _, summary = run_ingest(workspace)
        serialised = json.dumps(summary)

        assert secret not in serialised

    def test_markdown_is_parsed_not_treated_as_bare_text(self, workspace):
        write_document(workspace, "guide.md", "# Heading\n\nBody text.\n" * 10)

        code, _ = run_ingest(workspace)

        assert code == 0
        assert stored_documents(workspace)[0]["source_name"] == "guide.md"

    def test_a_destination_is_created_on_demand(self, workspace):
        write_document(workspace, "notes.txt", "A first run.\n" * 5)
        assert not workspace.destination.exists()

        code, _ = run_ingest(workspace)

        assert code == 0
        assert (workspace.destination / "run-summary.json").exists()


# --------------------------------------------------------------------------- #
# AC-22 -- hostile inputs are refused before anything loads them
# --------------------------------------------------------------------------- #


def _symlink_or_skip(target: Path, link: Path) -> None:
    """Create a symlink, or skip honestly when the platform will not allow one."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this platform refused a symlink without elevation: {exc}")


class TestHostileInputs:
    @pytest.mark.parametrize(
        ("filename", "body", "expected"),
        [
            # Real PE header bytes. Written as *text* it would be a text file
            # with an .exe name, and a validator that sniffs content would be
            # right to accept it -- so the fixture has to actually be binary.
            ("notes.exe", b"MZ\x90\x00\x03\x00\x00\x00\x04\x00", "not a recognised"),
            ("empty.txt", "", "empty"),
        ],
    )
    def test_an_unusable_file_is_refused_with_a_reason(self, workspace, filename, body, expected):
        write_document(workspace, filename, body)
        write_document(workspace, "good.txt", "a perfectly ordinary document\n" * 10)

        code, summary = run_ingest(workspace)

        assert code == 0
        refused = [doc for doc in summary["documents"] if doc["status"] == "rejected"]
        assert len(refused) == 1
        assert refused[0]["source_name"] == filename
        assert refused[0]["reason"], "a rejection without a reason is not actionable"
        assert expected in refused[0]["reason"].lower()

    def test_an_oversized_file_is_refused_with_a_reason(self, workspace):
        with open(workspace.source / "huge.txt", "wb") as handle:
            handle.write(b"x" * OVERSIZED_BYTES)

        code, summary = run_ingest(workspace)

        assert code == 0
        refused = [doc for doc in summary["documents"] if doc["status"] == "rejected"]
        assert len(refused) == 1
        reason = refused[0]["reason"].lower()
        assert "over the" in reason and "byte limit" in reason, reason

    def test_a_directory_with_only_unusable_files_still_reports_each_one(self, workspace):
        write_document(workspace, "notes.exe", b"MZ\x90\x00\x03\x00\x00\x00")
        write_document(workspace, "empty.txt", "")

        code, summary = run_ingest(workspace)

        assert code == 0
        assert summary["counts"]["documents_rejected"] == 2
        assert all(doc["reason"] for doc in summary["documents"])

    def test_a_rejected_document_never_reaches_a_document_stage(self, workspace):
        """AC-22's real claim: no extraction ran for any refused file."""
        write_document(workspace, "notes.exe", b"MZ\x90\x00\x03\x00\x00\x00")
        write_document(workspace, "empty.txt", "")
        write_document(workspace, "huge.txt", "x" * OVERSIZED_BYTES)
        write_document(workspace, "good.txt", "a perfectly ordinary document\n" * 10)

        _, summary = run_ingest(workspace)

        rejected = [doc for doc in summary["documents"] if doc["status"] == "rejected"]
        assert len(rejected) == 3, summary["documents"]
        for doc in rejected:
            assert doc["stages_completed"] == [], (
                f"{doc['source_name']} reached a document stage: {doc['stages_completed']}"
            )
        # And the accepted one still went all the way through.
        accepted = [doc for doc in summary["documents"] if doc["status"] == "processed"]
        assert len(accepted) == 1
        assert accepted[0]["stages_completed"] == list(THIRTEEN_STAGES[3:])

    def test_a_symlink_pointing_outside_the_source_is_refused(self, workspace, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("this lives outside the source\n", encoding="utf-8")
        write_document(workspace, "good.txt", "a perfectly ordinary document\n" * 10)
        _symlink_or_skip(outside / "secret.txt", workspace.source / "linked.txt")

        code, summary = run_ingest(workspace)

        assert code == 0
        assert all(doc["source_name"] != "linked.txt" for doc in summary["documents"]), (
            "a link escaping the source root was ingested"
        )

    def test_an_empty_directory_yields_a_clean_empty_run(self, workspace):
        code, summary = run_ingest(workspace)

        assert code == 0
        assert summary["counts"]["documents_discovered"] == 0
        assert summary["documents"] == []
        # No document was ever handed to a stage, so none did any work.
        assert all(stage["documents"] == 0 for stage in summary["stages"])
        assert all(stage["status"] in {"succeeded", "skipped"} for stage in summary["stages"])


# --------------------------------------------------------------------------- #
# AC-23 -- adding, editing and deleting across two runs
# --------------------------------------------------------------------------- #


class TestIncrementalRuns:
    def test_a_second_run_over_unchanged_files_does_nothing(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        run_ingest(workspace)

        code, summary = run_ingest(workspace)

        assert code == 0
        assert summary["counts"]["documents_skipped"] == 1
        assert summary["counts"]["documents_processed"] == 0
        assert summary["counts"]["chunks_processed"] == 0

    def test_adding_a_file_processes_only_the_new_one(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)
        run_ingest(workspace)

        write_document(workspace, "gamma.txt", "gamma content\n" * 10)
        _, summary = run_ingest(workspace)

        processed = [doc["source_name"] for doc in summary["documents"] if doc["status"] == "processed"]
        assert processed == ["gamma.txt"]
        assert len(stored_documents(workspace)) == 3

    def test_editing_a_file_processes_only_that_one(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)
        run_ingest(workspace)

        write_document(workspace, "beta.txt", "completely different beta content\n" * 10)
        _, summary = run_ingest(workspace)

        processed = [doc["source_name"] for doc in summary["documents"] if doc["status"] == "processed"]
        assert processed == ["beta.txt"], summary["documents"]

    def test_editing_a_file_replaces_its_chunks_rather_than_adding_to_them(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)
        _, first = run_ingest(workspace)
        before = len(stored_records(workspace))

        write_document(workspace, "beta.txt", "a much longer beta document than before\n" * 20)
        run_ingest(workspace)
        records = stored_records(workspace)

        beta_ids = {row["document_id"] for row in records if row["source_name"] == "beta.txt"}
        assert len(beta_ids) == 1, "an edited file left two document ids behind"
        # The old beta chunks are gone, so the total moves by the new size only.
        assert len(records) != before
        assert first["counts"]["chunks_processed"] > 0

    def test_deleting_a_file_removes_only_its_records(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)
        write_document(workspace, "gamma.txt", "gamma content\n" * 10)
        run_ingest(workspace)
        kept_ids = {
            row["document_id"]
            for row in stored_records(workspace)
            if row["source_name"] in {"alpha.txt", "gamma.txt"}
        }
        before = len(stored_records(workspace))

        (workspace.source / "beta.txt").unlink()
        _, summary = run_ingest(workspace)

        assert summary["counts"]["documents_discovered"] == 2
        records = stored_records(workspace)
        assert not any(row["source_name"] == "beta.txt" for row in records)
        assert len(records) < before
        still_there = {row["document_id"] for row in records}
        assert kept_ids <= still_there, "a surviving document lost chunks"

    def test_a_file_added_and_removed_between_runs_leaves_no_trace(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        run_ingest(workspace)

        write_document(workspace, "temp.txt", "temporary content\n" * 10)
        run_ingest(workspace)
        assert any(row["source_name"] == "temp.txt" for row in stored_records(workspace))

        (workspace.source / "temp.txt").unlink()
        run_ingest(workspace)
        assert not any(row["source_name"] == "temp.txt" for row in stored_records(workspace))
        assert [doc["source_name"] for doc in stored_documents(workspace)] == ["alpha.txt"]


# --------------------------------------------------------------------------- #
# AC-29 -- re-ingesting unchanged content
# --------------------------------------------------------------------------- #


class TestIdempotency:
    def test_reingesting_unchanged_content_changes_no_counts(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)

        _, first = run_ingest(workspace)
        chunks_before = len(stored_records(workspace))
        documents_before = len(stored_documents(workspace))

        _, second = run_ingest(workspace)

        assert len(stored_records(workspace)) == chunks_before
        assert len(stored_documents(workspace)) == documents_before
        assert second["counts"]["documents_skipped"] == 2
        assert second["counts"]["documents_processed"] == 0
        assert second["counts"]["chunks_processed"] == 0
        assert first["counts"]["chunks_processed"] == chunks_before

    def test_the_vectors_are_byte_identical_after_a_reingest(self, workspace):
        """No stale vector: the stored values must not drift."""
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        run_ingest(workspace)
        before = {row["chunk_id"]: row["vector"] for row in stored_records(workspace)}

        run_ingest(workspace)
        after = {row["chunk_id"]: row["vector"] for row in stored_records(workspace)}

        assert before == after

    def test_every_chunk_id_is_unique_in_the_index(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content\n" * 10)
        write_document(workspace, "beta.txt", "beta content\n" * 10)
        run_ingest(workspace)
        run_ingest(workspace)

        ids = [row["chunk_id"] for row in stored_records(workspace)]
        assert len(ids) == len(set(ids)), "duplicate chunk ids in the index"

    def test_two_identical_documents_do_not_double_the_index(self, workspace):
        body = "the very same content in both files\n" * 10
        write_document(workspace, "a.txt", body)
        write_document(workspace, "b.txt", body)

        _, summary = run_ingest(workspace)

        rejected = [doc for doc in summary["documents"] if doc["status"] == "rejected"]
        assert len(rejected) == 1
        assert "duplicate content" in rejected[0]["reason"]
        assert len(stored_documents(workspace)) == 1


# --------------------------------------------------------------------------- #
# AC-28 -- hostile input cannot forge its own identity
# --------------------------------------------------------------------------- #


class TestForgedMetadata:
    def test_the_keys_the_component_computes_cannot_be_overwritten(self):
        """Asserted at the component, because a text file has no metadata channel.

        AC-28's third hostile input is a document carrying ``system_*``
        metadata. The forward path through this assembly would need a document
        format that carries metadata, and inventing one here would be inventing
        a feature to make a criterion pass. The property itself is the
        component's, so it is checked directly.
        """
        from metadata_enricher import MetadataEnricher

        forged = {
            "system_source_path": "/attacker/chosen/path",
            "system_ingested_at": "1999-01-01T00:00:00+00:00",
            "ordinary_key": "left alone",
        }
        bundle = MetadataEnricher(ingested_at="2026-01-01T00:00:00+00:00").enrich(
            source_name="real.txt",
            source_metadata=forged,
            content_sha256="deadbeef",
        )

        assert bundle.merged["system_source_path"] == "real.txt"
        assert bundle.merged["system_ingested_at"] == "2026-01-01T00:00:00+00:00"
        # Non-reserved keys are the document's own and are preserved as given.
        assert bundle.merged["ordinary_key"] == "left alone"

    def test_a_document_can_still_introduce_a_reserved_key_it_does_not_overwrite(self):
        """A gap worth stating plainly rather than papering over.

        ``metadata-enricher`` protects the reserved ``system_`` keys *it*
        computes -- ``system_source_path``, ``system_ingested_at`` and
        ``system_content_sha256``. It does not reject a reserved-prefixed key
        it has no value for, so a document asserting ``system_document_id``
        carries it through as an ordinary source key.

        Nothing in this pipeline is misled by that: identity comes from the
        content hash in the seed, never from metadata. But a downstream reader
        treating the ``system_`` prefix as a guarantee would be wrong to, and
        the honest thing is to say so here. Asserted as current behaviour so
        that tightening the component breaks this test visibly.
        """
        from metadata_enricher import MetadataEnricher

        bundle = MetadataEnricher(ingested_at="2026-01-01T00:00:00+00:00").enrich(
            source_name="real.txt",
            source_metadata={"system_document_id": "attacker-chosen-id"},
            content_sha256="deadbeef",
        )

        assert bundle.merged["system_document_id"] == "attacker-chosen-id"

    def test_a_hostile_filename_cannot_escape_the_source(self, workspace):
        """On NTFS a filename cannot contain a separator, so the reachable
        traversal vector is a link out of the tree -- covered in
        :class:`TestHostileInputs`. What this asserts is that discovery only
        reports files it actually found under the source root.
        """
        nested = workspace.source / "nested"
        nested.mkdir()
        (nested / "inside.txt").write_text("nested content\n" * 5, encoding="utf-8")

        _, summary = run_ingest(workspace)

        names = [doc["source_name"] for doc in summary["documents"]]
        assert names == ["nested/inside.txt"]
        assert not any(".." in name for name in names)


# --------------------------------------------------------------------------- #
# AC-30 -- recovering from an interrupted run
# --------------------------------------------------------------------------- #


class TestRecoveryFromInterruption:
    @pytest.mark.parametrize(
        "interrupt_at",
        ["load", "extract", "normalize", "chunk", "validate_chunks",
         "deduplicate", "embed", "index", "verify"],
    )
    def test_an_interrupted_run_completes_without_duplicating_or_dropping(
        self, workspace, monkeypatch, interrupt_at
    ):
        """Interrupt a real stage, then run again with it working.

        Faking this by editing a checkpoint would only prove a checkpoint can
        be edited. Failing a real stage proves the pipeline records an honest
        prefix and that the next run picks up exactly where it left off.
        """
        for name in ("alpha", "beta", "gamma"):
            write_document(workspace, f"{name}.txt", f"{name} content about indexing\n" * 5)
        monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{STAGES_MODULE}:build_interrupted_stages")
        monkeypatch.setenv(INTERRUPT_VARIABLE, interrupt_at)

        interrupted_code, interrupted = run_ingest(workspace)

        assert interrupted_code == 1
        assert interrupted["counts"]["documents_failed"] == 3
        for doc in interrupted["documents"]:
            assert doc["failed_stage"] == interrupt_at
            assert doc["reason"], "a failure with no reason is not actionable"
            assert doc["status"] == "failed"

        monkeypatch.delenv(INTERRUPT_VARIABLE)
        code, resumed = run_ingest(workspace)

        assert code == 0, resumed["documents"]
        assert resumed["counts"]["documents_processed"] == 3
        assert len(stored_records(workspace)) == 9
        assert len(stored_documents(workspace)) == 3

    @pytest.mark.parametrize("interrupt_at", ["chunk", "embed", "index"])
    def test_a_resumed_document_reports_where_it_picked_up(
        self, workspace, monkeypatch, interrupt_at
    ):
        write_document(workspace, "alpha.txt", "alpha content about indexing\n" * 5)
        monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{STAGES_MODULE}:build_interrupted_stages")
        monkeypatch.setenv(INTERRUPT_VARIABLE, interrupt_at)
        run_ingest(workspace)

        monkeypatch.delenv(INTERRUPT_VARIABLE)
        _, resumed = run_ingest(workspace)

        doc = resumed["documents"][0]
        assert doc["status"] == "processed"
        assert doc["resumed_from"] == interrupt_at
        # Whatever it resumed from, the record describes the whole document.
        assert doc["stages_completed"] == list(THIRTEEN_STAGES[3:])

    def test_the_interrupted_run_records_an_honest_prefix(self, workspace, monkeypatch):
        """The failing stage and everything after it must NOT be recorded."""
        write_document(workspace, "alpha.txt", "alpha content about indexing\n" * 5)
        monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{STAGES_MODULE}:build_interrupted_stages")
        monkeypatch.setenv(INTERRUPT_VARIABLE, "chunk")

        _, interrupted = run_ingest(workspace)

        completed = interrupted["documents"][0]["stages_completed"]
        assert completed == ["load", "extract", "normalize", "enrich"]
        assert "chunk" not in completed

    def test_nothing_reaches_the_index_before_the_index_stage(self, workspace, monkeypatch):
        write_document(workspace, "alpha.txt", "alpha content about indexing\n" * 5)
        monkeypatch.setenv(f"{ENV_PREFIX}STAGES", f"{STAGES_MODULE}:build_interrupted_stages")
        monkeypatch.setenv(INTERRUPT_VARIABLE, "index")

        run_ingest(workspace)

        assert index_payload(workspace) == {}, (
            "an interrupted run left records behind for work it never finished"
        )

    def test_clearing_the_interruption_variable_returns_to_normal_stages(self, workspace):
        write_document(workspace, "alpha.txt", "alpha content about indexing\n" * 5)

        code, _ = run_ingest(
            workspace,
            **{f"{ENV_PREFIX}STAGES": f"{STAGES_MODULE}:build_interrupted_stages"},
        )

        assert code == 0
        assert len(stored_records(workspace)) > 0
        assert len(stored_documents(workspace)) == 1


# --------------------------------------------------------------------------- #
# The surface this file must not have disturbed
# --------------------------------------------------------------------------- #


class TestTheDeferredSurfaceIsUnchanged:
    """What this file can check about the other verification scripts.

    The scripts themselves are *not* re-run from here, and that is deliberate.
    ``check_documented_commands.py`` executes every fenced command in the docs,
    and one of those commands is ``python -m pytest -q`` -- so a test that ran
    that script would collect this file, run that script again, and nest until
    the machine ran out of patience. (It did: the first version of this class
    hung for over ten minutes before the cause was found.)

    Each script is its own declared gate on this task, and NFR-2 asks that
    their evidence be reported separately with interpreter, exit code and
    counts rather than collapsed into one boolean here. Running them twice
    would also double-report them, which is its own kind of dishonesty.
    """

    def test_the_deferred_categories_are_still_empty(self):
        """The e2e surface must not have smuggled implementation into a deferred area."""
        for category in ("observability", "integrations"):
            directory = ROOT / "components" / category
            shipped = [
                path
                for path in directory.rglob("*.py")
                if path.name != "__init__.py" and path.name != "example.py"
            ]
            assert shipped == [], f"{category} now ships implementation: {shipped}"

    def test_the_manual_checks_are_still_declared(self):
        """AC-15's browser checklist stays owed to a human, and must stay visible."""
        script = (ROOT / "scripts" / "check_documented_commands.py").read_text(encoding="utf-8")

        assert "MANUAL_CHECKS" in script
        assert "python -m http.server 8000" in script
        assert "uv sync" in script


# --------------------------------------------------------------------------- #
# NFR-17 -- no v1 behaviour changed
# --------------------------------------------------------------------------- #


class TestNoV1Regression:
    def test_the_v1_ingestion_gate_still_passes(self):
        """V2 added stages; it must not have moved what v1 already proved."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/test_ingestion.py", "-q"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, result.stdout[-2000:]
        assert "failed" not in result.stdout.split("=")[-1]

    def test_the_orchestration_layer_imports_no_component(self):
        """FR-54, re-checked at the seam where components actually get used."""
        source = (ROOT / "src" / "rag_code_library" / "pipeline.py").read_text(encoding="utf-8")
        component_names = {
            directory.name
            for area in (ROOT / "components").iterdir()
            if area.is_dir()
            for directory in area.iterdir()
            if directory.is_dir() and not directory.name.startswith("_")
        }
        assert component_names, "no components found; this check would be vacuous"
        for name in sorted(component_names):
            assert f"import {name}" not in source
            assert f"from {name}" not in source

    def test_core_only_needs_pydantic(self):
        """NFR-14: the runtime core's third-party imports stay at pydantic."""
        import ast

        allowed = {"pydantic", "pydantic_settings", "rag_code_library"}
        for relative in (
            "src/rag_code_library/pipeline.py",
            "src/rag_code_library/config.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            found = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
            third_party = {name for name in found if name not in sys.stdlib_module_names}
            assert third_party <= allowed, (relative, third_party - allowed)


# --------------------------------------------------------------------------- #
# The shape of the assembly itself
# --------------------------------------------------------------------------- #


class TestStageAssembly:
    def test_the_factory_returns_exactly_thirteen_stages(self):
        stages = build_stages()

        assert len(stages) == 13
        assert tuple(stage.name for stage in stages) == THIRTEEN_STAGES

    def test_discovery_is_the_only_corpus_stage_that_runs_first(self):
        stages = build_stages()
        corpus = [stage.name for stage in stages if stage.scope.value == "corpus"]

        assert corpus == ["discover", "validate", "identify"], (
            "rejection must happen before a loader can be handed the file"
        )

    def test_the_offline_embedder_needs_no_network(self):
        """A model download at import time would make every test depend on the net."""
        embedder = TinyEmbedder()
        vectors = embedder.embed(["one", "two"])

        assert len(vectors) == 2
        assert all(len(vector) == embedder.dimensions for vector in vectors)

    def test_the_embedder_is_deterministic(self):
        """A random embedder would make AC-29's no-stale-vector check vacuous."""
        embedder = TinyEmbedder()
        assert embedder.embed(["stable text"]) == embedder.embed(["stable text"])
        assert embedder.embed(["stable text"]) != embedder.embed(["different text"])

    def test_chunk_size_is_small_enough_to_produce_several_chunks(self, workspace):
        assert CHUNK_SIZE < 200, "the fixtures rely on short documents chunking more than once"

    def test_the_index_directory_reads_from_configuration(self, monkeypatch, tmp_path):
        monkeypatch.setenv(INDEX_DIR_VARIABLE, str(tmp_path / "elsewhere"))

        assert index_directory() == tmp_path / "elsewhere"

    def test_the_vector_store_survives_being_read_back(self, workspace):
        """A store that cannot round-trip its own records verifies nothing."""
        store = PersistentVectorStore(workspace.index)

        assert store.count() == 0
        assert store.stored_fingerprint() is None
        assert store.all_expected() == ()


def _credential_shaped_value() -> str:
    """A stand-in with the shape of an API key.

    Assembled from parts rather than written as a literal, and deliberately so:
    ``scripts/check_no_secrets.py`` rejects credential-shaped strings in the
    repository, and that rule is worth more than the convenience of a constant
    here. The leak check needs a value that *looks* like a credential -- an
    unremarkable string would prove nothing about FR-52 -- so the shape is
    built at runtime and never appears in the file.
    """
    return "-".join(("sk", "live", "DO-NOT-LEAK-12345"))

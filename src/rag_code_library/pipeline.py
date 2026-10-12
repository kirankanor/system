"""The ingestion orchestration layer (v2.md section 4.B; FR-51..FR-54).

What this module is
-------------------
It runs stages. That is the whole of it: a list of named callables applied in
order -- once over a corpus, then once per document -- with the output of each
handed to the next and the document's identity carried alongside them all.

What it deliberately is not
---------------------------
It does not know what a stage does, and it must not. There is no branch on a
storage backend, a parser or a provider anywhere in this file, and no import
of any ``components/`` module or of a workflow framework. FR-54 forbids all
three, and forbidding them is what makes the layer worth having: a component
is swapped by handing the pipeline a different callable, not by editing this
file. Nothing here needs to be re-reviewed when a parser changes.

That is also why ``ragkit ingest`` does not assemble stages itself. It
resolves a factory by dotted path from configuration and calls it, so the
composition root lives wherever the user's project keeps such things.

The two scopes
--------------
v2.md's twenty steps are not all the same shape. Discovery runs once over a
directory; chunking runs once per document. So a stage declares a scope:

* ``CORPUS`` stages run in order against one shared :class:`CorpusState`.
  The stage that discovers documents records them under
  :data:`DOCUMENTS_KEY`, which is how they reach the per-document phase.
* ``DOCUMENT`` stages run in order against one :class:`DocumentState` per
  discovered document.

:func:`Pipeline.run_document` accepts a named contiguous range of the document
stages, so "load through deduplicate" can be run without re-running discovery.

Failures are the caller's policy, not this module's
---------------------------------------------------
Whether one bad document aborts a run, is skipped, or is set aside for later
is a :class:`FailurePolicy` decision, not a hardcoded behaviour. What this
module guarantees regardless: no document is silently lost. Every document
that entered the run appears in the summary with a status and, if it failed,
the reason -- including the documents that were skipped because they were
unchanged, so "nothing happened" is distinguishable from "nothing was tried".

Errors are never swallowed and never re-typed. The exception a stage raised
propagates unchanged, because a caller classifying failures (see the retry
policy in the config layer) can only do so from the type it observes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .security.path_safety import SafeDestination

__all__ = [
    "CHUNKS_KEY",
    "DOCUMENTS_KEY",
    "CheckpointStore",
    "CorpusState",
    "DocumentOutcome",
    "DocumentRecord",
    "DocumentState",
    "DocumentStatus",
    "FailurePolicy",
    "Pipeline",
    "PipelineConfigurationError",
    "PipelineConcurrencyError",
    "PipelineError",
    "RunSummary",
    "Stage",
    "StageOutcome",
    "StageScope",
    "StageStatus",
    "StageSummary",
]

#: The :class:`CorpusState` key under which a discovery stage hands the
#: pipeline its documents. A documented convention rather than an injected
#: hook: one name, no configuration, and a stage that puts its list anywhere
#: else simply is not a discovery stage.
DOCUMENTS_KEY = "documents"

#: The :meth:`DocumentState.provenance` key under which a document stage may
#: record how many chunks it produced. The pipeline reads it only when no
#: ``count_chunks`` hook was supplied, so a project that counts chunks some
#: other way can say so instead. A number, not a list: the pipeline must not
#: hold every chunk alive just to ask how many there were.
CHUNKS_KEY = "chunks"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class PipelineError(Exception):
    """Base for every error this layer raises on purpose.

    :func:`rag_code_library.cli.main` reports anything raised from this package
    as a domain error with an actionable message, so these carry that message
    rather than a bare code.
    """


class PipelineConfigurationError(PipelineError):
    """The pipeline was asked to do something it cannot do as configured."""


class PipelineConcurrencyError(PipelineError):
    """Another run already holds the destination's write lock."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class StageScope(str, Enum):
    """How often a stage runs."""

    #: Once per run, against the shared :class:`CorpusState`.
    CORPUS = "corpus"
    #: Once per discovered document, against that document's state.
    DOCUMENT = "document"


class FailurePolicy(str, Enum):
    """What one document's failure does to the rest of the run."""

    #: Stop the run. The failing document and every later one are reported,
    #: and :meth:`Pipeline.run` raises the stage's own exception.
    STOP = "stop"
    #: Record the failure and continue with the next document.
    SKIP = "skip"
    #: Record the failure, mark the document quarantined, and continue.
    QUARANTINE = "quarantine"


class DocumentStatus(str, Enum):
    """Where a document ended up."""

    PROCESSED = "processed"
    #: Every document stage already succeeded in an earlier run with the same
    #: content hash and fingerprint, so nothing was attempted.
    SKIPPED = "skipped"
    FAILED = "failed"
    #: Held aside by :attr:`FailurePolicy.QUARANTINE` for later inspection.
    QUARANTINED = "quarantined"
    #: The corpus phase refused it before any document stage ran.
    REJECTED = "rejected"


class StageStatus(str, Enum):
    """Outcome of one stage over one document."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: Not run: an earlier stage of this document failed first.
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass
class CorpusState:
    """What a corpus-scope stage receives, and what it leaves behind.

    ``artifacts`` is the hand-off: a stage writes its output under its own
    name and reads whatever it needs from any earlier one.
    """

    #: Directory (or other seed) the run started from.
    source: str
    #: Identity and provenance visible to every corpus stage. Seeded by the
    #: pipeline with the run's correlation id and the source path.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Stage name -> that stage's output.
    artifacts: dict[str, Any] = field(default_factory=dict)
    #: Names of the corpus stages that succeeded, in order.
    completed: list[str] = field(default_factory=list)


@dataclass
class DocumentState:
    """What a document-scope stage receives.

    The same instance is threaded through every stage of one document, so a
    stage sees what earlier stages recorded -- including their artefacts --
    and the pipeline never has to know what any of those artefacts are.

    ``document_id``, ``source_name`` and ``content_hash`` are immutable
    identity. Everything a stage records downstream of them is provenance, and
    the pipeline guarantees all three are present in every stage's input, which
    is FR-51's "propagates document identity and provenance to every downstream
    artifact".
    """

    #: Stable identity for this document, carried to every artifact.
    document_id: str
    #: How the document was found (usually its file name). Recorded, never
    #: trusted: a downstream stage must not treat it as a path.
    source_name: str
    #: Hash of the document's content, which is what "unchanged" is decided on.
    content_hash: str
    #: Set when the corpus phase refused this document before any document
    #: stage ran -- a validation stage naming an unsupported format, an empty
    #: file or an oversized one. The pipeline runs no document stages for it
    #: and reports it as rejected with this as the reason.
    #:
    #: The *classification* stays with the stage that made it. This layer
    #: cannot know what an unsupported format is without importing a parser,
    #: which FR-54 forbids, so it records the verdict rather than forming one.
    rejected_reason: str | None = None
    #: Identity plus anything earlier stages chose to record. Always contains
    #: the three fields above under the same names.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Stage name -> that stage's output.
    artifacts: dict[str, Any] = field(default_factory=dict)
    #: Names of the document stages that succeeded, in order.
    completed: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Identity is written into provenance once, here, rather than being
        # asked of every stage. A stage that forgets to stamp its artifact
        # still leaves a state whose provenance names the document.
        self.provenance.setdefault("document_id", self.document_id)
        self.provenance.setdefault("source_name", self.source_name)
        self.provenance.setdefault("content_hash", self.content_hash)


@dataclass(frozen=True)
class Stage:
    """One named, injected callable. The pipeline knows nothing else about it.

    ``run`` receives the current state and returns its output; the pipeline
    stores that output under ``name`` and hands the state to the next stage.
    It never inspects, validates or wraps the output, because doing so would
    mean knowing what the stage was.
    """

    name: str
    run: Callable[[Any], Any]
    scope: StageScope = StageScope.DOCUMENT


@dataclass(frozen=True)
class DocumentRecord:
    """What an earlier run recorded about one document (FR-53).

    The three fields together are the whole resumability contract: identical
    content hash means the input did not change, identical fingerprint means
    the configuration that consumes it did not, and ``stages_completed`` says
    how far the last run got.
    """

    document_id: str
    content_hash: str
    #: Opaque string describing the configuration that produced the artifacts
    #: (see the config layer's ``IndexFingerprint``). Compared for equality
    #: only: this module must not parse it, or it would be coupling.
    fingerprint: str
    stages_completed: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "fingerprint": self.fingerprint,
            "stages_completed": list(self.stages_completed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DocumentRecord:
        return cls(
            document_id=str(payload["document_id"]),
            content_hash=str(payload["content_hash"]),
            fingerprint=str(payload.get("fingerprint", "")),
            stages_completed=tuple(str(s) for s in payload.get("stages_completed", ())),
        )


class CheckpointStore(MutableMapping):
    """Per-document records that outlive a single run (FR-53).

    A resume mechanism that forgets between invocations is not a resume
    mechanism: :class:`Pipeline` only ever sees a mapping, so whether the
    records survive the process is this class's business. It is a small JSON
    document in the destination, written through the same guarded path as any
    other write (NFR-3), and rewritten on each change so a run killed
    mid-document keeps everything it had already finished.

    Deliberately not a database. The file is rewritten whole because it is
    small -- one record per document, three short fields -- and a store that
    needed a server would be a store the minimal build could not run.
    """

    def __init__(
        self,
        destination: str | os.PathLike[str],
        name: str = "_ragkit.checkpoints.json",
    ) -> None:
        self._safe = SafeDestination(destination)
        self._name = name
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        """Read the previous run's records, tolerating an absent or broken file.

        A corrupt file is treated as empty rather than raised on: the
        consequence is reprocessing, which is always safe, whereas refusing to
        start would make a damaged file permanently block the destination.
        """
        if not self._safe.exists(self._name):
            return
        try:
            payload = json.loads(self._safe.resolve(self._name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, list):
            return
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            try:
                record = DocumentRecord.from_dict(entry)
            except (KeyError, TypeError):
                continue  # one bad entry must not discard the rest
            self._records[record.document_id] = record

    def _save(self) -> None:
        self._safe.write_text(
            self._name,
            json.dumps([r.to_dict() for r in self._records.values()], indent=2) + "\n",
            force=True,
        )

    # -- Mapping ----------------------------------------------------------- #

    def __getitem__(self, key: str) -> DocumentRecord:
        return self._records[key]

    def __setitem__(self, key: str, value: DocumentRecord) -> None:
        self._records[key] = value
        self._save()

    def __delitem__(self, key: str) -> None:
        del self._records[key]
        self._save()

    def __iter__(self) -> Iterable[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True)
class StageOutcome:
    """One stage's result over one document."""

    name: str
    status: StageStatus
    duration_s: float
    #: ``str()`` of the stage's exception. Never the content of anything the
    #: stage handled -- see :class:`RunSummary` on leaking secrets.
    error: str | None = None


@dataclass(frozen=True)
class StageSummary:
    """One stage's result across the whole run (FR-52's per-stage figures).

    Durations are summed across documents, because that is the number that
    says where a run actually spent its time; ``max_duration_s`` sits beside it
    so a single slow document is not averaged out of existence.
    """

    name: str
    #: ``FAILED`` if any document failed this stage, ``SKIPPED`` if no
    #: document reached it at all, otherwise ``SUCCEEDED``.
    #:
    #: The middle case matters. A stage that no document got to has not
    #: "succeeded over zero documents" -- that phrasing reads as a pass and is
    #: how a run whose first stage failed could look healthy. It did not run,
    #: so it is reported as not run.
    status: StageStatus
    documents: int
    failed: int
    duration_s: float
    max_duration_s: float


@dataclass(frozen=True)
class DocumentOutcome:
    """Where one document ended up, and why."""

    document_id: str
    source_name: str
    status: DocumentStatus
    stages_completed: tuple[str, ...]
    duration_s: float
    #: The stage that failed, if any.
    failed_stage: str | None = None
    reason: str | None = None
    #: Stage the document resumed from rather than starting over, if any.
    resumed_from: str | None = None
    #: Count reported by the pipeline's ``count_chunks`` hook (0 when none).
    chunks: int = 0


@dataclass(frozen=True)
class RunSummary:
    """Everything one run wants to say, and nothing it must not say.

    FR-52 requires the summary to carry no credentials and no full document
    content. This type holds only counts, stage names, statuses, durations and
    short reasons; the values a stage produced stay in its artefact and are
    never copied in here. The one place a message could leak is ``reason``,
    which is the ``str()`` of an exception -- deliberately short, and the
    reason the pipeline never prints document text itself.
    """

    correlation_id: str
    source: str
    started_at: float
    duration_s: float
    failure_policy: FailurePolicy
    #: Per-stage status and durations, in run order (FR-52).
    stages: tuple[StageSummary, ...]
    #: What the corpus phase found, NOT how many documents were attempted.
    #: Under :attr:`FailurePolicy.STOP` the two differ, and deriving this from
    #: the outcome list would make a run that stopped early look like it
    #: discovered less than it did -- quietly erasing the documents it never
    #: reached from the record.
    documents_discovered: int
    documents: tuple[DocumentOutcome, ...]

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages)

    # -- counts ------------------------------------------------------------ #

    @property
    def documents_rejected(self) -> int:
        return sum(1 for d in self.documents if d.status is DocumentStatus.REJECTED)

    @property
    def documents_accepted(self) -> int:
        """Discovered minus rejected: the documents work was attempted on.

        Under :attr:`FailurePolicy.STOP` this is an upper bound -- the run
        stopped, so a rejected document later in the list was never reached and
        is neither counted as rejected nor as anything else. ``discovered``
        stays exact regardless; it is the breakdown that STOP makes partial.
        """
        return self.documents_discovered - self.documents_rejected

    @property
    def documents_processed(self) -> int:
        return sum(1 for d in self.documents if d.status is DocumentStatus.PROCESSED)

    @property
    def documents_failed(self) -> int:
        return sum(1 for d in self.documents if d.status is DocumentStatus.FAILED)

    @property
    def documents_skipped(self) -> int:
        return sum(1 for d in self.documents if d.status is DocumentStatus.SKIPPED)

    @property
    def documents_quarantined(self) -> int:
        return sum(1 for d in self.documents if d.status is DocumentStatus.QUARANTINED)

    @property
    def chunks_processed(self) -> int:
        """Chunks reported by documents that reached the end.

        Counts are contributed by the stages that produce chunks, via the
        pipeline's ``count_chunks`` hook. A rejected document contributes none,
        because nothing was written for it -- which is the difference between
        "processed zero" and "not processed".
        """
        return sum(d.chunks for d in self.documents if d.status is DocumentStatus.PROCESSED)

    @property
    def chunks_failed(self) -> int:
        return sum(
            d.chunks
            for d in self.documents
            if d.status in (DocumentStatus.FAILED, DocumentStatus.QUARANTINED)
        )

    # -- presentation ------------------------------------------------------ #

    def counts(self) -> dict[str, int]:
        """The counts as a plain mapping, for a run summary file."""
        return {
            "documents_discovered": self.documents_discovered,
            "documents_accepted": self.documents_accepted,
            "documents_rejected": self.documents_rejected,
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "documents_skipped": self.documents_skipped,
            "documents_quarantined": self.documents_quarantined,
            "chunks_processed": self.chunks_processed,
            "chunks_failed": self.chunks_failed,
        }

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of the run.

        Serializes by hand rather than through a dataclass encoder so the
        shape is explicit: whatever a stage returned in its artefacts cannot
        reach this file, because this file never mentions artefacts.
        """
        return {
            "correlation_id": self.correlation_id,
            "source": self.source,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 6),
            "failure_policy": self.failure_policy.value,
            "stages": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "documents": s.documents,
                    "failed": s.failed,
                    "duration_s": round(s.duration_s, 6),
                    "max_duration_s": round(s.max_duration_s, 6),
                }
                for s in self.stages
            ],
            "counts": self.counts(),
            "documents": [
                {
                    "document_id": d.document_id,
                    "source_name": d.source_name,
                    "status": d.status.value,
                    "stages_completed": list(d.stages_completed),
                    "duration_s": round(d.duration_s, 6),
                    "failed_stage": d.failed_stage,
                    "reason": d.reason,
                    "resumed_from": d.resumed_from,
                    "chunks": d.chunks,
                }
                for d in self.documents
            ],
        }

    def report(self) -> str:
        """A short human-readable summary, one line per headline figure."""
        lines = [
            f"run {self.correlation_id}  source={self.source}  "
            f"policy={self.failure_policy.value}  {self.duration_s:.2f}s",
        ]
        for key, value in self.counts().items():
            lines.append(f"  {key}: {value}")
        for stage in self.stages:
            lines.append(
                f"  stage {stage.name}: {stage.status.value} over "
                f"{stage.documents} document(s), {stage.duration_s:.3f}s"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    """Run injected stages over a corpus and then over each document.

    Nothing in the constructor names a backend, a parser or a provider. What it
    takes instead is the stage list itself, an opaque configuration
    fingerprint, a place to keep per-document records, and a destination whose
    write lock this class takes for the duration of a run.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        *,
        failure_policy: FailurePolicy = FailurePolicy.SKIP,
        fingerprint: str = "",
        checkpoints: MutableMapping[str, DocumentRecord] | None = None,
        destination: str | os.PathLike[str] | None = None,
        lock_file: str = "_ragkit.lock",
        count_chunks: Callable[[DocumentState], int] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        correlation_id: str | None = None,
    ) -> None:
        self.stages = tuple(stages)
        self.failure_policy = FailurePolicy(failure_policy)
        self.fingerprint = fingerprint
        self.checkpoints: MutableMapping[str, DocumentRecord] = (
            checkpoints if checkpoints is not None else {}
        )
        self.destination = Path(destination) if destination is not None else None
        self.lock_file = lock_file
        self.count_chunks = count_chunks
        self.clock = clock
        self.correlation_id = correlation_id or uuid.uuid4().hex
        self._validate()

    # -- configuration ----------------------------------------------------- #

    def _validate(self) -> None:
        """Refuse a stage list the pipeline cannot run honestly.

        Checked once, at construction, so a misconfigured pipeline fails before
        a single document is touched rather than partway through a run.
        """
        if not self.stages:
            raise PipelineConfigurationError(
                "the pipeline has no stages; supply at least one Stage or "
                "configure where to load them from"
            )
        names = [stage.name for stage in self.stages]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise PipelineConfigurationError(
                f"duplicate stage name(s): {', '.join(duplicates)}; stage names "
                "are the keys artefacts and records are stored under, so they "
                "must be unique"
            )
        for stage in self.stages:
            if not stage.name:
                raise PipelineConfigurationError("a stage has an empty name")
            if not callable(stage.run):
                raise PipelineConfigurationError(
                    f"stage {stage.name!r} is not callable; stages are injected "
                    "as callables (FR-54), so the pipeline only ever calls them"
                )

    @property
    def corpus_stages(self) -> tuple[Stage, ...]:
        return tuple(s for s in self.stages if s.scope is StageScope.CORPUS)

    @property
    def document_stages(self) -> tuple[Stage, ...]:
        return tuple(s for s in self.stages if s.scope is StageScope.DOCUMENT)

    def stage_names(self, *, scope: StageScope | None = None) -> tuple[str, ...]:
        """Stage names in run order, optionally restricted to one scope."""
        if scope is None:
            return tuple(s.name for s in self.stages)
        return tuple(s.name for s in self.stages if s.scope is scope)

    # -- the public entry points ------------------------------------------- #

    def run(self, source: str | os.PathLike[str]) -> RunSummary:
        """Run the whole pipeline: corpus stages, then every document stage.

        Takes the destination's write lock for the duration, so two runs over
        the same destination cannot interleave. The lock is released even when
        the run raises, so a failed run does not block the next one.

        Always returns a summary. Under :attr:`FailurePolicy.STOP` the failing
        document ends the run and later documents are simply absent from it --
        the counts say so rather than the summary being thrown away along with
        the error. A caller wanting a non-zero exit asks the summary, which is
        why ``documents_failed`` exists; the stage's own exception type is seen
        by the stage wrapper that raised it, so classification stays with
        whoever injected the stage.
        """
        started = self.clock()
        source_str = str(source)
        with self._write_lock():
            corpus = self._run_corpus(source_str)
            seeds = tuple(corpus.artifacts.get(DOCUMENTS_KEY) or ())
            outcomes: list[DocumentOutcome] = []
            measured: list[StageOutcome] = []
            for seed in seeds:
                outcome, per_document = self._run_one(seed, corpus.provenance)
                outcomes.append(outcome)
                measured.extend(per_document)
                if (
                    outcome.status is DocumentStatus.FAILED
                    and self.failure_policy is FailurePolicy.STOP
                ):
                    break
            summary = RunSummary(
                correlation_id=self.correlation_id,
                source=source_str,
                started_at=started,
                duration_s=self.clock() - started,
                failure_policy=self.failure_policy,
                stages=_aggregate_stages(
                    measured, self.stage_names(scope=StageScope.DOCUMENT)
                ),
                documents_discovered=len(seeds),
                documents=tuple(outcomes),
            )
        return summary

    def run_document(
        self,
        document: DocumentState | Mapping[str, Any] | Any,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> DocumentOutcome:
        """Run a named contiguous range of document stages over one document.

        ``start`` and ``end`` are inclusive stage names, so "load through
        deduplicate" is ``start="load", end="dedup"``. Either may be omitted to
        run from the first stage or to the last. A name that is not a document
        stage, or an inverted range, is a configuration error rather than a
        silently shortened run.

        Deliberately does NOT write a checkpoint record. A range is the
        caller's choice, so declaring "these stages are done" here would let a
        partial, hand-picked run masquerade as a finished document on the next
        :meth:`run`. Resumability is a property of running the whole pipeline.
        """
        state = _as_document_state(document)
        started = self.clock()
        names = self.stage_names(scope=StageScope.DOCUMENT)
        if not names:
            raise PipelineConfigurationError(
                "the pipeline declares no document stages, so there is nothing "
                "to run for a document"
            )
        first, last = self._range(names, start, end)
        outcome, _measured = self._apply_range(
            state, names, first, last, started, record=False
        )
        return replace(outcome, duration_s=self.clock() - started)

    # -- internals --------------------------------------------------------- #

    def _run_corpus(self, source: str) -> CorpusState:
        """Run the corpus stages, then raise whatever any of them raised."""
        state = CorpusState(
            source=source,
            provenance={"correlation_id": self.correlation_id, "source": source},
        )
        for stage in self.corpus_stages:
            state.artifacts[stage.name] = stage.run(state)
            state.completed.append(stage.name)
        return state

    def _run_one(
        self, seed: Any, base_provenance: Mapping[str, Any] | None = None
    ) -> tuple[DocumentOutcome, tuple[StageOutcome, ...]]:
        """Decide whether a document needs work, then do it.

        Returns the document's outcome alongside this document's per-stage
        measurements, which :meth:`run` folds into the run's stage summaries.
        Never raises for a stage failure -- that decision belongs to the caller
        reading the summary, not to this layer.

        ``base_provenance`` is the run-level provenance (correlation id and
        source) from the corpus phase, copied onto every document. Without it a
        document's artifact could not be traced back to the run that produced
        it, which is most of what FR-51 means by propagating provenance.
        """
        state = _as_document_state(seed, base_provenance)
        started = self.clock()
        names = self.stage_names(scope=StageScope.DOCUMENT)
        if not names:
            raise PipelineConfigurationError(
                "the pipeline declares no document stages; discovery produced "
                "a document with nowhere to put it"
            )

        if state.rejected_reason is not None:
            # The corpus phase already judged this one. No document stage runs,
            # and nothing is recorded for it -- a rejected document has no work
            # to resume, and recording it would make the next run believe it
            # had been processed partway.
            return (
                DocumentOutcome(
                    document_id=state.document_id,
                    source_name=state.source_name,
                    status=DocumentStatus.REJECTED,
                    stages_completed=(),
                    duration_s=self.clock() - started,
                    reason=state.rejected_reason,
                ),
                (),
            )

        # FR-53: skip what an earlier run already finished with the same
        # content and the same configuration, resume a partially finished one,
        # and otherwise start over. The record is authoritative only while both
        # the hash and the fingerprint still match -- a changed embedding model
        # or chunking size must not reuse artifacts produced under the old one.
        record = self.checkpoints.get(state.document_id)
        first, resumed_from = 0, None
        if record is not None and record.content_hash == state.content_hash:
            if record.fingerprint == self.fingerprint:
                if record.stages_completed == names:
                    # Nothing to do, and no stage ran -- so this document
                    # contributes nothing to the stage measurements either.
                    return (
                        DocumentOutcome(
                            document_id=state.document_id,
                            source_name=state.source_name,
                            status=DocumentStatus.SKIPPED,
                            stages_completed=record.stages_completed,
                            duration_s=self.clock() - started,
                            reason="unchanged since the previous run",
                            chunks=self._count_chunks(state),
                        ),
                        (),
                    )
                if record.stages_completed == names[: len(record.stages_completed)]:
                    # A strict, in-order prefix: an interrupted run. Continue
                    # where it stopped rather than repeating finished work.
                    #
                    # The prefix is seeded back onto the state, so the record
                    # written when this run finishes describes the WHOLE
                    # document rather than only the part this run did. Without
                    # it the next run would see a truncated record and start
                    # over -- resume would silently never converge.
                    state.completed = list(record.stages_completed)
                    first = len(record.stages_completed)
                    resumed_from = names[first] if first < len(names) else None

        outcome, measured = self._apply_range(
            state, names, first, len(names) - 1, started, record=True
        )
        if resumed_from is not None:
            outcome = replace(outcome, resumed_from=resumed_from)
        return outcome, measured

    def _apply_range(
        self,
        state: DocumentState,
        names: tuple[str, ...],
        first: int,
        last: int,
        started: float,
        *,
        record: bool,
    ) -> tuple[DocumentOutcome, tuple[StageOutcome, ...]]:
        """Run document stages ``names[first..last]`` inclusive.

        Returns the outcome and this document's per-stage measurements. Stages
        the document never reached are simply absent, which is what makes the
        run's per-stage counts mean "documents that ran this stage" rather than
        "documents that existed".
        """
        by_name = {s.name: s for s in self.stages}
        names_set = set(names)
        failing: Exception | None = None
        failed_stage: str | None = None
        status = DocumentStatus.PROCESSED
        measured: list[StageOutcome] = []

        for index in range(first, last + 1):
            stage = by_name[names[index]]
            stage_started = self.clock()
            try:
                state.artifacts[stage.name] = stage.run(state)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                measured.append(
                    StageOutcome(
                        name=stage.name,
                        status=StageStatus.FAILED,
                        duration_s=self.clock() - stage_started,
                        error=str(exc) or type(exc).__name__,
                    )
                )
                # The exception is recorded, not replaced: nothing here needs
                # the type, and the stage that raised it still owns it.
                #
                # ``state.completed`` is deliberately NOT extended here. It
                # means "stages that succeeded", and the failing stage did not.
                # Later stages were never attempted and must not be counted as
                # done either -- a list called ``completed`` that quietly
                # included stages that never ran would make the resume logic
                # believe an interrupted document was finished.
                failing, failed_stage = exc, stage.name
                status = (
                    DocumentStatus.QUARANTINED
                    if self.failure_policy is FailurePolicy.QUARANTINE
                    else DocumentStatus.FAILED
                )
                break
            else:
                state.completed.append(stage.name)
                measured.append(
                    StageOutcome(
                        name=stage.name,
                        status=StageStatus.SUCCEEDED,
                        duration_s=self.clock() - stage_started,
                    )
                )

        outcome = DocumentOutcome(
            document_id=state.document_id,
            source_name=state.source_name,
            status=status,
            stages_completed=tuple(n for n in state.completed if n in names_set),
            duration_s=self.clock() - started,
            failed_stage=failed_stage,
            reason=None if failing is None else str(failing) or type(failing).__name__,
            chunks=self._count_chunks(state) if status is DocumentStatus.PROCESSED else 0,
        )

        if record:
            # Always record the honest prefix -- never only on success.
            #
            # A document that finished writes all its stage names, so the next
            # run skips it as unchanged. One that failed partway writes only
            # the stages that actually succeeded, which is exactly a strict
            # prefix, so the next run resumes after them instead of repeating
            # work already done -- FR-53's "recoverable work can be resumed
            # without duplicating successful writes".
            #
            # This is safe only because ``state.completed`` never includes the
            # stage that failed. Recording that one would turn "got halfway"
            # into "finished that stage", which is the fabricated completeness
            # the whole record exists to avoid.
            self.checkpoints[state.document_id] = DocumentRecord(
                document_id=state.document_id,
                content_hash=state.content_hash,
                fingerprint=self.fingerprint,
                stages_completed=tuple(state.completed),
            )
        return outcome, tuple(measured)

    def _count_chunks(self, state: DocumentState) -> int:
        """How many chunks this document produced, or 0 when nothing said.

        A hook wins when one was supplied; otherwise a stage that recorded a
        count under :data:`CHUNKS_KEY` is believed. Returning 0 rather than
        guessing is deliberate: a wrong chunk count in a run summary is worse
        than an absent one, because it looks like an answer.
        """
        if self.count_chunks is not None:
            try:
                return int(self.count_chunks(state))
            except Exception:  # noqa: BLE001 - a counting hook must not fail a run
                return 0
        recorded = state.provenance.get(CHUNKS_KEY, 0)
        return recorded if isinstance(recorded, int) and not isinstance(recorded, bool) else 0

    def _range(
        self, names: tuple[str, ...], start: str | None, end: str | None
    ) -> tuple[int, int]:
        """Turn inclusive stage names into inclusive indices."""
        def index_of(name: str) -> int:
            if name not in names:
                raise PipelineConfigurationError(
                    f"{name!r} is not a document stage in this pipeline; "
                    f"the document stages are: {', '.join(names) or '(none)'}"
                )
            return names.index(name)

        first = index_of(start) if start is not None else 0
        last = index_of(end) if end is not None else len(names) - 1
        if first > last:
            raise PipelineConfigurationError(
                f"stage range is inverted: start={start!r} runs after end={end!r}"
            )
        return first, last

    # -- the write lock ---------------------------------------------------- #

    def _write_lock(self) -> _WriteLock:
        if self.destination is None:
            # No destination means no shared mutable state to guard, so there
            # is nothing two runs could corrupt and a lock would be theatre.
            return _WriteLock(None, self.correlation_id)
        return _WriteLock(
            SafeDestination(self.destination), self.correlation_id, self.lock_file
        )


@dataclass
class _WriteLock:
    """An exclusive file lock over one destination (FR-53's enforcement).

    The policy is stated, not implied: one run owns the destination at a time,
    so two runs can never mutate the same document concurrently. The lock is a
    file rather than an in-process guard because two ``Pipeline`` objects in
    two processes are exactly the case that matters.

    It uses ``O_CREAT | O_EXCL``, which is atomic on POSIX and on Windows, and
    records who holds it so a refusal can name the holder instead of merely
    saying "no". A process killed mid-run leaves the file behind; that is the
    known limitation, and the error says so and says how to clear it rather
    than guessing on the caller's behalf whether the holder is still alive.
    """

    destination: SafeDestination | None
    correlation_id: str
    name: str = "_ragkit.lock"
    acquired: bool = False

    def __enter__(self) -> _WriteLock:
        if self.destination is not None:
            self.destination.ensure_root()
            payload = json.dumps(
                {
                    "correlation_id": self.correlation_id,
                    "pid": os.getpid(),
                    "acquired_at": time.time(),
                },
                indent=2,
            )
            target = str(self.destination.resolve(self.name))
            try:
                descriptor = os.open(
                    target, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError as exc:
                holder = _describe_holder(target)
                raise PipelineConcurrencyError(
                    f"another run already holds the lock for this destination "
                    f"({holder}); refusing to start a second run that could "
                    f"write the same documents. If no run is active, delete "
                    f"{target} and retry"
                ) from exc
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            self.acquired = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.acquired and self.destination is not None:
            try:
                os.remove(self.destination.resolve(self.name))
            except OSError:
                # Failing to clean up must not mask the run's real outcome.
                pass
            self.acquired = False


def _aggregate_stages(
    measured: Sequence[StageOutcome], names: Sequence[str]
) -> tuple[StageSummary, ...]:
    """Fold per-document stage measurements into per-stage run figures.

    Every declared document stage appears, even one no document reached, so a
    reader can tell "this stage ran on nothing" from "this stage is missing".
    """
    totals: dict[str, dict[str, Any]] = {
        name: {
            "documents": 0,
            "failed": 0,
            "duration_s": 0.0,
            "max_duration_s": 0.0,
        }
        for name in names
    }
    for outcome in measured:
        entry = totals.get(outcome.name)
        if entry is None:  # pragma: no cover - a stage the pipeline declared
            continue
        entry["documents"] += 1
        entry["duration_s"] += outcome.duration_s
        entry["max_duration_s"] = max(entry["max_duration_s"], outcome.duration_s)
        if outcome.status is StageStatus.FAILED:
            entry["failed"] += 1
    return tuple(
        StageSummary(
            name=name,
            status=(
                StageStatus.FAILED
                if entry["failed"]
                else StageStatus.SKIPPED
                if not entry["documents"]
                else StageStatus.SUCCEEDED
            ),
            documents=entry["documents"],
            failed=entry["failed"],
            duration_s=entry["duration_s"],
            max_duration_s=entry["max_duration_s"],
        )
        for name, entry in totals.items()
    )


def _describe_holder(target: str) -> str:
    """Read enough of a lock file to name the run that holds it."""
    try:
        payload = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"held by an unreadable lock file {target}"
    return (
        f"held by run {payload.get('correlation_id', 'unknown')!r} "
        f"(pid {payload.get('pid', 'unknown')}, since "
        f"{payload.get('acquired_at', 'unknown')})"
    )


def _as_document_state(
    seed: Any, base_provenance: Mapping[str, Any] | None = None
) -> DocumentState:
    """Accept a :class:`DocumentState`, or build one from a plain seed.

    Discovery stages should return whatever their own contract says; the
    pipeline only asks for ``document_id``, ``source_name`` and
    ``content_hash``, read from an attribute or a mapping key. That keeps the
    pipeline from importing the discovery component's types -- which FR-54
    forbids -- without forcing every caller to construct the state itself.

    Everything else on the seed is kept as provenance, as is the run-level
    provenance passed in: a validation stage's stated reason and the run's
    correlation id both belong on every artifact downstream of them.
    """
    if isinstance(seed, DocumentState):
        return seed

    def read(name: str) -> str:
        value = (
            seed.get(name)
            if isinstance(seed, Mapping)
            else getattr(seed, name, None)
        )
        if not isinstance(value, str) or not value:
            raise PipelineConfigurationError(
                f"a discovered document must provide a non-empty {name!r}; got "
                f"{value!r} from {type(seed).__name__}"
            )
        return value

    identity = ("document_id", "source_name", "content_hash")
    extra = {
        key: value
        for key, value in _seed_fields(seed, identity).items()
        if value is not None
    }
    reason = extra.pop("rejected", None)
    return DocumentState(
        document_id=read("document_id"),
        source_name=read("source_name"),
        content_hash=read("content_hash"),
        rejected_reason=reason if isinstance(reason, str) else None,
        # The seed's own fields win over the run's on a collision, because a
        # document-specific fact (its page count) is more specific than a
        # run-level one (the source path) and the run's is still reachable
        # under its own name.
        provenance={**dict(base_provenance or {}), **extra},
    )


def _seed_fields(seed: Any, taken: tuple[str, ...]) -> Mapping[str, Any]:
    """The seed's remaining fields, once the identity ones are removed."""
    if isinstance(seed, Mapping):
        return {k: v for k, v in seed.items() if k not in taken}
    fields_of_seed = getattr(seed, "__dict__", None)
    if isinstance(fields_of_seed, dict):
        return {k: v for k, v in fields_of_seed.items() if k not in taken}
    return {}

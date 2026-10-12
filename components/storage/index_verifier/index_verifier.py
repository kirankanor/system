"""Verification: a read of the index, not a receipt for the write.

Why this component has to exist
--------------------------------
A write that returns success proves the write returned success. It says
nothing about whether the rows are there now, whether they are the rows that
were expected, whether their vectors are the width the index claims, or
whether a query against them returns anything at all. Pipelines that treat
"the batch completed" as "the index is good" discover the difference in
production, usually as missing answers rather than as an error.

So this module does something deliberately different: it takes the
*expectation* -- the records that should be in the index -- and reads the
index back until it can say, per record, whether that expectation holds. Two
things follow from that.

**Nothing here can write.** The class exposes ``verify()`` and no ``add``,
``put`` or ``delete``. There is no configuration in which a successful write
becomes a successful verification, because the writer is not reachable from
this object at all (FR-50).

**Readiness is stricter than passing.** ``verdict.ok`` means no fatal
finding. ``verdict.ready`` means ``ok`` *and* every check actually ran. A
verdict that could not examine something reports it in ``checks_not_run()``
rather than quietly passing -- the thing this library exists to avoid is a
green result that was never earned.

The seven comparisons
---------------------
The six FR-49 names: ``count`` (expected against actual), ``presence`` (is
every expected record really there), ``embedding`` (does each record have a
usable vector), ``dimension`` (is every vector the width the index is built
for), ``metadata`` (is what we declared compatible with what is stored), and
``relationship`` (does document -> chunk -> source still hold).

Plus ``retrieval``, the smoke check FR-50 asks for: query the index that was
just examined and confirm the probe record comes back.

Why a count match is not verification
--------------------------------------
Some backends report a count that includes stale rows, so the count can be
exactly right while a record has quietly gone. ``presence`` is the check that
names the missing record; ``count`` is the check that notices the total
moved. Both run, because neither alone is sufficient.

Fatal versus warning
--------------------
A **fatal** finding means the index cannot be trusted for retrieval: a
missing record, a dimension mismatch, a count that disagrees, a broken
relationship at the root, a failed smoke check. A **warning** means the index
will answer, but something about the answer is degraded: no source file to
cite, no location within the document, metadata declared but empty. The split
matters because a pipeline that aborts on warnings learns nothing, and one
that ignores fatals learns nothing either.

Eventual consistency (FR-50)
-----------------------------
A write may not be visible the instant it returns. ``wait_ms`` bounds how
long verification will keep re-checking *only* the findings that consistency
can fix -- count, presence, retrieval. A dimension mismatch will not resolve
itself, so waiting for one would only delay an inevitable failure. The bound
is a count of attempts derived from the configuration, not just the clock, so
a clock that does not advance cannot turn the wait into an infinite loop.

Self-contained on purpose
-------------------------
Depends on ``core-types`` and ``vector-store-protocol``, both declared in
the manifest so ``ragkit add`` copies them along.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from core_types import EmbeddedChunk, ScoredDocument

__all__ = [
    "ALL_CHECKS",
    "COUNT",
    "DIMENSION",
    "EMBEDDING",
    "FATAL",
    "Finding",
    "IndexVerifier",
    "METADATA",
    "PRESENCE",
    "RELATIONSHIP",
    "RETRIEVAL",
    "VerificationError",
    "VerificationVerdict",
    "WARNING",
]

#: The six comparisons FR-49 names, in the order they are attempted.
COUNT = "count"
PRESENCE = "presence"
EMBEDDING = "embedding"
DIMENSION = "dimension"
METADATA = "metadata"
RELATIONSHIP = "relationship"
RETRIEVAL = "retrieval"

ALL_CHECKS: tuple[str, ...] = (
    COUNT,
    PRESENCE,
    EMBEDDING,
    DIMENSION,
    METADATA,
    RELATIONSHIP,
    RETRIEVAL,
)

#: Findings an eventual-consistency wait is allowed to retry. Anything not in
#: here is structural: waiting would only postpone the same failure.
_TRANSIENT_CHECKS = frozenset({COUNT, PRESENCE, RETRIEVAL})

FATAL = "fatal"
WARNING = "warning"

#: Reserved key prefix, matching ``storage-adapters``. Visible metadata
#: should never contain it; if it does, the adapter contract has leaked.
RESERVED_PREFIX = "_rag_"


class VerificationError(Exception):
    """Base for every error raised by this component."""


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong, or one thing degraded.

    ``record`` names the offending record when the finding is about a specific
    one, and is ``None`` when it is about the index as a whole. AC-27 asks
    for the record to be named, so ``describe()`` prints ``<index>`` rather
    than leaving a blank where the name should be.
    """

    check: str
    severity: str
    reason: str
    record: str | None = None

    @property
    def fatal(self) -> bool:
        return self.severity == FATAL

    @property
    def where(self) -> str:
        return self.record if self.record is not None else "<index>"

    def describe(self) -> str:
        label = "FATAL" if self.fatal else "warn"
        return f"[{label}] {self.check} {self.where}: {self.reason}"


@dataclass(frozen=True, slots=True)
class VerificationVerdict:
    """What verification found, and what it was able to look at.

    ``checks_run`` is part of the answer, not bookkeeping: a verdict that did
    not examine something cannot claim that thing passed.
    """

    findings: tuple[Finding, ...]
    expected_records: int
    actual_records: int
    checks_run: tuple[str, ...]
    waited_ms: int = 0
    attempts: int = 1

    @property
    def fatal(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.fatal)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if not f.fatal)

    @property
    def ok(self) -> bool:
        """No fatal finding. This is *not* readiness."""
        return not self.fatal

    @property
    def ready(self) -> bool:
        """Every check ran and passed, including the retrieval smoke check.

        Deliberately unreachable while ``checks_not_run()`` is non-empty: a
        verdict missing evidence is not a verdict that passed.
        """
        return self.ok and not self.checks_not_run()

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    def checks_not_run(self) -> tuple[str, ...]:
        return tuple(c for c in ALL_CHECKS if c not in self.checks_run)

    def named(self, record: str) -> tuple[Finding, ...]:
        """Every finding about one record -- AC-27's "name the record"."""
        return tuple(f for f in self.findings if f.record == record)

    def raise_if_failed(self) -> None:
        """Raise if any fatal finding exists.

        Findings are a *result*, so nothing here raises on its own; this is
        the opt-in that lets a caller turn a verdict into a failure.
        """
        if self.ok:
            return
        first = self.fatal[0]
        raise VerificationError(
            f"index-verifier: {len(self.fatal)} fatal finding(s); first is "
            f"{first.describe()}. {self.summary()}"
        )

    def summary(self) -> str:
        state = "READY" if self.ready else ("ok" if self.ok else "NOT READY")
        return (
            f"{state}: {self.actual_records}/{self.expected_records} records, "
            f"{len(self.fatal)} fatal, {len(self.warnings)} warning, "
            f"checked {len(self.checks_run)}/{len(ALL_CHECKS)} "
            f"in {self.attempts} pass(es), waited {self.waited_ms}ms"
        )

    def report(self) -> str:
        """The verdict as readable lines -- findings first."""
        lines = [self.summary()]
        lines.extend(f"  {f.describe()}" for f in self.findings)
        missing = self.checks_not_run()
        if missing:
            lines.append(
                f"  [not run] {', '.join(missing)} -- readiness withheld "
                f"because these were never examined"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The verifier (FR-49 / FR-50)
# --------------------------------------------------------------------------- #


class IndexVerifier:
    """Read an index back against what is expected to be in it.

        verifier = IndexVerifier(store, expected=items, dimensions=1536)
        verdict = verifier.verify()
        print(verdict.report())
        verdict.raise_if_failed()

    Args:
        store: Anything ``VectorStoreProtocol``-shaped. ``count`` and
            ``query`` are what this uses; both are part of the protocol, so a
            real backend and a test double work identically.
        expected: The records that should be stored.
        dimensions: Width every stored vector must have.
        required_metadata: Metadata keys every stored record must carry.
        wait_ms: How long to keep re-checking findings that eventual
            consistency could still fix. ``0`` means exactly one pass.
        interval_ms: Pause between passes.
        probe_k: Rows to ask for per presence query. Raised to a document's
            own record count, so a large document cannot be mistaken for an
            incomplete one.
        probe: The vector to smoke-check with. Defaults to the first expected
            record's; supply one explicitly when ``expected`` is empty.
        fetch: Optional ``chunk_id -> stored vector | None`` readback. When
            the store offers one, presence and dimension are confirmed by
            reading the record directly and the finding can name it exactly.
            ``store.get`` is used automatically when present. Without a
            readback both are established by the vector query instead --
            which still names a record it could not find, but cannot inspect
            a stored vector's width, and that gap is reported in
            ``checks_not_run()`` only if the check could not run at all.
        sleep/now: Injected so tests never actually wait.
    """

    def __init__(
        self,
        store: Any,
        *,
        expected: Sequence[EmbeddedChunk],
        dimensions: int,
        required_metadata: Sequence[str] = (),
        wait_ms: int = 0,
        interval_ms: int = 100,
        probe_k: int = 16,
        probe: Sequence[float] | None = None,
        fetch: Callable[[str], Sequence[float] | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if store is None:
            raise VerificationError(
                "index-verifier: store must be provided, got None"
            )
        if expected is None:
            raise VerificationError(
                "index-verifier: expected must be provided, got None"
            )
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise VerificationError(
                f"index-verifier: dimensions must be an int, got "
                f"{type(dimensions).__name__} ({dimensions!r})"
            )
        if dimensions <= 0:
            raise VerificationError(
                f"index-verifier: dimensions must be > 0, got {dimensions}"
            )
        for name, value in (
            ("wait_ms", wait_ms),
            ("interval_ms", interval_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VerificationError(
                    f"index-verifier: {name} must be a non-negative int, "
                    f"got {value!r}"
                )
        if isinstance(probe_k, bool) or not isinstance(probe_k, int) or probe_k <= 0:
            raise VerificationError(
                f"index-verifier: probe_k must be an int > 0, got {probe_k!r}"
            )
        for method in ("count", "query"):
            if not callable(getattr(store, method, None)):
                raise VerificationError(
                    f"index-verifier: store must provide {method}(), got "
                    f"{type(store).__name__}"
                )

        self.store = store
        self.expected: tuple[EmbeddedChunk, ...] = tuple(expected)
        self.dimensions = dimensions
        self.required_metadata = tuple(required_metadata)
        self.wait_ms = wait_ms
        self.interval_ms = interval_ms
        self.probe_k = probe_k
        self._sleep = sleep
        self._now = now

        if probe is not None:
            self._probe: tuple[float, ...] | None = tuple(float(v) for v in probe)
        elif self.expected:
            self._probe = tuple(float(v) for v in self.expected[0].vector)
        else:
            self._probe = None

        if fetch is not None:
            self._fetch = fetch
        else:
            getter = getattr(store, "get", None)
            self._fetch = getter if callable(getter) else None

    @property
    def probe(self) -> tuple[float, ...] | None:
        """The vector the smoke check queries with, if one is available."""
        return self._probe

    @property
    def has_readback(self) -> bool:
        """Whether a per-record vector readback is available.

        Exposed because it changes *how* presence and dimension are
        established, and a caller is entitled to know which evidence was used.
        """
        return self._fetch is not None

    def verify(self) -> VerificationVerdict:
        """Run every check, re-running only what a wait could still fix.

        Never raises for a finding -- a failed verification is a result, and
        the caller decides what to do with it via ``raise_if_failed()``.
        """
        started = self._now()
        max_attempts = self._max_attempts()
        verdict = self._attempt()

        attempt = 1
        while attempt < max_attempts:
            fatal = verdict.fatal
            if not fatal or not all(f.check in _TRANSIENT_CHECKS for f in fatal):
                break
            remaining = (started + self.wait_ms / 1000.0) - self._now()
            if remaining <= 0:
                break
            attempt += 1
            self._sleep(min(self.interval_ms / 1000.0, remaining))
            verdict = self._attempt()

        waited_ms = int(round((self._now() - started) * 1000.0))
        return replace(verdict, waited_ms=max(waited_ms, 0), attempts=attempt)

    def _max_attempts(self) -> int:
        """A bound derived from configuration, so a dead clock cannot stall it."""
        if self.wait_ms <= 0 or self.interval_ms <= 0:
            return 1
        return 1 + math.ceil(self.wait_ms / self.interval_ms)

    # -- one pass ---------------------------------------------------------- #

    def _attempt(self) -> VerificationVerdict:
        findings: list[Finding] = []
        run: list[str] = []
        expected_count = len(self.expected)

        actual = self._count(findings)
        if actual is not None:
            run.append(COUNT)
            if actual != expected_count:
                findings.append(
                    Finding(
                        check=COUNT,
                        severity=FATAL,
                        reason=(
                            f"index holds {actual} record(s) but "
                            f"{expected_count} were expected -- a difference "
                            f"of {abs(actual - expected_count)}"
                        ),
                    )
                )

        run.append(RELATIONSHIP)
        findings.extend(self._check_relationships())

        retrieved: dict[str, ScoredDocument] = {}
        self._smoke_and_collect(findings, run, retrieved)
        self._check_presence(findings, run)
        self._check_embedding(findings, run)
        self._check_metadata(findings, run, retrieved)

        return VerificationVerdict(
            findings=tuple(findings),
            expected_records=expected_count,
            actual_records=actual if actual is not None else expected_count,
            checks_run=tuple(dict.fromkeys(run)),
        )

    def _count(self, findings: list[Finding]) -> int | None:
        try:
            return int(self.store.count())
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            findings.append(
                Finding(
                    check=COUNT,
                    severity=FATAL,
                    reason=f"the index could not report a count: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return None

    def _check_relationships(self) -> list[Finding]:
        """document -> chunk -> source, on the side we can actually inspect."""
        findings: list[Finding] = []
        seen: set[str] = set()
        for item in self.expected:
            chunk = item.chunk
            record = chunk.chunk_id
            if not chunk.document_id:
                findings.append(
                    Finding(
                        check=RELATIONSHIP,
                        severity=FATAL,
                        record=record,
                        reason="chunk carries no document_id, so it cannot be "
                        "traced back to the document it came from",
                    )
                )
            elif not chunk.source_name:
                findings.append(
                    Finding(
                        check=RELATIONSHIP,
                        severity=WARNING,
                        record=record,
                        reason=f"chunk belongs to document "
                        f"{chunk.document_id!r} but names no source file, so "
                        f"an answer citing it cannot point at a file",
                    )
                )
            if chunk.page_start is None and chunk.page_end is None and not _section(chunk):
                findings.append(
                    Finding(
                        check=RELATIONSHIP,
                        severity=WARNING,
                        record=record,
                        reason="chunk has neither a page range nor a section, "
                        "so a reader is told which document but not where in "
                        "it",
                    )
                )
            if record is not None and record in seen:
                findings.append(
                    Finding(
                        check=RELATIONSHIP,
                        severity=FATAL,
                        record=record,
                        reason="two expected records share this chunk_id; an "
                        "upsert would collapse them, so the expectation "
                        "cannot be verified as stated",
                    )
                )
            elif record is not None:
                seen.add(record)
        return findings

    def _smoke_and_collect(
        self,
        findings: list[Finding],
        run: list[str],
        retrieved: dict[str, ScoredDocument],
    ) -> None:
        """The FR-50 retrieval smoke check, doubling as the read pass.

        One query answers three questions at once: is the index queryable at
        the expected width (``dimension``), does it return anything
        (``retrieval``), and what do those rows actually contain.
        """
        if self._probe is None:
            findings.append(
                Finding(
                    check=RETRIEVAL,
                    severity=FATAL,
                    reason="no probe vector is available, so no retrieval "
                    "could be attempted. Pass probe=... when the expectation "
                    "is empty; an index nothing was queried has not been "
                    "verified.",
                )
            )
            return

        try:
            hits = self.store.query(self._probe, k=max(self.probe_k, 1))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            findings.append(
                Finding(
                    check=DIMENSION,
                    severity=FATAL,
                    record=self._probe_record(),
                    reason=f"the index refused a {self.dimensions}-wide probe "
                    f"vector: {type(exc).__name__}: {exc}. The vectors it "
                    f"holds are not the width this verifier expects.",
                )
            )
            return

        run.extend((DIMENSION, RETRIEVAL))

        if not hits:
            findings.append(
                Finding(
                    check=RETRIEVAL,
                    severity=FATAL,
                    record=self._probe_record(),
                    reason=f"a {self.dimensions}-wide probe returned no rows "
                    f"at all, so this index cannot answer a query even though "
                    f"it reports {self._safe_count()} record(s)",
                )
            )
            return

        for hit in hits:
            if hit.chunk_id is not None:
                retrieved[hit.chunk_id] = hit

        returned = {hit.chunk_id for hit in hits}
        expected_ids = {item.chunk.chunk_id for item in self.expected}
        expected_documents = {
            item.chunk.document_id for item in self.expected if item.chunk.document_id
        }

        probe_id = self._probe_record()
        if probe_id is not None and probe_id not in returned:
            findings.append(
                Finding(
                    check=RETRIEVAL,
                    severity=FATAL,
                    record=probe_id,
                    reason="the record the probe was taken from did not come "
                    "back among the results, so this index cannot retrieve "
                    "the content it was given",
                )
            )

        for hit in hits:
            if hit.document_id and hit.document_id not in expected_documents:
                findings.append(
                    Finding(
                        check=RELATIONSHIP,
                        severity=FATAL,
                        record=hit.chunk_id,
                        reason=f"the index returned a record from document "
                        f"{hit.document_id!r}, which is not in the "
                        f"expectation",
                    )
                )
            if hit.chunk_id is not None and hit.chunk_id not in expected_ids:
                findings.append(
                    Finding(
                        check=PRESENCE,
                        severity=FATAL,
                        record=hit.chunk_id,
                        reason="this record is in the index but was never "
                        "expected -- a stale row that will be retrieved and "
                        "cannot be accounted for",
                    )
                )

    def _check_presence(self, findings: list[Finding], run: list[str]) -> None:
        run.append(PRESENCE)
        if not self.expected:
            return

        by_document: dict[str, list[EmbeddedChunk]] = {}
        for item in self.expected:
            by_document.setdefault(item.chunk.document_id or "", []).append(item)

        for document, items in by_document.items():
            present, reasons = self._present(document, items)
            for item in items:
                chunk_id = item.chunk.chunk_id
                if chunk_id is None or chunk_id in present:
                    continue
                reason = reasons.get(chunk_id) or (
                    f"a filtered query for document {document!r} returned no "
                    f"such record"
                )
                findings.append(
                    Finding(
                        check=PRESENCE,
                        severity=FATAL,
                        record=chunk_id,
                        reason=f"expected record is not in the index: "
                        f"{reason}. Either it was never written or it was "
                        f"removed behind the pipeline's back",
                    )
                )

    def _present(
        self, document: str, items: Sequence[EmbeddedChunk]
    ) -> tuple[set[str], dict[str, str]]:
        """Which expected ids are demonstrably stored, and why not the rest.

        A readback is stronger evidence than a query, so when one exists it
        is the only thing consulted -- falling back to a query after a direct
        read said "absent" would be second-guessing the better witness.
        """
        present: set[str] = set()
        if self._fetch is not None:
            reasons: dict[str, str] = {}
            for item in items:
                chunk_id = item.chunk.chunk_id
                if chunk_id is None:
                    continue
                try:
                    stored = self._fetch(chunk_id)
                except Exception as exc:  # noqa: BLE001 - reported as a reason
                    reasons[chunk_id] = (
                        f"the store's readback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if stored is not None:
                    present.add(chunk_id)
                else:
                    reasons[chunk_id] = "the store's readback returned nothing " \
                        f"for this id"
            return present, reasons

        if self._probe is None:
            return present, {}
        try:
            hits = self.store.query(
                self._probe,
                k=max(self.probe_k, len(items)),
                filter={"document_id": document} if document else None,
            )
        except Exception as exc:  # noqa: BLE001 - reported as a reason
            return present, {
                item.chunk.chunk_id: (
                    f"the filtered query failed: {type(exc).__name__}: {exc}"
                )
                for item in items
                if item.chunk.chunk_id is not None
            }
        present.update(
            hit.chunk_id for hit in hits if hit.chunk_id is not None
        )
        return present, {}

    def _check_embedding(self, findings: list[Finding], run: list[str]) -> None:
        run.append(EMBEDDING)
        if self._fetch is None:
            # A record the vector query returned demonstrably has an
            # embedding -- that is how it was found. There is nothing further
            # to claim here, and nothing to invent.
            return
        for item in self.expected:
            chunk_id = item.chunk.chunk_id
            if chunk_id is None:
                continue
            try:
                stored = self._fetch(chunk_id)
            except Exception as exc:  # noqa: BLE001 - reported as a finding
                findings.append(
                    Finding(
                        check=EMBEDDING,
                        severity=FATAL,
                        record=chunk_id,
                        reason=f"the stored vector could not be read: "
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if stored is None:
                continue  # absence belongs to presence; do not double-count
            if len(stored) == 0:
                findings.append(
                    Finding(
                        check=EMBEDDING,
                        severity=FATAL,
                        record=chunk_id,
                        reason="the stored record has no embedding at all",
                    )
                )
                continue
            if len(stored) != self.dimensions:
                findings.append(
                    Finding(
                        check=DIMENSION,
                        severity=FATAL,
                        record=chunk_id,
                        reason=f"the stored vector is {len(stored)}-wide but "
                        f"the index is built for {self.dimensions}",
                    )
                )
                continue
            bad = _first_unusable(stored)
            if bad is not None:
                findings.append(
                    Finding(
                        check=EMBEDDING,
                        severity=FATAL,
                        record=chunk_id,
                        reason=f"the stored vector is not usable at position "
                        f"{bad[0]}: {bad[1]!r}",
                    )
                )

    def _check_metadata(
        self,
        findings: list[Finding],
        run: list[str],
        retrieved: dict[str, ScoredDocument],
    ) -> None:
        run.append(METADATA)
        for record, hit in retrieved.items():
            metadata = hit.metadata
            if not isinstance(metadata, Mapping):
                findings.append(
                    Finding(
                        check=METADATA,
                        severity=FATAL,
                        record=record,
                        reason=f"stored metadata is {type(metadata).__name__}, "
                        f"not a mapping",
                    )
                )
                continue
            leaked = [k for k in metadata if str(k).startswith(RESERVED_PREFIX)]
            if leaked:
                findings.append(
                    Finding(
                        check=METADATA,
                        severity=WARNING,
                        record=record,
                        reason=f"reserved key(s) "
                        f"{', '.join(map(repr, leaked))} are visible to the "
                        f"caller",
                    )
                )
            for key in self.required_metadata:
                if key not in metadata:
                    findings.append(
                        Finding(
                            check=METADATA,
                            severity=FATAL,
                            record=record,
                            reason=f"required metadata key {key!r} is absent "
                            f"from the stored record",
                        )
                    )
                elif metadata[key] is None:
                    findings.append(
                        Finding(
                            check=METADATA,
                            severity=WARNING,
                            record=record,
                            reason=f"required metadata key {key!r} is present "
                            f"but empty",
                        )
                    )

    # -- small helpers ----------------------------------------------------- #

    def _probe_record(self) -> str | None:
        if not self.expected:
            return None
        return self.expected[0].chunk.chunk_id

    def _safe_count(self) -> int | None:
        try:
            return int(self.store.count())
        except Exception:  # noqa: BLE001 - only used in a message
            return None


def _section(chunk: Any) -> str | None:
    """The chunk's section reference, if it has a non-blank one."""
    metadata = getattr(chunk, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("section")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _first_unusable(vector: Sequence[float]) -> tuple[int, Any] | None:
    """The first position that is not a finite number, if any."""
    for position, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return position, value
        if not math.isfinite(float(value)):
            return position, value
    return None

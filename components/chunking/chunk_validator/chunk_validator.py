"""Decide which chunks may reach the embedder, and say why for the rest.

The whole point
---------------
Chunking produces candidates. Something has to decide which of them are safe
to spend money and compute on, because a whitespace-only chunk, a chunk with no
document behind it, or a chunk four times the model's context window will not
fail loudly at the point of embedding -- they will fail expensively, or worse,
succeed and quietly poison the index.

This component is the gate. It returns **one result per chunk, always**, with
a rule id, a severity and a reason, and it guarantees the property the
orchestration layer actually depends on:

    Nothing reaches ``outcome.accepted`` carrying an unaddressed fatal finding.

"Unaddressed" matters. A fatal finding can be *repaired* when a repair is
deterministic and invents nothing -- deriving a missing ``chunk_id`` from the
chunk's own content and position is a derivation, not a fabrication. Repairing
a missing page range by writing ``page 1`` would be a fabrication, so it is
not offered: the chunk is held instead, with the reason recorded.

Warning is not fatal
--------------------
Not every finding should cost a chunk. A chunk that is merely short, or that
carries no page range, is still perfectly retrievable -- it just cannot cite a
page. Those are warnings by default, and a warning never removes a chunk from
``accepted``. Severities are configurable per rule, so a caller who does want
short chunks refused can say so and be recorded as having said so.

Held, never dropped
-------------------
v2.md Step 9 is explicit that useful chunks must not be silently discarded.
So a refused chunk is always returned -- in ``rejected`` or ``quarantined``,
depending on the configured policy -- paired with the findings that refused
it. There is no code path in this component that returns fewer results than
it was given.

Duck typing, by design
----------------------
Chunks are read by attribute, not by import: ``text``, ``document_id``,
``chunk_id``, ``source_name``, ``chunk_index``, ``page_start``, ``page_end``,
``metadata``. That is the same structural contract ``core-types`` documents,
and it is why this module imports neither ``rag_code_library`` nor any
chunker. Both ``core_types.Chunk`` and a chunker's own local ``Chunk`` --
whose identity fields default to ``None`` -- are validated as they are.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum

__all__ = [
    "DEFAULT_ANOMALY_MARKERS",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MIN_CHARS",
    "ChunkValidation",
    "ChunkValidator",
    "FatalPolicy",
    "Finding",
    "Rule",
    "Severity",
    "UnrepairableChunk",
    "ValidationError",
    "ValidationOutcome",
    "ValidationResult",
]

DEFAULT_MIN_CHARS = 1
DEFAULT_MAX_CHARS = 8_000

#: Metadata keys whose presence signals that extraction went wrong somewhere.
#: Explicit rather than heuristic: guessing that a key "looks like a warning"
#: would make the validator's behaviour depend on naming conventions nobody
#: wrote down.
DEFAULT_ANOMALY_MARKERS: tuple[str, ...] = (
    "extraction_warning",
    "extraction_failed",
    "ocr_failed",
)


class ValidationError(Exception):
    """Base for every error this component raises by contract."""


class UnrepairableChunk(ValidationError):
    """A chunk could not be validated at all, because its shape is unusable."""


class Severity(str, Enum):
    WARNING = "warning"
    FATAL = "fatal"


class Rule(str, Enum):
    """Every rule this validator applies. Stable ids, safe to key config on."""

    EMPTY_TEXT = "empty-text"
    WHITESPACE_ONLY_TEXT = "whitespace-only-text"
    TEXT_TOO_SHORT = "text-too-short"
    TEXT_TOO_LONG = "text-too-long"
    MISSING_DOCUMENT_ID = "missing-document-id"
    MISSING_CHUNK_ID = "missing-chunk-id"
    INVALID_CHUNK_ID = "invalid-chunk-id"
    MISSING_SOURCE_LOCATION = "missing-source-location"
    INVALID_PAGE_RANGE = "invalid-page-range"
    MISSING_REQUIRED_METADATA = "missing-required-metadata"
    EXTRACTION_ANOMALY = "extraction-anomaly"


#: What each rule costs by default. Overridable per rule at construction.
DEFAULT_SEVERITIES: Mapping[Rule, Severity] = {
    Rule.EMPTY_TEXT: Severity.FATAL,
    Rule.WHITESPACE_ONLY_TEXT: Severity.FATAL,
    Rule.TEXT_TOO_SHORT: Severity.WARNING,
    Rule.TEXT_TOO_LONG: Severity.FATAL,
    Rule.MISSING_DOCUMENT_ID: Severity.FATAL,
    Rule.MISSING_CHUNK_ID: Severity.FATAL,
    Rule.INVALID_CHUNK_ID: Severity.FATAL,
    Rule.MISSING_SOURCE_LOCATION: Severity.WARNING,
    Rule.INVALID_PAGE_RANGE: Severity.FATAL,
    Rule.MISSING_REQUIRED_METADATA: Severity.FATAL,
    Rule.EXTRACTION_ANOMALY: Severity.WARNING,
}


class FatalPolicy(str, Enum):
    """What to do with a chunk that still carries a fatal finding."""

    #: Hold it aside with its findings. The default: nothing is lost.
    QUARANTINE = "quarantine"
    #: Drop it from the accepted set, still returned with its findings.
    REJECT = "reject"
    #: Try a deterministic repair first; hold it if that is not possible.
    REPAIR = "repair"


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule, one severity, one sentence explaining it to a human."""

    rule: Rule
    severity: Severity
    message: str

    @property
    def is_fatal(self) -> bool:
        return self.severity is Severity.FATAL


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Everything found about one chunk. One per chunk in, one per chunk out."""

    #: Position in the input sequence, so a result can always be traced back.
    index: int
    #: Identifier of the chunk, when it has one. ``None`` is itself a finding.
    chunk_id: str | None
    findings: tuple[Finding, ...]

    @property
    def is_fatal(self) -> bool:
        return any(finding.is_fatal for finding in self.findings)

    @property
    def is_valid(self) -> bool:
        return not self.findings

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(finding.rule for finding in self.findings)


@dataclass(frozen=True, slots=True)
class ChunkValidation:
    """A chunk held with the findings that held it. Never discarded silently."""

    index: int
    chunk: object
    findings: tuple[Finding, ...]
    #: ``True`` when a repair was attempted and failed.
    repair_failed: bool = False


@dataclass(frozen=True, slots=True)
class RepairRecord:
    """An audit line: a repair was applied, and it was this one."""

    index: int
    rule: Rule
    description: str


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """The full result of a validation run."""

    #: Chunks safe to embed: no unaddressed fatal finding. This is the set
    #: the orchestration layer is allowed to continue with.
    accepted: tuple[object, ...]
    #: Chunks held or dropped, each with the findings that caused it.
    rejected: tuple[ChunkValidation, ...]
    quarantined: tuple[ChunkValidation, ...]
    repairs: tuple[RepairRecord, ...]
    results: tuple[ValidationResult, ...]

    @property
    def fatal_count(self) -> int:
        return sum(1 for result in self.results if result.is_fatal)

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for result in self.results
            if any(not finding.is_fatal for finding in result.findings)
        )

    @property
    def is_clean(self) -> bool:
        """True when no chunk carried any finding at all."""
        return all(result.is_valid for result in self.results)

    def held(self) -> tuple[ChunkValidation, ...]:
        """Every chunk that did not make it through, regardless of policy."""
        return (*self.quarantined, *self.rejected)


def _present(value: object) -> bool:
    """A value counts as missing when absent, ``None``, or blank text."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


class ChunkValidator:
    """Validate chunks against configurable rules and a fatal-outcome policy.

        ChunkValidator(min_chars=20, max_chars=2000).validate(chunks)
    """

    def __init__(
        self,
        *,
        min_chars: int = DEFAULT_MIN_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
        required_metadata: Iterable[str] = (),
        severities: Mapping[Rule, Severity] | None = None,
        fatal_policy: FatalPolicy = FatalPolicy.QUARANTINE,
        anomaly_markers: Iterable[str] = DEFAULT_ANOMALY_MARKERS,
        size_of: Callable[[str], int] | None = None,
    ) -> None:
        for name, value in (("min_chars", min_chars), ("max_chars", max_chars)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(f"{name} must be a non-negative int, got {value!r}")
        if min_chars > max_chars:
            raise ValidationError(
                f"min_chars ({min_chars}) cannot exceed max_chars ({max_chars})"
            )
        if not isinstance(fatal_policy, FatalPolicy):
            raise ValidationError(
                f"fatal_policy must be a FatalPolicy, got {fatal_policy!r}"
            )
        for key in required_metadata:
            if not isinstance(key, str) or not key.strip():
                raise ValidationError(
                    f"required_metadata must contain non-empty strings, got {key!r}"
                )
        if size_of is not None and not callable(size_of):
            raise ValidationError("size_of must be callable, taking str -> int")

        self._min = min_chars
        self._max = max_chars
        self._required = tuple(required_metadata)
        self._policy = fatal_policy
        self._anomalies = tuple(anomaly_markers)
        # A caller-supplied size function replaces the char count entirely, so
        # a token-aware caller is never silently given a character count.
        self._size_of: Callable[[str], int] = size_of or len
        self._severities: dict[Rule, Severity] = dict(DEFAULT_SEVERITIES)
        for rule, severity in (severities or {}).items():
            rule = Rule(rule)
            if not isinstance(severity, Severity):
                raise ValidationError(
                    f"severity for {rule.value} must be a Severity, got {severity!r}"
                )
            self._severities[rule] = severity

    @property
    def fatal_policy(self) -> FatalPolicy:
        return self._policy

    @property
    def severities(self) -> Mapping[Rule, Severity]:
        """The effective severities, after any overrides."""
        return dict(self._severities)

    def _severity(self, rule: Rule) -> Severity:
        return self._severities[rule]

    def _finding(self, rule: Rule, message: str) -> Finding:
        return Finding(rule=rule, severity=self._severity(rule), message=message)

    def inspect(self, chunk: object, *, index: int = 0) -> ValidationResult:
        """Find everything wrong with one chunk. Never raises for a bad chunk.

        A chunk missing the attributes this validator reads is a caller error
        rather than a validation finding, so it raises
        :class:`UnrepairableChunk` naming the attribute.
        """
        for attribute in ("text", "metadata"):
            if not hasattr(chunk, attribute):
                raise UnrepairableChunk(
                    f"chunk at index {index} has no {attribute!r} attribute; "
                    f"got {type(chunk).__name__}. This validator reads chunks "
                    "by attribute, matching the core-types contract."
                )

        findings: list[Finding] = []
        text = getattr(chunk, "text", None)
        if not isinstance(text, str):
            findings.append(
                self._finding(
                    Rule.EMPTY_TEXT,
                    f"text is {type(text).__name__}, not str; there is nothing to index",
                )
            )
            text = ""
        elif not text:
            findings.append(self._finding(Rule.EMPTY_TEXT, "text is empty"))
        elif not text.strip():
            findings.append(
                self._finding(
                    Rule.WHITESPACE_ONLY_TEXT,
                    "text is whitespace only; it carries no retrievable content",
                )
            )
        else:
            size = self._size_of(text)
            if size < self._min:
                findings.append(
                    self._finding(
                        Rule.TEXT_TOO_SHORT,
                        f"text measures {size}, below the minimum of {self._min}",
                    )
                )
            if size > self._max:
                findings.append(
                    self._finding(
                        Rule.TEXT_TOO_LONG,
                        f"text measures {size}, above the maximum of {self._max}",
                    )
                )

        document_id = getattr(chunk, "document_id", None)
        if not _present(document_id):
            findings.append(
                self._finding(
                    Rule.MISSING_DOCUMENT_ID,
                    "document_id is missing, so this chunk cannot be traced to a source",
                )
            )

        chunk_id = getattr(chunk, "chunk_id", None)
        if not _present(chunk_id):
            findings.append(
                self._finding(
                    Rule.MISSING_CHUNK_ID,
                    "chunk_id is missing; it can be derived deterministically",
                )
            )
        elif not isinstance(chunk_id, str):
            findings.append(
                self._finding(
                    Rule.INVALID_CHUNK_ID,
                    f"chunk_id is {type(chunk_id).__name__}, not str",
                )
            )

        findings.extend(self._check_location(chunk, index))

        metadata = getattr(chunk, "metadata", None)
        metadata = metadata if isinstance(metadata, Mapping) else {}
        for key in self._required:
            if key not in metadata or not _present(metadata[key]):
                findings.append(
                    self._finding(
                        Rule.MISSING_REQUIRED_METADATA,
                        f"required metadata key {key!r} is absent or empty",
                    )
                )
        for marker in self._anomalies:
            if _present(metadata.get(marker)):
                findings.append(
                    self._finding(
                        Rule.EXTRACTION_ANOMALY,
                        f"metadata records an extraction problem ({marker}: "
                        f"{metadata[marker]!r}); the text may be incomplete",
                    )
                )

        return ValidationResult(
            index=index,
            # A blank string is not an identifier the chunk "has", so it is
            # recorded as None -- matching the MISSING_CHUNK_ID finding rather
            # than contradicting it in the same record.
            chunk_id=(
                chunk_id if isinstance(chunk_id, str) and chunk_id.strip() else None
            ),
            findings=tuple(findings),
        )

    def _check_location(self, chunk: object, index: int) -> list[Finding]:
        """Page range: absent is a warning, contradictory is fatal."""
        findings: list[Finding] = []
        page_start = getattr(chunk, "page_start", None)
        page_end = getattr(chunk, "page_end", None)

        if page_start is None or page_end is None:
            findings.append(
                self._finding(
                    Rule.MISSING_SOURCE_LOCATION,
                    "page range is absent, so this chunk cannot cite a location",
                )
            )
            return findings

        if not isinstance(page_start, int) or not isinstance(page_end, int):
            findings.append(
                self._finding(
                    Rule.INVALID_PAGE_RANGE,
                    f"page range is not integral ({type(page_start).__name__}, "
                    f"{type(page_end).__name__})",
                )
            )
            return findings

        if page_start < 1 or page_end < 1:
            findings.append(
                self._finding(
                    Rule.INVALID_PAGE_RANGE,
                    f"page range starts below 1 ({page_start}-{page_end}); "
                    "pages are counted from 1",
                )
            )
        elif page_end < page_start:
            findings.append(
                self._finding(
                    Rule.INVALID_PAGE_RANGE,
                    f"page range is inverted ({page_start}-{page_end}); the "
                    "two values can be swapped",
                )
            )
        return findings

    # ------------------------------------------------------------------
    # Repairs
    # ------------------------------------------------------------------

    def _repair(
        self, chunk: object, rule: Rule, result: ValidationResult
    ) -> tuple[object, RepairRecord] | None:
        """Apply one deterministic repair, returning ``(repaired, record)``.

        A repair must be derivable from the chunk itself. Anything that would
        invent a fact -- a page number, a document id -- is deliberately not
        repairable, and the chunk is held instead.

        A frozen chunk is never mutated: ``dataclasses.replace`` produces a new
        value so the caller's original stays exactly as it was. Only a plainly
        mutable object is edited in place.
        """
        if rule is Rule.MISSING_CHUNK_ID or rule is Rule.INVALID_CHUNK_ID:
            text = getattr(chunk, "text", "") or ""
            document_id = getattr(chunk, "document_id", "") or ""
            chunk_index = getattr(chunk, "chunk_index", 0)
            digest = _digest(f"{document_id}\x1f{chunk_index}\x1f{text}")
            identifier = f"{document_id or 'unknown'}-{chunk_index}-{digest[:12]}"
            repaired = _with(chunk, chunk_id=identifier)
            if repaired is None:
                return None
            return repaired, RepairRecord(
                index=result.index,
                rule=rule,
                description=f"derived chunk_id {identifier!r} from document, index and text",
            )

        if rule is Rule.INVALID_PAGE_RANGE:
            start = getattr(chunk, "page_start", None)
            end = getattr(chunk, "page_end", None)
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and start >= 1
                and end >= 1
                and end < start
            ):
                repaired = _with(chunk, page_start=end, page_end=start)
                if repaired is None:
                    return None
                return repaired, RepairRecord(
                    index=result.index,
                    rule=rule,
                    description=f"swapped the inverted page range to {end}-{start}",
                )
        return None

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    def validate(self, chunks: Sequence[object]) -> ValidationOutcome:
        """Validate a batch and partition it by the configured policy.

        Returns exactly one :class:`ValidationResult` per chunk given, in
        order. No chunk is dropped without appearing in ``rejected`` or
        ``quarantined`` alongside the findings that caused it.
        """
        accepted: list[object] = []
        rejected: list[ChunkValidation] = []
        quarantined: list[ChunkValidation] = []
        repairs: list[RepairRecord] = []
        results: list[ValidationResult] = []

        for index, chunk in enumerate(chunks):
            result = self.inspect(chunk, index=index)
            results.append(result)

            if not result.is_fatal:
                accepted.append(chunk)
                continue

            repaired, hold = self._apply_policy(chunk, result, repairs)
            if repaired is not None:
                accepted.append(repaired)
            elif self._policy is FatalPolicy.REJECT:
                rejected.append(hold)
            else:
                quarantined.append(hold)

        return ValidationOutcome(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            quarantined=tuple(quarantined),
            repairs=tuple(repairs),
            results=tuple(results),
        )

    def _apply_policy(
        self,
        chunk: object,
        result: ValidationResult,
        repairs: list[RepairRecord],
    ) -> tuple[object, ChunkValidation | None]:
        """Return ``(chunk to accept, hold record)``; at most one is non-None."""
        if self._policy is not FatalPolicy.REPAIR:
            return None, ChunkValidation(
                index=result.index, chunk=chunk, findings=result.findings
            )

        current: object = chunk
        remaining: list[Finding] = list(result.findings)
        for finding in tuple(remaining):
            if not finding.is_fatal:
                continue
            outcome = self._repair(current, finding.rule, result)
            if outcome is None:
                continue
            current, record = outcome
            repairs.append(record)
            remaining = [item for item in remaining if item.rule is not finding.rule]

        if any(item.is_fatal for item in remaining):
            return None, ChunkValidation(
                index=result.index,
                chunk=current,
                findings=tuple(remaining),
                repair_failed=True,
            )
        return current, None


def _with(chunk: object, **changes: object) -> object | None:
    """Return a chunk carrying ``changes``, or ``None`` if that is impossible.

    A dataclass is copied with :func:`dataclasses.replace`, so a frozen chunk
    stays frozen and the caller's original is untouched. Anything else is
    mutated in place, which is the only option a non-dataclass leaves.
    """
    if is_dataclass(chunk) and not isinstance(chunk, type):
        try:
            return replace(chunk, **changes)  # type: ignore[type-var]
        except (TypeError, ValueError):
            return None
    try:
        for name, value in changes.items():
            setattr(chunk, name, value)
    except Exception:  # noqa: BLE001 - a slots object may simply refuse
        return None
    return chunk


def _digest(text: str) -> str:
    """Deterministic 64-hex digest, stdlib only."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

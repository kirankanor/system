"""A labelled evaluation dataset, and the assertion that a change did not regress.

What this component is
----------------------
One of the three evaluation components (FR-9). It supplies the two things
FR-9 asks of evaluation: *a dataset format*, and *regression-test support*.

**The format.** An :class:`EvaluationCase` is one labelled question: what was
asked, what the retriever returned, what should have been returned, and
optionally the answer and the evidence it was given. An
:class:`EvaluationDataset` is a named, ordered collection of those, with a
versioned JSON representation.

**The regression support.** :func:`assert_no_regression` takes the scores from
two runs and raises, naming every metric that got worse, by how much, and
against what.

Where the numbers come from
---------------------------
This component deliberately computes **no metrics**. It holds labels and
compares scores; the scoring itself belongs to ``retrieval-metrics`` and
``answer-metrics``.

That is not an omission to apologise for. The three manifests all declare
``dependencies: []``, so each component copies out alone -- and if this one
imported the metric components, adding a dataset to a project would drag two
metric implementations in with it whether or not the caller wanted them.
Wiring is three lines and is shown in the README:

    dataset = EvaluationDataset.from_json(text)
    metrics = RetrievalMetrics(k=10)
    current = metrics.score_many(dataset.retrieval_examples())
    dataset.assert_no_regression(current=current, baseline=baseline)

Nothing here reaches a network, a model, a store, or the filesystem. There is
no ``to_file``: this component returns and accepts JSON *strings*, and where
they are written is the caller's decision. A library that writes files on
request has to answer where it is allowed to write, and that question belongs
to the application, not to a metrics helper.

The regression rule
-------------------
A regression is a metric that got **worse by more than ``tolerance``**.
``tolerance`` is absolute and in metric units, so a ``0.01`` tolerance on
``hit_rate`` means "a one-point drop is noise".

Three properties make this trustworthy rather than merely convenient:

- **Every baseline metric must still be reported.** If the baseline had
  ``hit_rate``, ``mrr`` and ``recall`` and the new run reports only
  ``hit_rate``, this raises rather than comparing the one metric that
  happened to survive. Comparing a subset is precisely how a real regression
  hides: hit rate can hold steady while MRR collapses, and the subset
  comparison calls it a pass.

- **Metrics absent from the baseline are not regressions.** A new metric has
  no history to regress from, so it is reported but never fails the check.

- **Lower-is-better metrics are opt-in.** ``loss`` and ``latency`` improve as
  they fall, so they must be named, or the check would fail every improvement.

A regression raises :class:`RegressionError` carrying every offending metric,
not just the first, so one run tells you the whole story instead of a
whack-a-mole sequence of fixes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = [
    "FORMAT_VERSION",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationDatasetError",
    "EvaluationInputError",
    "Regression",
    "RegressionError",
]

#: Bumped only when the JSON shape changes incompatibly. A reader that sees a
#: version it does not know says so instead of guessing at the fields.
FORMAT_VERSION: Final[int] = 1


class EvaluationDatasetError(Exception):
    """Base for every failure in this component."""


class EvaluationInputError(EvaluationDatasetError):
    """The caller passed something this component cannot use."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One labelled example.

    ``retrieved``/``relevant`` feed retrieval metrics; ``answer``/``evidence``
    feed answer metrics. All four are optional independently, so a dataset can
    hold retrieval-only cases, answer-only cases, or both.
    """

    id: str
    query: str
    retrieved: tuple[str, ...] = ()
    relevant: tuple[str, ...] = ()
    answer: str | None = None
    evidence: tuple[str, ...] = ()

    @property
    def has_retrieval(self) -> bool:
        return bool(self.relevant)

    @property
    def has_answer(self) -> bool:
        return self.answer is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "query": self.query,
            "retrieved": list(self.retrieved),
            "relevant": list(self.relevant),
        }
        if self.answer is not None:
            data["answer"] = self.answer
            data["evidence"] = list(self.evidence)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, where: str = "case") -> EvaluationCase:
        if not isinstance(data, Mapping):
            raise EvaluationInputError(
                f"{where} must be a mapping, got {type(data).__name__}"
            )
        missing = [key for key in ("id", "query") if key not in data]
        if missing:
            raise EvaluationInputError(
                f"{where} is missing required field(s) {missing}. Every case "
                f"needs an 'id' (to name it in a failure) and a 'query'"
            )
        answer = data.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise EvaluationInputError(
                f"{where} field 'answer' must be a string or null, got "
                f"{type(answer).__name__}"
            )
        case = cls(
            id=_text(data["id"], f"{where} field 'id'"),
            query=_text(data["query"], f"{where} field 'query'"),
            retrieved=_ids(data.get("retrieved", ()), f"{where} field 'retrieved'"),
            relevant=_ids(data.get("relevant", ()), f"{where} field 'relevant'"),
            answer=answer,
            evidence=_ids(data.get("evidence", ()), f"{where} field 'evidence'"),
        )
        if not case.id:
            raise EvaluationInputError(f"{where} field 'id' must not be empty")
        return case


@dataclass(frozen=True, slots=True)
class Regression:
    """One metric that got worse, with enough context to act on it."""

    metric: str
    baseline: float
    current: float
    tolerance: float
    lower_is_better: bool = False

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def worsening(self) -> float:
        """How far past the tolerance this metric moved the wrong way.

        In metric units, and always ``> 0`` exactly when the check fails. A
        higher-is-better metric worsens by ``baseline - current``; a
        lower-is-better one worsens by ``current - baseline``. The tolerance
        is subtracted in both cases, because that is how much slack the
        caller granted.
        """
        if self.lower_is_better:
            return self.delta - self.tolerance
        return -self.delta - self.tolerance

    def __repr__(self) -> str:
        return (
            f"Regression(metric={self.metric!r}, baseline={self.baseline}, "
            f"current={self.current}, delta={self.delta})"
        )


class RegressionError(EvaluationDatasetError):
    """One or more metrics got worse by more than the tolerance.

    Carries every offending :class:`Regression`, so a single run reports the
    whole picture instead of one metric at a time.
    """

    def __init__(self, message: str, *, regressions: tuple[Regression, ...]) -> None:
        super().__init__(message)
        self.regressions = regressions


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """An ordered, named collection of labelled cases."""

    cases: tuple[EvaluationCase, ...] = field(default_factory=tuple)
    name: str = "dataset"

    def __post_init__(self) -> None:
        if isinstance(self.cases, (str, bytes)):
            raise EvaluationInputError(
                f"cases must be a sequence of EvaluationCase, got "
                f"{type(self.cases).__name__}"
            )
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise EvaluationInputError(
                f"cases must be a sequence of EvaluationCase, got "
                f"{type(self.cases).__name__}"
            ) from exc
        if not cases:
            raise EvaluationInputError(
                "the dataset is empty. A baseline computed over nothing scores "
                "0.0, and that would read as a measured result rather than an "
                "absence of one"
            )
        if not isinstance(self.name, str) or not self.name:
            raise EvaluationInputError(
                f"name must be a non-empty string, got {self.name!r}"
            )
        for position, case in enumerate(cases):
            if not isinstance(case, EvaluationCase):
                raise EvaluationInputError(
                    f"case {position} must be an EvaluationCase, got "
                    f"{type(case).__name__}"
                )
        seen: set[str] = set()
        for case in cases:
            if case.id in seen:
                raise EvaluationInputError(
                    f"duplicate case id {case.id!r}. A failure must name one "
                    f"example, and a repeated id makes that ambiguous"
                )
            seen.add(case.id)
        object.__setattr__(self, "cases", cases)

    # ------------------------------------------------------------------ #
    # Reading the cases
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self.cases)

    def __contains__(self, case_id: object) -> bool:
        return any(case.id == case_id for case in self.cases)

    def get(self, case_id: str) -> EvaluationCase:
        """Look one case up by id, or raise naming what is available."""
        for case in self.cases:
            if case.id == case_id:
                return case
        raise EvaluationInputError(
            f"no case with id {case_id!r} in dataset {self.name!r} "
            f"({len(self.cases)} cases)"
        )

    def retrieval_examples(
        self,
    ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        """The labelled cases as ``(retrieved, relevant)`` pairs.

        Ordered as the dataset is, and only the cases that carry a label --
        an answer-only case has nothing to retrieve against. Ready for
        ``RetrievalMetrics.score_many``.
        """
        return [
            (case.retrieved, case.relevant) for case in self.cases if case.has_retrieval
        ]

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #

    def assert_no_regression(
        self,
        *,
        current: Mapping[str, float],
        baseline: Mapping[str, float],
        tolerance: float = 0.0,
        lower_is_better: Iterable[str] = (),
    ) -> tuple[Regression, ...]:
        """Raise if any metric got worse by more than ``tolerance``.

        Returns the (empty) tuple of regressions when the check passes, so the
        same call works as a non-throwing check for a dashboard.
        """
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise EvaluationInputError(
                f"tolerance must be a number, got {type(tolerance).__name__}"
            )
        if tolerance < 0:
            raise EvaluationInputError(
                f"tolerance must not be negative, got {tolerance}. A negative "
                f"tolerance would demand an improvement before accepting a "
                f"tie, which is not what 'no regression' means"
            )
        for label, values in (("current", current), ("baseline", baseline)):
            if not isinstance(values, Mapping):
                raise EvaluationInputError(
                    f"{label} must be a mapping of metric name to value, got "
                    f"{type(values).__name__}"
                )
        lower = set(lower_is_better)

        missing = [metric for metric in baseline if metric not in current]
        if missing:
            raise EvaluationInputError(
                f"the current run does not report {sorted(missing)!r}, which "
                f"the baseline does. Comparing only the metrics that survived "
                f"is how a real regression hides -- hit rate can hold steady "
                f"while MRR collapses"
            )

        found: list[Regression] = []
        for metric, before in baseline.items():
            if not isinstance(before, (int, float)) or isinstance(before, bool):
                raise EvaluationInputError(
                    f"baseline metric {metric!r} must be a number, got "
                    f"{type(before).__name__}"
                )
            after = current[metric]
            if not isinstance(after, (int, float)) or isinstance(after, bool):
                raise EvaluationInputError(
                    f"current metric {metric!r} must be a number, got "
                    f"{type(after).__name__}"
                )
            is_lower_better = metric in lower
            delta = float(after) - float(before)
            # Positive when the metric moved the wrong way by more than the
            # tolerance. Sign flips for lower-is-better metrics; the tolerance
            # is subtracted either way.
            worsening = (delta if is_lower_better else -delta) - tolerance
            if worsening > 0:
                found.append(
                    Regression(
                        metric=metric,
                        baseline=float(before),
                        current=float(after),
                        tolerance=float(tolerance),
                        lower_is_better=is_lower_better,
                    )
                )

        if found:
            detail = "; ".join(
                f"{item.metric} {item.baseline} -> {item.current} "
                f"(delta {item.delta:+g}, tolerance {item.tolerance}, "
                f"past it by {item.worsening:g})"
                for item in found
            )
            raise RegressionError(
                f"dataset {self.name!r}: {len(found)} metric(s) regressed "
                f"beyond tolerance -- {detail}",
                regressions=tuple(found),
            )
        return ()

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": FORMAT_VERSION,
            "cases": [case.to_dict() for case in self.cases],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise. Returns a string -- this component writes no files."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationDataset:
        if not isinstance(data, Mapping):
            raise EvaluationInputError(
                f"a dataset must be a mapping, got {type(data).__name__}"
            )
        version = data.get("version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise EvaluationInputError(
                f"dataset format version {version!r} is not this build's "
                f"({FORMAT_VERSION}). It is being read rather than guessed at"
            )
        raw = data.get("cases")
        if raw is None:
            raise EvaluationInputError(
                "dataset has no 'cases' field. There is nothing to evaluate"
            )
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise EvaluationInputError(
                f"'cases' must be a sequence, got {type(raw).__name__}"
            )
        name = data.get("name", "dataset")
        if not isinstance(name, str) or not name:
            raise EvaluationInputError(
                f"dataset field 'name' must be a non-empty string, got {name!r}"
            )
        return cls(
            cases=tuple(
                EvaluationCase.from_dict(item, where=f"cases[{position}]")
                for position, item in enumerate(raw)
            ),
            name=name,
        )

    @classmethod
    def from_json(cls, text: str) -> EvaluationDataset:
        """Parse JSON. A malformed document is a typed error, not a traceback."""
        if not isinstance(text, str):
            raise EvaluationInputError(
                f"from_json expects a string, got {type(text).__name__}"
            )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluationInputError(
                f"dataset is not valid JSON: {exc.msg} at line {exc.lineno}, "
                f"column {exc.colno}"
            ) from exc
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"EvaluationDataset(name={self.name!r}, cases={len(self.cases)}, "
            f"retrieval={sum(1 for c in self.cases if c.has_retrieval)}, "
            f"answers={sum(1 for c in self.cases if c.has_answer)})"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise EvaluationInputError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    return value


def _ids(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise EvaluationInputError(
            f"{label} must be a sequence of ids, got {type(value).__name__}. "
            f"A string is iterable, so passing one would be read as a list "
            f"of single characters"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise EvaluationInputError(
            f"{label} must be a sequence of ids, got {type(value).__name__}"
        ) from exc
    for item in items:
        if not isinstance(item, str):
            raise EvaluationInputError(
                f"{label} must contain only id strings, got {item!r} "
                f"({type(item).__name__})"
            )
    return items

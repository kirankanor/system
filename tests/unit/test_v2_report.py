"""The V2 completion report must be complete and honest, or fail loudly.

``v2.md`` asks for twenty steps. ``SPEC.md`` AC-32 and NFR-18 require that all
twenty are classified as complete, partial, missing or blocked, each with file
and line evidence, and that no step is reported complete because a class, file
or placeholder exists.

This module is the enforcement of that. It is deliberately unforgiving:

* a step missing from the report fails,
* a step classified without evidence fails,
* a ``complete`` step that cites no tests fails -- "a file exists" is not
  evidence that the behaviour works,
* a summary table that disagrees with the per-step headings fails, so the
  headline can never drift into optimism,
* a tally that disagrees with the classifications fails,
* a step claimed complete while the report separately says it is not
  implemented fails,
* a shell command block in the report fails, because the documented-commands
  script executes every fenced command it finds and this file is a report
  rather than a how-to.

Nothing here inspects the implementation. It inspects the claims, which is the
only thing a report can be wrong about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REPORT = _ROOT / "docs" / "v2-report.md"

#: ``### 4. Document Loading -- partial``
_HEADING = re.compile(
    r"^### (?P<number>\d+)\. (?P<title>.+?) \u2014 "
    r"(?P<state>complete|partial|missing|blocked)\s*$",
    re.MULTILINE,
)
#: ``| 4 | Document Loading | partial |`` -- partial rows are bolded for
#: emphasis, which the pattern tolerates rather than special-casing.
_TABLE_ROW = re.compile(
    r"^\| (?P<number>\d+) \| (?P<title>.+?) \| \*{0,2}"
    r"(?P<state>complete|partial|missing|blocked)\*{0,2} \|$",
    re.MULTILINE,
)
#: ``- `components/.../thing.py` L293 -- ...``
_CITATION = re.compile(r"`(?P<path>[A-Za-z0-9_./\\-]+\.py)` L(?P<line>\d+)")
#: A tally line, e.g. ``**Tally: 10 complete, 10 partial, 0 missing, 0 blocked.**``
_TALLY = re.compile(
    r"Tally: (\d+) complete, (\d+) partial, (\d+) missing, (\d+) blocked"
)
_STATES = ("complete", "partial", "missing", "blocked")

#: Names the report must state as unimplemented. AC-32 asks for "every
#: unimplemented file format, service, backend and verification check"; these
#: are the ones a reader would otherwise assume were present.
_REQUIRED_ABSENCES = (
    "docx",
    "xlsx",
    "pptx",
    "rtf",
    "qdrant",
    "pgvector",
    "chroma",
    "faiss",
    "near-duplicate",
)


@pytest.fixture(scope="module")
def report() -> str:
    """The report under test, read once."""
    assert _REPORT.exists(), f"missing {_REPORT.relative_to(_ROOT)}"
    return _REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def steps(report: str) -> list[dict[str, str]]:
    """Every ``### N. Title -- state`` section, in document order."""
    return [m.groupdict() for m in _HEADING.finditer(report)]


def _section(report: str, step: dict[str, str]) -> str:
    """The body of one step's section, up to the next step heading."""
    start = report.index(f"### {step['number']}. {step['title']}")
    rest = report[start + 4 :]
    nxt = _HEADING.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _evidence_bullets(body: str) -> list[str]:
    """The evidence bullets of a step, stopping where the gaps begin.

    Counting bullets rather than citations is deliberate: a step may cite one
    file several times, and several bullets over one file are still several
    independent claims, which is what "several pieces of evidence" means.
    """
    evidence = body.split("\nGaps", 1)[0]
    return [line for line in evidence.splitlines() if line.startswith("- ")]


def test_the_report_classifies_exactly_twenty_steps(steps: list[dict[str, str]]) -> None:
    """All twenty steps, numbered 1 to 20, none repeated or skipped."""
    numbers = [int(step["number"]) for step in steps]
    assert numbers == list(range(1, 21)), (
        "v2.md defines exactly twenty steps; the report must classify all of "
        f"them, and only them. Found {numbers}"
    )
    titles = [step["title"] for step in steps]
    assert len(set(titles)) == len(titles), "a step title appears twice"


def test_every_step_is_classified_with_a_recognised_state(
    steps: list[dict[str, str]],
) -> None:
    """No free text stands in for a classification."""
    for step in steps:
        assert step["state"] in _STATES, step


def test_every_step_carries_file_and_line_evidence(
    report: str, steps: list[dict[str, str]]
) -> None:
    """AC-32: each step names a file and a line that actually exists."""
    for step in steps:
        citations = _CITATION.findall(_section(report, step))
        assert citations, (
            f"step {step['number']} ({step['title']}) carries no file and line "
            "evidence"
        )
        for path, line in citations:
            source = _ROOT / path.replace("\\", "/")
            assert source.exists(), (
                f"step {step['number']} cites `{path}`, which does not exist"
            )
            total = len(source.read_text(encoding="utf-8").splitlines())
            assert 1 <= int(line) <= total, (
                f"step {step['number']} cites `{path}` L{line}, but that file "
                f"has only {total} lines"
            )


def test_a_step_claimed_complete_must_cite_the_tests_that_prove_it(
    report: str, steps: list[dict[str, str]]
) -> None:
    """NFR-18: a file's existence is not evidence that its behaviour works."""
    for step in steps:
        if step["state"] != "complete":
            continue
        body = _section(report, step)
        bullets = _evidence_bullets(body)
        assert len(bullets) >= 3, (
            f"step {step['number']} ({step['title']}) is claimed complete on "
            f"{len(bullets)} piece(s) of evidence; a complete step must show "
            "several independent ones"
        )
        assert "test" in body.lower(), (
            f"step {step['number']} ({step['title']}) is claimed complete "
            "without naming any test. A class existing is not proof."
        )


def test_the_summary_table_agrees_with_the_per_step_headings(
    report: str, steps: list[dict[str, str]]
) -> None:
    """The headline cannot drift into optimism behind the detail."""
    rows = [m.groupdict() for m in _TABLE_ROW.finditer(report)]
    assert [int(row["number"]) for row in rows] == list(range(1, 21)), (
        "the summary table must list all twenty steps"
    )
    by_number = {int(step["number"]): step["state"] for step in steps}
    for row in rows:
        number = int(row["number"])
        assert row["state"] == by_number[number], (
            f"step {number} is summarised as {row['state']!r} but classified "
            f"{by_number[number]!r} in its own section"
        )


def test_the_stated_tally_matches_the_classifications(
    report: str, steps: list[dict[str, str]]
) -> None:
    """The count in prose must be the count on the page."""
    match = _TALLY.search(report)
    assert match, "the report must state a tally"
    claimed = [int(value) for value in match.groups()]
    actual = [sum(1 for s in steps if s["state"] == state) for state in _STATES]
    assert claimed == actual, f"tally says {claimed}, the page says {actual}"
    assert sum(claimed) == 20, "a tally over twenty steps must add up to twenty"


def test_no_complete_step_is_contradicted_by_the_absence_lists(
    report: str, steps: list[dict[str, str]]
) -> None:
    """A step cannot be complete while the report also calls it absent."""
    absences = report.split("## Not implemented", 1)
    assert len(absences) == 2, "the report must have a 'Not implemented' section"
    unimplemented = absences[1].lower()

    # A format that the loading step admits it cannot parse cannot leave that
    # step complete. If this ever fires, the step was upgraded past its proof.
    for step in steps:
        if step["state"] != "complete":
            continue
        body = _section(report, step).lower()
        for word in ("docx", "xlsx", "pptx", "rtf"):
            assert not re.search(rf"\b{word}\b[^.]*\bnot implemented\b", body), (
                f"step {step['number']} is complete but says {word} is not "
                "implemented"
            )
    assert unimplemented, "'Not implemented' is empty, which cannot be true"


def test_the_report_names_every_unimplemented_format_and_backend(report: str) -> None:
    """AC-32: name every unimplemented file format, service and backend."""
    tail = report.split("## Not implemented", 1)[1].lower()
    for name in _REQUIRED_ABSENCES:
        assert name in tail, (
            f"'Not implemented' does not mention {name!r}; AC-32 requires "
            "every unimplemented format, service and backend to be named"
        )


def test_the_report_contains_no_shell_command_block(report: str) -> None:
    """Pre-flight: the documented-commands gate executes ```console fences.

    ``docs/v2-report.md`` is a report, not a how-to. A fenced command here
    would be run by ``check_documented_commands.py`` against the live tree,
    which is not what a completion report is for.
    """
    assert "```console" not in report, (
        "a console block in the report would be executed by "
        "scripts/check_documented_commands.py"
    )


def test_the_deferred_steps_are_reported_on_their_merits(
    report: str, steps: list[dict[str, str]]
) -> None:
    """DECISION-1aedada9 deferred 16, 17 and 20; deferral is not completion."""
    by_number = {int(step["number"]): step for step in steps}
    for number in (16, 17, 20):
        step = by_number[number]
        assert step["state"] != "complete", (
            f"step {number} ({step['title']}) was deferred by "
            "DECISION-1aedada9 and must not be reported complete"
        )
    assert "DECISION-1aedada9" in report, (
        "the report must name the decision that deferred those steps"
    )

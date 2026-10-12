"""Prove the documentation by executing it (FR-26, FR-27, AC-16).

Every shell command the README and the docs pages show is extracted here,
run for real, and compared against the output those pages claim it produces.
A command that is documented but does not run is a documentation bug, and this
is what finds it.

The markdown convention
-----------------------
Documentation writes its commands as ``console`` blocks::

    ```console
    $ ragkit --version
    ragkit 0.1.0
    ```

* A line beginning ``$ `` is a command.
* Every other non-blank line is an **expected output substring**: the command
  is only considered correct if that text appears in what it actually printed.
* ``# exit:N`` on a command line declares the exit code the documentation
  promises. Without it, success (0) is assumed, because the common case should
  need no annotation.
* ``{scratch}`` is replaced with a fresh temporary directory per command, so a
  documented example that writes files cannot contaminate the repository or the
  next command.

Why substrings rather than exact matches: exact matching would make every
documentation edit fragile against incidental detail -- a temp path, a row
count, a version bump. Substrings still catch the failure that matters, which
is documentation claiming a command does something it no longer does.

Unrunnable commands
-------------------
A handful of things genuinely cannot be checked unattended -- a long-running
web server, an interactive browser, anything that mutates the developer's
environment. Those are not skipped silently: each has an entry in
:data:`MANUAL_CHECKS` naming the command and the reason it needs a human.
The check fails if a documented command is neither runnable nor declared, so a
new example cannot quietly escape verification.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = [_ROOT / "README.md", *sorted((_ROOT / "docs").glob("*.md"))]

#: How long one documented command may run before it is treated as hung.
_TIMEOUT_SECONDS = 120

#: Commands that cannot be verified unattended, with the reason a human is
#: required. Keys match the documented command text.
MANUAL_CHECKS: dict[str, str] = {
    "python -m http.server 8000": (
        "Long-running foreground server; it never exits, so there is no exit "
        "code to check. Verify by opening http://localhost:8000/cookbook/ in a "
        "browser -- this is also AC-15, which is owed as a human run."
    ),
    "uv sync": (
        "Mutates the developer's virtual environment and network state. Run it "
        "by hand when setting up; the suite runs against the working tree."
    ),
}

_FENCE = re.compile(r"^```console\s*$")
_EXPECTED_EXIT = re.compile(r"#\s*exit:(\d+)\s*$")


@dataclass
class Command:
    """One documented command, and what its documentation claims."""

    source: Path
    line_number: int
    text: str
    expected_exit: int
    #: Which console fence this command came from. Commands in one fence share
    #: a scratch destination, so a page can document a *sequence* -- add a
    #: component, then re-add it to show the overwrite safeguard -- rather than
    #: only isolated one-liners.
    block_index: int
    expected_output: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.source.relative_to(_ROOT)}:{self.line_number}"


@dataclass
class Result:
    """What happened when a documented command was actually run."""

    command: Command
    status: str  # "pass" | "fail" | "manual"
    detail: str = ""


def _is_manual(command_text: str) -> str | None:
    """The declared reason this command needs a human, if it is declared."""
    stripped = command_text.strip()
    for prefix, reason in MANUAL_CHECKS.items():
        if stripped == prefix or stripped.startswith(prefix):
            return reason
    return None


def _venv_path_env() -> dict[str, str]:
    """The developer environment, with the project virtualenv on PATH.

    Only ``.venv/Scripts`` (Windows) or ``.venv/bin`` (POSIX) is added. Nothing
    else is adjusted, so any machine-specific output still surfaces rather than
    being papered over -- if a documented command depends on something the
    developer's machine happens to have, this will not hide it.
    """
    env = dict(os.environ)
    candidates = [_ROOT / ".venv" / "Scripts", _ROOT / ".venv" / "bin"]
    for candidate in candidates:
        if candidate.is_dir():
            env["PATH"] = str(candidate) + os.pathsep + env.get("PATH", "")
            break
    return env


def parse(path: Path) -> list[Command]:
    """Pull every documented command out of one markdown file.

    Only ``console`` fences are treated as executable. A shell-tagged fence
    would be ambiguous about which lines are commands, and an ambiguous
    convention is one that silently stops being checked.
    """
    commands: list[Command] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    in_console = False
    current: Command | None = None
    block_index = -1

    for offset, raw in enumerate(lines, start=1):
        if raw.lstrip().startswith("```"):
            if not in_console:
                in_console = bool(_FENCE.match(raw.strip()))
                if in_console:
                    current = None
                    block_index += 1
                continue
            # Closing the fence ends the current command.
            in_console = False
            current = None
            continue

        if not in_console:
            continue

        stripped = raw.strip()
        if not stripped:
            current = None
            continue

        if stripped.startswith("$ "):
            body = stripped[2:].strip()
            exit_match = _EXPECTED_EXIT.search(body)
            expected_exit = int(exit_match.group(1)) if exit_match else 0
            if exit_match:
                body = body[: exit_match.start()].strip()
            current = Command(
                source=path,
                line_number=offset,
                text=body,
                expected_exit=expected_exit,
                block_index=block_index,
            )
            commands.append(current)
        elif current is not None:
            # An indented continuation of the command, not its output.
            if stripped.startswith("#"):
                continue
            current.expected_output.append(raw.rstrip())
        else:
            raise ValueError(
                f"{path}:{offset}: output line with no command above it. "
                "Output must follow a '$ ' line inside the same console block."
            )

    return commands


def run(command: Command, scratch: Path) -> Result:
    """Execute one documented command and compare against its promise."""
    reason = _is_manual(command.text)
    if reason is not None:
        return Result(command, "manual", reason)

    resolved = command.text.replace("{scratch}", str(scratch))
    try:
        completed = subprocess.run(
            resolved,
            cwd=_ROOT,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            # Explicitly False: the exit code is *what this check is about*, so
            # letting subprocess raise on a non-zero status would abort before
            # the documented exit code could be compared against the real one.
            check=False,
            # Keep the developer's own environment, but put the project's own
            # virtualenv on PATH. Documentation says `ragkit`, and a user who
            # has just run `uv sync` and activated the environment has exactly
            # that on their PATH. Without this the checker would fail every
            # command for a reason that is about the test harness rather than
            # the documentation -- which is the failure mode this script exists
            # to avoid confusing with a real one.
            env=_venv_path_env(),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return Result(
            command,
            "fail",
            f"timed out after {_TIMEOUT_SECONDS}s -- a documented example that "
            "hangs is worse than one that fails",
        )

    output = completed.stdout + completed.stderr

    if completed.returncode != command.expected_exit:
        return Result(
            command,
            "fail",
            f"exited {completed.returncode}, documentation says "
            f"{command.expected_exit}\n      output: {output.strip()[:400]}",
        )

    for expected in command.expected_output:
        if expected.strip() and expected.strip() not in output:
            return Result(
                command,
                "fail",
                f"documented output line is missing: {expected.strip()!r}\n"
                f"      actual: {output.strip()[:400]}",
            )

    return Result(command, "pass")


def main() -> int:
    if not _SOURCES[0].exists():
        print(f"missing {_SOURCES[0].relative_to(_ROOT)}")
        return 1

    missing_sources = [p for p in _SOURCES if not p.exists()]
    if missing_sources:
        for path in missing_sources:
            print(f"missing {path.relative_to(_ROOT)}")
        return 1

    commands: list[Command] = []
    for path in _SOURCES:
        commands.extend(parse(path))

    if not commands:
        print(
            "no documented commands were found. That is a failure, not a pass: "
            "FR-26 requires the README to document commands, and a docs tree "
            "with no runnable examples proves nothing."
        )
        return 1

    print(f"check_documented_commands: {len(commands)} command(s) across "
          f"{len(_SOURCES)} file(s)")

    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="ragkit-doccmds-") as tmp:
        base = Path(tmp)
        # One scratch destination per console fence, shared by that fence's
        # commands, so a page can document a sequence that depends on the step
        # before it. Separate fences stay separate so no page can contaminate
        # another.
        scratches: dict[tuple[Path, int], Path] = {}
        for command in commands:
            key = (command.source, command.block_index)
            scratch = scratches.get(key)
            if scratch is None:
                scratch = base / f"dest-{len(scratches):03d}"
                scratch.mkdir(parents=True, exist_ok=True)
                scratches[key] = scratch
            results.append(run(command, scratch))

    failures = [r for r in results if r.status == "fail"]
    manual = [r for r in results if r.status == "manual"]
    passed = [r for r in results if r.status == "pass"]

    for result in results:
        marker = {"pass": "ok  ", "fail": "FAIL", "manual": "note"}[result.status]
        print(f"  {marker} {result.command.label}: {result.command.text}")
        if result.status == "fail":
            print(f"        {result.detail}")
        elif result.status == "manual":
            print(f"        manual check required: {result.detail}")

    print(
        f"\n{len(passed)} passed, {len(manual)} manual, {len(failures)} failed"
    )

    if failures:
        print(
            "\nA documented command did not do what the documentation says. "
            "Fix the documentation or the command -- do not delete the check."
        )
        return 1

    # Guard against a documentation tree that quietly stops exercising
    # anything, which would turn this gate into a formality.
    if not passed and manual:
        print(
            "\nEvery documented command was declared unrunnable. At least one "
            "example must actually execute, or the docs prove nothing."
        )
        return 1

    print("OK: every documented command ran and matched what the docs claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

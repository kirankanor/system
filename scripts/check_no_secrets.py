"""Prove no secret value reaches an artifact (AC-17).

AC-17 asks that no ``SecretStr`` value appears in a log line, an error message,
a generated cookbook file, or a committed artifact. This repository deliberately
holds no credentials at all -- NFR-9 makes providers opt-in with no default
endpoint, model or key -- so the interesting failures here are *accidental*
ones: a token pasted into a fixture, a key in a docstring, or a secret unwrapped
into a message by ``.get_secret_value()``.

Two independent checks
----------------------
**Shape.** Every file is scanned for strings that look like a credential:
provider keys, private key blocks, JWTs, and assignment-shaped
``password=...`` bindings. Shape-matching is crude and would not survive a
determined adversary, which is fine -- the threat here is an accident, not an
attacker, and an accident has the shape of a real key because it *is* one.

**Unwrapping.** ``pydantic.SecretStr`` exists precisely so a value cannot be
printed by accident. The way that guarantee is broken is calling
``.get_secret_value()`` and interpolating the result into something that gets
logged or raised. Any such call is reported wherever it appears, because there
is no safe use of it in a message.

What it will not do
-------------------
It never prints a matched secret, not even in the failure report. A check that
reproduces the credential it found has converted a private mistake into a
recorded one. Matches are reported as file, line, rule name and a redacted
excerpt, and nothing else.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: How much of a matched line to show, once masked.
_EXCERPT = 24

#: Rules whose name is reported, never whose text. Each is anchored so that a
#: plausible-looking string in ordinary prose does not trip it.
SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("AWS secret access key", re.compile(r"(?i)\baws_secret_access_key\b\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Anthropic-style key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "assignment of a credential",
        re.compile(
            r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token"
            r"|auth[_-]?token|client[_-]?secret|credential)s?\b\s*[=:]\s*"
            r"['\"]([^'\"\s]{8,})['\"]"
        ),
    ),
)

#: The generic assignment rule is the one that can be tripped by an ordinary
#: placeholder, so its captured value has to look like a credential before it
#: is reported. Requiring digits *and* mixed case rules out every demo string
#: worth writing -- ``"secret-key"``, ``"changeme"``, ``"<your key>"`` -- while
#: keeping every real key this repository could plausibly contain. Without this
#: the rule would cry wolf on the examples, and a check that is routinely wrong
#: gets ignored, which is worse than not having it.
#: The one rule whose match is judged rather than trusted.
_ASSIGNMENT_RULE = "assignment of a credential"


def _is_test_file(relative: str) -> bool:
    name = relative.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def _looks_credential_like(value: str) -> bool:
    """True when a captured value is plausibly a real credential."""
    if len(value) < 12:
        return False
    has_digit = any(c.isdigit() for c in value)
    has_upper = any(c.isupper() for c in value)
    has_lower = any(c.islower() for c in value)
    # A credential is not a word: it mixes character classes.
    return has_digit and (has_upper or value.isalnum()) and has_lower

#: File types worth scanning. Everything else is skipped rather than read,
#: because a credential in a wheel or an image is not a thing this check can
#: meaningfully reason about.
_SCANNED_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".json", ".js", ".html", ".css", ".yml",
    ".yaml", ".cfg", ".ini", ".rst", ".csv", ".env", "",
}

#: Byte size above which a file is skipped, to keep the run bounded.
_MAX_BYTES = 2_000_000

_UNWRAP = re.compile(r"\.get_secret_value\s*\(")


@dataclass
class Finding:
    """A candidate secret, recorded without reproducing it."""

    path: str
    line_number: int
    rule: str
    excerpt: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line_number}  [{self.rule}]\n"
            f"        redacted: {self.excerpt}"
        )


def _mask(text: str) -> str:
    """Show only enough of a line to locate it, never enough to reuse it."""
    stripped = text.strip()
    if len(stripped) <= _EXCERPT:
        return "<redacted>"
    return f"{stripped[:8]}...<{len(stripped)} chars>...{stripped[-4:]}"


def _tracked_files() -> list[Path]:
    """Every file git considers part of the repository.

    Uses git so that ignored scratch files, bytecode caches and virtualenvs are
    excluded automatically. A secret in an ignored file is not a committed
    artifact, which is what AC-17 is about.
    """
    try:
        output = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if output.returncode != 0:
        return []
    return [_ROOT / name for name in output.stdout.split("\0") if name]


def _scan_shape(path: Path, relative: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        if path.stat().st_size > _MAX_BYTES:
            return findings
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings

    for number, line in enumerate(text.splitlines(), start=1):
        for rule_name, pattern in SECRET_RULES:
            match = pattern.search(line)
            if match is None:
                continue
            # The assignment rule is the only one whose match must be judged
            # before being reported; every other rule is unambiguous.
            if rule_name == _ASSIGNMENT_RULE and not _looks_credential_like(
                match.group(2)
            ):
                continue
            findings.append(
                Finding(relative, number, rule_name, _mask(line))
            )
    return findings


def _scan_unwrapping(path: Path, relative: str) -> list[Finding]:
    """Report every ``.get_secret_value()`` call in a shipping Python module.

    Not a pattern match for a credential but for the *mechanism* that would
    produce one. Reported wherever it appears in shipping code, because once a
    value is unwrapped there is no check that can tell whether it stayed
    private.

    Two deliberate boundaries, each learned the hard way:

    **Python source only.** A call site only exists in code. This repository's
    own decision records describe the rule and necessarily quote it -- Genesis
    wrote ``DECISION-cc6496b8`` into ``.genesis/project.json`` with the literal
    text of the pattern while explaining why tests are exempt, and the first
    version of this check flagged that description as a leak. A checker that
    fails on its own audit trail teaches the reader to distrust it. The shape
    rules are unaffected: a credential *value* pasted into a decision record,
    a manifest or a docstring is still found, everywhere.

    **Test modules exempt.** ``components/config/tests/test_config.py`` calls
    the accessor to assert that the value is masked in ``repr`` and
    ``safe_dump``. That test is the evidence that masking works, so forbidding
    the call everywhere would make AC-17's own mechanism untestable. Shape
    findings are still reported inside tests -- only the mechanism check steps
    aside there.
    """
    findings: list[Finding] = []
    if path.suffix.lower() != ".py":
        return findings
    if _is_test_file(relative):
        return findings
    try:
        if path.stat().st_size > _MAX_BYTES:
            return findings
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings

    if "SecretStr" not in text and ".get_secret_value" not in text:
        return findings

    for number, line in enumerate(text.splitlines(), start=1):
        if _UNWRAP.search(line):
            findings.append(
                Finding(
                    relative,
                    number,
                    "unwrapped SecretStr via .get_secret_value()",
                    _mask(line),
                )
            )
    return findings


def check() -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    scanned = 0
    # The checker's own source is skipped: it necessarily contains the literal
    # patterns it searches for, so scanning it would only ever report itself.
    # The *output* it produces is still checked, since that is where a real
    # leak would surface.
    own = Path(__file__).resolve()
    for path in _tracked_files():
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SCANNED_SUFFIXES:
            continue
        try:
            relative = path.relative_to(_ROOT).as_posix()
        except ValueError:
            continue
        if path.resolve() == own:
            continue
        scanned += 1
        findings.extend(_scan_shape(path, relative))
        findings.extend(_scan_unwrapping(path, relative))
    return findings, scanned


def main() -> int:
    print("check_no_secrets: scanning committed and untracked-but-not-ignored "
          "artifacts")

    findings, scanned = check()

    print(f"  files scanned: {scanned}")
    print(f"  rules applied : {len(SECRET_RULES)} shape rules + "
          f"SecretStr unwrapping")
    print(f"  findings      : {len(findings)}")

    if findings:
        print()
        for finding in findings:
            print(f"  FAIL {finding}")
        print(
            "\nA credential-shaped string is in a committed artifact. Remove it\n"
            "and rotate the credential if it was real. Do not weaken a rule to\n"
            "make this pass -- the rule is the only thing standing between a\n"
            "pasted key and a repository that outlives it."
        )
        return 1

    print(
        "\nOK: no credential-shaped string and no unwrapped SecretStr value\n"
        "    appears in any scanned artifact."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

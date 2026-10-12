"""Treat retrieved text as untrusted data, and mark where the boundary is.

The premise
-----------
Retrieved content is text someone else wrote, fetched by a system you built,
and handed to a model that has been told to follow instructions. Those two
facts are in direct tension, and prompt injection is the name for the collision.

There is no fix for it. Not a good one, not a complete one -- no amount of
filtering makes untrusted text safe to treat as trusted, because the text can
always say something you did not anticipate in a way you did not anticipate.

So this component does not pretend to fix it. It does two smaller, honest
things: **it reports what looks like an attack, and it makes the boundary
explicit.**

Flag, do not strip
------------------
Everything here is *detection*, never deletion. Three reasons, in order of
how much they matter:

**A denylist is a coin flip at best.** ``"Ignore previous instructions"`` is
catchable. ``"Actually, let's start over -- new task:"`` is the same attack in
different words. Worse, authors defeat keyword matching deliberately: writing
``"ig\\u200bore prev\\u200bious instructions"`` inserts zero-width spaces that
break every substring filter on the way in while rendering identically to a
model. Deleting what you recognise does not remove the threat; it removes your
ability to see it, and it teaches the reader that "scanned" meant "safe".

**Retrieved content legitimately contains instructions.** Documents quote
policies. Code comments say "do not do X". A manual's whole subject is what
the user should do. Text that matches an injection pattern is *often* exactly
what the document was for, and a filter would delete the useful part.

**Silent removal is its own failure mode.** A caller who asked for a document
and got a quietly shorter one has no way to know what was taken or why.

The alternative is structural, and it is the one that actually helps: tell the
model that retrieved text is *data*, not instruction, and mark where it starts
and stops. :meth:`ContentSafety.wrap` renders that boundary;
:meth:`ContentSafety.inspect` tells you what is inside it.

False positives are expected
----------------------------
Every rule here will fire on legitimate documents -- an API reference that
documents a ``system`` role, a security paper quoting an attack. That is not a
defect in the rules. It is why the output is a *report* and the caller decides,
rather than a verdict this component imposes on text it does not understand.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_RULES",
    "DEFAULT_WRAP_END",
    "DEFAULT_WRAP_START",
    "ContentSafety",
    "Finding",
    "Rule",
    "SafetyError",
    "SafetyInputError",
    "SafetyReport",
    "UnsafeContentError",
    "invisible_characters",
]

DEFAULT_WRAP_START: Final[str] = "<retrieved_document>"
DEFAULT_WRAP_END: Final[str] = "</retrieved_document>"

#: Zero-width and invisible characters. These are the tell that a denylist was
#: defeated on purpose: they render as nothing, so "ig<ZWSP>ore" matches no
#: keyword filter while a model reads "ignore" perfectly well.
INVISIBLE: Final[frozenset[str]] = frozenset(
    "\u200b\u200c\u200d\u2060\ufeff\u00ad"
)


@dataclass(frozen=True)
class Rule:
    """One detection rule: a category, a human explanation, a pattern."""

    category: str
    why: str
    pattern: re.Pattern[str]


def _rule(category: str, why: str, pattern: str) -> Rule:
    # MULTILINE, not DOTALL: ``^`` must anchor at the start of *every* line,
    # because a role marker buried on line 3 of a retrieved passage is still
    # a role marker. ``.`` still refuses to cross a newline, so no single rule
    # can swallow an entire document.
    return Rule(category=category, why=why,
                pattern=re.compile(pattern, re.IGNORECASE | re.MULTILINE))


#: Every rule states what it catches and why it is worth catching. Patterns are
#: deliberately loose: this component reports, so a false positive costs a
#: glance while a false negative costs an answer nobody can trace.
DEFAULT_RULES: Final[tuple[Rule, ...]] = (
    _rule(
        "instruction_to_model",
        "an imperative addressed to the model rather than a description of "
        "anything",
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|"
        r"earlier|all|your)\b|\bnew instructions?\b|\byou are now\b|"
        r"\bact as (?:if|though)\b|\bfrom now on\b|\brole[- ]play as\b",
    ),
    _rule(
        "role_override",
        "an attempt to occupy a privileged role the retrieved text has no "
        "right to",
        r"^\s*(system|assistant|developer)\s*[:\-]|\[/?system\]|"
        r"\byou are (?:a|an|the)\b.{0,30}\b(with no|unrestricted|jailbroken)\b",
    ),
    _rule(
        "concealment",
        "zero-width characters, which exist to break keyword matching",
        r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]",
    ),
    _rule(
        "exfiltration",
        "an instruction to move data somewhere, or a credential-shaped string",
        r"\b(send|post|upload|email|forward|exfiltrate)\b.{0,40}\b(secrets?|"
        r"keys?|tokens?|credentials?|environment|\.env|history|passwords?)\b|"
        r"\bsk-[A-Za-z0-9]{16,}\b|\bAKIA[0-9A-Z]{16}\b",
    ),
    _rule(
        "format_escape",
        "an attempt to break out of the wrapper this component provides",
        r"</?(?:retrieved_document|context|document|instructions)\s*>|"
        r"^\s*```(?:system|assistant|developer)",
    ),
    _rule(
        "hidden_directive",
        "a directive dressed as housekeeping -- the shape most likely to "
        "survive a skim",
        r"\bwithout (?:telling|informing|mentioning to) (?:the )?user\b|"
        r"\bdo not (?:tell|mention|reveal)\b.{0,30}\buser\b|"
        r"\bkeep this (?:secret|hidden|between us)\b|"
        r"\boutput your (?:instructions|system prompt)\b",
    ),
)


@dataclass(frozen=True)
class Finding:
    """One thing that looked like an attack, and where it was."""

    category: str
    matched: str
    offset: int
    line: int
    why: str


@dataclass(frozen=True)
class SafetyReport:
    """What :meth:`ContentSafety.inspect` found. ``safe`` is the summary."""

    text: str
    findings: tuple[Finding, ...]

    @property
    def safe(self) -> bool:
        return not self.findings

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(f.category for f in self.findings)

    def report(self) -> str:
        """A readable summary: one line per finding, or an all-clear."""
        if not self.findings:
            return "no findings"
        lines = [
            f"line {f.line:>4}  offset {f.offset:>6}  {f.category:<22} "
            f"{f.matched!r}"
            for f in self.findings
        ]
        return "\n".join([*lines, f"{len(self.findings)} finding(s)"])


class SafetyError(Exception):
    """Base for every failure in this component."""


class SafetyInputError(SafetyError):
    """The caller passed something this component cannot inspect."""


class UnsafeContentError(SafetyError):
    """Retrieved content matched a rule and the caller asked it not to.

    Carries ``findings`` so every match is reported at once -- discovering
    them one per request is a slow way to learn what is in a corpus.
    """

    def __init__(self, message: str, *, findings: Sequence[Finding]) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


class ContentSafety:
    """Scan retrieved text for attack shapes, and mark the trust boundary.

    ``rules`` is replaceable so a caller can add corpus-specific detections --
    an internal doc that routinely documents a ``system`` role may want that
    rule off, and a corpus with a known attack should get a rule for it.
    """

    def __init__(
        self,
        *,
        rules: Sequence[Rule] = DEFAULT_RULES,
        wrap_start: str = DEFAULT_WRAP_START,
        wrap_end: str = DEFAULT_WRAP_END,
        name: str = "retrieved-content-safety",
    ) -> None:
        if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
            raise SafetyInputError(
                f"retrieved-content-safety: rules must be a sequence of Rule, "
                f"got {type(rules).__name__}"
            )
        for position, rule in enumerate(rules):
            if not isinstance(rule, Rule):
                raise SafetyInputError(
                    f"retrieved-content-safety: rules[{position}] is "
                    f"{type(rule).__name__}, expected a Rule"
                )
        for label, value in (("wrap_start", wrap_start), ("wrap_end", wrap_end)):
            if not isinstance(value, str) or not value:
                raise SafetyInputError(
                    f"retrieved-content-safety: {label} must be a non-empty "
                    f"string, got {value!r}"
                )
            if any(char in INVISIBLE for char in value):
                raise SafetyInputError(
                    f"retrieved-content-safety: {label} contains a zero-width "
                    f"character. A delimiter nobody can see is a delimiter "
                    f"nobody can check"
                )
        if not isinstance(name, str) or not name:
            raise SafetyInputError(
                f"retrieved-content-safety: name must be a non-empty string, "
                f"got {name!r}"
            )
        self.rules = tuple(rules)
        self.wrap_start = wrap_start
        self.wrap_end = wrap_end
        self.name = name

    def inspect(self, text: str) -> SafetyReport:
        """Find every match. Never raises on hostile text -- that is the point.

        Reporting is the job. A scanner that raised on the first hit would
        make it impossible to see the whole payload, which is the part worth
        understanding.
        """
        if not isinstance(text, str):
            raise SafetyInputError(
                f"retrieved-content-safety: text must be a string, got "
                f"{type(text).__name__}"
            )

        findings: list[Finding] = []
        for rule in self.rules:
            for match in rule.pattern.finditer(text):
                findings.append(
                    Finding(
                        category=rule.category,
                        matched=match.group(0)[:120],
                        offset=match.start(),
                        line=text.count("\n", 0, match.start()) + 1,
                        why=rule.why,
                    )
                )

        findings.sort(key=lambda f: f.offset)
        return SafetyReport(text=text, findings=tuple(findings))

    def is_safe(self, text: str) -> bool:
        """Whether nothing matched. Never raises."""
        return self.inspect(text).safe

    def require_safe(self, text: str) -> str:
        """Return ``text``, or raise ``UnsafeContentError`` listing every hit.

        Use this only where a finding genuinely must stop the request. Most
        callers should inspect and decide -- see the module docstring for why
        a finding is often legitimate.
        """
        report = self.inspect(text)
        if report.safe:
            return text
        categories = ", ".join(sorted(report.categories))
        raise UnsafeContentError(
            f"{self.name}: {len(report.findings)} finding(s) in retrieved "
            f"content ({categories}). Retrieved text is untrusted, and no "
            f"filter makes it trusted -- a finding is a reason to look, not "
            f"proof of an attack, so this is raised only because you asked "
            f"for it. Use inspect() to review instead",
            findings=report.findings,
        )

    def wrap(self, text: str) -> str:
        """Render ``text`` inside explicit delimiters.

        This is the defence that actually works, and it is not a filter: it
        tells the model that what follows is *data quoted from a source*, not
        instructions to follow. No pattern matching is involved, so no
        paraphrase evades it.

        Content that tries to close the wrapper is flagged by the
        ``format_escape`` rule, which is deliberate coherence between the two
        halves of this component: the delimiter is fixed and public, so an
        attempt to terminate it early is visible in the report.
        """
        if not isinstance(text, str):
            raise SafetyInputError(
                f"retrieved-content-safety: text must be a string, got "
                f"{type(text).__name__}"
            )
        return f"{self.wrap_start}\n{text}\n{self.wrap_end}"

    def __repr__(self) -> str:
        return (
            f"ContentSafety(name={self.name!r}, rules={len(self.rules)}, "
            f"wrap_start={self.wrap_start!r})"
        )


def invisible_characters(text: str) -> list[tuple[str, int]]:
    """Every zero-width character in ``text`` as ``(codepoint, offset)``.

    Public because concealment is worth seeing on its own: a single zero-width
    character in a document that should not have one is the signal, long
    before any pattern matches.
    """
    if not isinstance(text, str):
        raise SafetyInputError(
            f"retrieved-content-safety: text must be a string, got "
            f"{type(text).__name__}"
        )
    return [
        (f"U+{ord(char):04X}", offset)
        for offset, char in enumerate(text)
        if char in INVISIBLE
    ]

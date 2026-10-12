"""Normalize extracted text with explicit rules, and say what changed.

Two jobs, deliberately different from the security area
-------------------------------------------------------
This component **strips** control characters. ``input-validation`` **refuses**
them. Both are right, and the difference is the job:

* ``input-validation`` guards a **boundary**. Text arriving from outside has
  no business containing a NUL or a bidi override, so refusing it is the only
  posture that does not quietly lose. See that component for the attack
  behind every family.
* This component cleans **already-ingested** text. A PDF legitimately hands
  you a form feed at every page break, a soft hyphen at every line end, and
  CRLF from anything produced on Windows. Refusing those would make this
  component useless -- the whole point is that extraction produces them.

So the rule here is not "reject the hostile" but "make the text comparable",
and every transformation is named and reported rather than applied silently.

Why idempotence is a hard requirement
-------------------------------------
Normalization runs inside ingestion pipelines, where a stage is very often
re-run -- a retry, a re-index, a resumed job over a file that was already
processed. If ``normalize`` were not idempotent, the second pass would produce
different text than the first, and the index would disagree with what a fresh
run produced. Two documents ingested at different times would not be
comparable, which defeats the purpose of normalizing at all.

Every rule below was chosen with that in mind, and
``test_normalize_is_idempotent`` asserts it over a corpus of nasty inputs --
including a double pass, a triple pass, and every supported form.

What the rules are, in order
----------------------------
1. **Line endings.** ``\\r\\n`` and lone ``\\r`` become ``\\n``.
2. **Vertical tab** ``\\x0b`` becomes ``\\n``.
3. **Form feed** ``\\x0c`` becomes ``\\n\\n``. This one is worth explaining:
   a form feed in extracted text is a *page break*, and collapsing it to a
   space would delete the only signal that a new page began. Two newlines
   keep the break visible and survive the whitespace pass below.
4. **Remaining C0 controls, DEL and C1** are removed. ``\\t`` and ``\\n`` are
   kept -- indentation and structure are content.
5. **Invisible format characters** -- soft hyphen ``U+00AD``, byte-order mark
   ``U+FEFF`` and zero-width space ``U+200B`` -- are removed. These are *not*
   controls (Unicode category ``Cf``), so rule 4 never caught them, yet they
   render as nothing while defeating an exact match: for an index that is a
   correctness problem, not a cosmetic one. ``U+200D`` (ZWJ) is deliberately
   left alone -- removing it changes what an emoji *says*, and this component
   reports repairs rather than reinterpreting content.
6. **Unicode normalization** in the chosen form.
7. **Horizontal whitespace** runs collapse to one space; trailing whitespace
   is dropped per line. Leading indentation is preserved, because in a code
   corpus it is meaning.
8. **Three or more blank lines** collapse to two.

NFC by default, on purpose
--------------------------
``NFC`` preserves compatibility distinctions: ``\\ufb01`` (the ``fi`` ligature)
stays a ligature. ``NFKC`` folds it to ``fi``.

NFC is the default because it is the least destructive choice -- this is a
component that should not surprise you by changing what a document says. But
if you index for *matching*, NFKC is very likely what you want: it unifies
full-width and half-width forms, so ``ｐｙｔｈｏｎ`` and ``python`` become the
same term. Pass ``form="NFKC"`` and read the trade-off in the README.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_FORM",
    "DEFAULT_MAX_LENGTH",
    "FORMS",
    "NormalizationResult",
    "NormalizerError",
    "NormalizerInputError",
    "TextNormalizer",
    "normalize_text",
]

DEFAULT_FORM: Final[str] = "NFC"

#: The four forms ``unicodedata.normalize`` accepts. Exposed so a caller can
#: validate a form before constructing a normalizer.
FORMS: Final[tuple[str, ...]] = ("NFC", "NFD", "NFKC", "NFKD")

#: Generous, like ``input-validation``: a ceiling meaning "this is not a real
#: document", not a policy. Extracted pages are rarely this large.
DEFAULT_MAX_LENGTH: Final[int] = 5_000_000

#: C0 controls minus ``\\t``, ``\\n``, ``\\x0b`` and ``\\x0c`` -- the three the
#: rules above handle first, plus the two that are content. DEL and C1 follow.
_REMOVED: Final[frozenset[str]] = frozenset(
    "".join(chr(c) for c in range(0x00, 0x09))
    + "".join(chr(c) for c in range(0x0D, 0x20))
    + "\x7f"
    + "".join(chr(c) for c in range(0x80, 0xA0))
)

_HORIZONTAL: Final[frozenset[str]] = frozenset(" \t")

#: Invisible *format* characters (Unicode category ``Cf``), not controls --
#: which is exactly why rule 4 never touched them and why they are called out
#: by name here. All three render as nothing while defeating substring
#: matching, so for an index they are a correctness problem, not a cosmetic
#: one: ``soft hyphen`` in a line-wrapped PDF splits a word at every wrap
#: point, a ``byte-order mark`` sitting anywhere in the text breaks an exact
#: match on the first token it precedes, and a ``zero-width space`` inserted
#: by a hostile author defeats a keyword filter (see ``retrieved-content-
#: safety``, which reports the same characters as ``concealment`` when they
#: arrive at query time instead).
#:
#: ``U+200D`` (ZWJ) is deliberately NOT here. It joins emoji and grapheme
#: clusters; removing it changes what the text *says*, and this component
#: reports its repairs rather than reinterpreting content. Neither is
#: ``U+00A0`` (no-break space) -- that one is whitespace with a rule, and the
#: whitespace pass already reports what it does with it.
_INVISIBLE: Final[frozenset[str]] = frozenset("\u00ad\ufeff\u200b")


class NormalizerError(Exception):
    """Base for every failure in this component."""


class NormalizerInputError(NormalizerError):
    """The caller passed something this component cannot normalize."""


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """The normalized text plus a record of which rules fired.

    ``rules`` is what makes this non-destructive in the sense that matters:
    the caller can see *what was changed*, not merely receive different text.
    A component that rewrote your corpus without saying so would be a
    component you could not debug.
    """

    text: str
    original: str
    rules: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def original_length(self) -> int:
        return len(self.original)

    @property
    def normalized_length(self) -> int:
        return len(self.text)

    def report(self) -> str:
        """A readable summary: what fired, and the size change."""
        if not self.rules:
            return f"no rules applied ({self.original_length:,} characters)"
        applied = ", ".join(self.rules)
        delta = self.normalized_length - self.original_length
        sign = "+" if delta >= 0 else ""
        return (
            f"applied {applied}; {self.original_length:,} -> "
            f"{self.normalized_length:,} characters ({sign}{delta:,})"
        )


class TextNormalizer:
    """Normalize text with explicit, idempotent, reported rules.

    ``form`` chooses the Unicode normalization form; ``collapse_whitespace``
    and ``collapse_blank_lines`` turn individual rules off without reaching
    for a different class. They default to on because they are the ones that
    make extracted text comparable.
    """

    def __init__(
        self,
        *,
        form: str = DEFAULT_FORM,
        collapse_whitespace: bool = True,
        collapse_blank_lines: bool = True,
        max_length: int = DEFAULT_MAX_LENGTH,
        name: str = "text-normalizer",
    ) -> None:
        if form not in FORMS:
            raise NormalizerInputError(
                f"text-normalizer: form must be one of {FORMS}, got {form!r}. "
                f"NFC preserves compatibility distinctions; NFKC folds them "
                f"(the fi ligature becomes 'fi'), which is usually what you "
                f"want when indexing for matching"
            )
        for label, value in (
            ("collapse_whitespace", collapse_whitespace),
            ("collapse_blank_lines", collapse_blank_lines),
        ):
            if not isinstance(value, bool):
                raise NormalizerInputError(
                    f"text-normalizer: {label} must be a bool, got "
                    f"{type(value).__name__}"
                )
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise NormalizerInputError(
                f"text-normalizer: max_length must be an int, got "
                f"{type(max_length).__name__}"
            )
        if max_length < 1:
            raise NormalizerInputError(
                f"text-normalizer: max_length must be at least 1, got "
                f"{max_length}"
            )
        if not isinstance(name, str) or not name:
            raise NormalizerInputError(
                f"text-normalizer: name must be a non-empty string, got {name!r}"
            )
        self.form = form
        self.collapse_whitespace = collapse_whitespace
        self.collapse_blank_lines = collapse_blank_lines
        self.max_length = max_length
        self.name = name

    # ------------------------------------------------------------------ #
    # The rules, in order
    # ------------------------------------------------------------------ #

    @staticmethod
    def _line_endings(text: str) -> str:
        """``\\r\\n`` and lone ``\\r`` become ``\\n``.

        ``\\r\\n`` first, so a lone ``\\r`` pass cannot leave a doubled
        newline behind. Doing it the other way round would turn every Windows
        file into blank lines between content.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _vertical_tab(text: str) -> str:
        return text.replace("\x0b", "\n")

    @staticmethod
    def _form_feed(text: str) -> str:
        """A form feed is a page break: keep it visible as a blank line."""
        return text.replace("\x0c", "\n\n")

    @staticmethod
    def _controls(text: str) -> str:
        return "".join(char for char in text if char not in _REMOVED)

    @staticmethod
    def _invisible(text: str) -> str:
        """Drop the three format characters documented on ``_INVISIBLE``.

        A static method like ``_controls`` so the rule reads as a pure
        function of its input -- no configuration can make it keep a
        character it has decided is an artifact.
        """
        return "".join(char for char in text if char not in _INVISIBLE)

    def _normalize_form(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)

    # ``lstrip``/``rstrip`` take a *string* of characters, not a set.
    _HORIZONTAL_CHARS: Final[str] = " \t"

    def _horizontal(self, text: str) -> str:
        """Collapse runs of spaces/tabs, preserving leading indentation.

        Indented leading whitespace is left alone because in a code corpus it
        is meaning, not layout noise. Trailing whitespace on every line is
        dropped -- it is never meaning, and it defeats line-based diffing.
        """
        out: list[str] = []
        for line in text.split("\n"):
            stripped = line.lstrip(self._HORIZONTAL_CHARS)
            indent = line[: len(line) - len(stripped)]
            body = stripped.rstrip(self._HORIZONTAL_CHARS)
            collapsed: list[str] = []
            for char in body:
                if char in _HORIZONTAL:
                    if not collapsed or collapsed[-1] != " ":
                        collapsed.append(" ")
                else:
                    collapsed.append(char)
            out.append(indent + "".join(collapsed))
        return "\n".join(out)

    def _blank_lines(self, text: str) -> str:
        out: list[str] = []
        blank_run = 0
        for line in text.split("\n"):
            if not line.strip():
                blank_run += 1
                if blank_run > 2:
                    continue
            else:
                blank_run = 0
            out.append(line)
        return "\n".join(out)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def normalize(self, text: str) -> NormalizationResult:
        """Normalize ``text`` and report which rules changed it.

        Returns a :class:`NormalizationResult`; the transformed text is
        ``result.text``. Use :meth:`clean` if you only want the string.
        """
        if not isinstance(text, str):
            raise NormalizerInputError(
                f"{self.name}: text must be a string, got "
                f"{type(text).__name__}"
            )
        if len(text) > self.max_length:
            raise NormalizerInputError(
                f"{self.name}: text is {len(text):,} characters, over the "
                f"{self.max_length:,}-character limit"
            )

        applied: list[str] = []
        current = text

        for label, rule in (
            ("line-endings", self._line_endings),
            ("vertical-tab", self._vertical_tab),
            ("form-feed", self._form_feed),
            ("controls", self._controls),
            ("invisible", self._invisible),
            ("unicode-form", self._normalize_form),
        ):
            result = rule(current)
            if result != current:
                applied.append(label)
                current = result

        if self.collapse_whitespace:
            result = self._horizontal(current)
            if result != current:
                applied.append("horizontal-whitespace")
                current = result

        if self.collapse_blank_lines:
            result = self._blank_lines(current)
            if result != current:
                applied.append("blank-lines")
                current = result

        return NormalizationResult(
            text=current, original=text, rules=tuple(applied)
        )

    def clean(self, text: str) -> str:
        """Just the normalized text, for callers that do not need the report."""
        return self.normalize(text).text

    def is_idempotent(self, text: str) -> bool:
        """Whether normalizing twice gives the same answer as normalizing once.

        Exposed rather than only tested, because a pipeline that re-runs a
        stage needs to be able to assert this about *its own* corpus, not
        just ours.
        """
        once = self.clean(text)
        return self.clean(once) == once

    def __repr__(self) -> str:
        return (
            f"TextNormalizer(form={self.form!r}, "
            f"collapse_whitespace={self.collapse_whitespace}, "
            f"collapse_blank_lines={self.collapse_blank_lines})"
        )


def normalize_text(text: str, *, form: str = DEFAULT_FORM) -> str:
    """One-call convenience for the common case.

    A function, not a default-constructed instance reused at module scope:
    callers who need the rules or the report should use :class:`TextNormalizer`
    and read the result.
    """
    return TextNormalizer(form=form).clean(text)

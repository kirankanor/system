"""Validate inputs against size limits, and refuse text that is hostile.

What "hostile" means here, concretely
-------------------------------------
Not "offensive", not "wrong" -- text this component rejects is text that
*changes meaning downstream* in a way the caller cannot see by reading it.
Four families, each with a real attack behind it:

**NUL bytes.** ``"report.pdf\\x00.jpg"`` is one string to Python and two
filenames to anything that crosses into C, which is how extension checks get
bypassed. A NUL is never present in legitimate text, so rejecting it costs
nothing.

**C0/C1 control characters.** These are not invisible decoration -- they are
commands. ``\\x1b[2J`` clears a terminal, ``\\x1b]0;pwned\\x07`` silently
rewrites the window title, and ``\\r`` rewrites the start of a line so a log
reads as something it did not say. Text that reaches a terminal or a log file
is a control channel unless someone has stopped it being one.

**Unicode bidirectional overrides.** U+202E makes everything after it render
right-to-left, so ``"exe.txt"`` displays as ``"txt.exe"``. This is not
theoretical: it is the shape of the git CVE-2017-1000112 filename-spoofing
attack, and it works on any surface that renders Unicode faithfully -- which
is most of them.

**Line and paragraph separators.** U+2028 and U+2029 are line breaks to some
parsers and ordinary characters to others, so a string can contain a "newline"
that a downstream check never sees.

All four are invisible in a normal editor and survive copy-paste. That is
exactly why a validator, not a human, is the right place to catch them.

Reject, do not silently repair
------------------------------
This component **raises** rather than stripping the offending characters. A
filter that quietly removes them produces text the caller did not ask for,
with no record that anything changed -- and a NUL-stripping filter is itself
the bypass, because ``"a\\x00b"`` becomes ``"ab"`` and the caller never learns
the input was malformed in the first place.

Callers who *want* to normalise can call :func:`normalise` explicitly and see
what it returned.

Size limits are a safety property
---------------------------------
``max_length`` exists because an unbounded string is a resource problem long
before it is a security one: it is what turns a text field into a memory
exhaustion vector, and what makes every downstream pass over the text cost
whatever the attacker chooses. The default is generous -- it is a ceiling for
"something has gone wrong", not a policy. Set your own.
"""

from __future__ import annotations

import unicodedata
from typing import Final

__all__ = [
    "BIDI_OVERRIDES",
    "CONTROLS",
    "DEFAULT_MAX_TEXT_LENGTH",
    "ForbiddenCharacter",
    "InputError",
    "InputValidator",
    "OversizedInputError",
    "contains_forbidden",
    "forbidden_ranges",
    "normalise",
    "what_is_forbidden",
]

#: 1,000,000 characters. Generous: a ceiling that means "this is not a real
#: document", not a formatting policy. Override per call or per validator.
DEFAULT_MAX_TEXT_LENGTH: Final[int] = 1_000_000

#: C0 controls (minus the ones that are ordinary whitespace), DEL, C1.
#: NUL is included deliberately and is called out in the error message.
CONTROLS: Final[frozenset[str]] = frozenset(
    "".join(chr(c) for c in range(0x00, 0x09))
    + "".join(chr(c) for c in range(0x0B, 0x20))
    + "\x7f"
    + "".join(chr(c) for c in range(0x80, 0xA0))
)

#: Directional overrides and isolates. U+202E is the RTLO attack character.
BIDI_OVERRIDES: Final[frozenset[str]] = frozenset(
    "".join(chr(c) for c in range(0x202A, 0x202F))
    + "".join(chr(c) for c in range(0x2066, 0x206A))
)

#: Characters that split text for some parsers but not others.
SEPARATORS: Final[frozenset[str]] = frozenset("\u2028\u2029")


class InputError(Exception):
    """Base for every failure in this component."""


class OversizedInputError(InputError):
    """The input exceeded its allowed size."""


class ForbiddenCharacter(InputError):
    """The input contains a character that changes meaning downstream.

    Carries ``characters`` -- every offending codepoint as ``"U+XXXX"``, so a
    caller can report all of them at once rather than discovering them one per
    request.
    """

    def __init__(self, message: str, *, characters: tuple[str, ...]) -> None:
        super().__init__(message)
        self.characters = characters


def _name(char: str) -> str:
    """A readable name, handling the characters that have no official one."""
    if char in CONTROLS:
        # NUL, most C0 controls and C1 controls are unnamed in UnicodeData.txt.
        # Calling them "unassigned" would be wrong -- they are assigned, just
        # nameless -- so describe what they actually are.
        return "control character"
    return unicodedata.name(char, "unassigned")


def what_is_forbidden(text: str) -> list[tuple[str, str, int]]:
    """Every forbidden character in ``text`` as ``(codepoint, name, offset)``.

    Public and non-raising, so a caller can inspect input -- log it, quarantine
    it, report it -- without being forced into an exception path.
    """
    if not isinstance(text, str):
        raise InputError(
            f"input-validation: expected a string, got {type(text).__name__}"
        )
    return [
        (f"U+{ord(char):04X}", _name(char), offset)
        for offset, char in enumerate(text)
        if char in CONTROLS or char in BIDI_OVERRIDES or char in SEPARATORS
    ]


def contains_forbidden(text: str) -> bool:
    """Whether ``text`` carries a character from any forbidden family."""
    if not isinstance(text, str):
        raise InputError(
            f"input-validation: expected a string, got {type(text).__name__}"
        )
    return any(char in CONTROLS or char in BIDI_OVERRIDES or char in SEPARATORS
               for char in text)


def normalise(text: str) -> str:
    """NFKC-normalise and drop nothing.

    Provided so normalisation is an explicit, visible step rather than a side
    effect of validation. NFKC collapses compatibility forms -- a full-width
    ``ｂ`` becomes ``b`` -- which is the right direction to unify in, and it
    leaves control characters and bidi overrides untouched, so this is *not* a
    cleaning function. Pair it with :func:`what_is_forbidden` if you need to
    know what is still there.
    """
    if not isinstance(text, str):
        raise InputError(
            f"input-validation: expected a string, got {type(text).__name__}"
        )
    return unicodedata.normalize("NFKC", text)


class InputValidator:
    """Check size and character safety of text before it goes anywhere.

    ``max_text_length`` is the default ceiling; pass ``max_length=`` per call
    when a particular field has its own limit. Both are enforced *before* the
    character scan, so an oversized string is reported as oversized rather
    than making the caller scroll to the first control character.

    ``name`` appears in every message, because "which validator rejected this"
    is the first thing anyone asks.
    """

    def __init__(
        self,
        *,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        name: str = "input-validation",
    ) -> None:
        if isinstance(max_text_length, bool) or not isinstance(max_text_length, int):
            raise InputError(
                f"input-validation: max_text_length must be an int, got "
                f"{type(max_text_length).__name__} ({max_text_length!r})"
            )
        if max_text_length < 1:
            raise InputError(
                f"input-validation: max_text_length must be at least 1, got "
                f"{max_text_length}"
            )
        if not isinstance(name, str) or not name:
            raise InputError(
                f"input-validation: name must be a non-empty string, got {name!r}"
            )
        self.max_text_length = max_text_length
        self.name = name

    def _limit(self, field: str, max_length: int | None) -> int:
        if max_length is None:
            return self.max_text_length
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise InputError(
                f"{self.name}: {field} max_length must be an int, got "
                f"{type(max_length).__name__} ({max_length!r})"
            )
        if max_length < 1:
            raise InputError(
                f"{self.name}: {field} max_length must be at least 1, got {max_length}"
            )
        return max_length

    def check_size(
        self, text: str, *, field: str = "text", max_length: int | None = None
    ) -> int:
        """Raise ``OversizedInputError`` if ``text`` is over the limit."""
        if not isinstance(text, str):
            raise InputError(
                f"{self.name}: {field} must be a string, got {type(text).__name__}"
            )
        limit = self._limit(field, max_length)
        if len(text) > limit:
            raise OversizedInputError(
                f"{self.name}: {field} is {len(text):,} characters, over the "
                f"{limit:,}-character limit. Raise max_length if this really is "
                f"a document of that size; an unbounded field is a memory "
                f"exhaustion vector, and every pass downstream costs whatever "
                f"the input decides it should"
            )
        return len(text)

    def validate_text(
        self,
        text: str,
        *,
        field: str = "text",
        max_length: int | None = None,
    ) -> str:
        """Check size and characters; return ``text`` unchanged.

        Unchanged matters: the caller gets back exactly what passed, so nothing
        is silently altered. Raise is the only way this reports a problem.
        """
        self.check_size(text, field=field, max_length=max_length)

        offenders: list[str] = []
        families: set[str] = set()
        for offset, char in enumerate(text):
            family = (
                "a control character" if char in CONTROLS
                else "a bidirectional override" if char in BIDI_OVERRIDES
                else "a line separator" if char in SEPARATORS
                else None
            )
            if family is None:
                continue
            families.add(family)
            offenders.append(f"U+{ord(char):04X} at offset {offset}")

        if offenders:
            preview = ", ".join(offenders[:5])
            more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
            raise ForbiddenCharacter(
                f"{self.name}: {field} contains {preview}{more} -- "
                f"{' and '.join(sorted(families))}. These are invisible in an "
                f"editor and change meaning downstream: control characters are "
                f"terminal and log commands, bidirectional overrides reverse "
                f"what a reader sees, and line separators split text for some "
                f"parsers but not others. Call normalise() explicitly if you "
                f"meant to transform the input",
                characters=tuple(o.split(" at ")[0] for o in offenders),
            )
        return text

    def validate_non_empty(
        self, text: str, *, field: str = "text", max_length: int | None = None
    ) -> str:
        """As :meth:`validate_text`, but a blank string is also an error."""
        value = self.validate_text(text, field=field, max_length=max_length)
        if not value.strip():
            raise InputError(
                f"{self.name}: {field} is empty or whitespace. There is nothing "
                f"to act on, which is a caller bug rather than a legitimate "
                f"empty value"
            )
        return value

    def validate_identifier(
        self, value: str, *, field: str = "identifier", max_length: int = 256
    ) -> str:
        """Validate a short identifier: safe characters, bounded size.

        Identifiers are a narrower problem than text -- they are interpolated
        into paths, filenames, SQL and prompts, so the character set is
        restricted rather than merely screened. ASCII letters, digits, ``_``,
        ``-`` and ``.`` only.
        """
        checked = self.validate_text(value, field=field, max_length=max_length)
        if not checked.strip():
            raise InputError(f"{self.name}: {field} is empty or whitespace")
        bad = [char for char in checked if not (char.isascii() and (char.isalnum() or char in "_.-"))]
        if bad:
            raise InputError(
                f"{self.name}: {field} contains "
                f"{', '.join(sorted({repr(c) for c in bad}))}. Identifiers are "
                f"interpolated into paths and queries, so only ASCII letters, "
                f"digits, '_', '-' and '.' are allowed"
            )
        return checked

    def __repr__(self) -> str:
        return (
            f"InputValidator(name={self.name!r}, "
            f"max_text_length={self.max_text_length})"
        )

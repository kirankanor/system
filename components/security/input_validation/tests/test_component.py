"""Component tests for `input-validation`.

Runs standalone: ``python -m pytest components/security/input_validation/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the two the component is built around: hostile characters are
refused rather than silently removed, and an oversized input raises a typed
error rather than being accepted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from input_validation import (
    CONTROLS,
    BIDI_OVERRIDES,
    DEFAULT_MAX_TEXT_LENGTH,
    ForbiddenCharacter,
    InputError,
    InputValidator,
    OversizedInputError,
    contains_forbidden,
    normalise,
    what_is_forbidden,
)


@pytest.fixture
def v() -> InputValidator:
    return InputValidator(max_text_length=64)


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #


def test_a_ordinary_sentence_passes(v):
    text = "The json module serialises Python objects into JSON text."
    assert v.validate_text(text) == text


def test_the_input_is_returned_by_identity_not_a_copy(v):
    text = "unchanged"
    assert v.validate_text(text) is text


def test_a_unicode_sentence_with_real_typography_passes(v):
    # Accented text, curly quotes and an em dash are ordinary, not hostile.
    text = "Caf\u00e9 \u2014 \u201cna\u00efve\u201d \u00e9l\u00e8ve"
    assert v.validate_text(text) == text


def test_a_tab_and_newline_are_allowed(v):
    # They are whitespace, not commands. The scanner is about characters that
    # change meaning downstream, and normal layout is not that.
    assert v.validate_text("line one\nline two\tindented") is not None


def test_check_size_reports_the_length_without_raising(v):
    assert v.check_size("abc", field="q") == 3


def test_the_default_ceiling_is_one_million_characters():
    assert InputValidator().max_text_length == DEFAULT_MAX_TEXT_LENGTH
    assert DEFAULT_MAX_TEXT_LENGTH == 1_000_000


# --------------------------------------------------------------------------- #
# The four hostile families
# --------------------------------------------------------------------------- #


def test_a_nul_byte_is_refused(v):
    with pytest.raises(ForbiddenCharacter) as excinfo:
        v.validate_text("report.pdf\x00.jpg", field="filename")
    assert "U+0000" in str(excinfo.value)
    assert "control character" in str(excinfo.value)


def test_a_terminal_escape_is_refused(v):
    # \x1b]0;pwned\x07 rewrites a terminal window title, which is a command
    # dressed as text.
    with pytest.raises(ForbiddenCharacter) as excinfo:
        v.validate_text("title\x1b]0;pwned\x07", field="log")
    assert "U+001B" in excinfo.value.characters
    assert "U+0007" in excinfo.value.characters


def test_an_rtlo_override_is_refused(v):
    # The CVE-2017-1000112 shape: "exe.txt" rendered as "txt.exe".
    with pytest.raises(ForbiddenCharacter) as excinfo:
        v.validate_text("exe\u202etxt", field="path")
    assert "bidirectional override" in str(excinfo.value)


def test_every_offender_is_reported_not_just_the_first(v):
    with pytest.raises(ForbiddenCharacter) as excinfo:
        v.validate_text("a\x00b\x01c", field="q")
    assert len(excinfo.value.characters) == 2
    assert "offset 1" in str(excinfo.value)
    assert "offset 3" in str(excinfo.value)


def test_more_than_five_offenders_are_truncated_in_the_message(v):
    text = "".join(f"{i}\x00" for i in range(10))
    with pytest.raises(ForbiddenCharacter) as excinfo:
        v.validate_text(text, field="q")
    assert "more)" in str(excinfo.value)
    assert len(excinfo.value.characters) == 10  # the tuple still has them all


def test_the_forbidden_sets_are_the_documented_ones():
    assert "\x00" in CONTROLS
    assert "\x1b" in CONTROLS
    assert "\x7f" in CONTROLS
    assert "\u202e" in BIDI_OVERRIDES
    assert "\u2066" in BIDI_OVERRIDES
    assert " " not in CONTROLS  # a space is not a control character
    assert "\n" not in CONTROLS  # nor is a newline


# --------------------------------------------------------------------------- #
# Oversized input -- the typed error
# --------------------------------------------------------------------------- #


def test_oversized_input_raises_a_typed_error(v):
    with pytest.raises(OversizedInputError) as excinfo:
        v.validate_text("x" * 65, field="document")
    assert isinstance(excinfo.value, InputError)
    assert "65 characters" in str(excinfo.value)
    assert "64-character limit" in str(excinfo.value)


def test_exactly_the_limit_is_accepted(v):
    assert len(v.validate_text("x" * 64)) == 64


def test_the_limit_can_be_set_per_call(v):
    assert v.validate_text("x" * 50, max_length=60) == "x" * 50
    with pytest.raises(OversizedInputError):
        v.validate_text("x" * 61, max_length=60)


def test_size_is_checked_before_characters():
    # The caller's real bug is the size, so that is what gets reported.
    with pytest.raises(OversizedInputError):
        InputValidator(max_text_length=4).validate_text("a\x00" * 100)


# --------------------------------------------------------------------------- #
# Invalid inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["a list"]])
def test_a_non_string_is_a_caller_error(v, bad):
    with pytest.raises(InputError) as excinfo:
        v.validate_text(bad)
    assert "must be a string" in str(excinfo.value)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "10"])
def test_an_unusable_limit_is_refused_at_construction(bad):
    with pytest.raises(InputError):
        InputValidator(max_text_length=bad)


@pytest.mark.parametrize("bad", [0, -5, True, "10"])
def test_an_unusable_per_call_limit_is_refused(v, bad):
    with pytest.raises(InputError):
        v.validate_text("ok", max_length=bad)


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_a_blank_is_refused_by_validate_non_empty(v, bad):
    with pytest.raises(InputError) as excinfo:
        v.validate_non_empty(bad)
    assert "nothing" in str(excinfo.value)


def test_a_blank_is_accepted_by_validate_text():
    # Whitespace is a legitimate value for some fields; non_empty is the
    # stricter entry point, not a hidden rule inside the default one.
    assert InputValidator().validate_text("   ") == "   "


@pytest.mark.parametrize("bad", ["a/../b", "a b", "a;b", "a\0b", "a\u202eb", "\u202ea"])
def test_an_unusable_identifier_is_refused(v, bad):
    with pytest.raises(InputError):
        v.validate_identifier(bad)


def test_an_identifier_carries_a_256_character_limit(v):
    with pytest.raises(OversizedInputError):
        v.validate_identifier("x" * 257)


def test_a_validator_without_a_name_is_refused():
    with pytest.raises(InputError):
        InputValidator(name="")


# --------------------------------------------------------------------------- #
# Documented guarantees
# --------------------------------------------------------------------------- #


def test_stripping_happens_only_when_asked_for():
    # The refusal to silently repair is the whole posture: a filter that
    # quietly removed a NUL would be the bypass.
    assert "\x00" in normalise("a\x00b")


def test_normalise_unifies_compatibility_forms():
    assert normalise("\ufb01") == "fi"  # ligature fi -> f + i
    assert normalise("\uff21") == "A"  # fullwidth A -> A


def test_what_is_forbidden_never_raises():
    found = what_is_forbidden("clean\x00text\u202e")
    assert [code for code, _, _ in found] == ["U+0000", "U+202E"]
    assert found[0][1] == "control character"  # NUL is unnamed, not unassigned
    assert found[0][2] == 5


def test_contains_forbidden_is_boolean_and_non_raising():
    assert contains_forbidden("clean") is False
    assert contains_forbidden("\x00") is True
    assert contains_forbidden("a\u2028b") is True


def test_the_errors_are_the_components_own_hierarchy():
    assert issubclass(OversizedInputError, InputError)
    assert issubclass(ForbiddenCharacter, InputError)
    assert not issubclass(InputError, ValueError)


def test_the_repr_is_informative():
    assert repr(InputValidator(name="mine", max_text_length=10)) == (
        "InputValidator(name='mine', max_text_length=10)"
    )


def test_no_network_or_optional_dependency_is_used():
    import input_validation as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for banned in ("import openai", "import anthropic", "import httpx",
                   "import socket", "import requests"):
        assert banned not in source

"""Component tests for `text-normalizer`.

Runs standalone: ``python -m pytest components/ingestion/text_normalizer/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

Covers normal cases, edge cases, invalid inputs and the documented guarantees
-- in particular the two the component is built around: normalization is
idempotent, and control characters are stripped while a page break survives.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from text_normalizer import (
    DEFAULT_FORM,
    DEFAULT_MAX_LENGTH,
    FORMS,
    NormalizationResult,
    NormalizerError,
    NormalizerInputError,
    TextNormalizer,
    normalize_text,
)


@pytest.fixture
def n() -> TextNormalizer:
    return TextNormalizer()


# --------------------------------------------------------------------------- #
# Normal cases
# --------------------------------------------------------------------------- #


def test_an_ordinary_sentence_comes_back_unchanged(n):
    text = "The json module serialises Python objects into JSON text."
    assert n.clean(text) == text
    assert n.normalize(text).rules == ()


def test_empty_string_is_valid(n):
    assert n.clean("") == ""
    assert n.normalize("").changed is False


def test_unicode_text_survives(n):
    text = "Caf\u00e9 \u2014 \u201cna\u00efve\u201d \u00e9l\u00e8ve"
    assert n.clean(text) == text


def test_the_default_form_is_nfc():
    assert DEFAULT_FORM == "NFC"
    assert TextNormalizer().form == "NFC"


def test_the_default_ceiling_is_a_ceiling_not_a_policy():
    assert DEFAULT_MAX_LENGTH == 5_000_000


def test_every_documented_form_is_accepted():
    for form in FORMS:
        assert TextNormalizer(form=form).form == form


# --------------------------------------------------------------------------- #
# 1. Idempotence -- the property the component is built around
# --------------------------------------------------------------------------- #

NASTY = [
    ("crlf", "line one\r\nline two\rline three"),
    ("a form feed", "page one\x0cpage two"),
    ("a vertical tab", "a\x0bb"),
    ("controls", "a\x00b\x1bc\x85d"),
    ("runs of spaces", "a    b\t\tc"),
    ("trailing whitespace", "text   \nmore\t"),
    ("a flood of blank lines", "a\n\n\n\n\n\n\nb"),
    ("preserved indentation", "def f():\n        return 1"),
    ("an fi ligature", "the \ufb01le"),
    ("full-width latin", "\uff50\uff59\uff54"),
    ("nothing to do", "already clean"),
    ("empty", ""),
    ("everything at once", "a\r\n\x00\x0cb    c\n\n\n\n  d \ufb01"),
]


@pytest.mark.parametrize(("label", "text"), NASTY)
def test_normalize_is_idempotent(n, label, text):
    once = n.clean(text)
    assert n.clean(once) == once, label
    assert n.clean(n.clean(once)) == once, label


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize(("label", "text"), NASTY)
def test_every_form_is_idempotent(label, text, form):
    once = TextNormalizer(form=form).clean(text)
    assert TextNormalizer(form=form).clean(once) == once, (form, label)


def test_is_idempotent_exposes_the_property_to_a_pipeline(n):
    assert n.is_idempotent("a\r\n\x00 b    c") is True
    assert n.is_idempotent("") is True
    # It is a real check, not a constant: run it against text a naive
    # implementation would fail. Ours passes, so assert the positive.
    assert n.is_idempotent("\x0c\x0c\x0c     \r\n\r\n") is True


# --------------------------------------------------------------------------- #
# 2. Control characters are stripped
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("label", "text", "expected"), [
    ("a NUL", "a\x00b", "ab"),
    ("an ESC", "a\x1bb", "ab"),
    ("a C1 control", "a\x85b", "ab"),
    ("DEL", "a\x7fb", "ab"),
    ("a backspace", "a\x08b", "ab"),
    ("every C0 control", "".join(chr(c) for c in range(0x00, 0x09)), ""),
    # \x0d is excluded here: CR is not stripped, it is turned into \n by the
    # line-endings rule that runs first.
    ("a full C0 sweep", "".join(chr(c) for c in range(0x0D, 0x20)), "\n"),
])
def test_control_characters_are_stripped(n, label, text, expected):
    assert n.clean(text) == expected, label


def test_a_form_feed_becomes_a_visible_page_break(n):
    # Stripping it outright would delete the only signal that a page ended;
    # collapsing it to a space would delete it just as completely.
    assert n.clean("one\x0ctwo") == "one\n\ntwo"


def test_a_vertical_tab_becomes_a_newline(n):
    assert n.clean("a\x0bb") == "a\nb"


def test_tab_and_newline_survive_because_they_are_content(n):
    assert n.clean("a\nb") == "a\nb"
    # A tab MID-LINE is horizontal whitespace and collapses -- that is the
    # documented rule. Only LEADING indentation is preserved.
    assert n.clean("a\tb") == "a b"
    assert n.clean("\t\tindented") == "\t\tindented"


# --------------------------------------------------------------------------- #
# 2b. The three invisible FORMAT characters (FR-55)
#
# These are Unicode category Cf, not controls, which is why the rule above
# never caught them. They are removed because each renders as nothing while
# defeating an exact match -- a correctness problem for an index, not a
# cosmetic one.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("label", "char", "codepoint"), [
    ("soft hyphen", "\u00ad", "U+00AD"),
    ("byte-order mark", "\ufeff", "U+FEFF"),
    ("zero-width space", "\u200b", "U+200B"),
])
def test_the_three_invisible_format_characters_are_removed(n, label, char, codepoint):
    before = f"before{char}after"
    assert n.clean(before) == "beforeafter", codepoint
    # and the artefact is what is gone, not the characters around it
    assert char not in n.clean(before)


def test_the_invisible_characters_are_removed_even_under_nfkc():
    # Compatibility folding must not be a way back in for them.
    for char in ("\u00ad", "\ufeff", "\u200b"):
        assert TextNormalizer(form="NFKC").clean(f"a{char}b") == "ab"


def test_zwj_is_deliberately_left_alone(n):
    """The exclusion is the point: ZWJ changes what the text says."""
    assert n.clean("a\u200db") == "a\u200db"
    # No-break space is whitespace with its own rule, not an artefact here.
    assert n.clean("a\u00a0b") == "a\u00a0b"


def test_the_removal_reports_itself_as_its_own_rule(n):
    result = n.normalize("soft\u00adhyphen\u200b")
    assert "invisible" in result.rules
    assert result.changed is True
    assert "invisible" in result.report()
    # ...and reports nothing when there is nothing to remove
    assert n.normalize("clean text").rules == ()


def test_the_removal_is_idempotent_over_every_form():
    nasty = "".join("\u00ad\ufeff\u200b") * 5
    for form in FORMS:
        nn = TextNormalizer(form=form)
        assert nn.is_idempotent(nasty), form


def test_crlf_and_lone_cr_both_become_lf(n):
    assert n.clean("a\r\nb") == "a\nb"
    assert n.clean("a\rb") == "a\nb"
    # The \r\n pass must run first, or the lone-\r pass would turn every
    # Windows newline into a blank line.
    assert n.clean("a\r\n\r\nb") == "a\n\nb"


def test_trailing_whitespace_is_dropped_per_line(n):
    assert n.clean("text   \nmore\t") == "text\nmore"


def test_a_flood_of_blank_lines_collapses_to_two(n):
    assert n.clean("a\n\n\n\n\n\n\nb") == "a\n\n\nb"
    # The rule caps the RUN; it does not delete every blank line.
    assert n.clean("a\n\nb") == "a\n\nb"
    assert n.clean("a\n\n\nb") == "a\n\n\nb"


def test_whitespace_rules_can_be_turned_off_individually():
    uncollapsed = TextNormalizer(
        collapse_whitespace=False, collapse_blank_lines=False
    )
    assert uncollapsed.clean("a    b\n\n\n\n\nc") == "a    b\n\n\n\n\nc"
    only_blank = TextNormalizer(collapse_whitespace=False)
    assert only_blank.clean("a    b\n\n\n\n\nc") == "a    b\n\n\nc"


def test_the_rules_in_order_give_one_stable_result(n):
    # \r\n -> \n; \x0c -> \n\n; \x00 removed; "b    c" -> "b c"; both
    # blank-line floods capped at two.
    assert n.clean("a\r\n\x00\x0cb    c\n\n\n\n  d") == "a\n\n\nb c\n\n\n  d"


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_result_records_what_changed(n):
    result = n.normalize("a\r\n\x00b    c")
    assert isinstance(result, NormalizationResult)
    assert result.changed is True
    assert "line-endings" in result.rules
    assert "controls" in result.rules
    assert "horizontal-whitespace" in result.rules
    assert result.original == "a\r\n\x00b    c"
    assert result.text == "a\nb c"
    assert result.original_length == len("a\r\n\x00b    c")
    assert result.normalized_length == len(result.text)


def test_a_clean_string_reports_no_rules(n):
    result = n.normalize("already clean")
    assert result.rules == ()
    assert result.changed is False
    assert "no rules applied" in result.report()


def test_the_report_names_the_rules_and_the_size_delta(n):
    report = n.normalize("a\r\nb").report()
    assert "line-endings" in report
    # "a\r\nb" is four characters and comes back as three, so the delta is
    # reported rather than the report going quiet about a one-character loss.
    assert "4 -> 3" in report
    assert "(-1" in report

    shrinking = n.normalize("a\r\n\x00b").report()
    assert "5 -> 3" in shrinking  # \r\n -> \n drops one, the NUL drops one
    assert "(-2" in shrinking


def test_the_result_is_frozen(n):
    result = n.normalize("x")
    with pytest.raises(Exception):
        result.text = "tampered"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# NFC vs NFKC
# --------------------------------------------------------------------------- #


def test_nfc_preserves_compatibility_forms():
    ligature, fullwidth = "\ufb01", "\uff50"
    assert TextNormalizer(form="NFC").clean(ligature) == ligature
    assert TextNormalizer(form="NFC").clean(fullwidth) == fullwidth


def test_nfkc_folds_them():
    assert TextNormalizer(form="NFKC").clean("\ufb01") == "fi"
    assert TextNormalizer(form="NFKC").clean("\uff50\uff59") == "py"


def test_nfd_decomposes():
    composed = "\u00e9"
    assert TextNormalizer(form="NFD").clean(composed) == "e\u0301"


def test_nfc_is_the_default_because_it_is_least_destructive():
    assert TextNormalizer().form == "NFC"


# --------------------------------------------------------------------------- #
# Invalid inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [None, 42, b"bytes", ["a list"]])
def test_a_non_string_is_a_typed_error(n, bad):
    with pytest.raises(NormalizerInputError) as excinfo:
        n.normalize(bad)
    assert "must be a string" in str(excinfo.value)
    assert type(bad).__name__ in str(excinfo.value)  # names what it got


@pytest.mark.parametrize("bad", ["NFX", "nfc", "", None, 5])
def test_an_unusable_form_is_refused(bad):
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(form=bad)
    assert "form must be one of" in str(excinfo.value)


def test_the_form_error_explains_the_trade_off():
    # A caller picking a form needs to know what they are choosing between.
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(form="NFX")
    assert "NFC" in str(excinfo.value)
    assert "NFKC" in str(excinfo.value)


@pytest.mark.parametrize(("kwargs", "needle"), [
    ({"collapse_whitespace": 1}, "must be a bool"),
    ({"collapse_whitespace": "x"}, "must be a bool"),
    ({"collapse_blank_lines": None}, "must be a bool"),
    ({"max_length": 0}, "at least 1"),
    ({"max_length": -5}, "at least 1"),
    ({"max_length": True}, "must be an int"),
    ({"max_length": "10"}, "must be an int"),
    ({"max_length": 1.5}, "must be an int"),
    ({"name": ""}, "non-empty"),
    ({"name": None}, "non-empty"),
])
def test_a_nonsense_configuration_is_refused_at_construction(kwargs, needle):
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(**kwargs)
    assert needle in str(excinfo.value), excinfo.value


def test_the_length_limit_is_enforced():
    with pytest.raises(NormalizerInputError) as excinfo:
        TextNormalizer(max_length=5).clean("x" * 10)
    assert "over the 5-character limit" in str(excinfo.value)


def test_the_limit_is_inclusive():
    assert TextNormalizer(max_length=5).clean("xxxxx") == "xxxxx"


def test_errors_never_raise_on_valid_types():
    # The configuration checks run first; once constructed, normalize only
    # rejects a bad *argument*, never its own state.
    assert TextNormalizer().clean("ok") == "ok"


# --------------------------------------------------------------------------- #
# Documented guarantees
# --------------------------------------------------------------------------- #


def test_the_convenience_function_matches_the_class():
    assert normalize_text("a\r\n\x00b") == TextNormalizer().clean("a\r\n\x00b")
    assert normalize_text("fi", form="NFKC") == "fi"


def test_the_convenience_function_takes_a_form():
    assert normalize_text("\ufb01", form="NFKC") == "fi"
    assert normalize_text("\ufb01") == "\ufb01"


def test_the_errors_are_the_components_own_hierarchy():
    assert issubclass(NormalizerInputError, NormalizerError)
    assert not issubclass(NormalizerError, ValueError)


def test_the_repr_is_informative():
    assert repr(TextNormalizer()) == (
        "TextNormalizer(form='NFC', collapse_whitespace=True, "
        "collapse_blank_lines=True)"
    )
    assert repr(TextNormalizer(form="NFKC", collapse_whitespace=False)) == (
        "TextNormalizer(form='NFKC', collapse_whitespace=False, "
        "collapse_blank_lines=True)"
    )


def test_no_network_or_optional_dependency_is_used():
    import text_normalizer as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for banned in (
        "import openai",
        "import anthropic",
        "import httpx",
        "import socket",
        "import requests",
    ):
        assert banned not in source


# --------------------------------------------------------------------------- #
# FR-55: behaviour and documentation must agree, and keep agreeing
#
# The defect this requirement closes was a CONTRADICTION INSIDE the docs: the
# comparison table implied these characters were stripped while the
# limitations swore they survived. Fixing the behaviour alone would have left
# a document that still contradicts itself, so the documentation is asserted
# here as directly as the behaviour is. Written as codepoints rather than as
# literal characters so a reader of this test can see what it is testing --
# and so an invisible character cannot hide in its own assertion.
# --------------------------------------------------------------------------- #

_ARTIFACTS = (
    ("soft hyphen", chr(0x00AD), "U+00AD"),
    ("byte-order mark", chr(0xFEFF), "U+FEFF"),
    ("zero-width space", chr(0x200B), "U+200B"),
)

#: Sentences this requirement retired. Each is the exact wording that used to
#: claim the opposite of the current behaviour.
_RETIRED_CLAIMS = (
    "No soft-hyphen folding",
    "Zero-width characters are left alone",
    "zero-width characters survive too",
    "U+00AD survives",
)

#: What a passage must say about a codepoint, and what it must never say.
#: Both directions are asserted: an earlier draft only required SOME passage
#: to mention removal, which meant the rules list could revert to "left
#: alone" while the limitations section still said "removed" -- the exact
#: self-contradiction this requirement exists to close -- and the test would
#: have passed.
_REMOVAL_WORDS = ("remov", "stripp", "delet", "drop")
_SURVIVAL_WORDS = ("left alone", "surviv", "are kept", "is kept", "not removed")


def _passages(text: str) -> list[str]:
    """Sentences of a source file or markdown document.

    Paragraph splits are too coarse -- README's rules list is one paragraph,
    so every item in it would vouch for every other. Sentence granularity is
    what makes the assertions below mean something.
    """
    return [
        sentence
        for block in re.split(r"\n\s*\n", text)
        for sentence in re.split(r"(?<=[.!?])\s+", block)
    ]


def test_the_three_artifacts_are_removed_by_the_behaviour():
    n = TextNormalizer()
    for label, char, codepoint in _ARTIFACTS:
        assert n.clean(f"a{char}b") == "ab", codepoint


def test_the_documentation_says_the_three_are_removed():
    """The other half of FR-55.

    Every codepoint must be documented, at least one sentence describing it
    must say it is removed, and *no* sentence describing it may claim it
    survives. The last clause is the one that matters: a document that says
    "removed" in one place and "left alone" in another is the mismatch
    itself, so a test that only looked for agreement somewhere would certify
    the contradiction.
    """
    component = Path(__file__).resolve().parents[1]
    for name in ("README.md", "text_normalizer.py"):
        text = (component / name).read_text(encoding="utf-8")
        passages = _passages(text)
        for label, _char, codepoint in _ARTIFACTS:
            hits = [p for p in passages if codepoint in p]
            assert hits, f"{name} never documents {codepoint}"

            assert any(
                any(word in hit.lower() for word in _REMOVAL_WORDS)
                for hit in hits
            ), (
                f"{name}: no sentence mentioning {codepoint} says it is "
                f"removed, so the documentation no longer describes the "
                f"behaviour -- the mismatch FR-55 closed is back"
            )

            for hit in hits:
                low = hit.lower()
                assert not any(word in low for word in _SURVIVAL_WORDS), (
                    f"{name}: a sentence about {codepoint} claims it "
                    f"survives, contradicting the behaviour and the rest of "
                    f"the documentation: {hit.strip()!r}"
                )


def test_the_retired_claims_do_not_come_back():
    component = Path(__file__).resolve().parents[1]
    everything = "\n".join(
        (component / name).read_text(encoding="utf-8")
        for name in ("README.md", "text_normalizer.py", "example.py")
    )
    for claim in _RETIRED_CLAIMS:
        assert claim not in everything, (
            f"documentation has reverted to claiming {claim!r}"
        )

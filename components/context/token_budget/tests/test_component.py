"""Tests for the token-budget component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

This component imports nothing from the library, so it needs no sibling on
sys.path -- which is itself the point: it copies out and runs.

`tiktoken` is NOT installed here, and installing it would modify dependency
files, which this library forbids. So the tokenizer path is exercised through
an injected counter and the real wheel is never touched; the missing-wheel
path is exercised for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT))

from token_budget import (  # noqa: E402
    MANIFEST_MESSAGE,
    MissingDependency,
    TokenBudget,
    TokenBudgetError,
    TokenBudgetInputError,
    word_count,
)

import pytest  # noqa: E402


@pytest.fixture
def budget():
    return TokenBudget(counter=word_count)


WORDS = " ".join(f"w{i}" for i in range(50))


# --- success paths ------------------------------------------------------- #


def test_word_count_counts_whitespace_separated_words():
    assert word_count("a  b\tc\nd") == 4
    assert word_count("") == 0
    assert word_count("   \n\t ") == 0


def test_count_uses_the_supplied_counter(budget):
    assert budget.count("one two three") == 3


def test_fits_agrees_with_count(budget):
    assert budget.fits("a b c", 3) is True
    assert budget.fits("a b c", 2) is False


def test_remaining_is_negative_when_already_over(budget):
    assert budget.remaining("a b c", 5) == 2
    assert budget.remaining("a b c", 1) == -2


def test_an_injected_counter_means_the_wheel_is_loaded(budget):
    assert budget.is_loaded() is True


def test_a_wheelless_budget_constructs_without_loading():
    assert TokenBudget().is_loaded() is False


def test_truncate_returns_text_that_already_fits_unchanged(budget):
    assert budget.truncate("a b c", 10) == "a b c"


@pytest.mark.parametrize("limit", range(1, 40))
def test_truncate_never_exceeds_the_budget(budget, limit):
    assert budget.count(budget.truncate(WORDS, limit)) <= limit


def test_truncate_reaches_the_budget_when_it_can(budget):
    cut = budget.truncate(WORDS, 10)
    assert budget.count(cut) <= 10
    assert budget.count(cut) >= 8  # not needlessly short


def test_truncate_cuts_at_a_word_boundary(budget):
    cut = budget.truncate("alpha beta gamma delta epsilon", 4)
    assert not cut.endswith(" ")
    assert all(word.isalnum() for word in cut.split())


def test_truncate_can_be_told_to_cut_anywhere(budget):
    cut = budget.truncate("alpha beta gamma", 2, at_word_boundary=False)
    assert budget.count(cut) <= 2


def test_truncate_handles_text_with_no_whitespace_at_all(budget):
    cut = budget.truncate("a" * 100, 5)
    assert budget.count(cut) <= 5


def test_pack_returns_indices_in_order(budget):
    assert budget.pack(["a b", "c d e", "f g h"], 5) == [0, 1]


def test_pack_counts_the_separator(budget):
    items = ["aa bb", "cc dd", "ee ff"]
    taken = budget.pack(items, 4)
    assert budget.count("\n\n".join(items[i] for i in taken)) <= 4


def test_pack_takes_nothing_when_even_the_first_item_is_too_big(budget):
    assert budget.pack(["a b c d e"], 2) == []


def test_pack_of_an_empty_sequence_is_empty(budget):
    assert budget.pack([], 10) == []


def test_the_budget_counts_itself_as_a_counter_for_another_budget():
    inner = TokenBudget(counter=word_count)
    outer = TokenBudget(counter=inner.count)
    assert outer.count("a b c") == 3


def test_the_repr_names_the_encoding(budget):
    assert "cl100k_base" in repr(budget)


# --- failure paths ------------------------------------------------------- #


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_budget_is_rejected(budget, limit):
    with pytest.raises(TokenBudgetInputError) as exc:
        budget.fits("x", limit)
    assert "at least 1" in str(exc.value)


@pytest.mark.parametrize("limit", [True, 1.5, "10", None, []])
def test_a_non_int_budget_is_rejected(budget, limit):
    with pytest.raises(TokenBudgetInputError):
        budget.fits("x", limit)


@pytest.mark.parametrize("text", [None, 3, ["a"], {}])
def test_a_non_string_text_is_rejected(budget, text):
    with pytest.raises(TokenBudgetInputError):
        budget.count(text)


@pytest.mark.parametrize("encoding", ["", None, 5])
def test_an_unusable_encoding_is_rejected(encoding):
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter=word_count, encoding=encoding)


def test_a_non_callable_counter_is_rejected():
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter="count me")


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter=word_count, name=name)


def test_a_counter_returning_a_string_is_rejected():
    with pytest.raises(TokenBudgetError) as exc:
        TokenBudget(counter=lambda text: "no").count("x")
    assert "non-negative int" in str(exc.value)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_a_counter_returning_a_bad_number_is_rejected(value):
    with pytest.raises(TokenBudgetError):
        TokenBudget(counter=lambda text, v=value: v).count("x")


def test_word_count_rejects_a_non_string():
    with pytest.raises(TokenBudgetInputError):
        word_count(3)


@pytest.mark.parametrize("texts", ["a string", 3, None, {"a": 1}])
def test_pack_rejects_a_non_sequence(budget, texts):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(texts, 10)


def test_pack_rejects_a_non_string_item(budget):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(["ok", 3], 10)


def test_pack_rejects_a_non_string_separator(budget):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(["a"], 10, separator=3)


# --- the guarded dependency (FR-3) --------------------------------------- #


def test_a_missing_wheel_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError):
        TokenBudget().count("x")


def test_the_error_is_an_import_error_and_names_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingDependency) as exc:
        TokenBudget().count("x")
    assert isinstance(exc.value, ImportError)
    assert "tiktoken" in str(exc.value)


def test_the_message_is_the_manifest_verbatim(monkeypatch):
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingDependency) as exc:
        TokenBudget().count("x")
    assert str(exc.value) == MANIFEST_MESSAGE == manifest["missing_dependency_message"]


def test_the_original_import_error_is_kept_as_the_cause(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingDependency) as exc:
        TokenBudget().count("x")
    assert isinstance(exc.value.__cause__, ImportError)


def test_an_injected_counter_never_touches_the_wheel(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert TokenBudget(counter=word_count).count("a b") == 2


def test_the_errors_form_a_hierarchy():
    assert issubclass(TokenBudgetInputError, TokenBudgetError)
    assert not issubclass(TokenBudgetError, ValueError)
    assert not issubclass(MissingDependency, TokenBudgetInputError)


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_tiktoken_optional_not_required():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["optional_packages"] == [
        {"name": "tiktoken", "specifier": ">=0.7.0"}
    ]
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == []
    assert manifest["category"] == "context"

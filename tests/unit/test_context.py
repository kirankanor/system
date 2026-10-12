"""Gate tests for the context area (T-09c).

Three components:

* ``token-budget``      -- measure and cut text to a model's budget;
* ``context-selector``  -- deduplicate and pack chunks (ContextBuilderProtocol);
* ``citation-formatter`` -- resolve ``[n]`` spans against what was shown.

The four properties the task names as observable are asserted here by name:

1. the token budget is never exceeded,
2. duplicate chunks are deduplicated,
3. an all-duplicate input still returns context,
4. an orphan citation span raises a typed error.

`sentence-transformers` and `tiktoken` are NOT installed here, and installing
either would modify dependency files -- which this library forbids. So
``token-budget`` is exercised with an injected counter, and its missing-wheel
path is exercised for real. The real tokenizer is unexecuted; the component
README says so plainly.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "components" / "context"
CORE_TYPES = ROOT / "components" / "storage" / "core_types"

for directory in (
    CORE_TYPES,
    CONTEXT / "token_budget",
    CONTEXT / "context_selector",
    CONTEXT / "citation_formatter",
):
    sys.path.insert(0, str(directory))

import core_types as core  # noqa: E402
from citation_formatter import (  # noqa: E402
    CitationFormatter,
    CitationInputError,
    CitationOrphanSpanError,
)
from context_selector import (  # noqa: E402
    ContextError,
    ContextInputError,
    ContextSelector,
)
from token_budget import (  # noqa: E402
    MANIFEST_MESSAGE,
    MissingDependency,
    TokenBudget,
    TokenBudgetError,
    TokenBudgetInputError,
    word_count,
)

import pytest  # noqa: E402


def scored(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("source_name", f"{cid}.md")
    return core.ScoredDocument(chunk_id=cid, text=text, score=1.0, **kwargs)


def context(text: str, chunks=(), *, budget: int = 100, **kwargs) -> core.Context:
    return core.Context(
        text=text,
        chunks=tuple(chunks),
        token_count=kwargs.pop("token_count", 0),
        budget=budget,
        truncated=kwargs.pop("truncated", False),
    )


@pytest.fixture
def budget():
    return TokenBudget(counter=word_count)


@pytest.fixture
def selector():
    return ContextSelector(counter=word_count)


@pytest.fixture
def candidates():
    return [
        scored("c0", "alpha beta", chunk_index=0),
        scored("c1", "gamma delta", chunk_index=1),
        scored("c2", "epsilon zeta", chunk_index=2),
    ]


# =========================================================================== #
# Observable 1: the token budget is never exceeded
# =========================================================================== #


def test_a_packed_context_never_exceeds_its_budget(selector, candidates):
    for limit in range(1, 12):
        built = selector.select("alpha", candidates, limit)
        assert built.token_count <= built.budget, limit


def test_the_returned_token_count_is_the_real_cost_of_the_text(selector, candidates):
    built = selector.select("alpha", candidates, 100)
    assert built.token_count == word_count(built.text)


def test_the_separator_between_passages_is_counted(candidates):
    """The classic overshoot: the passages fit, the joins do not.

    A *word* counter charges nothing for a whitespace join, so this uses a
    character counter, where the separator has an unmistakable cost. That
    difference is the point: the guarantee is always relative to the counter
    the caller chose.
    """
    per_character = len
    no_sep = ContextSelector(counter=per_character, separator="")
    with_sep = ContextSelector(counter=per_character, separator="\n\n")

    assert no_sep.select("q", candidates, 21).token_count == 21  # c0 + c1
    assert len(no_sep.select("q", candidates, 21).chunks) == 2

    assert with_sep.select("q", candidates, 21).token_count == 10  # c0 only
    assert len(with_sep.select("q", candidates, 21).chunks) == 1


def test_a_whitespace_separator_is_free_to_a_word_counter(candidates):
    """Stated plainly rather than assumed: the unit is the counter's."""
    assert word_count("\n\n") == 0
    assert ContextSelector(counter=word_count).select("q", candidates, 6).token_count == 6


def test_a_passage_that_does_not_fit_is_skipped_not_truncated(selector, candidates):
    built = selector.select("alpha", candidates, 5)
    assert built.token_count <= built.budget
    assert built.truncated is True
    assert len(built.chunks) == 2  # "epsilon zeta" did not fit


def test_nothing_is_crammed_in_when_the_first_passage_is_too_big(selector, candidates):
    """A half-passage is worse than none: the model cannot tell it is a fragment."""
    built = selector.select("alpha", candidates, 1)
    assert built.chunks == ()
    assert built.text == ""
    assert built.truncated is True


def test_truncate_never_returns_more_than_asked_for(budget):
    words = " ".join(f"w{i}" for i in range(50))
    for limit in range(1, 40):
        assert budget.count(budget.truncate(words, limit)) <= limit, limit


def test_truncate_returns_the_whole_text_when_it_already_fits(budget):
    assert budget.truncate("a b c", 10) == "a b c"


def test_truncate_cuts_at_a_word_boundary_by_default(budget):
    cut = budget.truncate("alpha beta gamma delta", 4)
    assert not cut.endswith(" ")  # no trailing separator left behind
    assert budget.count(cut) <= 4
    assert all(word.isalnum() for word in cut.split())


def test_truncate_can_be_told_to_cut_anywhere(budget):
    cut = budget.truncate("alpha beta gamma delta", 4, at_word_boundary=False)
    assert budget.count(cut) <= 4


def test_pack_returns_the_indices_that_fit_in_order(budget):
    assert budget.pack(["a b", "c d e", "f g h"], 5) == [0, 1]


def test_pack_counts_the_separator(budget):
    items = ["aa bb", "cc dd", "ee ff"]
    taken = budget.pack(items, 4)
    assert taken == [0, 1]
    joined = "\n\n".join(items[i] for i in taken)
    assert budget.count(joined) <= 4


def test_a_budget_is_measured_in_the_counters_own_unit(selector):
    """token_count and budget are in the same unit, so Context is self-describing."""
    built = ContextSelector(counter=lambda text: len(text)).select(
        "q", [scored("c0", "abcd"), scored("c1", "efgh")], 10
    )
    assert built.token_count == len(built.text)  # characters, by that counter


def test_fits_and_remaining_agree_with_count(budget):
    assert budget.fits("a b c", 3) is True
    assert budget.fits("a b c", 2) is False
    assert budget.remaining("a b c", 5) == 2
    assert budget.remaining("a b c", 1) == -2  # negative means already over


def test_word_count_is_a_named_honest_approximation():
    assert word_count("a  b\tc\nd") == 4
    assert word_count("") == 0
    assert word_count("   ") == 0


# =========================================================================== #
# Observable 2: duplicate chunks are deduplicated
# =========================================================================== #


def test_the_same_passage_returned_twice_is_kept_once(selector):
    dup = [scored("c0", "same words", chunk_index=0), scored("c1", "same words", chunk_index=1)]
    assert len(selector.select("q", dup, 100).chunks) == 1


def test_duplicates_arriving_under_two_ids_are_still_duplicates(selector):
    """The real case: dense and sparse returned the same passage separately."""
    dup = [scored("dense-7", "identical evidence", chunk_index=0),
           scored("sparse-3", "identical evidence", chunk_index=1)]
    kept = selector.select("q", dup, 100).chunks
    assert len(kept) == 1
    assert kept[0].chunk_id == "dense-7"  # the better-ranked copy survives


def test_layout_differences_do_not_defeat_deduplication(selector):
    dup = [scored("c0", "alpha   beta\n\ngamma", chunk_index=0),
           scored("c1", "alpha beta gamma", chunk_index=1)]
    assert len(selector.select("q", dup, 100).chunks) == 1


def test_a_repeated_chunk_id_is_deduplicated_even_with_different_text(selector):
    dup = [scored("dup", "first wording", chunk_index=0), scored("dup", "second wording", chunk_index=1)]
    assert [c.chunk_id for c in selector.select("q", dup, 100).chunks] == ["dup"]


def test_distinct_passages_are_all_kept(selector, candidates):
    assert len(selector.select("q", candidates, 100).chunks) == 3


def test_deduplication_happens_before_the_budget_is_spent(selector):
    """Otherwise a duplicate would consume budget and push real evidence out."""
    dup = [scored("c0", "alpha beta", chunk_index=0), scored("c1", "alpha beta", chunk_index=1),
           scored("c2", "gamma delta", chunk_index=2)]
    built = selector.select("q", dup, 4)
    assert [c.chunk_id for c in built.chunks] == ["c0", "c2"]


def test_deduplicate_is_callable_on_its_own(selector):
    dup = [scored("c0", "same", chunk_index=0), scored("c1", "same", chunk_index=1)]
    assert [c.chunk_id for c in selector.deduplicate(dup)] == ["c0"]


# =========================================================================== #
# Observable 3: an all-duplicate input still returns context
# =========================================================================== #


def test_an_all_duplicate_input_still_returns_context(selector):
    dup = [scored("c0", "same words", chunk_index=0), scored("c1", "same words", chunk_index=1),
           scored("c2", "same words", chunk_index=2)]
    built = selector.select("q", dup, 100)
    assert built.text == "same words"
    assert len(built.chunks) == 1
    assert built.token_count == 2


def test_an_all_duplicate_input_is_not_an_empty_context(selector):
    dup = [scored(f"c{i}", "repeated evidence", chunk_index=i) for i in range(5)]
    built = selector.select("q", dup, 100)
    assert built.text != ""
    assert built.chunks != ()


def test_an_all_duplicate_input_still_respects_the_budget(selector):
    dup = [scored(f"c{i}", "same words", chunk_index=i) for i in range(5)]
    built = selector.select("q", dup, 1)
    assert built.token_count <= built.budget  # deduped to 2 tokens, which is over 1
    assert built.chunks == ()


def test_a_genuinely_empty_input_returns_an_empty_context_not_an_error(selector):
    built = selector.select("q", [], 10)
    assert built.text == ""
    assert built.chunks == ()
    assert built.token_count == 0
    assert built.budget == 10
    assert built.truncated is False  # nothing was dropped, so nothing was truncated


# =========================================================================== #
# Observable 4: an orphan citation span raises a typed error
# =========================================================================== #


@pytest.fixture
def formatter():
    return CitationFormatter()


@pytest.fixture
def cited():
    return context(
        "Alpha says [1]. Beta adds [2]. See also [1].",
        [scored("c0", "alpha"), scored("c1", "beta")],
    )


def test_an_orphan_citation_span_raises_a_typed_error(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(context("a [1] and ghost [3]", [scored("c0", "a")]))


def test_the_orphan_error_names_the_offending_span(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.validate(context("a [1] ghost [3]", [scored("c0", "a")]))
    assert "[3]" in str(exc.value)
    assert "[1] through [1]" in str(exc.value)  # tells the caller what IS valid


def test_the_orphan_error_carries_every_offending_span(formatter):
    built = context("[1] [9] [2] [8]", [scored("c0", "a"), scored("c1", "b")])
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.validate(built)
    assert [number for number, _ in exc.value.spans] == [9, 8]


def test_a_span_pointing_at_zero_is_an_orphan(formatter):
    """Nobody writes [0] in prose; 1-based is decided once, here."""
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(context("zero [0]", [scored("c0", "a")]))


def test_a_span_with_no_chunks_at_all_is_an_orphan(formatter):
    with pytest.raises(CitationOrphanSpanError) as exc:
        formatter.format(context("claims [1]", []))
    assert "none at all" in str(exc.value)


def test_a_span_beyond_the_last_chunk_is_an_orphan(formatter):
    with pytest.raises(CitationOrphanSpanError):
        formatter.validate(context("[1] [2] [3]", [scored("c0", "a"), scored("c1", "b")]))


def test_valid_spans_do_not_raise(formatter, cited):
    formatter.validate(cited)  # must not raise


def test_text_with_no_spans_is_valid(formatter):
    """A model is allowed to answer without citing."""
    assert formatter.format(context("a plain answer", [])) == "a plain answer"


def test_format_returns_the_text_unchanged(formatter, cited):
    """Appending a bibliography would silently blow the budget upstream enforced."""
    assert formatter.format(cited) == cited.text


def test_spans_report_their_character_offsets(formatter):
    assert formatter.spans("ab [12] cd [3]") == [(12, 3), (3, 11)]


def test_references_lists_cited_numbers_in_order(formatter, cited):
    assert formatter.references(cited) == [1, 2, 1]


def test_bracketed_non_numbers_are_ordinary_text(formatter):
    assert formatter.spans("see [note], [1.5] and [2]") == [(2, 22)]


def test_the_bibliography_lists_distinct_cited_sources_in_order(formatter, cited):
    assert formatter.bibliography(cited) == ["c0.md", "c1.md"]


def test_the_bibliography_omits_uncited_passages(formatter):
    built = context("[2]", [scored("c0", "a"), scored("c1", "b")])
    assert formatter.bibliography(built) == ["c1.md"]


def test_a_bibliography_label_falls_back_when_a_chunk_has_no_source(formatter):
    bare = core.ScoredDocument(chunk_id="c9", text="x", score=1.0, chunk_index=0)
    built = context("[1]", [bare])
    assert formatter.bibliography(built) == ["c9"]


def test_a_bibliography_label_falls_back_to_a_positional_name(formatter):
    bare = core.ScoredDocument(chunk_id=None, text="x", score=1.0, chunk_index=0)
    assert formatter.bibliography(context("[1]", [bare])) == ["passage 1"]


def test_an_unreachable_label_never_renders_blank(formatter):
    """bibliography() must not include a span it cannot resolve."""
    assert formatter.bibliography(context("[9]", [scored("c0", "a")])) == []


# =========================================================================== #
# Cross-component: the three compose
# =========================================================================== #


def test_select_then_format_is_the_pipeline(budget, selector, formatter, candidates):
    built = selector.select("alpha", candidates, 6)
    cited = context(f"{built.text} [1]", built.chunks,
                    budget=built.budget, token_count=built.token_count + 2)
    assert formatter.format(cited) == cited.text
    assert formatter.bibliography(cited) == ["c0.md"]


def test_a_budget_count_feeds_a_selector_directly(budget, candidates):
    """TokenBudget.count is a valid ContextSelector counter -- that is the wiring."""
    built = ContextSelector(counter=budget.count).select("alpha", candidates, 6)
    assert built.token_count <= built.budget


def test_the_selector_satisfies_context_builder_protocol(selector):
    assert isinstance(selector, core.ContextBuilderProtocol)


def test_the_protocol_signature_is_the_documented_one(selector):
    import inspect

    assert list(inspect.signature(selector.build).parameters) == [
        "query", "chunks", "budget",
    ]


# =========================================================================== #
# Failure paths -- token-budget
# =========================================================================== #


@pytest.mark.parametrize("budget_value", [0, -1])
def test_a_non_positive_budget_is_rejected(budget, budget_value):
    with pytest.raises(TokenBudgetInputError) as exc:
        budget.fits("x", budget_value)
    assert "at least 1" in str(exc.value)


@pytest.mark.parametrize("budget_value", [True, 1.5, "10", None])
def test_a_non_int_budget_is_rejected(budget, budget_value):
    with pytest.raises(TokenBudgetInputError):
        budget.fits("x", budget_value)


@pytest.mark.parametrize("text", [None, 3, ["a"]])
def test_a_non_string_text_is_rejected(budget, text):
    with pytest.raises(TokenBudgetInputError):
        budget.count(text)


@pytest.mark.parametrize("encoding", ["", None, 5])
def test_an_unusable_encoding_is_rejected(encoding):
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter=word_count, encoding=encoding)


def test_a_non_callable_counter_is_rejected():
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter="not callable")


def test_a_counter_returning_a_non_int_is_rejected(budget):
    broken = TokenBudget(counter=lambda text: "no")
    with pytest.raises(TokenBudgetError) as exc:
        broken.count("x")
    assert "non-negative int" in str(exc.value)


def test_a_counter_returning_a_negative_int_is_rejected():
    with pytest.raises(TokenBudgetError):
        TokenBudget(counter=lambda text: -1).count("x")


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(TokenBudgetInputError):
        TokenBudget(counter=word_count, name=name)


@pytest.mark.parametrize("texts", ["a string", 3, None])
def test_pack_rejects_a_non_sequence(budget, texts):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(texts, 10)


def test_pack_rejects_a_non_string_item(budget):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(["ok", 3], 10)


def test_pack_rejects_a_non_string_separator(budget):
    with pytest.raises(TokenBudgetInputError):
        budget.pack(["a"], 10, separator=3)


def test_word_count_rejects_a_non_string():
    with pytest.raises(TokenBudgetInputError):
        word_count(3)


# =========================================================================== #
# Failure paths -- the guarded tiktoken dependency (FR-3)
# =========================================================================== #


def test_a_missing_tiktoken_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError):
        TokenBudget().count("x")


def test_the_missing_dependency_error_names_the_package_and_matches_the_manifest(
    monkeypatch,
):
    manifest = json.loads(
        (CONTEXT / "token_budget" / "manifest.json").read_text(encoding="utf-8")
    )
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(MissingDependency) as exc:
        TokenBudget().count("x")
    assert isinstance(exc.value, ImportError)
    assert "tiktoken" in str(exc.value)
    assert str(exc.value) == MANIFEST_MESSAGE == manifest["missing_dependency_message"]


def test_the_wheel_is_never_a_top_level_import():
    """DECISION-bf2566ea, applied to this component too."""
    tree = ast.parse(
        Path(sys.modules[TokenBudget.__module__].__file__).read_text(encoding="utf-8")
    )
    top_level = set()
    for node in tree.body:  # direct children of the module only
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "tiktoken" not in top_level


def test_an_injected_counter_never_touches_the_wheel(budget):
    assert budget.is_loaded() is True  # a counter counts as loaded
    assert budget.count("a b") == 2


def test_the_budget_constructs_with_the_wheel_absent():
    assert TokenBudget().is_loaded() is False  # nothing imported yet


def test_the_manifest_declares_tiktoken_optional_not_required():
    manifest = json.loads(
        (CONTEXT / "token_budget" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["optional_packages"] == [
        {"name": "tiktoken", "specifier": ">=0.7.0"}
    ]
    assert manifest["required_packages"] == []
    assert manifest["dependencies"] == []  # this one needs no core-types


# =========================================================================== #
# Failure paths -- context-selector
# =========================================================================== #


def test_an_absent_counter_is_rejected_with_the_reason():
    """No silent default: a guessed token count overshoots real budgets."""
    with pytest.raises(ContextInputError) as exc:
        ContextSelector()
    assert "counter is required" in str(exc.value)
    assert "overshoots" in str(exc.value)


def test_a_non_callable_counter_is_rejected():
    with pytest.raises(ContextInputError):
        ContextSelector(counter=12)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(selector, candidates, query):
    with pytest.raises(ContextInputError) as exc:
        selector.select(query, candidates, 10)
    assert "context-selector" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(selector, candidates, query):
    with pytest.raises(ContextInputError):
        selector.select(query, candidates, 10)


@pytest.mark.parametrize("budget_value", [0, -1, True, 1.5, "10", None])
def test_an_unusable_budget_is_rejected(selector, candidates, budget_value):
    with pytest.raises(ContextInputError):
        selector.select("q", candidates, budget_value)


def test_a_non_document_in_the_list_is_rejected_by_position(selector, candidates):
    with pytest.raises(ContextInputError) as exc:
        selector.select("q", candidates + ["nope"], 10)
    assert "[3]" in str(exc.value)


@pytest.mark.parametrize("bad", ["a string", 3, None, {"a": 1}])
def test_a_non_sequence_of_chunks_is_rejected(selector, bad):
    with pytest.raises(ContextInputError):
        selector.select("q", bad, 10)


@pytest.mark.parametrize("separator", [3, None, ["x"]])
def test_an_unusable_separator_is_rejected(separator):
    with pytest.raises(ContextInputError):
        ContextSelector(counter=word_count, separator=separator)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_selector_name_is_rejected(name):
    with pytest.raises(ContextInputError):
        ContextSelector(counter=word_count, name=name)


def test_a_broken_counter_is_reported(candidates):
    broken = ContextSelector(counter=lambda text: "not a number")
    with pytest.raises(ContextError) as exc:
        broken.select("q", candidates, 10)
    assert "non-negative int" in str(exc.value)


# =========================================================================== #
# Failure paths -- citation-formatter
# =========================================================================== #


def test_a_pattern_that_is_not_a_valid_regex_is_rejected():
    with pytest.raises(CitationInputError):
        CitationFormatter(pattern="[")


def test_a_pattern_without_exactly_one_group_is_rejected():
    with pytest.raises(CitationInputError) as exc:
        CitationFormatter(pattern=r"\[(\d+)\]-(\d+)")
    assert "exactly one capture group" in str(exc.value)


def test_a_pattern_with_no_group_is_rejected():
    with pytest.raises(CitationInputError):
        CitationFormatter(pattern=r"\[\d+\]")


@pytest.mark.parametrize("one_based", ["yes", 1, None])
def test_an_unusable_one_based_is_rejected(one_based):
    with pytest.raises(CitationInputError):
        CitationFormatter(one_based=one_based)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_formatter_name_is_rejected(name):
    with pytest.raises(CitationInputError):
        CitationFormatter(name=name)


def test_a_non_context_is_rejected(formatter):
    with pytest.raises(CitationInputError) as exc:
        formatter.validate("not a context")
    assert "must be a Context" in str(exc.value)


@pytest.mark.parametrize("text", [None, 3, ["a"]])
def test_spans_reject_a_non_string(formatter, text):
    with pytest.raises(CitationInputError):
        formatter.spans(text)


# =========================================================================== #
# Component independence
# =========================================================================== #


def test_the_selector_and_formatter_import_only_core_types():
    """Each is copy-ready on its own."""
    for module_name in ("context_selector", "citation_formatter"):
        module = sys.modules[module_name]
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "token_budget" not in imported, module_name
        assert "context_selector" not in imported or module_name == "context_selector"
        assert "citation_formatter" not in imported or module_name == "citation_formatter"


def test_token_budget_imports_no_component_at_all():
    module = sys.modules[TokenBudget.__module__]
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "core_types" not in imported


def test_every_context_manifest_declares_reranking_category():
    for directory in ("token_budget", "context_selector", "citation_formatter"):
        manifest = json.loads((CONTEXT / directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["category"] == "context", directory


def test_the_two_core_types_consumers_declare_it():
    for directory in ("context_selector", "citation_formatter"):
        manifest = json.loads((CONTEXT / directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dependencies"] == ["core-types"], directory

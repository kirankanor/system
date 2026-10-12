"""Tests for the prompt-builder component.

Runs without importing `rag_code_library` (AC-13): the component is copyable
and testable on its own. Covers success paths and deliberate failure paths
(AC-4).

No optional package and no network: this returns a string.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types, which lives in its own component
# directory under components/storage. The manifest declares it a dependency,
# so `ragkit add` copies it beside this file; here we reach across to the
# sibling so the test runs from the component tree.
sys.path.insert(0, str(COMPONENT.parent.parent / "storage" / "core_types"))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
from prompt_builder import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    PromptBuilder,
    PromptError,
    PromptInputError,
    render_sources,
)

import pytest  # noqa: E402


def scored(cid: str, text: str, **kwargs) -> core.ScoredDocument:
    kwargs.setdefault("chunk_index", 0)
    kwargs.setdefault("metadata", {})
    kwargs.setdefault("source_name", f"{cid}.md")
    return core.ScoredDocument(chunk_id=cid, text=text, score=1.0, **kwargs)


def ctx(chunks, *, budget: int = 100) -> core.Context:
    items = tuple(chunks)
    return core.Context(
        text="\n\n".join(c.text for c in items),
        chunks=items,
        token_count=5,
        budget=budget,
    )


SOURCES = [
    scored("c0", "Python raises TypeError for a wrong argument type.", chunk_index=0),
    scored("c1", "The json module serialises Python objects.", chunk_index=1),
    scored("c2", "JSON text is easy to read and write.", chunk_index=2),
]


@pytest.fixture
def builder():
    return PromptBuilder()


@pytest.fixture
def context():
    return ctx(SOURCES)


# --- success paths ------------------------------------------------------- #


def test_every_source_gets_a_number(builder, context):
    prompt = builder.build("q", context)
    for number in range(1, len(SOURCES) + 1):
        assert f"[{number}]" in prompt


def test_the_numbers_are_sequential_with_no_gaps(builder, context):
    prompt = builder.build("q", context)
    assert sorted({int(n) for n in re.findall(r"\[(\d+)\]", prompt)}) == [1, 2, 3]


def test_no_number_beyond_the_last_source_appears(builder, context):
    assert "[4]" not in builder.build("q", context)


def test_every_source_text_is_present_verbatim(builder, context):
    prompt = builder.build("q", context)
    for chunk in SOURCES:
        assert chunk.text in prompt


def test_a_source_and_its_number_share_a_line(builder, context):
    prompt = builder.build("q", context)
    for number, chunk in enumerate(SOURCES, start=1):
        assert f"[{number}] {chunk.text}" in prompt


def test_sources_appear_in_context_order(builder, context):
    prompt = builder.build("q", context)
    positions = [prompt.index(f"[{n}] {c.text}") for n, c in enumerate(SOURCES, 1)]
    assert positions == sorted(positions)


def test_render_sources_is_public_and_matches(builder):
    assert render_sources(SOURCES) == [
        f"[{n}] {c.text}" for n, c in enumerate(SOURCES, 1)
    ]


def test_numbers_reports_the_available_citations(builder, context):
    assert builder.numbers(context) == [1, 2, 3]


def test_the_prompt_asks_for_citations(builder, context):
    prompt = builder.build("q", context)
    assert "square brackets" in prompt
    assert "Never invent a source number" in prompt
    assert "instead of guessing" in prompt


def test_the_instruction_contains_no_literal_citation(builder, context):
    body = builder.build("q", context)
    for number, chunk in enumerate(SOURCES, start=1):
        body = body.replace(f"[{number}] {chunk.text}", "")
    assert not re.search(r"\[\d+\]", body)


def test_the_question_is_named(builder, context):
    assert "Question: how does json work?" in builder.build("how does json work?", context)


def test_the_prompt_ends_ready_for_an_answer(builder, context):
    assert builder.build("q", context).rstrip().endswith("Answer:")


def test_building_is_deterministic(builder, context):
    assert builder.build("q", context) == builder.build("q", context)


def test_the_question_label_is_configurable(context):
    assert "User asked: q" in PromptBuilder(question_label="User asked").build("q", context)


def test_a_custom_instruction_replaces_the_default(context):
    custom = PromptBuilder(instruction="Use the sources.")
    prompt = custom.build("q", context)
    assert "Use the sources." in prompt
    assert "Do not use prior knowledge" not in prompt


def test_the_default_instruction_is_not_empty():
    assert DEFAULT_INSTRUCTION.strip()


def test_the_repr_names_the_component(builder):
    assert "prompt-builder" in repr(builder)


# --- failure paths ------------------------------------------------------- #


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(builder, context, query):
    with pytest.raises(PromptInputError) as exc:
        builder.build(query, context)
    assert "prompt-builder" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"], {}])
def test_a_non_string_query_is_rejected(builder, context, query):
    with pytest.raises(PromptInputError):
        builder.build(query, context)


def test_a_missing_context_is_rejected(builder):
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", None)
    assert "context is required" in str(exc.value)
    assert "hallucinate" in str(exc.value)


def test_an_empty_context_is_rejected(builder):
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", ctx([]))
    assert "no passages" in str(exc.value)


def test_a_non_context_is_rejected(builder):
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", "not a context")
    assert "must be a Context" in str(exc.value)


def test_numbers_refuses_an_empty_context(builder):
    with pytest.raises(PromptInputError):
        builder.numbers(ctx([]))


@pytest.mark.parametrize("instruction", ["", "   ", None, 5])
def test_an_unusable_instruction_is_rejected(instruction):
    with pytest.raises(PromptInputError):
        PromptBuilder(instruction=instruction)


@pytest.mark.parametrize("label", ["", None, 5])
def test_an_unusable_question_label_is_rejected(label):
    with pytest.raises(PromptInputError):
        PromptBuilder(question_label=label)


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_name_is_rejected(name):
    with pytest.raises(PromptInputError):
        PromptBuilder(name=name)


def test_render_sources_rejects_a_non_sequence():
    with pytest.raises(PromptInputError):
        render_sources("a string")


def test_render_sources_rejects_a_non_document_by_position():
    with pytest.raises(PromptInputError) as exc:
        render_sources([SOURCES[0], "nope"])
    assert "[1]" in str(exc.value)


def test_the_errors_form_a_hierarchy():
    assert issubclass(PromptInputError, PromptError)
    assert not issubclass(PromptError, ValueError)


# --- manifest ------------------------------------------------------------ #


def test_the_manifest_declares_the_core_types_dependency():
    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["category"] == "generation"
    assert manifest["optional_packages"] == []
    assert manifest["required_packages"] == []
    assert manifest["external_services"] == []

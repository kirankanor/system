"""Gate tests for the generation area (T-09d).

Two components, both network-free by design (FR-10):

* ``prompt-builder``              -- a query plus a Context, as one grounded prompt;
* ``grounded-answer-contract``    -- reject claims the evidence does not support.

The three properties the task names as observable are asserted here by name:

1. the grounded prompt cites every supplied source,
2. missing context raises a typed error,
3. an answer claim unsupported by the supplied evidence is rejected by the contract.

There is no LLM anywhere in this library in v1. Nothing here calls a provider,
and nothing needs a network: the prompt builder returns a string, and the
contract is arithmetic over the evidence vocabulary. That is the whole point
of shipping generation as two components instead of a client.

Runs without importing `rag_code_library` (AC-13).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATION = ROOT / "components" / "generation"
CORE_TYPES = ROOT / "components" / "storage" / "core_types"

for directory in (
    CORE_TYPES,
    GENERATION / "prompt_builder",
    GENERATION / "grounded_answer",
):
    sys.path.insert(0, str(directory))

import core_types as core  # noqa: E402
from grounded_answer import (  # noqa: E402
    Claim,
    GroundednessContract,
    GroundednessResult,
    GroundingError,
    GroundingInputError,
    tokenize,
)
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


def ctx(chunks, *, budget: int = 100, text: str | None = None) -> core.Context:
    items = tuple(chunks)
    return core.Context(
        text=text if text is not None else "\n\n".join(c.text for c in items),
        chunks=items,
        token_count=5,
        budget=budget,
    )


EVIDENCE = [
    scored(
        "c0",
        "Python raises TypeError when a function receives an argument of the "
        "wrong type.",
        chunk_index=0,
    ),
    scored(
        "c1",
        "The json module serialises Python objects into JSON text.",
        chunk_index=1,
    ),
]

GROUNDABLE = (
    "The json module serialises Python objects into JSON text. "
    "Python raises TypeError for a wrong argument type."
)

FABRICATED = (
    "Python raises TypeError for a wrong argument type. "
    "The Zephyr framework uses quantum entanglement for IPC."
)


@pytest.fixture
def context():
    return ctx(EVIDENCE)


@pytest.fixture
def builder():
    return PromptBuilder()


@pytest.fixture
def contract():
    return GroundednessContract()


# =========================================================================== #
# Observable 1: the grounded prompt cites every supplied source
# =========================================================================== #


def test_every_supplied_source_gets_a_number(builder, context):
    prompt = builder.build("What is a TypeError?", context)
    for number in range(1, len(EVIDENCE) + 1):
        assert f"[{number}]" in prompt, number


def test_the_numbers_are_sequential_from_one_with_no_gaps(builder, context):
    prompt = builder.build("What is a TypeError?", context)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", prompt)})
    assert cited == [1, 2]


def test_a_number_never_beyond_the_last_source_appears(builder, context):
    """A [3] with two sources is exactly the orphan citation-formatter catches."""
    prompt = builder.build("What is a TypeError?", context)
    assert "[3]" not in prompt


def test_every_source_text_is_present_verbatim(builder, context):
    prompt = builder.build("What is a TypeError?", context)
    for chunk in EVIDENCE:
        assert chunk.text in prompt


def test_the_source_and_the_number_stay_on_one_line(builder, context):
    prompt = builder.build("What is a TypeError?", context)
    for number, chunk in enumerate(EVIDENCE, start=1):
        assert f"[{number}] {chunk.text}" in prompt


def test_render_sources_is_the_same_numbering(builder, context):
    assert render_sources(EVIDENCE) == [
        "[1] " + EVIDENCE[0].text,
        "[2] " + EVIDENCE[1].text,
    ]


def test_numbers_reports_what_the_prompt_makes_available(builder, context):
    assert builder.numbers(context) == [1, 2]


def test_the_prompt_tells_the_model_to_cite(builder, context):
    prompt = builder.build("q", context)
    assert "square brackets" in prompt
    assert "Never invent a source number" in prompt


def test_the_instruction_carries_no_literal_citation_of_its_own(builder, context):
    """'like [1]' in an instruction reads as a real citation once assembled."""
    body = builder.build("q", context)
    for number, chunk in enumerate(EVIDENCE, start=1):
        body = body.replace(f"[{number}] {chunk.text}", "")
    assert not re.search(r"\[\d+\]", body), "the instruction invented a citation"


def test_the_prompt_is_deterministic(builder, context):
    assert builder.build("q", context) == builder.build("q", context)


def test_the_prompt_names_the_question(builder, context):
    assert "Question: What is a TypeError?" in builder.build("What is a TypeError?", context)


def test_the_question_label_is_configurable(context):
    custom = PromptBuilder(question_label="User asked")
    assert "User asked: q" in custom.build("q", context)


def test_a_custom_instruction_replaces_the_default_wholesale(context):
    custom = PromptBuilder(instruction="Use the sources. Cite them.")
    prompt = custom.build("q", context)
    assert "Use the sources. Cite them." in prompt
    assert "Do not use prior knowledge" not in prompt


# =========================================================================== #
# Observable 2: missing context raises a typed error
# =========================================================================== #


def test_a_missing_context_raises_a_typed_error(builder):
    with pytest.raises(PromptInputError):
        builder.build("q", None)


def test_the_missing_context_error_explains_the_danger(builder):
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", None)
    assert "context is required" in str(exc.value)
    assert "hallucinate" in str(exc.value)


def test_an_empty_context_is_also_rejected(builder):
    """A grounded prompt with nothing to ground it is the same trap."""
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", ctx([]))
    assert "no passages" in str(exc.value)


def test_the_prompt_error_is_not_a_value_error(builder):
    assert not issubclass(PromptError, ValueError)
    assert issubclass(PromptInputError, PromptError)


def test_numbers_also_refuses_an_empty_context(builder):
    with pytest.raises(PromptInputError):
        builder.numbers(ctx([]))


# =========================================================================== #
# Observable 3: an unsupported claim is rejected by the contract
# =========================================================================== #


def test_a_fabricated_claim_is_rejected(contract, context):
    result = contract.check(FABRICATED, context)
    assert result.grounded is False


def test_the_fabricated_claim_is_the_one_reported(contract, context):
    result = contract.check(FABRICATED, context)
    assert result.unsupported == (
        "The Zephyr framework uses quantum entanglement for IPC.",
    )


def test_the_fabricated_claim_scores_zero(contract, context):
    result = contract.check(FABRICATED, context)
    failed = [c for c in result.claims if not c.supported]
    assert len(failed) == 1
    assert failed[0].support == 0.0


def test_an_entirely_invented_answer_scores_zero(contract, context):
    result = contract.check("Zephyr quantum entanglement teleportation.", context)
    assert result.grounded is False
    assert result.claims[0].support == 0.0


def test_one_bad_claim_fails_the_whole_answer(contract, context):
    mixed = "The json module serialises objects. GPT-9 runs at 400 teraflops."
    result = contract.check(mixed, context)
    assert result.grounded is False
    assert len(result.claims) == 2
    assert result.claims[0].supported is True
    assert result.claims[1].supported is False


def test_an_invented_number_is_expensive(contract, context):
    """Numbers are distinctive, so an absent one is weighted accordingly."""
    result = contract.check("Python raises TypeError 918273645.", context)
    assert result.claims[0].support < 0.5
    assert result.grounded is False


def test_a_fully_supported_answer_passes(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert result.grounded is True
    assert result.unsupported == ()


def test_every_supported_claim_scores_one(contract, context):
    for claim in contract.check(GROUNDABLE, context).claims:
        assert claim.support == pytest.approx(1.0)


def test_is_grounded_is_the_boolean_form(contract, context):
    assert contract.is_grounded(GROUNDABLE, context) is True
    assert contract.is_grounded(FABRICATED, context) is False


def test_the_verdict_requires_every_claim(contract, context):
    assert contract.check(GROUNDABLE, context).grounded is True
    assert contract.check(FABRICATED, context).grounded is False


def test_a_claim_must_be_a_sentence(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert len(result.claims) == 2
    assert result.claims[0].text.endswith(".")


def test_support_is_bounded_to_the_unit_interval(contract, context):
    samples = [
        GROUNDABLE,
        FABRICATED,
        "The json module alone.",
        "Python json TypeError module objects serialises wrong argument type.",
    ]
    for answer in samples:
        for claim in contract.check(answer, context).claims:
            assert 0.0 <= claim.support <= 1.0, claim


def test_a_near_quote_scores_higher_than_a_paraphrase(contract, context):
    """Stated limit: support is lexical, so a paraphrase costs something."""
    near = contract.check("The json module serialises Python objects.", context)
    paraphrased = contract.check("JSON conversion turns Python values into text.", context)
    assert near.claims[0].support > paraphrased.claims[0].support


def test_the_report_names_each_claim_and_the_verdict(contract, context):
    report = contract.check(FABRICATED, context).report()
    assert "FAIL" in report
    assert "NOT GROUNDED" in report


def test_the_result_exposes_its_claims(contract, context):
    result = contract.check(GROUNDABLE, context)
    assert all(isinstance(c, Claim) for c in result.claims)
    assert isinstance(result, GroundednessResult)


# =========================================================================== #
# Cross-component: the two agree on numbering
# =========================================================================== #


def test_a_prompt_built_here_resolves_in_citation_formatter(builder, context):
    """prompt-builder numbers [1]..[n]; citation-formatter resolves exactly that."""
    citation_dir = ROOT / "components" / "context" / "citation_formatter"
    sys.path.insert(0, str(citation_dir))
    from citation_formatter import CitationFormatter

    prompt = builder.build("What is a TypeError?", context)
    answer = prompt + " Python raises TypeError [1]."
    CitationFormatter().validate(core.Context(
        text=answer, chunks=context.chunks, token_count=9, budget=context.budget,
    ))  # must not raise


def test_a_prompt_builder_and_contract_compose(builder, contract, context):
    prompt = builder.build("What does the json module do?", context)
    assert "[1]" in prompt and "[2]" in prompt
    assert contract.is_grounded("The json module serialises Python objects.", context)


# =========================================================================== #
# Failure paths -- prompt-builder
# =========================================================================== #


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_a_blank_query_is_rejected(builder, context, query):
    with pytest.raises(PromptInputError) as exc:
        builder.build(query, context)
    assert "prompt-builder" in str(exc.value)


@pytest.mark.parametrize("query", [None, 3, ["q"]])
def test_a_non_string_query_is_rejected(builder, context, query):
    with pytest.raises(PromptInputError):
        builder.build(query, context)


def test_a_non_context_is_rejected(builder):
    with pytest.raises(PromptInputError) as exc:
        builder.build("q", "not a context")
    assert "must be a Context" in str(exc.value)


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
        render_sources([EVIDENCE[0], "nope"])
    assert "[1]" in str(exc.value)


def test_the_default_instruction_is_not_empty():
    assert DEFAULT_INSTRUCTION.strip()


# =========================================================================== #
# Failure paths -- grounded-answer-contract
# =========================================================================== #


@pytest.mark.parametrize("answer", ["", "   ", "\t\n"])
def test_an_empty_answer_is_rejected(contract, context, answer):
    """An empty answer makes no claims and would pass vacuously."""
    with pytest.raises(GroundingInputError) as exc:
        contract.check(answer, context)
    assert "vacuously" in str(exc.value)


@pytest.mark.parametrize("answer", [None, 3, ["a"]])
def test_a_non_string_answer_is_rejected(contract, context, answer):
    with pytest.raises(GroundingInputError):
        contract.check(answer, context)


def test_a_non_context_is_rejected(contract):
    with pytest.raises(GroundingInputError) as exc:
        contract.check("an answer", "not a context")
    assert "must be a Context" in str(exc.value)


def test_an_empty_context_is_rejected(contract):
    """Every claim failing tells the caller nothing they did not know."""
    with pytest.raises(GroundingInputError) as exc:
        contract.check("an answer", ctx([]))
    assert "no passages" in str(exc.value)


@pytest.mark.parametrize("value", [1.5, -0.1, 2, True, "0.5", None, float("nan")])
def test_an_unusable_min_support_is_rejected(value):
    with pytest.raises(GroundingInputError):
        GroundednessContract(min_support=value)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5])
def test_a_valid_min_support_is_accepted(value):
    assert GroundednessContract(min_support=value).min_support == value


def test_a_non_frozenset_stopwords_is_rejected():
    with pytest.raises(GroundingInputError):
        GroundednessContract(stopwords=set())


@pytest.mark.parametrize("name", ["", None, 5])
def test_an_unusable_contract_name_is_rejected(name):
    with pytest.raises(GroundingInputError):
        GroundednessContract(name=name)


def test_tokenize_rejects_a_non_string():
    with pytest.raises(GroundingInputError):
        tokenize(3)


def test_the_grounding_errors_form_a_hierarchy():
    assert issubclass(GroundingInputError, GroundingError)
    assert not issubclass(GroundingError, ValueError)


# =========================================================================== #
# The threshold is a real knob, and lower means more lenient
# =========================================================================== #


def test_a_low_threshold_admits_a_weak_claim(contract, context):
    weak = "The json module does something with objects perhaps."
    lenient_result = GroundednessContract(min_support=0.0).check(weak, context)
    strict_result = GroundednessContract(min_support=1.0).check(weak, context)
    assert lenient_result.grounded is True
    assert strict_result.grounded is False
    assert 0.0 < strict_result.claims[0].support < 1.0  # weak, not fabricated


def test_a_zero_support_claim_fails_at_every_positive_threshold(context):
    """A fabricated claim stays fabricated however low the bar goes.

    ``min_support`` is a plain threshold with no special case for zero, so
    ``0.0`` disables the check entirely -- an opt-out, not a stricter setting.
    """
    fabricated = (
        "Python raises TypeError for a wrong argument type. "
        "The Zephyr framework uses quantum entanglement for IPC."
    )
    for threshold in (0.01, 0.25, 0.5, 1.0):
        contract = GroundednessContract(min_support=threshold)
        result = contract.check(fabricated, context)
        claim = [c for c in result.claims if "Zephyr" in c.text][0]
        assert claim.support == 0.0
        assert claim.supported is False, threshold
        assert result.grounded is False, threshold


def test_min_support_zero_disables_the_check_entirely(context):
    """Documented, not accidental: 0.0 is an explicit opt-out."""
    fabricated = "The Zephyr framework uses quantum entanglement for IPC."
    result = GroundednessContract(min_support=0.0).check(fabricated, context)
    assert result.claims[0].support == 0.0
    assert result.claims[0].supported is True
    assert result.grounded is True


# =========================================================================== #
# Network-free by design (FR-10)
# =========================================================================== #


def test_no_generation_module_imports_a_provider_client():
    """No LLM client exists in this library in v1."""
    banned = {"openai", "anthropic", "httpx", "requests", "aiohttp", "urllib3"}
    for module_name in ("prompt_builder", "grounded_answer"):
        module = sys.modules[module_name]
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & banned, f"{module_name} imports {imported & banned}"


def test_the_prompt_builder_imports_only_core_types():
    module = sys.modules[PromptBuilder.__module__]
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"core_types", "__future__", "collections"}


def test_the_two_components_do_not_import_each_other():
    """Each is copy-ready on its own."""
    module = sys.modules[GroundednessContract.__module__]
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "prompt_builder" not in imported


# =========================================================================== #
# Manifests
# =========================================================================== #


def test_the_manifests_declare_the_core_types_dependency():
    for directory in ("prompt_builder", "grounded_answer"):
        manifest = json.loads(
            (GENERATION / directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["dependencies"] == ["core-types"], directory
        assert manifest["category"] == "generation", directory
        assert manifest["optional_packages"] == [], directory
        assert manifest["required_packages"] == [], directory
        assert manifest["external_services"] == [], directory

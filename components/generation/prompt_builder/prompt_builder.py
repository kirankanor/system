"""Turn a query and a built ``Context`` into one grounded prompt.

What this component is for
--------------------------
Between context assembly and generation there is a formatting step that is
easy to get wrong in a way that silently costs grounding: the passages have to
be *labelled* so the model can refer to them. An unlabelled context is a wall
of text the model must quote verbatim to point at anything, and a model that
cannot point will paraphrase -- which is exactly how an answer drifts away from
its evidence while sounding perfectly confident.

So this builds a prompt in which every passage carries a number, and the
instruction tells the model to use those numbers.

The numbering matches the context exactly
-----------------------------------------
``[1]`` here means ``context.chunks[0]``, and the numbers appear in the order
the passages appear in ``context.text``. That is the same convention
``citation-formatter`` validates against, so a prompt built here and an answer
checked there agree without either component knowing about the other.

That is the whole contract: build the prompt, let the model answer, then run
the answer past :mod:`citation_formatter` and :mod:`grounded_answer`.

No context, no prompt
---------------------
A missing ``Context``, and an empty one, are both rejected with a typed error
rather than producing a prompt. "Answer using ONLY the sources below" with
nothing below is a trap: the model will answer anyway, and every word of it
will be unsourced. A caller that genuinely wants an open-ended answer wants a
different prompt, and should not reach for a grounded one to get it.

This component is network-free and provider-free by design (``FR-10``): it
returns a string. Sending that string to a model is the caller's business, and
no LLM client exists anywhere in this library in v1.
"""

from __future__ import annotations

from collections.abc import Sequence

from core_types import Context, ScoredDocument

__all__ = [
    "DEFAULT_INSTRUCTION",
    "PromptBuilder",
    "PromptError",
    "PromptInputError",
    "render_sources",
]

#: The instruction the model is given. Deliberately explicit about citing,
#: because a model that is not told to cite will not. It contains no literal
#: bracketed numbers: an example like "[1]" in the instruction reads exactly
#: like a real citation once the prompt is assembled.
DEFAULT_INSTRUCTION = """Answer the question using ONLY the numbered sources below.

Rules:
- Cite the sources you use by their number in square brackets, combining several in one bracket when they agree.
- Never invent a source number. Every number below corresponds to exactly one passage.
- If the sources do not contain the answer, say so plainly instead of guessing.
- Do not use prior knowledge, even if you are sure of it."""


class PromptError(Exception):
    """Base for every failure in this component."""


class PromptInputError(PromptError):
    """The caller passed something this component cannot turn into a prompt."""


def render_sources(chunks: Sequence[ScoredDocument]) -> list[str]:
    """One ``[n] passage`` line per chunk, numbered from 1, in order.

    Public so a caller can render the same numbering in a UI -- a footnote, a
    hover card -- without reimplementing the convention and drifting from it.
    """
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
        raise PromptInputError(
            f"prompt-builder: chunks must be a sequence of ScoredDocument, got "
            f"{type(chunks).__name__}"
        )
    rendered: list[str] = []
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, ScoredDocument):
            raise PromptInputError(
                f"prompt-builder: chunks[{position}] is {type(chunk).__name__}, "
                f"expected ScoredDocument"
            )
        rendered.append(f"[{position + 1}] {chunk.text}")
    return rendered


class PromptBuilder:
    """Assemble a grounded prompt from a query and a ``Context``.

    ``instruction`` replaces the default wholesale rather than being appended
    to, so a caller with a house style is not fighting a template. It is
    validated for being a non-empty string, and nothing more -- this component
    does not second-guess the caller's prompt.

    The rendered prompt is deterministic: same query, same context, same
    string. No timestamps, no ordering by score, no randomness.
    """

    def __init__(
        self,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        question_label: str = "Question",
        name: str = "prompt-builder",
    ) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise PromptInputError(
                f"prompt-builder: instruction must be a non-empty string, got "
                f"{instruction!r}"
            )
        if not isinstance(question_label, str) or not question_label:
            raise PromptInputError(
                f"prompt-builder: question_label must be a non-empty string, "
                f"got {question_label!r}"
            )
        if not isinstance(name, str) or not name:
            raise PromptInputError(
                f"prompt-builder: name must be a non-empty string, got {name!r}"
            )
        self.instruction = instruction
        self.question_label = question_label
        self.name = name

    def _validate_query(self, query: object) -> str:
        if not isinstance(query, str):
            raise PromptInputError(
                f"prompt-builder: query must be a string, got "
                f"{type(query).__name__}"
            )
        if not query.strip():
            raise PromptInputError(
                f"prompt-builder: query is empty or whitespace. A grounded "
                f"prompt with no question gives the model nothing to answer"
            )
        return query

    def _validate_context(self, context: object) -> Context:
        if context is None:
            raise PromptInputError(
                f"prompt-builder: context is required. Without it there is "
                f"nothing to ground the answer in, and a prompt that says "
                f"'use only the sources below' with no sources invites the "
                f"model to hallucinate"
            )
        if not isinstance(context, Context):
            raise PromptInputError(
                f"prompt-builder: context must be a Context, got "
                f"{type(context).__name__}"
            )
        if not context.chunks:
            raise PromptInputError(
                f"prompt-builder: context has no passages (token_count="
                f"{context.token_count}, budget={context.budget}). A grounded "
                f"prompt needs grounding; an empty context should reach the "
                f"answer stage as an abstain decision, not as a prompt"
            )
        return context

    def build(self, query: str, context: Context) -> str:
        """Render the grounded prompt.

        Every passage in ``context.chunks`` gets exactly one number, and the
        numbers are sequential from 1 -- so an answer citing ``[n]`` can always
        be resolved, and one citing a number that is not there can always be
        caught.
        """
        text = self._validate_query(query)
        ctx = self._validate_context(context)
        sources = "\n".join(render_sources(ctx.chunks))
        return (
            f"{self.instruction}\n\n"
            f"Sources:\n{sources}\n\n"
            f"{self.question_label}: {text}\n\n"
            f"Answer:"
        )

    def numbers(self, context: Context) -> list[int]:
        """The citation numbers this prompt will make available, in order.

        Lets a caller check an answer's citations, or render a legend, before
        any model has run.
        """
        count = len(self._validate_context(context).chunks)
        return list(range(1, count + 1))

    def __repr__(self) -> str:
        return f"PromptBuilder(name={self.name!r}, question_label={self.question_label!r})"

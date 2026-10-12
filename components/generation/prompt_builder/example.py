"""Runnable example for `prompt-builder`.

    python example.py

Shows a grounded prompt being assembled, and the numbering that makes an
answer's citations resolvable later. Runs standalone with `core_types.py`
beside it -- the manifest declares that component a dependency, so
`ragkit add` copies it along.

Nothing is installed to make this run. This component returns a string: no
LLM client, no transport, no network. Sending the string to a model is the
caller's business, which is why this works with any provider or none.

The five guarantees this example demonstrates:

1. every supplied source gets exactly one number, sequential from 1;
2. no number beyond the last source appears anywhere in the prompt;
3. a source and its number share one line, in context order;
4. the instruction itself contains no literal citation of its own;
5. a missing or empty context is rejected, never rendered.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# core-types sits in its own component directory under components/storage.
# `ragkit add` copies it beside this file; here we reach across to the sibling
# so the example runs from the component tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "storage" / "core_types")
)

import core_types as core  # noqa: E402

from prompt_builder import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    PromptBuilder,
    PromptError,
    PromptInputError,
    render_sources,
)

SOURCES = [
    core.ScoredDocument(chunk_id="c0", source_name="python/errors.md",
                        text="Python raises TypeError when a function receives an "
                             "argument of the wrong type.",
                        score=0.9, chunk_index=0, metadata={}),
    core.ScoredDocument(chunk_id="c1", source_name="json/module.md",
                        text="The json module serialises Python objects into JSON text.",
                        score=0.8, chunk_index=1, metadata={}),
    core.ScoredDocument(chunk_id="c2", source_name="json/overview.md",
                        text="JSON text is easy for both humans and programs to read.",
                        score=0.7, chunk_index=2, metadata={}),
]


def context_of(chunks) -> core.Context:
    items = tuple(chunks)
    return core.Context(
        text="\n\n".join(c.text for c in items),
        chunks=items,
        token_count=5,
        budget=200,
    )


def main() -> None:
    builder = PromptBuilder()
    context = context_of(SOURCES)
    print(f"{builder}\n")

    prompt = builder.build("How does Python report a type error?", context)

    print("The prompt:")
    for line in prompt.splitlines():
        print(f"  | {line}")

    print("\n1. Every supplied source gets exactly one number, sequential from 1:")
    print(f"   numbers available : {builder.numbers(context)}")
    print(f"   all present       : "
          f"{all(f'[{n}]' in prompt for n in builder.numbers(context))}")

    print("\n2. No number beyond the last source appears:")
    present = sorted({int(n) for n in re.findall(r"\[(\d+)\]", prompt)})
    print(f"   bracketed numbers found: {present}")
    print(f"   matches the source count exactly: {present == [1, 2, 3]}")

    print("\n3. A source and its number share one line, in context order:")
    for number, rendered in enumerate(render_sources(SOURCES), start=1):
        print(f"   found line {number}: {rendered in prompt}  {rendered[:58]!r}")

    print("\n4. The instruction carries no literal citation of its own:")
    body = prompt
    for number, chunk in enumerate(SOURCES, start=1):
        body = body.replace(f"[{number}] {chunk.text}", "")
    leftover = re.findall(r"\[\d+\]", body)
    print(f"   bracketed numbers outside the sources: {leftover or 'none'}")
    print("   (an example like [1] would read as a real citation once assembled)")

    print("\n   The instruction itself:")
    for line in DEFAULT_INSTRUCTION.splitlines():
        print(f"     {line}")

    print("\n   It is deterministic -- same input, same string:")
    print(f"     build twice, equal: "
          f"{builder.build('q', context) == builder.build('q', context)}")

    print("\n   instruction= replaces wholesale rather than appending:")
    custom = PromptBuilder(instruction="Use the sources. Cite them.", question_label="Asked")
    custom_prompt = custom.build("q", context)
    print(f"     keeps its own: {'Use the sources. Cite them.' in custom_prompt}")
    print(f"     drops the default: {'Do not use prior knowledge' not in custom_prompt}")
    print(f"     relabels the question: {'Asked: q' in custom_prompt}")

    print("\n5. A missing or empty context is rejected, never rendered:")
    for label, call in [
        ("context=None", lambda: builder.build("q", None)),
        ("context with no passages", lambda: builder.build("q", context_of([]))),
        ("a blank query", lambda: builder.build("   ", context)),
        ("a non-Context", lambda: builder.build("q", "not a context")),
        ("a blank instruction", lambda: PromptBuilder(instruction="  ")),
    ]:
        try:
            call()
        except PromptError as exc:
            print(f"   {label:<26} -> {str(exc)[:62]}...")

    print("\n   Why an empty context is refused rather than rendered:")
    print("     'Answer using ONLY the sources below' with nothing below")
    print("     invites the model to answer anyway -- and every word unsourced.")

    print("\nThis is one link in a chain, none of which import each other:")
    print("   prompt-builder          numbers the passages  [1]..[n]")
    print("   citation-formatter      resolves those numbers, rejects orphans")
    print("   grounded-answer-contract scores claims against the evidence")
    print("   (the numbering convention is shared, not the code)")


if __name__ == "__main__":
    main()

"""Runnable example for `context-selector`.

    python example.py

Shows deduplication, budget packing, and the empty context. Runs standalone
with `core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

Nothing is installed to make this run. The token counter is injected, so this
runs anywhere Python does -- and so the budget guarantee is measured rather
than assumed.

The five guarantees this example demonstrates:

1. the same passage arriving twice is kept once, best copy first;
2. the token budget is never exceeded, at any budget;
3. the separator between passages is charged, so joins cannot overflow;
4. an input that is entirely duplicates still returns context;
5. a context too small for anything returns empty rather than a fragment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory under components/storage.
# `ragkit add` copies it beside this file; here we reach across to the sibling
# so the example runs from the component tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "storage" / "core_types")
)

import core_types as core  # noqa: E402

from context_selector import ContextInputError, ContextSelector  # noqa: E402

CANDIDATES = [
    core.ScoredDocument(chunk_id="c0", source_name="python/errors.md", text="alpha beta",
                        score=0.9, chunk_index=0, metadata={}),
    core.ScoredDocument(chunk_id="c1", source_name="cooking/pasta.md", text="gamma delta",
                        score=0.8, chunk_index=1, metadata={}),
    core.ScoredDocument(chunk_id="c2", source_name="java/vs.md", text="epsilon zeta",
                        score=0.7, chunk_index=2, metadata={}),
]


def words(text: str) -> int:
    """A stand-in counter, so this example needs no wheel. Swap in a real one:
    ContextSelector(counter=TokenBudget().count)."""
    return len(text.split())


def show(label, ctx):
    print(f"  {label}")
    print(f"    chunks     : {[c.chunk_id for c in ctx.chunks]}")
    print(f"    tokens     : {ctx.token_count} / budget {ctx.budget}  "
          f"(fits: {ctx.token_count <= ctx.budget})")
    print(f"    truncated  : {ctx.truncated}")
    print(f"    text       : {ctx.text!r}")


def main() -> None:
    selector = ContextSelector(counter=words)
    print(f"selector: {selector}")

    print("\n1. Selection keeps retrieval's order and stops when the budget is full:")
    show("select('alpha', 3 passages, budget 100)", selector.select("alpha", CANDIDATES, 100))

    print("\n2. The budget is never exceeded -- measured, at every budget:")
    for limit in range(1, 9):
        built = selector.select("alpha", CANDIDATES, limit)
        flag = "ok " if built.token_count <= built.budget else "OVER"
        print(f"    budget {limit} -> {len(built.chunks)} chunk(s), "
              f"{built.token_count} tokens, truncated={built.truncated}  [{flag}]")

    print("\n3. The separator is charged, so joins cannot overflow:")
    per_character = ContextSelector(counter=len, separator="\n\n")
    no_joins = ContextSelector(counter=len, separator="")
    print(f"    11 chars: with joins -> "
          f"{len(no_joins.select('q', CANDIDATES[:2], 100).text)} chars, "
          f"{len(per_character.select('q', CANDIDATES[:2], 100).text)} chars")
    print("    the joins are real text and are counted like any other")

    print("\n4. The same passage arriving twice is kept once:")
    duplicated = CANDIDATES + [
        core.ScoredDocument(chunk_id="sparse-3", source_name="python/errors.md",
                            text="alpha beta", score=0.6, chunk_index=3, metadata={}),
    ]
    show("dense + sparse both returned 'alpha beta'",
         selector.select("alpha", duplicated, 100))
    print("    -> the better-ranked copy survives")

    print("\n   Whitespace differences do not defeat it:")
    layout = [
        core.ScoredDocument(chunk_id="a", text="alpha   beta\n\ngamma", score=1.0,
                            chunk_index=0, metadata={}),
        core.ScoredDocument(chunk_id="b", text="alpha beta gamma", score=0.9,
                            chunk_index=1, metadata={}),
    ]
    print(f"    -> {[c.chunk_id for c in selector.select('q', layout, 100).chunks]}")

    print("\n5. An input that is ENTIRELY duplicates still returns context:")
    all_dup = [
        core.ScoredDocument(chunk_id=f"c{i}", text="repeated evidence", score=1.0,
                            chunk_index=i, metadata={})
        for i in range(5)
    ]
    show("five copies of one passage", selector.select("q", all_dup, 100))
    print("    -> not empty: one copy is still evidence")

    print("\n6. Too small a budget returns empty, never a fragment:")
    show("budget 1 (a passage costs 2)", selector.select("alpha", CANDIDATES, 1))
    print("    -> a half-passage is worse than none; the model cannot tell it is one")

    print("\nAn empty input is an empty context, not an error:")
    show("no candidates", selector.select("q", [], 10))

    print("\nThe counter defines the unit, and Context says so:")
    show("a character counter", ContextSelector(counter=len).select("q", CANDIDATES, 20))

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("no counter at all", lambda: ContextSelector()),
        ("a blank query", lambda: selector.select("   ", CANDIDATES, 10)),
        ("a budget of 0", lambda: selector.select("q", CANDIDATES, 0)),
        ("a non-document", lambda: selector.select("q", CANDIDATES + ["nope"], 10)),
    ]:
        try:
            call()
        except ContextInputError as exc:
            print(f"    {label:<20} -> {str(exc)[:58]}...")

    print("\nIt satisfies ContextBuilderProtocol, so it plugs in anywhere:")
    print(f"    isinstance(..., ContextBuilderProtocol) -> "
          f"{isinstance(selector, core.ContextBuilderProtocol)}")


if __name__ == "__main__":
    main()

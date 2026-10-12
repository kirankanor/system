"""Runnable example for `token-budget`.

    python example.py

Shows measuring and cutting text against a budget, and the guarded-dependency
contract.

This component imports nothing from the library, so there is no sibling to
reach for -- it runs exactly as it would after `ragkit add` copies it out.

Nothing is installed to make this run. `tiktoken` is optional and is NOT
installed in this repository, so:

* the real tokenizer is never exercised here;
* counting is demonstrated with an injected counter, which is exactly how a
  caller with a HuggingFace tokenizer or a remote counting service uses it;
* the missing-dependency path IS exercised for real, because the wheel is
  genuinely absent.

The six guarantees this example demonstrates:

1. the module constructs with the wheel absent, and loads it lazily;
2. a supplied counter is used, and tiktoken is never imported;
3. `truncate` never returns more tokens than asked for;
4. `truncate` cuts at a word boundary, not mid-word;
5. `pack` counts the separator between items, so the total really fits;
6. a missing wheel raises ImportError naming the package, verbatim.
"""

from __future__ import annotations

from token_budget import (
    MANIFEST_MESSAGE,
    MissingDependency,
    TokenBudget,
    TokenBudgetInputError,
    word_count,
)

ARTICLE = (
    "Retrieval augmented generation fetches passages, assembles them into a "
    "single context, and asks a model to answer from that evidence. The "
    "context window is finite, so something has to decide what fits."
)

PASSAGES = [
    "alpha beta gamma",
    "delta epsilon",
    "zeta eta theta iota",
]


def main() -> None:
    print("word_count is a NAMED approximation -- never a silent default:")
    print(f"  word_count('a  b\\tc\\nd') -> {word_count('a  b\tc\nd')}")

    print("\n1. The module constructs with the wheel absent:")
    lazy = TokenBudget()
    print(f"   {lazy}")
    print("   (nothing imported yet -- the load happens on first count)")

    print("\n2. A supplied counter is used, and tiktoken is never imported:")
    budget = TokenBudget(counter=word_count)
    print(f"   count(ARTICLE) -> {budget.count(ARTICLE)} words")
    print(f"   is_loaded()    -> {budget.is_loaded()}")

    print("\n   The counter is any callable. Characters work too:")
    per_character = TokenBudget(counter=len)
    print(f"   count(ARTICLE) -> {per_character.count(ARTICLE)} characters")

    print("\n3. truncate never returns more than asked for:")
    for limit in (5, 10, 20):
        cut = budget.truncate(ARTICLE, limit)
        print(f"   budget {limit:>2} -> {budget.count(cut):>2} tokens, "
              f"{len(cut):>2} chars  (never over: {budget.count(cut) <= limit})")

    print("\n4. It cuts at a word boundary, not mid-word:")
    cut = budget.truncate(ARTICLE, 10)
    print(f"   ends with ...{cut[-18:]!r}")

    print("\n5. pack counts the separator, so the total really fits:")
    for limit in (5, 8, 20):
        taken = budget.pack(PASSAGES, limit)
        joined = "\n\n".join(PASSAGES[i] for i in taken)
        print(f"   budget {limit:>2} -> items {taken}, total {budget.count(joined):>2} "
              f"(fits: {budget.count(joined) <= limit})")
    print("   the separator is charged, which is where naive contexts overshoot")

    print("\n   remaining() is negative when already over:")
    print(f"   remaining(ARTICLE, 3) -> {budget.remaining(ARTICLE, 3)}")

    print("\n6. A missing wheel raises ImportError naming the package:")
    try:
        TokenBudget().count(ARTICLE)
    except MissingDependency as exc:
        print(f"   caught  : {type(exc).__name__} (an ImportError: {isinstance(exc, ImportError)})")
        print(f"   message : {exc}")
        print(f"   verbatim from the manifest: {str(exc) == MANIFEST_MESSAGE}")
        print(f"   __cause__ is the original ImportError: {isinstance(exc.__cause__, ImportError)}")

    print("\nThis is what makes token-budget and context-selector compose:")
    print("   ContextSelector(counter=TokenBudget().count)  <- your counter, their budget")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("a budget of 0", lambda: budget.fits("x", 0)),
        ("a bool budget", lambda: budget.fits("x", True)),
        ("a non-string", lambda: budget.count(3)),
        ("an empty encoding", lambda: TokenBudget(counter=word_count, encoding="")),
    ]:
        try:
            call()
        except TokenBudgetInputError as exc:
            print(f"   {label:<18} -> {str(exc)[:56]}...")


if __name__ == "__main__":
    main()

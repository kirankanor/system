"""Assemble retrieved chunks into one context that fits a token budget.

Implements ``ContextBuilderProtocol`` from core-types, so it plugs in
anywhere a ``ContextBuilder`` is expected.

What it does, and what it deliberately does not
-----------------------------------------------
Between retrieval and generation there is a gap: a retriever returns a ranked
list of *separate* passages, and a model needs *one* string, short enough for
its context window. This component closes that gap and does three things:

1. **Deduplicates.** Retrieval routinely returns the same passage twice --
   once from dense, once from sparse, once after a reranker that did not know
   they were the same. Showing a model its own evidence twice is not
   harmless: it double-counts in attention and inflates the sense that many
   sources agree.
2. **Selects greedily, in the order given.** The caller already ranked these
   chunks. Selection keeps that order and stops at the first passage that
   does not fit, because skipping ahead to a shorter passage would silently
   reorder the evidence the retriever chose.
3. **Never exceeds the budget.** By measuring, not estimating.

**It selects; it does not truncate.** A passage too large to fit is skipped
and ``truncated`` is set, rather than cut mid-sentence. Cutting text is
:mod:`token_budget`'s job, and a half-passage is usually worse than none --
the model cannot tell it is looking at a fragment. A caller that wants a
single oversized passage included can truncate it first and pass the result.

The counter is required, and that is the point
----------------------------------------------
There is no default token counter here, on purpose. ``len(text) // 4`` is a
plausible-looking estimate that overshoots on code, tables and non-English
text -- precisely where overshooting is most expensive, because models drop
the far end of the context, where the answer was.

So the caller must say what a token is::

    budget = TokenBudget()
    ContextSelector(counter=budget.count)

:class:`token_budget.TokenBudget` is one answer to that; a HuggingFace
tokenizer's callable is another; ``token_budget.word_count`` is a cheap one
for tests. None of them is assumed.

``Context.token_count`` and ``Context.budget`` are in the counter's unit, so
a returned ``Context`` is self-describing about what it was measured in.

The query is part of the protocol signature
-------------------------------------------
``build`` takes a query because ``ContextBuilderProtocol`` says so. This
selector does not use it: selection is driven by the order retrieval already
produced, and inventing query-dependent reordering here would contradict
every ranking done upstream. It is validated and then not applied, which is
said plainly rather than hidden behind a parameter that does nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from core_types import Context, ScoredDocument

__all__ = [
    "ContextError",
    "ContextInputError",
    "ContextSelector",
]

#: Read once at import, for the documented messages.
MANIFEST_MESSAGE = json.loads(
    (Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8")
)["missing_dependency_message"]


class ContextError(Exception):
    """Base for every failure in this component."""


class ContextInputError(ContextError):
    """The caller passed something this component cannot assemble."""


def _validate_budget(budget: object) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ContextInputError(
            f"context-selector: budget must be an int, got "
            f"{type(budget).__name__} ({budget!r})"
        )
    if budget < 1:
        raise ContextInputError(
            f"context-selector: budget must be at least 1 token, got {budget}"
        )
    return budget


def _validate_counter(counter: object) -> Callable[[str], int]:
    if counter is None:
        raise ContextInputError(
            "context-selector: counter is required. It must be callable, "
            "taking a string and returning a token count -- for example "
            "`TokenBudget().count` from the token-budget component. It is not "
            "optional because a guessed token count (len(text)//4, say) "
            "overshoots real budgets on code, tables and non-English text, "
            "and the overshoot lands at the far end of the context, where "
            "the answer was"
        )
    if not callable(counter):
        raise ContextInputError(
            f"context-selector: counter must be callable, taking a string and "
            f"returning an int, got {type(counter).__name__}"
        )
    return counter


def _normalise(text: str) -> str:
    """Collapse whitespace so layout differences do not defeat dedup."""
    return " ".join(text.split())


class ContextSelector:
    """Deduplicate, then pack ranked chunks into a ``Context``.

    ``counter`` is required -- see the module docstring. It is called once per
    candidate and once per separator, and its unit becomes ``Context``'s.

    ``separator`` is inserted between passages and **is counted**, which is
    the classic way a context overshoots: the passages fit, the joins do not.
    """

    def __init__(
        self,
        *,
        counter: Callable[[str], int] | None = None,
        separator: str = "\n\n",
        name: str = "context-selector",
    ) -> None:
        self._counter = _validate_counter(counter)
        if not isinstance(separator, str):
            raise ContextInputError(
                f"context-selector: separator must be a string, got "
                f"{type(separator).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise ContextInputError(
                f"context-selector: name must be a non-empty string, got {name!r}"
            )
        self.separator = separator
        self.name = name

    def _count(self, text: str) -> int:
        result = self._counter(text)
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise ContextError(
                f"context-selector: the counter returned {result!r}, which is "
                f"not a non-negative int"
            )
        return result

    def deduplicate(
        self, chunks: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        """First occurrence of each distinct passage, in order.

        Two things make a passage a duplicate:

        * its ``chunk_id`` has already been seen;
        * its text, whitespace-collapsed, has already been seen.

        The second catches the real case -- the same passage returned by two
        retrievers under two ids. Keeping the first occurrence preserves the
        best-ranked copy.
        """
        items = self._validate_chunks(chunks)
        seen_ids: set[str] = set()
        seen_text: set[str] = set()
        kept: list[ScoredDocument] = []
        for item in items:
            key_id = item.chunk_id if item.chunk_id is not None else ""
            key_text = _normalise(item.text)
            if (key_id and key_id in seen_ids) or key_text in seen_text:
                continue
            if key_id:
                seen_ids.add(key_id)
            seen_text.add(key_text)
            kept.append(item)
        return kept

    def _validate_chunks(self, chunks: object) -> list[ScoredDocument]:
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise ContextInputError(
                f"context-selector: chunks must be a sequence of "
                f"ScoredDocument, got {type(chunks).__name__}"
            )
        items = list(chunks)
        for position, item in enumerate(items):
            if not isinstance(item, ScoredDocument):
                raise ContextInputError(
                    f"context-selector: chunks[{position}] is "
                    f"{type(item).__name__}, expected ScoredDocument"
                )
        return items

    def select(
        self, query: str, chunks: Sequence[ScoredDocument], budget: int
    ) -> Context:
        """Deduplicate and pack ``chunks`` into one ``Context``.

        Returns an empty ``Context`` -- ``text=""``, ``chunks=()`` -- rather
        than raising, when nothing fits or nothing was passed. The answer
        stage is the one that decides whether an empty context means abstain;
        a selector that raised here would force every caller to special-case
        the same condition.
        """
        if not isinstance(query, str):
            raise ContextInputError(
                f"context-selector: query must be a string, got "
                f"{type(query).__name__}"
            )
        if not query.strip():
            raise ContextInputError(
                f"context-selector: query is empty or whitespace. Nothing is "
                f"being selected for, which is a caller bug rather than a "
                f"legitimate empty result"
            )
        limit = _validate_budget(budget)

        candidates = self.deduplicate(chunks)
        if not candidates:
            return Context(text="", chunks=(), token_count=0, budget=limit)

        taken: list[ScoredDocument] = []
        used = 0
        stopped = False
        for item in candidates:
            cost = self._count(item.text)
            if taken:
                cost += self._count(self.separator)
            if used + cost > limit:
                stopped = True
                break
            used += cost
            taken.append(item)

        text = self.separator.join(item.text for item in taken)
        return Context(
            text=text,
            chunks=tuple(taken),
            token_count=used,
            budget=limit,
            truncated=stopped,
        )

    def build(
        self, query: str, chunks: Sequence[ScoredDocument], budget: int
    ) -> Context:
        """``ContextBuilderProtocol``'s name for :meth:`select`."""
        return self.select(query, chunks, budget)

    def __repr__(self) -> str:
        return f"ContextSelector(name={self.name!r}, separator={self.separator!r})"

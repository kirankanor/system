"""Count, measure against, and cut text to a model's token budget.

Why this is a component and not a helper
----------------------------------------
Every RAG pipeline eventually has to answer "how much context fits?". The
answer depends entirely on the model: the same passage is ~240 tokens to
`cl100k_base` and a different number to any other encoding, and a budget
guessed with `len(text) // 4` will quietly overshoot on code, tables and
non-English text -- which is exactly where an overshoot is most expensive,
because the model truncates the far end of the context, where the answer was.

So this component refuses to guess. It counts with a real tokenizer, and it
says plainly what it counted with.

The dependency is optional, and genuinely guarded
------------------------------------------------
`tiktoken` is the reference tokenizer for OpenAI-family models and is a
reasonable default -- but it is not the only tokenizer, and a library that
hard-requires one would be wrong for everyone using something else. So:

* ``counter=`` accepts any ``Callable[[str], int]``, and when given, tiktoken
  is never imported;
* without it, the encoding is loaded **lazily, on first count**;
* a missing wheel raises ``MissingDependency`` -- an ``ImportError`` -- whose
  message is the manifest's ``missing_dependency_message`` **verbatim**.

A caller counting with HuggingFace tokenizers, or with a word count during
tests, never touches the wheel.

The guarantee, stated honestly
------------------------------
``truncate`` and ``pack`` never return more tokens than asked for. That is
enforced by *measuring the result*, not by predicting it -- the returned
prefix is the one that was counted and found to fit.

Truncation cuts at a character boundary found by binary search, then walks
back to a word boundary when one is available, because handing a model a
context ending in `handl` costs more than the tokens it saves. Binary search
assumes token count rises with text length; BPE merges make that very nearly
true but not strictly guaranteed, so the search finds a prefix that *fits*
rather than claiming to find the longest one. It is never too long. It may,
on pathological input, be shorter than the theoretical maximum.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ENCODING",
    "MissingDependency",
    "TokenBudget",
    "TokenBudgetError",
    "TokenBudgetInputError",
    "word_count",
]

#: Read once at import; the wheel is never touched to produce this message.
MANIFEST_MESSAGE = json.loads(
    (Path(__file__).resolve().parent / "manifest.json").read_text(encoding="utf-8")
)["missing_dependency_message"]

DEFAULT_ENCODING = "cl100k_base"


class TokenBudgetError(Exception):
    """Base for every failure in this component."""


class TokenBudgetInputError(TokenBudgetError):
    """The caller passed something this component cannot measure."""


class MissingDependency(TokenBudgetError, ImportError):
    """``tiktoken`` is not installed and no ``counter`` was supplied.

    Subclasses ``ImportError`` so a caller catching the standard exception
    works. Deliberately *not* a subclass of ``ValueError`` or ``TypeError``:
    the machine is the problem, not the caller's arguments.
    """


def word_count(text: str) -> int:
    """Count whitespace-separated words. A named, honest approximation.

    Public and exported so that a caller who wants a cheap counter can
    *choose* it explicitly rather than inheriting a silent one. It is a word
    count, not a token count: for a real model budget pass a real counter.
    """
    if not isinstance(text, str):
        raise TokenBudgetInputError(
            f"token-budget: word_count expects a string, got {type(text).__name__}"
        )
    return len(text.split())


def _load_encoding(name: str) -> Any:
    try:
        import tiktoken  # noqa: PLC0415
    except ImportError as exc:
        raise MissingDependency(MANIFEST_MESSAGE) from exc
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - unknown encoding name
        raise TokenBudgetError(
            f"token-budget: tiktoken has no encoding named {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _validate_budget(budget: object) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TokenBudgetInputError(
            f"token-budget: budget must be an int, got {type(budget).__name__} "
            f"({budget!r})"
        )
    if budget < 1:
        raise TokenBudgetInputError(
            f"token-budget: budget must be at least 1 token, got {budget}. A "
            f"zero budget cannot hold any context, so it should not be a budget"
        )
    return budget


def _validate_text(text: object, *, label: str) -> str:
    if not isinstance(text, str):
        raise TokenBudgetInputError(
            f"token-budget: {label} must be a string, got {type(text).__name__}"
        )
    return text


class TokenBudget:
    """Measure and cut text against a specific model's tokenizer.

    Pass ``counter=`` to count with anything you like -- a HuggingFace
    tokenizer's callable, a service's endpoint, a word count in tests. When
    given, tiktoken is never imported, so this object works on a machine
    without the wheel.

    Without ``counter``, the ``encoding`` is loaded lazily on first use, so
    constructing a budget is free.

    ``count`` is itself a valid ``counter`` for another ``TokenBudget``, and
    for :class:`context_selector.ContextSelector` -- that is the intended way
    to wire a real tokenizer into context assembly.
    """

    def __init__(
        self,
        *,
        encoding: str = DEFAULT_ENCODING,
        counter: Callable[[str], int] | None = None,
        name: str = "token-budget",
    ) -> None:
        if not isinstance(encoding, str) or not encoding:
            raise TokenBudgetInputError(
                f"token-budget: encoding must be a non-empty string, got {encoding!r}"
            )
        if counter is not None and not callable(counter):
            raise TokenBudgetInputError(
                f"token-budget: counter must be a callable taking a string and "
                f"returning an int, got {type(counter).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise TokenBudgetInputError(
                f"token-budget: name must be a non-empty string, got {name!r}"
            )
        self.encoding = encoding
        self.name = name
        self._counter = counter
        self._tokenizer: Any | None = None

    def is_loaded(self) -> bool:
        """Whether the tokenizer is loaded, without triggering a load."""
        return self._counter is not None or self._tokenizer is not None

    def _ensure_counter(self) -> Callable[[str], int]:
        if self._counter is not None:
            return self._counter
        if self._tokenizer is None:
            encoding = _load_encoding(self.encoding)
            # disallowed_special=() so text containing a literal special token
            # such as "<<" is counted rather than raising.
            self._tokenizer = lambda text: len(  # noqa: E731
                encoding.encode(text, disallowed_special=())
            )
        return self._tokenizer

    def count(self, text: str) -> int:
        """Token count of ``text`` under this budget's counter."""
        value = _validate_text(text, label="text")
        result = self._ensure_counter()(value)
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise TokenBudgetError(
                f"token-budget: the counter returned {result!r}, which is not a "
                f"non-negative int. A counter must return a token count"
            )
        return result

    def fits(self, text: str, budget: int) -> bool:
        """Whether ``text`` is within ``budget`` tokens."""
        return self.count(text) <= _validate_budget(budget)

    def remaining(self, text: str, budget: int) -> int:
        """Tokens left under ``budget``; negative when already over."""
        return _validate_budget(budget) - self.count(text)

    def truncate(self, text: str, budget: int, *, at_word_boundary: bool = True) -> str:
        """The longest prefix of ``text`` that fits ``budget``, or all of it.

        The result is measured before it is returned, so the guarantee holds
        for any counter -- including one whose token count is not perfectly
        monotonic in text length.

        ``at_word_boundary`` walks the cut back to whitespace, because a
        context ending mid-word reads badly and costs tokens to no benefit.
        When no whitespace exists in the cut region the character boundary
        stands.
        """
        value = _validate_text(text, label="text")
        limit = _validate_budget(budget)
        if self.count(value) <= limit:
            return value

        low, high, best = 0, len(value), ""
        while low <= high:
            middle = (low + high) // 2
            candidate = value[:middle]
            if self.count(candidate) <= limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1

        if not at_word_boundary or not best:
            return best

        # Walk back to the last whitespace. Re-measure, because a shorter
        # prefix is not guaranteed to hold fewer tokens for every tokenizer.
        trimmed = best.rstrip()
        boundary = trimmed.rfind(" ")
        if boundary == -1:
            return best
        shortened = trimmed[: boundary + 1].rstrip()
        return shortened if self.count(shortened) <= limit else best

    def pack(
        self,
        texts: Sequence[str],
        budget: int,
        *,
        separator: str = "\n\n",
    ) -> list[int]:
        """Indices of the items that fit ``budget``, in order.

        Greedy, in the order given: an item that does not fit ends packing,
        because a later, shorter item would reorder the evidence. That is the
        caller's decision to make, not this component's.

        The separator between items is counted, so the total never exceeds the
        budget by the joins alone -- the classic way a context overshoots.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TokenBudgetInputError(
                f"token-budget: texts must be a sequence of strings, got "
                f"{type(texts).__name__}"
            )
        limit = _validate_budget(budget)
        _validate_text(separator, label="separator")

        taken: list[int] = []
        used = 0
        for position, item in enumerate(texts):
            value = _validate_text(item, label=f"texts[{position}]")
            cost = self.count(value)
            if taken:
                cost += self.count(separator)
            if used + cost > limit:
                break
            used += cost
            taken.append(position)
        return taken

    def __repr__(self) -> str:
        return (
            f"TokenBudget(encoding={self.encoding!r}, name={self.name!r}, "
            f"loaded={self.is_loaded()})"
        )

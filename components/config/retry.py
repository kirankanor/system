"""Bounded, configurable retry for transient failures (FR-57).

Why here, and not inside a component
------------------------------------
``max_retries`` existed on ``EmbeddingSettings`` and ``LLMSettings`` and was
read by nothing: configuration that promised behaviour it never delivered.
The obvious repair -- a loop around the embedding call and another around the
storage call -- would put control flow inside components, and components are
meant to be copyable on their own with their own exceptions. Every component
would reinvent the loop and every caller would have to trust each one.

So the loop lives in the config layer, which the orchestration code imports
and components deliberately do not (see ``config/__init__.py``). One
implementation serves every stage, and no component grows control flow it did
not ask for.

What counts as worth retrying
-----------------------------
The decision already exists and is not this module's to make: ``RagError``
carries ``retryable`` and ``is_retryable`` reads it. A timeout or a 429 says
the next attempt may differ; a bad API key, a malformed argument or a chunk
that failed validation says it will not. So nothing here inspects message
text or classifies exceptions -- it honours a flag the error already carries,
and an exception it does not recognise is permanent, because ``is_retryable``
returns False for unknown types.

Bounded means bounded
---------------------
The worst case is ``1 + max_retries`` calls, each separated by a delay capped
at ``backoff_max``: a known quantity before the first attempt is made. Backoff
is computed from the retry *number* rather than read from a clock, so the
schedule is reproducible in a test and no clock quirk can extend it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .errors import ConfigurationError, is_retryable

__all__ = ["RetryPolicy"]

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """How long to keep trying a call that claims the next attempt may differ.

    Immutable, so a policy can be built once per stage and passed around
    without a caller quietly widening someone else's retry budget.
    """

    #: Extra attempts permitted after the first call fails.
    max_retries: int
    #: Seconds before retry 1; multiplied per retry after that.
    backoff_initial: float = 0.1
    #: Growth factor. 1.0 means a flat delay, never a shorter one.
    backoff_multiplier: float = 2.0
    #: Ceiling on a single delay, so a long failure run costs a known total.
    backoff_max: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ConfigurationError(
                "RetryPolicy.max_retries must be an int, got "
                f"{type(self.max_retries).__name__}"
            )
        if self.max_retries < 0:
            raise ConfigurationError(
                f"RetryPolicy.max_retries must be >= 0, got {self.max_retries}"
            )
        if self.backoff_initial < 0 or self.backoff_multiplier < 1:
            raise ConfigurationError(
                "RetryPolicy backoff must start non-negative and grow: "
                f"backoff_initial={self.backoff_initial}, "
                f"backoff_multiplier={self.backoff_multiplier}"
            )

    @property
    def max_attempts(self) -> int:
        """Calls made in the worst case: one try plus every permitted retry."""
        return 1 + self.max_retries

    def backoff_for(self, retry_number: int) -> float:
        """Seconds to sleep before ``retry_number`` (numbered from 1).

        Derived from the retry number rather than a running clock so a test
        can assert the exact schedule instead of a tolerance band.
        """
        if retry_number < 1:
            raise ConfigurationError(
                f"backoff_for() numbers retries from 1, got {retry_number}"
            )
        delay = self.backoff_initial * (self.backoff_multiplier ** (retry_number - 1))
        return min(delay, self.backoff_max)

    @classmethod
    def from_settings(cls, group: Any, **overrides: Any) -> RetryPolicy:
        """Read ``max_retries`` from a settings *group* -- ``settings.embedding``.

        This is the line that stops ``max_retries`` being unread
        configuration: the value comes from the settings object instead of
        being restated at the call site, so changing
        ``RAG_EMBEDDING__MAX_RETRIES`` changes how often a call is attempted.

        ``group`` is a group rather than the whole ``Settings`` so the caller
        says which stage's budget it wants and cannot silently inherit
        another stage's.
        """
        value = getattr(group, "max_retries", None)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(
                "RetryPolicy.from_settings needs a settings group carrying an "
                f"int max_retries; {type(group).__name__} has {value!r}. Pass "
                "max_retries= directly instead."
            )
        return cls(max_retries=value, **overrides)

    def call(
        self,
        fn: Callable[[], T],
        *,
        retry_if: Callable[[Exception], bool] = is_retryable,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> T:
        """Run ``fn``, repeating it only while ``retry_if`` accepts the failure.

        A permanent failure is raised on the first attempt and never comes
        back as anything else -- the original exception propagates unchanged,
        which is what "reported once" means: no wrapping, no second raising,
        no retry that could mask which call actually failed.

        ``sleep`` is injected so a test can assert the schedule without
        spending it, and ``on_retry(retry_number, error, delay)`` gives the
        caller a hook to log a retry as it happens.
        """
        retry_number = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                retry_number += 1
                # Attempts exhausted, or not worth retrying: raise exactly as
                # the callee raised it. retry_if is not consulted once the
                # budget is gone -- there is no decision left to make.
                if retry_number > self.max_retries or not retry_if(exc):
                    raise
                delay = self.backoff_for(retry_number)
                if on_retry is not None:
                    on_retry(retry_number, exc, delay)
                sleep(delay)

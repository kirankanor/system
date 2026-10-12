"""Configuration for ragkit's own commands.

Two different configuration layers exist in this repository and it is worth
being explicit about which this is. ``components/config`` is a *component*: it
ships to a user's project, carries pydantic models for the twenty-step
pipeline, and is what ``ragkit add settings`` copies. This module is ragkit's
own settings -- which stages to run, how to behave when one fails -- and it
must not import a component to have any (FR-54 forbids the orchestration layer
from statically importing component modules, and this file serves it).

So this is deliberately stdlib-only and flat. A settings layer that needed a
dependency to describe "load stages from this dotted path" would be a layer
that could not run in the minimal build it exists to support.

Stage resolution
----------------
The one interesting setting is :attr:`PipelineSettings.stages`, a dotted
reference of the form ``"package.module:factory"``. The orchestration layer
takes stages as callables and imports none, so *something* has to turn a name
into those callables. That something is here, by string, on purpose:

* The composition root stays in the user's project, where it can import the
  components it chose. Nothing in ``rag_code_library`` has to know which
  parser, chunker or store a project picked.
* Swapping a stage set is a configuration change rather than a code change,
  which is what FR-54 means by "stages are selected from configuration".

The factory is called with no arguments and returns an iterable of
:class:`~rag_code_library.pipeline.Stage`. Returning the stages rather than a
pipeline keeps the *assembly* -- which backend, which chunker -- out of this
package entirely.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

from .pipeline import FailurePolicy, Stage

__all__ = [
    "ENV_PREFIX",
    "PipelineSettings",
    "load_pipeline_settings",
    "resolve_stages",
]

#: Environment variables for these settings all start here, so the whole set
#: can be found in one place and none of them is mistaken for someone else's.
ENV_PREFIX = "RAG_PIPELINE__"

#: Names :func:`resolve_stages` accepts for the failure policy, so a typo in
#: an environment variable is reported instead of silently defaulting.
_POLICY_NAMES = {policy.value: policy for policy in FailurePolicy}


@dataclass(frozen=True)
class PipelineSettings:
    """How ``ragkit ingest`` should run. Every field has a safe default."""

    #: ``"package.module:factory"`` producing the stages. Empty means "not
    #: configured", which the CLI turns into an actionable error rather than
    #: running nothing and reporting success.
    stages: str = ""
    #: What one document's failure does to the rest of the run.
    failure_policy: str = FailurePolicy.SKIP.value
    #: Opaque configuration fingerprint recorded beside each document. Compare
    #: it for equality only; the orchestration layer never parses it.
    fingerprint: str = ""
    #: Name of the lock file taken inside the destination for a run's duration.
    lock_file: str = "_ragkit.lock"
    #: Where to write the run summary, relative to the destination.
    summary_file: str = "run-summary.json"

    # -- coercion ---------------------------------------------------------- #

    def policy(self) -> FailurePolicy:
        """The failure policy as an enum, or an actionable error if mistyped."""
        try:
            return _POLICY_NAMES[self.failure_policy]
        except KeyError:
            raise ValueError(
                f"unknown failure policy {self.failure_policy!r}; expected one "
                f"of: {', '.join(sorted(_POLICY_NAMES))}"
            ) from None

    @property
    def configured(self) -> bool:
        """Whether a stage source has been named at all."""
        return bool(self.stages.strip())


def load_pipeline_settings(
    environ: Mapping[str, str] | None = None, **overrides: Any
) -> PipelineSettings:
    """Build settings from the environment, with explicit overrides winning.

    Reads the environment itself rather than importing a settings framework:
    there are five values here, they are strings, and a validation library for
    five strings would be a dependency the minimal build cannot afford
    (NFR-14).
    """
    env = os.environ if environ is None else environ
    base = PipelineSettings()

    def read(name: str, default: str) -> str:
        return env.get(f"{ENV_PREFIX}{name}", default)

    values = {
        "stages": read("STAGES", base.stages),
        "failure_policy": read("FAILURE_POLICY", base.failure_policy).strip().lower(),
        "fingerprint": read("FINGERPRINT", base.fingerprint),
        "lock_file": read("LOCK_FILE", base.lock_file),
        "summary_file": read("SUMMARY_FILE", base.summary_file),
    }
    known = {f.name for f in fields(PipelineSettings)}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(
                f"unknown pipeline setting {key!r}; the settings are: "
                f"{', '.join(sorted(known))}"
            )
        values[key] = value
    settings = PipelineSettings(**values)
    settings.policy()  # validate now, not halfway through a run
    return settings


def resolve_stages(settings: PipelineSettings) -> tuple[Stage, ...]:
    """Import the named factory and take the stages it returns.

    Importing by string here rather than by ``import`` statement is the point:
    a static import anywhere in this package would name a component, and
    naming a component is what this architecture exists to avoid.

    Failure modes are all actionable. A missing module, a name with no
    factory, a factory that is not callable, or a return value that is not a
    list of stages each produce a message naming what was expected -- because
    the only way to reach this code is a configuration value the user wrote,
    and "cannot import X" alone would send them reading the wrong thing.
    """
    reference = settings.stages.strip()
    if not reference:
        raise ValueError(
            "no stages are configured; set "
            f"{ENV_PREFIX}STAGES to a 'module:factory' reference, or pass "
            "--stages. The factory returns the Pipeline Stage objects to run, "
            "which is how a project selects its own parser, chunker and store "
            "without this package importing any of them"
        )

    module_name, sep, attribute = reference.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(
            f"invalid stages reference {reference!r}; expected "
            "'module:factory', for example "
            "'myproject.pipeline:build_stages'"
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"could not import {module_name!r} named by {ENV_PREFIX}STAGES: "
            f"{exc}. Check the module is importable from where ragkit runs"
        ) from exc

    try:
        factory = getattr(module, attribute)
    except AttributeError:
        raise ValueError(
            f"{module_name!r} has no attribute {attribute!r} (stages reference "
            f"{reference!r})"
        ) from None

    if not callable(factory):
        raise ValueError(
            f"{reference!r} resolved to {type(factory).__name__}, which is not "
            "callable; the stages reference must name a function that returns "
            "the stages to run"
        )

    produced = factory()
    if isinstance(produced, Stage) or not isinstance(produced, Iterable):
        raise ValueError(
            f"{reference!r} returned {type(produced).__name__}, which is not an "
            "iterable of Stage objects"
        )

    stages = tuple(produced)
    if not stages:
        raise ValueError(f"{reference!r} returned no stages; there is nothing to run")
    wrong = [s for s in stages if not isinstance(s, Stage)]
    if wrong:
        raise ValueError(
            f"{reference!r} returned {len(wrong)} item(s) that are not Stage "
            f"objects (first: {type(wrong[0]).__name__}); a stages factory "
            "returns rag_code_library.pipeline.Stage instances"
        )
    return stages


def with_overrides(
    settings: PipelineSettings, **overrides: Any
) -> PipelineSettings:
    """Copy settings with replacements, for a command-line flag to win."""
    return replace(settings, **overrides)

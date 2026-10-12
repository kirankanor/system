"""Schema-faithful manifest model, and the Issue type every failure reports.

Why a pydantic mirror of the schema
-----------------------------------
``schemas/component-manifest.schema.json`` is canonical (FR-12), but FR-3 keeps
the runtime dependency set to pydantic + pydantic-settings and ``jsonschema`` is
not one of them, so this package cannot validate against the file directly.
This module mirrors it instead.

A mirror can drift, which is the exact failure FR-12 exists to prevent -- "no
component definition maintained in more than one place". So the mirror is kept
honest by ``tests/unit/test_registry.py``, which compares property names,
required sets and enum values between this module and the schema in *both*
directions. Add a field to either file without the other and the gate fails.

Errors are data, not exceptions
-------------------------------
Every manifest problem becomes an :class:`Issue` rather than being raised,
because FR-11 asks for actionable messages instead of tracebacks: a registry
holding many manifests should be able to report all that are broken in one
pass, not stop at the first. The one thing that *is* raised is an unknown component
id (see ``resolver.py``) -- that is a caller mistake, not data corruption.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, field_validator

__all__ = [
    "CATEGORIES",
    "STATUSES",
    "ComponentManifest",
    "ExternalService",
    "Issue",
    "Package",
    "RequiresPython",
    "StrictModel",
    "issue_from",
    "load_schema",
    "schema_path",
]

# Declared once as Literals so pydantic both enforces them and publishes them
# in model_json_schema() as an enum -- which is what lets the drift test compare
# them against the schema file's enums without a second copy of the values.
Category = Literal[
    "ingestion",
    "chunking",
    "embeddings",
    "retrieval",
    "reranking",
    "context",
    "generation",
    "evaluation",
    "security",
    "storage",
    "config",
    "observability",
    "integrations",
]
CATEGORIES: tuple[str, ...] = get_args(Category)

Status = Literal["stable", "experimental", "v2-deferred"]
STATUSES: tuple[str, ...] = get_args(Status)

# Patterns copied verbatim from the schema. pydantic compiles `pattern=` with
# the Rust regex crate, which does not support look-around, so _DESTINATION
# cannot use it at all -- see ComponentManifest._destination_is_contained.
#
# The rest are exercised rather than compared: if one drifts from the schema,
# either the real manifests stop validating or one of the corrupt fixtures
# stops being rejected, and both fail the gate. String-comparing them would
# only prove the two files were edited together, not that either is correct.
_KEBAB_ID = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
_SEMVER = r"^\d+\.\d+\.\d+$"
_DESTINATION = r"^(?![\\/])(?!.*(?:^|/)\.\.(?:/|$)).*$"
_PY_VERSION = r"^\d+\.\d+$"
_DISTRIBUTION = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

_NonEmpty = Annotated[str, StringConstraints(min_length=1)]
_KEBAB = Annotated[str, StringConstraints(pattern=_KEBAB_ID)]


def schema_path(root: Path | None = None) -> Path:
    """Locate the canonical schema file.

    ``root`` defaults to the repository root derived from this file, so the
    drift test and the CLI agree on one location without either hard-coding it.
    """
    base = root if root is not None else Path(__file__).resolve().parents[3]
    return base / "schemas" / "component-manifest.schema.json"


def load_schema(root: Path | None = None) -> dict:
    """Parse the canonical schema."""
    path = schema_path(root)
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class Issue:
    """One actionable reason a manifest was rejected (FR-11).

    ``path`` identifies the offending file; ``component`` is the declared id
    when it is readable, which is often *not* the case -- a malformed id is one
    of the things that brings a manifest here -- so it is optional.
    """

    reason: str
    path: str = ""
    component: str | None = None

    def __str__(self) -> str:
        if self.component and self.path:
            label = f"{self.component} ({self.path})"
        elif self.component:
            label = self.component
        elif self.path:
            label = self.path
        else:
            label = "registry"
        return f"{label}: {self.reason}"


def issue_from(exc: ValidationError, *, path: str, component: str | None = None) -> Issue:
    """Flatten pydantic's error tree into one readable line (FR-11).

    ``str(ValidationError)`` is a multi-line block quoting field paths and
    regexes back at the reader; that is a traceback in spirit if not in form.
    This keeps the same information as ``field: message`` pairs on one line.
    """
    detail = "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '<manifest>'}: {err['msg']}"
        for err in exc.errors()
    )
    return Issue(reason=f"schema violation: {detail}", path=path, component=component)


class StrictModel(BaseModel):
    """Base for every manifest model.

    ``extra="forbid"`` mirrors the schema's ``additionalProperties: false``:
    a typo like ``"dependancies"`` must fail loudly rather than being silently
    dropped, because a dropped ``dependencies`` reads as "no dependencies".
    """

    model_config = ConfigDict(extra="forbid")


class Package(StrictModel):
    """A third-party distribution the component imports."""

    name: Annotated[str, StringConstraints(pattern=_DISTRIBUTION)]
    specifier: _NonEmpty


class ExternalService(StrictModel):
    """A network service the component calls when configured."""

    name: _NonEmpty
    env_vars: tuple[_NonEmpty, ...]
    notes: _NonEmpty


class RequiresPython(StrictModel):
    """Interpreter bounds. ``max`` is optional, matching the schema."""

    min: Annotated[str, StringConstraints(pattern=_PY_VERSION)]
    max: Annotated[str, StringConstraints(pattern=_PY_VERSION)] | None = None


class ComponentManifest(StrictModel):
    """One copy-ready component's metadata.

    Every field below is required by the schema except ``documentation``,
    ``tests`` and ``tradeoffs``, so those three are the only ones with a
    default. Fields the schema marks required deliberately take *no* default
    here either: giving them one would let a manifest that omits them pass this
    model while failing the schema, which is drift the drift test is meant to
    catch.

    One difference the drift test does not cover: those four optional fields
    are typed ``str | None``, so an explicit JSON ``null`` is accepted here
    where the schema's ``"type": "string"`` rejects it. Nothing turns on the
    distinction -- absent, empty and ``null`` all read as "declares none"
    downstream (``resolver.resolve`` and ``validator._missing_files`` both
    treat falsy as absent) -- and no manifest in the tree uses ``null``, since
    every one of them passes the schema as written. Typing them ``str = ""``
    instead would reject ``null`` but trade it for a ``default: ""`` the
    schema does not declare, so neither modelling is strictly closer; this one
    is the idiomatic reading of "optional string".
    """

    id: _KEBAB
    name: _NonEmpty
    category: Category
    source: _NonEmpty
    version: Annotated[str, StringConstraints(pattern=_SEMVER)]
    dependencies: tuple[_KEBAB, ...]
    description: _NonEmpty
    destination: Annotated[str, StringConstraints(min_length=1)]
    documentation: str | None = None
    tests: str | None = None
    status: Status
    requires_python: RequiresPython
    required_packages: tuple[Package, ...]
    optional_packages: tuple[Package, ...]
    external_services: tuple[ExternalService, ...]
    install: _NonEmpty
    missing_dependency_message: _NonEmpty
    tradeoffs: str | None = None

    @field_validator("destination")
    @classmethod
    def _destination_is_contained(cls, value: str) -> str:
        """Apply the schema's ``destination`` pattern using Python's ``re``.

        This cannot be a ``pattern=`` constraint: pydantic compiles those with
        the Rust regex crate, and this pattern's look-around makes model
        construction fail with ``SchemaError: look-around ... is not supported``
        at *import* time -- so the whole registry package would be unloadable
        rather than merely stricter.

        Semantics stay those of JSON Schema, where ``pattern`` is an unanchored
        search; the pattern anchors itself with ``^``/``$``, so ``re.search``
        reproduces it exactly. The message names what was expected and what was
        given instead, because a bare "string should match pattern ..." tells
        the reader nothing they can act on.
        """
        if not re.search(_DESTINATION, value):
            raise ValueError(
                "must be relative to the destination root, use forward slashes, "
                f"and contain no '..' segment; got {value!r}"
            )
        return value

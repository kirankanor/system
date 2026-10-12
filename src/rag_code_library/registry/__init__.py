"""The component registry: discover, validate, and resolve manifests.

This is the stable surface the CLI (``ragkit list``/``info``/``add``), the test
suite and the cookbook generator all read, so that no component definition is
maintained in more than one place (FR-12).

Typical use::

    from rag_code_library.registry import Registry

    registry = Registry.load("components")
    if not registry.ok:
        for issue in registry.issues:
            print(issue)          # one actionable line each, no traceback
    chunking = registry.resolve("fixed-size")
    chunking.source               # the file to copy
    chunking.manifest.destination # the template to copy it to

Requirements: FR-11 (actionable reporting), FR-12 (one source of metadata),
FR-24 (``scripts/validate_components.py``), NFR-6 (no drift between the
manifests and everything derived from them).
"""

from __future__ import annotations

from .loader import ManifestEntry, Registry, discover, load_entries
from .manifest import (
    CATEGORIES,
    STATUSES,
    ComponentManifest,
    ExternalService,
    Issue,
    Package,
    RequiresPython,
    issue_from,
    load_schema,
    schema_path,
)
from .resolver import ResolvedComponent, UnknownComponentError, resolve
from .validator import validate_entries

__all__ = [
    "CATEGORIES",
    "STATUSES",
    "ComponentManifest",
    "ExternalService",
    "Issue",
    "ManifestEntry",
    "Package",
    "Registry",
    "RequiresPython",
    "ResolvedComponent",
    "UnknownComponentError",
    "discover",
    "issue_from",
    "load_entries",
    "load_schema",
    "resolve",
    "schema_path",
    "validate_entries",
]

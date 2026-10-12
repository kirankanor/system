"""Add provenance to a document's metadata without ever inventing any of it.

What this component will not do
-------------------------------
It will not supply a title, an author or a date it does not have. Every field
it adds is either *given* to it or *computed* from something already known --
a path, a digest, an injected timestamp. A field it cannot account for is
omitted and named in ``MetadataBundle.omitted``, never filled with a default,
because an invented author is worse than a missing one: it looks like evidence.

``ingested_at`` is a required constructor argument rather than a value this
component reads from a clock. A wall-clock timestamp would make re-ingesting
the same file produce different metadata every run, which contradicts the one
property content-derived ids exist to guarantee -- that re-ingesting is
idempotent. Injection keeps the component pure, and leaves the decision about
*when* ingestion happened to whoever ran it.

Three bags, kept apart
----------------------
* ``extracted`` -- the semantic fields (title, author, page_number), only where
  they were supplied and only where they are well-formed.
* ``system``     -- what this component computed (source path, ingestion time,
  content id), all under a reserved ``system_`` prefix so they can never
  shadow a document's own field.
* ``source``     -- the original metadata bag, verbatim, minus only the entries
  that were not scalars.
* ``merged``     -- all three, for storage, with ``source`` winning a collision
  so the document's own account of itself is never overwritten.

Why ``page_number`` refuses a ``bool``: ``True`` is an ``int`` equal to 1, and
a metadata filter that accepts it will quietly treat "page 1" and "yes" as the
same thing. This matches the rule ``security/metadata-validation`` already
enforces, so a bundle produced here validates there.

Optional ``pypdf``
------------------
``read_document_info`` can pull title and author out of a PDF's own document
information dictionary when the caller does not have them. It is optional and
separated from :meth:`MetadataEnricher.enrich`, which never touches a file --
so the required path stays stdlib-only and deterministic, and the FR-3 guard
surrounds the one function that needs the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "SYSTEM_PREFIX",
    "MalformedSourceMetadata",
    "Metadata",
    "MetadataBundle",
    "MetadataEnricher",
    "MetadataError",
    "MissingOptionalDependency",
    "read_document_info",
]

#: Every value this component emits lives under this prefix if this component
#: computed it. A document cannot have called itself ``system_ingested_at``.
SYSTEM_PREFIX = "system_"

MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


class MetadataError(Exception):
    """Base for every failure this component reports."""


class MalformedSourceMetadata(MetadataError):
    """The metadata *container* was unusable, as opposed to one bad entry."""


class MissingOptionalDependency(MetadataError):
    """A caller asked for a capability shipped as an optional dependency."""


def _is_scalar(value: Any) -> bool:
    """Flat scalars only. A ``bool`` counts, because ``MetadataValue`` allows it.

    Nested structures are rejected because most vector databases cannot filter
    on them, so accepting one here only defers the failure to query time.
    """
    return isinstance(value, str | int | float)


def _valid_text(value: Any) -> str | None:
    """A usable non-empty string, or ``None`` when the value says nothing."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _valid_page(value: Any) -> int | None:
    """A page number that is a real integer and not a ``True`` in disguise."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


@dataclass(frozen=True)
class MetadataBundle:
    """What was added, what it came from, and what was left out."""

    #: Semantic fields supplied by the caller. Absent fields are simply absent.
    extracted: Metadata
    #: Values this component computed, all prefixed with ``system_``.
    system: Metadata
    #: The document's own metadata, preserved rather than rewritten.
    source: Metadata
    #: The three combined, for storage. ``source`` wins any collision.
    merged: Metadata
    #: Fields dropped for being empty, wrong-typed or out of range, with the
    #: reason attached. Empty means nothing was wrong -- not that the check
    #: was skipped.
    omitted: tuple[str, ...]

    @property
    def title(self) -> str | None:
        """Convenience for the field most callers actually want."""
        value: MetadataValue | None = self.extracted.get("title")
        return value if isinstance(value, str) else None

    @property
    def author(self) -> str | None:
        value: MetadataValue | None = self.extracted.get("author")
        return value if isinstance(value, str) else None


def read_document_info(path: str | Path) -> dict[str, Any]:
    """Read title, author and friends from a PDF's own document information.

    Optional (FR-3), and deliberately *not* called by :meth:`MetadataEnricher.enrich`
    -- enrichment stays a pure function of its arguments, so this is offered as
    a way to *obtain* the values rather than smuggled into the required path.

    The refusal names the package, because an error that only says "something
    is missing" cannot be acted on.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingOptionalDependency(
            "pypdf is optional and is not installed, so document information "
            "cannot be read from the PDF. Install it with `pip install pypdf`. "
            "Every other part of this component runs without it, because "
            "enrichment does not otherwise touch a file."
        ) from exc

    try:
        reader = PdfReader(str(path))
        info = reader.metadata
    except Exception as exc:  # pypdf raises several unrelated types
        raise MalformedSourceMetadata(f"could not read {path}: {exc}") from exc

    if info is None:
        return {}
    # pypdf returns a DocumentInformation mapping whose keys are slash-prefixed;
    # normalise both the keys and the values to plain scalars.
    raw = {str(key).lstrip("/"): value for key, value in dict(info).items()}
    return {key: value for key, value in raw.items() if _is_scalar(value)}


class MetadataEnricher:
    """Attach provenance to metadata, and report what could not be attached.

    Constructed with the ingestion time rather than reading one, so the same
    inputs always yield byte-identical output and a replayed ingestion can
    reproduce a previous run exactly.
    """

    def __init__(self, *, ingested_at: str, system_prefix: str = SYSTEM_PREFIX) -> None:
        timestamp = _valid_text(ingested_at)
        if timestamp is None:
            raise MalformedSourceMetadata(
                "ingested_at must be a non-empty string; this component does not "
                "read a clock, because a wall-clock value would make re-ingesting "
                "the same file produce different metadata each run. Pass "
                "datetime.now(UTC).isoformat() yourself if you want one."
            )
        if not system_prefix or system_prefix.strip() != system_prefix:
            raise ValueError(
                "system_prefix must be a non-empty prefix with no surrounding "
                "whitespace, or system fields would not be distinguishable"
            )
        self.ingested_at = timestamp
        self.system_prefix = system_prefix

    # -- enrichment ---------------------------------------------------------
    def enrich(
        self,
        *,
        source_name: str,
        source_metadata: Mapping[str, Any] | None = None,
        title: Any = None,
        author: Any = None,
        page_number: Any = None,
        content_sha256: str | None = None,
    ) -> MetadataBundle:
        """Build the four bags. Never raises for a single bad field.

        A caller ingesting ten thousand documents must lose one record's title,
        not the batch -- so a malformed field is recorded in ``omitted`` and
        the rest of the bundle is still produced. Only an unusable *container*
        raises.
        """
        omitted: list[str] = []

        # -- source, preserved, minus non-scalars --------------------------
        if source_metadata is not None and not isinstance(source_metadata, Mapping):
            raise MalformedSourceMetadata(
                "source_metadata must be a mapping of flat scalars; "
                f"got {type(source_metadata).__name__}"
            )
        source: Metadata = {}
        for key, value in dict(source_metadata or {}).items():
            if not isinstance(key, str):
                omitted.append(f"{key!r}: key is not a string")
                continue
            if _is_scalar(value):
                source[key] = value
            else:
                omitted.append(f"{key}: {type(value).__name__} is not a flat scalar")

        # -- extracted, only where well-formed ------------------------------
        # A field that was never supplied is simply absent: that is the normal
        # case for title and author, so it is not an omission. Only a field
        # that was supplied and could not be used is reported, because that is
        # the one where information was actually lost.
        extracted: Metadata = {}
        for name, value in (("title", title), ("author", author)):
            if value is None:
                continue
            text = _valid_text(value)
            if text is None:
                omitted.append(f"{name}: expected a non-empty string, got {value!r}")
            else:
                extracted[name] = text

        if page_number is not None:
            page = _valid_page(page_number)
            if page is None:
                omitted.append(
                    f"page_number: expected an integer >= 0, got {page_number!r}"
                )
            else:
                extracted["page_number"] = page

        # -- system, computed ------------------------------------------------
        system: Metadata = {
            f"{self.system_prefix}source_path": source_name,
            f"{self.system_prefix}ingested_at": self.ingested_at,
        }
        if content_sha256 is not None:
            system[f"{self.system_prefix}content_sha256"] = content_sha256

        # -- merged, source winning -----------------------------------------
        merged: Metadata = {}
        merged.update(extracted)
        merged.update(system)
        merged.update(source)  # the document's own account is never overwritten

        return MetadataBundle(
            extracted=extracted,
            system=system,
            source=source,
            merged=merged,
            omitted=tuple(omitted),
        )

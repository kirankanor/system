"""Component tests for `metadata-enricher`.

Runs standalone: ``python -m pytest components/ingestion/metadata_enricher/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

The component's two load-bearing promises are opposite in kind, so both get
more attention than the rest:

* it **adds** the provenance it was given or could compute; and
* it **refuses to add** anything it could not account for.

A test that only exercised the first would pass on a component that invented
an author for every document, so the "nothing invented" cases are pinned as
hard as the enrichment cases -- including the subtle one where a malformed
field must be dropped without taking the rest of the record with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from metadata_enricher import (
    SYSTEM_PREFIX,
    MalformedSourceMetadata,
    MetadataBundle,
    MetadataEnricher,
    MetadataError,
    MissingOptionalDependency,
    read_document_info,
)

NOW = "2026-10-11T09:30:00+00:00"


@pytest.fixture
def enricher():
    return MetadataEnricher(ingested_at=NOW)


# --- success paths ----------------------------------------------------------


def test_every_supplied_field_reaches_the_right_bag(enricher):
    bundle = enricher.enrich(
        source_name="docs/report.pdf",
        title="Quarterly Report",
        author="A. Author",
        page_number=3,
        content_sha256="a" * 64,
    )

    assert isinstance(bundle, MetadataBundle)
    assert bundle.extracted == {
        "title": "Quarterly Report",
        "author": "A. Author",
        "page_number": 3,
    }
    assert bundle.system[f"{SYSTEM_PREFIX}source_path"] == "docs/report.pdf"
    assert bundle.system[f"{SYSTEM_PREFIX}ingested_at"] == NOW
    assert bundle.system[f"{SYSTEM_PREFIX}content_sha256"] == "a" * 64
    assert bundle.omitted == ()


def test_source_metadata_is_preserved_verbatim(enricher):
    """The document's own account must survive enrichment untouched."""
    original = {"title": "From the file", "language": "en", "pages": 12}

    bundle = enricher.enrich(source_name="a.txt", source_metadata=original)

    assert bundle.source == original
    assert original == {"title": "From the file", "language": "en", "pages": 12}


def test_source_wins_a_collision_so_its_metadata_is_never_overwritten(enricher):
    """`merged` must not let a computed or supplied value displace the source."""
    bundle = enricher.enrich(
        source_name="a.txt",
        title="Supplied by the caller",
        source_metadata={"title": "Written into the document"},
    )

    assert bundle.extracted["title"] == "Supplied by the caller"
    assert bundle.source["title"] == "Written into the document"
    assert bundle.merged["title"] == "Written into the document"


def test_everything_lands_in_merged(enricher):
    bundle = enricher.enrich(
        source_name="a.txt",
        title="T",
        page_number=1,
        source_metadata={"language": "en"},
    )

    assert bundle.merged["title"] == "T"
    assert bundle.merged["page_number"] == 1
    assert bundle.merged["language"] == "en"
    assert bundle.merged[f"{SYSTEM_PREFIX}source_path"] == "a.txt"


def test_a_document_cannot_overwrite_a_computed_system_field(enricher):
    """FR-56: provenance must not be writable by the thing being provenanced.

    Before this requirement the merge applied ``source`` last, so a document
    that supplied ``system_source_path`` had it copied straight into
    ``merged`` -- the field that gets stored. The old test asserted that the
    attacker's value survived; it now asserts the opposite.
    """
    bundle = enricher.enrich(
        source_name="a.txt",
        source_metadata={
            f"{SYSTEM_PREFIX}source_path": "attacker-controlled",
            f"{SYSTEM_PREFIX}ingested_at": "1999-01-01",
        },
    )

    # The stored bag carries the computed values, not the document's claims.
    assert bundle.merged[f"{SYSTEM_PREFIX}source_path"] == "a.txt"
    assert bundle.merged[f"{SYSTEM_PREFIX}ingested_at"] == NOW

    # Nothing is hidden: the document's own account is preserved verbatim in
    # `source`, so a caller can still see exactly what the file asserted.
    assert bundle.source[f"{SYSTEM_PREFIX}source_path"] == "attacker-controlled"
    assert bundle.source[f"{SYSTEM_PREFIX}ingested_at"] == "1999-01-01"

    # And the system bag was never at risk.
    assert bundle.system[f"{SYSTEM_PREFIX}source_path"] == "a.txt"
    assert bundle.system[f"{SYSTEM_PREFIX}ingested_at"] == NOW

    # The overrule is reported, not applied silently.
    assert any(
        reason.startswith(f"{SYSTEM_PREFIX}source_path:")
        for reason in bundle.omitted
    )
    assert any(
        reason.startswith(f"{SYSTEM_PREFIX}ingested_at:")
        for reason in bundle.omitted
    )
    assert all("reserved system field" in reason for reason in bundle.omitted
               if reason.startswith(SYSTEM_PREFIX))


def test_a_document_that_agrees_is_not_reported_as_a_conflict(enricher):
    """Agreement costs nothing, so there is nothing to report.

    The value is still the computed one -- agreement is not authority -- but
    filing a conflict when the outcome is identical would train a caller to
    ignore `omitted`.
    """
    same = enricher.enrich(
        source_name="a.txt",
        source_metadata={f"{SYSTEM_PREFIX}source_path": "a.txt"},
    )
    assert same.merged[f"{SYSTEM_PREFIX}source_path"] == "a.txt"
    assert not [r for r in same.omitted if r.startswith(SYSTEM_PREFIX)]


def test_non_system_fields_are_still_the_document_s_to_win(enricher):
    """The exception must not swallow the rule it is an exception to."""
    bundle = enricher.enrich(
        source_name="a.txt",
        title="Supplied by the caller",
        source_metadata={"title": "Written into the document"},
    )
    assert bundle.merged["title"] == "Written into the document"
    assert bundle.extracted["title"] == "Supplied by the caller"
    assert bundle.omitted == ()


def test_the_title_and_author_properties_read_through(enricher):
    bundle = enricher.enrich(source_name="a.txt", title="T", author="A")

    assert bundle.title == "T"
    assert bundle.author == "A"


def test_enrichment_is_deterministic():
    """Same inputs, byte-identical output -- the reason the timestamp is injected."""
    first = MetadataEnricher(ingested_at=NOW).enrich(source_name="a.txt", title="T")
    second = MetadataEnricher(ingested_at=NOW).enrich(source_name="a.txt", title="T")

    assert first.merged == second.merged
    assert first.omitted == second.omitted


def test_a_page_number_of_zero_is_accepted(enricher):
    """Zero-indexed systems exist; refusing it would be a policy, not a check."""
    assert enricher.enrich(source_name="a.txt", page_number=0).extracted[
        "page_number"
    ] == 0


def test_whitespace_around_a_title_is_stripped(enricher):
    bundle = enricher.enrich(source_name="a.txt", title="  Padded  ")

    assert bundle.extracted["title"] == "Padded"


def test_an_absent_title_is_omitted_rather_than_blank(enricher):
    """Missing is normal, so it is absence -- not an entry set to empty string."""
    bundle = enricher.enrich(source_name="a.txt")

    assert "title" not in bundle.extracted
    assert "author" not in bundle.extracted
    assert "title" not in bundle.merged
    assert bundle.omitted == ()


def test_the_bundle_is_immutable(enricher):
    bundle = enricher.enrich(source_name="a.txt")

    with pytest.raises(AttributeError):
        bundle.merged = {}


def test_every_metadata_value_stays_a_flat_scalar(enricher):
    """The vector databases downstream cannot filter on anything nested."""
    bundle = enricher.enrich(
        source_name="a.txt",
        source_metadata={"ok": 1, "flag": True, "ratio": 0.5, "label": "x"},
    )

    assert all(isinstance(v, str | int | float | bool) for v in bundle.merged.values())


# --- nothing is invented ----------------------------------------------------


def test_no_author_is_added_when_none_was_supplied(enricher):
    """The promise the component is named for."""
    bundle = enricher.enrich(source_name="a.txt", title="Title only")

    assert "author" not in bundle.extracted
    assert "author" not in bundle.merged
    assert bundle.author is None


def test_no_title_is_added_when_none_was_supplied(enricher):
    bundle = enricher.enrich(source_name="a.txt", author="Someone")

    assert "title" not in bundle.merged
    assert bundle.title is None


def test_an_empty_title_is_dropped_not_stored_as_blank(enricher):
    """An empty string is not a title, and storing it would read as one."""
    bundle = enricher.enrich(source_name="a.txt", title="   ")

    assert "title" not in bundle.merged
    assert any("title" in reason for reason in bundle.omitted)


def test_no_date_is_invented_beyond_the_injected_one(enricher):
    """No created/modified/issued field appears that nobody supplied."""
    bundle = enricher.enrich(source_name="a.txt")

    invented = [
        key
        for key in bundle.merged
        if any(
            word in key.lower()
            for word in ("created", "modified", "issued", "published", "date", "year")
        )
    ]
    assert invented == []
    # ...and the one time that is present is verbatim what the caller injected.
    assert bundle.merged[f"{SYSTEM_PREFIX}ingested_at"] == NOW


# --- malformed fields do not fail the record --------------------------------


def test_a_wrong_typed_title_is_reported_and_the_rest_survives(enricher):
    """Losing one field must not lose the batch."""
    bundle = enricher.enrich(
        source_name="a.txt",
        title=42,
        author="Real Author",
        source_metadata={"language": "en"},
    )

    assert "title" not in bundle.merged
    assert bundle.extracted["author"] == "Real Author"
    assert bundle.merged["language"] == "en"
    assert any(reason.startswith("title:") for reason in bundle.omitted)


@pytest.mark.parametrize("page", [-1, 3.5, True, "3"])
def test_a_malformed_page_number_is_dropped_with_a_reason(enricher, page):
    """`True` in particular: it is an int equal to 1, and would read as page 1."""
    bundle = enricher.enrich(source_name="a.txt", page_number=page)

    assert "page_number" not in bundle.extracted
    assert any(reason.startswith("page_number:") for reason in bundle.omitted)


def test_a_nested_source_value_is_dropped_but_its_neighbours_are_kept(enricher):
    bundle = enricher.enrich(
        source_name="a.txt",
        source_metadata={"language": "en", "authors": ["A", "B"]},
    )

    assert bundle.source == {"language": "en"}
    assert any("authors" in reason for reason in bundle.omitted)


def test_a_non_string_source_key_is_dropped(enricher):
    bundle = enricher.enrich(source_name="a.txt", source_metadata={1: "numeric key"})

    assert bundle.source == {}
    assert bundle.omitted


def test_omitted_is_empty_when_nothing_was_wrong(enricher):
    """Otherwise the field would be decorative and its absence untestable."""
    bundle = enricher.enrich(source_name="a.txt", title="T", author="A", page_number=2)

    assert bundle.omitted == ()


# --- failure paths ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_an_unusable_ingestion_time_is_refused_at_construction(bad):
    """The timestamp is required, so a bad one must not reach enrichment."""
    with pytest.raises(MalformedSourceMetadata) as excinfo:
        MetadataEnricher(ingested_at=bad)  # type: ignore[arg-type]

    assert "does not" in str(excinfo.value) or "non-empty" in str(excinfo.value)


def test_the_ingestion_time_refusal_explains_why_no_clock_is_used():
    """The message must carry the design reason, not just the failure."""
    with pytest.raises(MalformedSourceMetadata) as excinfo:
        MetadataEnricher(ingested_at="")

    message = str(excinfo.value)
    assert "clock" in message
    assert "idempotent" in message or "different metadata" in message


def test_source_metadata_that_is_not_a_mapping_is_a_container_failure(enricher):
    """A bad container is a caller error and does raise -- unlike one bad entry."""
    with pytest.raises(MalformedSourceMetadata):
        enricher.enrich(source_name="a.txt", source_metadata=["not", "a", "mapping"])


def test_a_bad_system_prefix_is_refused():
    with pytest.raises(ValueError):
        MetadataEnricher(ingested_at=NOW, system_prefix="")


def test_the_failures_share_a_base_type():
    """So a caller can catch everything this component raises in one place."""
    assert issubclass(MalformedSourceMetadata, MetadataError)
    assert issubclass(MissingOptionalDependency, MetadataError)


# --- optional dependency (FR-3) ---------------------------------------------


def test_enrichment_never_needs_pypdf():
    """`enrich()` must not touch a file, so it cannot need the parser."""
    MetadataEnricher(ingested_at=NOW).enrich(source_name="a.pdf", title="T")


def test_reading_document_info_without_pypdf_names_the_package(monkeypatch):
    """FR-3: the refusal has to say what to install, not merely that it is absent."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("No module named 'pypdf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(MissingOptionalDependency) as excinfo:
        read_document_info("anything.pdf")

    message = str(excinfo.value)
    assert "pypdf" in message
    assert "pip install pypdf" in message


def test_read_document_info_is_reachable_as_a_module_level_function():
    """Pins the public shape: callers use `read_document_info`, not a method."""
    assert callable(read_document_info)

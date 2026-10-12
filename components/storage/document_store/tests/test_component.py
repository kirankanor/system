"""Tests for the document-store component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

This component is a contract: it defines the checks and the error vocabulary a
document-store implementation owes its callers, and re-exports the protocol
from core-types. It defines no store of its own, so the store-facing rules --
notably that `get` returns None for an unknown id -- are asserted against
implementations as they arrive, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1]

# This component imports core-types, which lives in its own component
# directory. The manifest declares it a dependency, so `ragkit add` copies it
# beside this file; here we reach across to the sibling.
sys.path.insert(0, str(COMPONENT.parent / "core_types"))
sys.path.insert(0, str(COMPONENT))

import core_types as core  # noqa: E402
import document_store as ds  # noqa: E402
from document_store import (  # noqa: E402
    DocumentStoreBackendError,
    DocumentStoreError,
    DocumentStoreInputError,
    DocumentStoreProtocol,
    validate_filter,
    validate_ids,
    validate_source_document,
)

import pytest  # noqa: E402


def _doc(**overrides):
    fields = {
        "document_id": "doc-1",
        "source_name": "handbook.md",
        "sha256": "a" * 64,
        "page_count": 12,
    }
    fields.update(overrides)
    return core.SourceDocument(**fields)


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_the_protocol_is_re_exported_not_redeclared():
    assert DocumentStoreProtocol is core.DocumentStoreProtocol


def test_a_well_formed_document_passes_and_comes_back_unchanged():
    doc = _doc()
    assert validate_source_document(doc) is doc


def test_a_pageless_document_is_accepted():
    """A plain-text source has no pages; page_count=0 is legitimate."""
    assert validate_source_document(_doc(page_count=0)).document_id == "doc-1"


def test_none_filter_passes_through():
    assert validate_filter(None) is None


def test_a_flat_filter_comes_back_as_a_plain_dict():
    out = validate_filter({"source_name": "handbook.md", "lang": "en"})
    assert out == {"source_name": "handbook.md", "lang": "en"}
    assert isinstance(out, dict)


def test_a_flat_filter_accepts_scalars_of_any_type():
    assert validate_filter({"ok": True, "n": 2.5, "s": "x", "z": None}) == {
        "ok": True,
        "n": 2.5,
        "s": "x",
        "z": None,
    }


def test_ids_come_back_as_a_list():
    assert validate_ids(("doc-1", "doc-2")) == ["doc-1", "doc-2"]


# --------------------------------------------------------------------------- #
# Failure paths -- documents
# --------------------------------------------------------------------------- #


def test_a_non_document_is_rejected_by_type():
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_source_document({"document_id": "doc-1"})
    assert "SourceDocument" in str(exc.value)


@pytest.mark.parametrize("bad", [None, "text", 3, object()], ids=["none", "str", "int", "obj"])
def test_anything_but_a_document_is_rejected(bad):
    with pytest.raises(DocumentStoreInputError):
        validate_source_document(bad)


def test_an_invalid_document_is_rejected_with_the_components_name():
    """The model's own rules still apply; the message names document-store."""
    broken = core.SourceDocument.model_construct(
        document_id="", source_name="x", sha256="short"
    )
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_source_document(broken)
    assert "document-store" in str(exc.value)


# --------------------------------------------------------------------------- #
# Failure paths -- filters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad", [{"a": {"b": 1}}, {"a": [1, 2]}, {"a": (1,)}, {"a": {1, 2}}],
    ids=["mapping", "list", "tuple", "set"],
)
def test_nested_filter_values_are_rejected_not_dropped(bad):
    """Dropping a nested value narrows the result set invisibly."""
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_filter(bad)
    assert "flat" in str(exc.value)


@pytest.mark.parametrize("bad", ["str", [1, 2], 3, object()], ids=["str", "list", "int", "obj"])
def test_a_non_mapping_filter_is_rejected(bad):
    with pytest.raises(DocumentStoreInputError):
        validate_filter(bad)


@pytest.mark.parametrize("bad", [{"": 1}, {None: 1}, {5: 1}])
def test_an_unusable_filter_key_is_rejected(bad):
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_filter(bad)
    assert "field names" in str(exc.value)


# --------------------------------------------------------------------------- #
# Failure paths -- ids
# --------------------------------------------------------------------------- #


def test_a_bare_string_of_ids_is_rejected_because_it_is_a_sequence():
    """'doc-1' would mean ['d','o','c','-','1'] -- five lookups, not one."""
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_ids("doc-1")
    assert "character" in str(exc.value)


def test_a_bare_bytes_of_ids_is_rejected():
    with pytest.raises(DocumentStoreInputError):
        validate_ids(b"doc-1")


@pytest.mark.parametrize("bad", [None, 3, {1, 2}])
def test_a_non_sequence_of_ids_is_rejected(bad):
    with pytest.raises(DocumentStoreInputError):
        validate_ids(bad)


@pytest.mark.parametrize("bad", ["", 5, None], ids=["empty", "int", "none"])
def test_an_unusable_id_is_rejected_by_position(bad):
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_ids(["ok", bad])
    assert "[1]" in str(exc.value)


# --------------------------------------------------------------------------- #
# Error hierarchy and message provenance
# --------------------------------------------------------------------------- #


def test_the_hierarchy_is_catchable_at_the_base():
    assert issubclass(DocumentStoreInputError, DocumentStoreError)
    assert issubclass(DocumentStoreBackendError, DocumentStoreError)


def test_the_base_is_distinguishable_from_a_programming_error():
    assert not issubclass(DocumentStoreError, ValueError)
    assert not issubclass(DocumentStoreError, TypeError)


def test_the_component_defines_its_own_errors_rather_than_importing_them():
    """Copy-ready: nothing here reaches into vector-store-protocol."""
    assert not hasattr(ds, "StorageError")
    assert not hasattr(ds, "validate_vector")
    assert hasattr(ds, "DocumentStoreError")


def test_every_check_is_catchable_under_the_one_base_class():
    for check in (
        lambda: validate_filter({"a": {"b": 1}}),
        lambda: validate_ids("abc"),
        lambda: validate_source_document(object()),
    ):
        with pytest.raises(DocumentStoreError):
            check()


def test_messages_name_the_component():
    """NFR-7: a caller can tell which component complained."""
    with pytest.raises(DocumentStoreInputError) as exc:
        validate_ids("abc")
    assert "document-store" in str(exc.value)


def test_the_manifest_declares_the_core_types_dependency():
    import json

    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []

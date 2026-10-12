"""Tests for the vector-store-protocol component.

Runs without importing `rag_code_library` (AC-13): components are copyable and
testable on their own. Covers success paths and deliberate failure paths (AC-4)
against small, fully controlled inputs.

The backend-facing behaviour of these checks is asserted end to end in
`tests/unit/test_storage.py`; this file covers the checks themselves.
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
import vector_store_protocol as vsp  # noqa: E402
from vector_store_protocol import (  # noqa: E402
    MissingStorageDependency,
    StorageBackendError,
    StorageError,
    StorageInputError,
    VectorStoreProtocol,
    validate_embedded_chunks,
    validate_filter,
    validate_ids,
    validate_k,
    validate_vector,
)

import pytest  # noqa: E402


# --------------------------------------------------------------------------- #
# Success paths
# --------------------------------------------------------------------------- #


def test_the_protocol_is_re_exported_not_redeclared():
    """One definition, reachable from this component alone."""
    assert VectorStoreProtocol is core.VectorStoreProtocol


def test_none_filter_passes_through():
    assert validate_filter(None) is None


def test_a_flat_filter_comes_back_as_a_plain_dict():
    out = validate_filter({"source": "a.txt", "page": 3})
    assert out == {"source": "a.txt", "page": 3}
    assert isinstance(out, dict)


def test_a_flat_filter_accepts_scalars_of_any_type():
    assert validate_filter({"ok": True, "n": 2.5, "s": "x", "z": None}) == {
        "ok": True,
        "n": 2.5,
        "s": "x",
        "z": None,
    }


def test_ids_come_back_as_a_list():
    assert validate_ids(("a", "b")) == ["a", "b"]


def test_a_single_id_is_accepted_when_wrapped():
    assert validate_ids(["only"]) == ["only"]


def test_a_numeric_vector_is_normalised_to_floats():
    assert validate_vector([1, 2, 3.5]) == [1.0, 2.0, 3.5]


def test_a_one_element_vector_is_accepted():
    assert validate_vector([0.5]) == [0.5]


def test_k_is_returned_as_an_int():
    assert validate_k(7) == 7
    assert isinstance(validate_k(7), int)


def _embedded(width=2):
    return [
        core.EmbeddedChunk(
            chunk=core.Chunk(chunk_index=i, text=f"text {i}", chunk_id=f"c{i}"),
            vector=tuple(float(n) for n in range(width)),
            index_fingerprint="fp",
        )
        for i in range(3)
    ]


def test_uniform_widths_are_accepted():
    assert len(validate_embedded_chunks(_embedded(width=3))) == 3


# --------------------------------------------------------------------------- #
# Failure paths -- filters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad", [{"a": {"b": 1}}, {"a": [1, 2]}, {"a": (1,)}, {"a": {1, 2}}],
    ids=["mapping", "list", "tuple", "set"],
)
def test_nested_filter_values_are_rejected_not_dropped(bad):
    """Dropping a nested value narrows the result set invisibly."""
    with pytest.raises(StorageInputError) as exc:
        validate_filter(bad)
    assert "flat" in str(exc.value)


@pytest.mark.parametrize("bad", ["str", [1, 2], 3, object()], ids=["str", "list", "int", "obj"])
def test_a_non_mapping_filter_is_rejected(bad):
    with pytest.raises(StorageInputError):
        validate_filter(bad)


@pytest.mark.parametrize("bad", [{"": 1}, {None: 1}, {5: 1}])
def test_an_unusable_filter_key_is_rejected(bad):
    with pytest.raises(StorageInputError) as exc:
        validate_filter(bad)
    assert "field names" in str(exc.value)


# --------------------------------------------------------------------------- #
# Failure paths -- ids
# --------------------------------------------------------------------------- #


def test_a_bare_string_of_ids_is_rejected_because_it_is_a_sequence():
    """'abc' would mean ['a', 'b', 'c'] -- three deletes where one was meant."""
    with pytest.raises(StorageInputError) as exc:
        validate_ids("abc")
    assert "character" in str(exc.value)


def test_a_bare_bytes_of_ids_is_rejected():
    with pytest.raises(StorageInputError):
        validate_ids(b"abc")


@pytest.mark.parametrize("bad", [None, 3, {1, 2}])
def test_a_non_sequence_of_ids_is_rejected(bad):
    with pytest.raises(StorageInputError):
        validate_ids(bad)


@pytest.mark.parametrize("bad", ["", 5, None], ids=["empty", "int", "none"])
def test_an_unusable_id_is_rejected_by_position(bad):
    with pytest.raises(StorageInputError) as exc:
        validate_ids(["ok", bad])
    assert "[1]" in str(exc.value)


# --------------------------------------------------------------------------- #
# Failure paths -- vectors
# --------------------------------------------------------------------------- #


def test_an_empty_vector_is_rejected():
    with pytest.raises(StorageInputError) as exc:
        validate_vector([])
    assert "at least one" in str(exc.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_rejected(value):
    """NaN compares against nothing, so it poisons a search silently."""
    with pytest.raises(StorageInputError) as exc:
        validate_vector([1.0, value])
    assert "NaN" in str(exc.value) or "infinity" in str(exc.value)


@pytest.mark.parametrize("bad", ["abc", b"abc", bytearray(b"abc")])
def test_a_string_vector_is_rejected(bad):
    """A str is a sequence, and would otherwise become per-character floats."""
    with pytest.raises(StorageInputError):
        validate_vector(bad)


@pytest.mark.parametrize("bad", [["a", "b"], [True, 1.0], [None], [{1}]], ids=["strs", "bool", "none", "set"])
def test_a_non_numeric_vector_value_is_rejected(bad):
    """`bool` is an int in Python, and True-as-1.0 would go unnoticed."""
    with pytest.raises(StorageInputError):
        validate_vector(bad)


# --------------------------------------------------------------------------- #
# Failure paths -- k and embedded chunks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [0, -1, -10])
def test_a_non_positive_k_is_rejected(k):
    with pytest.raises(StorageInputError) as exc:
        validate_k(k)
    assert "no results" in str(exc.value)


@pytest.mark.parametrize("k", [True, 1.5, "3", None, object()], ids=["bool", "float", "str", "none", "obj"])
def test_a_non_int_k_is_rejected(k):
    with pytest.raises(StorageInputError):
        validate_k(k)


@pytest.mark.parametrize("bad", [[], "text", 3, None], ids=["empty", "str", "int", "none"])
def test_an_unusable_items_argument_is_rejected(bad):
    with pytest.raises(StorageInputError):
        validate_embedded_chunks(bad)


def test_a_non_EmbeddedChunk_item_is_rejected():
    with pytest.raises(StorageInputError) as exc:
        validate_embedded_chunks([core.Chunk(chunk_index=0, text="x")])
    assert "EmbeddedChunk" in str(exc.value)


def test_mixed_widths_are_rejected():
    items = _embedded(width=2) + [
        core.EmbeddedChunk(
            chunk=core.Chunk(chunk_index=9, text="wide", chunk_id="c9"),
            vector=(1.0, 2.0, 3.0),
            index_fingerprint="fp",
        )
    ]
    with pytest.raises(StorageInputError) as exc:
        validate_embedded_chunks(items)
    assert "3-wide" in str(exc.value)


def test_a_non_finite_vector_among_items_is_rejected():
    """Caught on the way in, not discovered after it is indexed."""
    items = [
        core.EmbeddedChunk(
            chunk=core.Chunk(chunk_index=0, text="x", chunk_id="c0"),
            vector=(float("nan"), 1.0),
            index_fingerprint="fp",
        )
    ]
    with pytest.raises(StorageInputError):
        validate_embedded_chunks(items)


# --------------------------------------------------------------------------- #
# Error hierarchy and message provenance
# --------------------------------------------------------------------------- #


def test_the_hierarchy_is_catchable_at_the_base():
    assert issubclass(StorageInputError, StorageError)
    assert issubclass(StorageBackendError, StorageError)


def test_the_base_is_distinguishable_from_a_programming_error():
    assert not issubclass(StorageError, ValueError)
    assert not issubclass(StorageError, TypeError)


def test_the_missing_dependency_error_is_an_import_error():
    assert issubclass(MissingStorageDependency, ImportError)
    assert not issubclass(MissingStorageDependency, StorageError)


def test_messages_name_the_component():
    """NFR-7: a caller can tell which component complained."""
    with pytest.raises(StorageInputError) as exc:
        validate_vector([])
    assert "vector-store-protocol" in str(exc.value)


def test_every_check_is_catchable_under_the_one_base_class():
    for check in (
        lambda: validate_filter({"a": {"b": 1}}),
        lambda: validate_ids("abc"),
        lambda: validate_vector([]),
        lambda: validate_k(0),
        lambda: validate_embedded_chunks([]),
    ):
        with pytest.raises(StorageError):
            check()


def test_the_manifest_declares_the_core_types_dependency():
    import json

    manifest = json.loads((COMPONENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == ["core-types"]
    assert manifest["required_packages"] == []
    assert manifest["optional_packages"] == []

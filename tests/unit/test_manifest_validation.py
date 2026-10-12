"""AC-13: every hostile manifest fixture is rejected, naming its own field.

What this file adds over ``test_registry.py``
---------------------------------------------
``test_registry.py`` already parametrises over these same eight fixtures and
asserts a per-fixture substring. Two things AC-13's wording -- "rejected with a
message naming the offending field" -- requires that a substring cannot express:

* **Attribution, not mention.** The fixtures are deliberately sparse
  (``KNOWLEDGE-6934aeda``), so every one of them also reports ``status``,
  ``requires_python`` and five further required fields as missing. A substring
  assertion passes for any of those boilerplate names whether or not the file
  was rejected *for* the field it names. Here the reason is parsed into
  ``field: detail`` entries and the boilerplate is discounted, so only the
  actual fault can satisfy the test.
* **Traceability from the criterion to the fixture set.**
  ``test_the_corruption_fixtures_are_all_present`` compares the on-disk files to
  the same dict that drives its own parametrisation, so deleting a fixture and
  its dict entry together passes unnoticed. The map below is written out from
  AC-13's own seven named corruptions, so a fixture cannot vanish without this
  test naming the criterion it was carrying.

Shape of a rejection
--------------------
Each fixture produces exactly ONE :class:`~rag_code_library.registry.Issue`,
whose reason is the words ``schema violation:`` followed by every pydantic
error joined with ``;`` in model-field order. Two consequences worth pinning
rather than assuming:

* The intended fault is **not always first**. For
  ``additional_property_rejected.json`` the unknown key ``author`` is reported
  last, after seven missing fields, because it is not a model field at all and
  pydantic emits model fields first. Asserting on the leading field would fail
  a correct implementation.
* A field name can appear in the **prose** as well as the label.
  ``path_traversal_destination.json`` reads "must be relative to the
  destination root", so the word "destination" occurs twice and a plain
  substring check proves nothing at all.

This file imports nothing from the component tree: it exercises
``rag_code_library.registry``, which is a package module, not a copy-ready
component.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import from the working tree rather than an installed copy, so the suite runs
# under any interpreter and tests what is actually here (AC-3 / NFR-5).
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_code_library.registry import Registry

_FIXTURES = _ROOT / "tests" / "fixtures" / "invalid_manifests"

#: AC-13's seven named corruptions, each mapped to the fixture carrying it.
#: Written out from the acceptance criterion rather than derived from
#: ``test_registry.CORRUPTIONS``, so the two cannot drift into agreeing with
#: each other while both drifting away from the criterion.
AC13_CORRUPTIONS = {
    "missing required field": "missing_required_field.json",
    "wrong type": "wrong_type_dependencies.json",
    "invalid category": "invalid_category_enum.json",
    "bad id pattern": "malformed_id_pattern.json",
    "bad version": "malformed_version_format.json",
    "extra property": "additional_property_rejected.json",
    "path-traversal destination": "path_traversal_destination.json",
}

#: The field each fixture must be blamed for. ``empty_required_string`` goes
#: beyond AC-13's seven -- it was added for its own sake, and is asserted here
#: because leaving it unattributed would let it rot.
EXPECTED_FIELD = {
    "additional_property_rejected.json": "author",
    "empty_required_string.json": "description",
    "invalid_category_enum.json": "category",
    "malformed_id_pattern.json": "id",
    "malformed_version_format.json": "version",
    "missing_required_field.json": "version",
    "path_traversal_destination.json": "destination",
    "wrong_type_dependencies.json": "dependencies",
}

#: Required fields that every sparse fixture reports as missing regardless of
#: what it was corrupted for. They are the noise the attribution test sees past;
#: discounting them is what stops the assertion from being satisfiable by the
#: cascade alone.
BOILERPLATE = {
    "status",
    "requires_python",
    "required_packages",
    "optional_packages",
    "external_services",
    "install",
    "missing_dependency_message",
}


def _labelled_fields(reason: str) -> list[str]:
    """The field names a reason blames, in the order it blames them.

    A reason reads ``schema violation: <field>: <detail>; <field>: <detail>``.
    Splitting on ``;`` and keeping only entries that *open* with a ``field:``
    label is what separates the blame from the surrounding prose -- necessary
    because a detail may itself contain ``;`` (the traversal message does, via
    ``; got '../../../escaped/...'``) and may itself mention the field name.
    """
    body = reason.partition("schema violation: ")[2]
    fields: list[str] = []
    for entry in body.split("; "):
        label, separator, _ = entry.partition(": ")
        # A label is a bare identifier; anything containing a space is prose
        # that merely happened to contain a colon.
        if separator and " " not in label:
            fields.append(label.strip())
    return fields


def _reason(fixture: str) -> str:
    """Load one fixture and return the reason it was rejected."""
    registry = Registry.load(_FIXTURES / fixture)
    assert not registry.ok, f"{fixture} was accepted"
    assert registry.issues, f"{fixture} failed but gave no reason at all"
    return registry.issues[0].reason


def test_every_corruption_named_by_ac13_still_has_a_fixture():
    """Each of AC-13's seven corruptions must still be carried by a fixture.

    This is the traceability half of AC-13, and the half that survives deletion:
    ``test_the_corruption_fixtures_are_all_present`` compares files to the dict
    that drives its own parametrisation, so removing both halves at once passes.
    Here the criterion is the source of truth, so a missing fixture fails with
    the name of the corruption it was holding up.
    """
    missing = {
        name: path
        for name, path in AC13_CORRUPTIONS.items()
        if not (_FIXTURES / path).is_file()
    }
    assert not missing, f"AC-13 names a corruption with no fixture: {missing}"


@pytest.mark.parametrize("fixture", sorted(EXPECTED_FIELD))
def test_the_fault_is_blamed_on_its_own_field_and_not_on_the_cascade(fixture):
    """AC-13: rejected with a message naming the offending field.

    Two assertions, and the second is the one with teeth. The first merely
    requires the field to be named somewhere; the second requires that the
    *only* field distinguishable from the sparse fixture's boilerplate is the
    one under test. Together they fail if a fixture is rejected for an
    unrelated reason while still mentioning the field its filename promises.
    """
    fields = set(_labelled_fields(_reason(fixture)))
    expected = EXPECTED_FIELD[fixture]

    assert expected in fields, f"{fixture} never names {expected!r}: {sorted(fields)}"
    assert fields - BOILERPLATE == {expected}, (
        f"{fixture} blames {sorted(fields - BOILERPLATE)}, but only {expected!r} "
        "should be distinguishable from the sparse-fixture cascade"
    )


def test_the_boilerplate_is_the_same_in_every_fixture():
    """Pin the noise the test above discounts.

    If a fixture were ever rewritten as a complete manifest plus one mutation,
    its cascade would shrink, the common set would shrink with it, and the
    attribution test would start comparing against a smaller exclusion set
    without saying so. Failing here forces that to be a deliberate choice
    instead of a silent weakening.
    """
    per_fixture = [
        set(_labelled_fields(_reason(fixture))) & BOILERPLATE
        for fixture in sorted(EXPECTED_FIELD)
    ]
    common = set.intersection(*per_fixture)
    assert common == BOILERPLATE, (
        "the cascade is no longer identical across fixtures; "
        f"only {sorted(common)} are shared. Boilerplate changed: "
        f"{sorted(BOILERPLATE - common) or 'none missing'}"
    )


def test_a_complete_manifest_reports_no_cascade_at_all(tmp_path):
    """The control that makes discounting the cascade legitimate.

    The exclusion set is only sound if those seven names appear *because*
    something is wrong. Written complete, a manifest comes back clean, so the
    boilerplate is confirmed to be a symptom of failure rather than a fact about
    how manifests are validated.
    """
    complete = {
        "id": "complete",
        "name": "Complete",
        "category": "chunking",
        "source": "complete.py",
        "version": "0.1.0",
        "dependencies": [],
        "description": "A manifest with nothing wrong with it.",
        "destination": "src/{project_package}/rag/complete.py",
        "status": "experimental",
        "requires_python": {"min": "3.12"},
        "required_packages": [],
        "optional_packages": [],
        "external_services": [],
        "install": "No additional packages required.",
        "missing_dependency_message": "complete declares nothing optional.",
    }
    directory = tmp_path / "complete"
    directory.mkdir()
    (directory / "complete.py").write_text("# fixture\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(complete), encoding="utf-8")

    registry = Registry.load(tmp_path)

    assert registry.ok, [str(issue) for issue in registry.issues]
    # `issues` is a tuple, so compare on truth rather than against a list.
    assert not registry.issues

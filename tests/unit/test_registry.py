"""The registry: discovery, validation, resolution and schema fidelity.

Covers FR-11 (actionable messages rather than tracebacks), FR-12 (one source of
component metadata), FR-24 (``validate_components.py``), NFR-6 (no drift) and
AC-5 (exit 0 over the real set, non-zero with a per-component reason when a
manifest is deliberately corrupted).

Three shapes of assertion carry the weight here:

* **Every failure produces an Issue, never an exception.** FR-11's "rather than
  tracebacks" is checked by loading things that are broken -- missing targets,
  empty directories, malformed JSON -- inside ``pytest.raises(...)``-free tests,
  so an escaping exception fails the test rather than being accepted as a result.
* **Exactly one issue, with exactly the intended reason.** When a test corrupts
  one thing, it asserts ``len(issues) == 1`` plus the specific message. Asserting
  only "not ok" would pass just as happily if the loader rejected everything for
  an unrelated reason.
* **The real component set loads.** Synthetic manifests prove the rules; the 32
  real ones prove the rules agree with reality.

This file imports nothing from the component tree as a component (AC-13): it
exercises ``rag_code_library.registry``, which is a package module, not a
copy-ready component.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Import from the working tree rather than from an installed copy.
#
# The gate command is literally `python -m pytest`, and `python` on this host is
# the system interpreter (3.14.5) where the package is NOT installed -- the
# installed copy lives in the uv venv (3.12.14). Prepending `src` makes the
# suite runnable under any interpreter, which is what AC-3 / NFR-5 require when
# the same suite must pass on 3.12 and 3.14, and it guarantees the working tree
# is tested rather than a stale installed snapshot.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_code_library.registry import (  # noqa: E402
    CATEGORIES,
    STATUSES,
    ComponentManifest,
    Registry,
    UnknownComponentError,
    load_schema,
)

_COMPONENTS = _ROOT / "components"
_FIXTURES = _ROOT / "tests" / "fixtures" / "invalid_manifests"
_SCRIPT = _ROOT / "scripts" / "validate_components.py"

# One corruption per fixture, with the signal that identifies it. The fixtures
# are deliberately sparse -- each was cut down to the field under test -- so
# they also report every other required field as missing. Asserting on the
# *intended* signal is what keeps these tests about the corruption they name
# rather than about whatever the message happens to contain.
CORRUPTIONS = {
    "additional_property_rejected.json": "author",
    "empty_required_string.json": "description: String should have at least 1 character",
    "invalid_category_enum.json": "category: Input should be",
    "malformed_id_pattern.json": "id: String should match pattern",
    "malformed_version_format.json": "version: String should match pattern",
    "missing_required_field.json": "version: Field required",
    "path_traversal_destination.json": "destination: Value error",
    "wrong_type_dependencies.json": "dependencies: Input should be a valid tuple",
}

# A manifest satisfying every field the schema requires. Tests override exactly
# one thing, so the single issue they assert is the thing they changed.
_BASE: dict = {
    "id": "sample",
    "name": "Sample",
    "category": "chunking",
    "source": "sample.py",
    "version": "0.1.0",
    "dependencies": [],
    "description": "A component used only by the registry tests.",
    "destination": "src/{project_package}/rag/sample.py",
    "status": "experimental",
    "requires_python": {"min": "3.12"},
    "required_packages": [],
    "optional_packages": [],
    "external_services": [],
    "install": "No additional packages required.",
    "missing_dependency_message": "sample declares no optional dependencies.",
}


def _component(root: Path, component_id: str = "sample", **overrides) -> Path:
    """Write a component directory: a valid manifest plus its declared source.

    ``sample.py`` is always written. Overriding ``source`` therefore makes the
    *declared* file the one that does not exist, which is how the missing-file
    test isolates itself from every other check.
    """
    directory = root / component_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sample.py").write_text("# fixture\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({**_BASE, "id": component_id, **overrides}), encoding="utf-8"
    )
    return directory


def _run_script(*args: str) -> subprocess.CompletedProcess:
    """Run validate_components.py as a real process, so its exit code is real."""
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        timeout=180,
    )


def _property(document: dict, name: str) -> dict:
    """Resolve one property of a JSON Schema, following ``$ref`` if present."""
    prop = document["properties"][name]
    ref = prop.get("$ref")
    if ref is None:
        return prop
    return document["$defs"][ref.rsplit("/", 1)[-1]]


# --------------------------------------------------------------------------- #
# Discovery and loading: broken input is reported, never raised (FR-11)
# --------------------------------------------------------------------------- #


def test_the_real_component_set_loads_cleanly():
    """The set the whole project reads must be valid, and entirely present."""
    registry = Registry.load(_COMPONENTS)

    assert registry.ok, [str(issue) for issue in registry.issues]
    # ids come from a dict, so equality with the entry count is what proves no
    # two manifests declared the same id -- the duplicate would have collapsed.
    assert len(registry.ids) == registry.count
    # And discovery really walked the tree rather than stopping early.
    assert registry.count == len(list(_COMPONENTS.rglob("manifest.json")))


def test_discovery_finds_every_manifest_and_ignores_unrelated_files(tmp_path):
    _component(tmp_path, "alpha")
    _component(tmp_path, "beta")
    (tmp_path / "README.md").write_text("not a manifest", encoding="utf-8")

    registry = Registry.load(tmp_path)

    assert registry.ids == ("alpha", "beta")
    assert registry.ok


def test_a_missing_target_is_an_issue_not_an_exception(tmp_path):
    registry = Registry.load(tmp_path / "does-not-exist")

    assert not registry.ok
    assert len(registry.issues) == 1
    assert "no such file or directory" in str(registry.issues[0])


def test_an_empty_directory_never_passes_as_valid(tmp_path):
    """Zero manifests found must be a failure, not "0 problems" (anti-vacuity).

    Otherwise pointing the validator at the wrong path would report success
    while checking nothing at all.
    """
    registry = Registry.load(tmp_path)

    assert not registry.ok
    assert len(registry.issues) == 1
    assert "no manifest files found" in str(registry.issues[0])


def test_unparseable_json_is_reported_with_the_file_name(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")

    registry = Registry.load(tmp_path)

    assert not registry.ok
    assert len(registry.issues) == 1
    assert "not valid JSON" in str(registry.issues[0])


def test_a_json_document_that_is_not_an_object_is_reported(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    registry = Registry.load(tmp_path)

    assert not registry.ok
    assert "must be a JSON object" in str(registry.issues[0])


def test_issue_str_names_both_the_component_and_the_file():
    """The message is the deliverable; it must be locatable as well as readable."""
    from rag_code_library.registry import Issue

    labelled = Issue(reason="broken", path="components/x/manifest.json", component="x")
    assert str(labelled) == "x (components/x/manifest.json): broken"

    unlabelled = Issue(reason="broken", path="components/x/manifest.json")
    assert str(unlabelled) == "components/x/manifest.json: broken"


# --------------------------------------------------------------------------- #
# Schema violations (FR-24, AC-5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture,signal", sorted(CORRUPTIONS.items()))
def test_each_deliberately_corrupt_fixture_is_rejected_with_its_reason(fixture, signal):
    """AC-5: a corrupted manifest must fail, naming the reason it failed."""
    registry = Registry.load(_FIXTURES / fixture)

    assert not registry.ok, f"{fixture} was accepted"
    assert len(registry.issues) == 1
    assert signal in str(registry.issues[0]), str(registry.issues[0])


def test_the_corruption_fixtures_are_all_present():
    """Otherwise a fixture could vanish and its parametrised test with it."""
    present = {path.name for path in _FIXTURES.glob("*.json")}
    assert present == set(CORRUPTIONS)


def test_an_unrecognised_field_is_rejected_rather_than_ignored(tmp_path):
    """A typo like `dependancies` must fail, not read as "no dependencies"."""
    _component(tmp_path, "typo", dependancies=[])

    registry = Registry.load(tmp_path)

    assert not registry.ok
    assert len(registry.issues) == 1
    assert "dependancies" in str(registry.issues[0])
    assert "typo" in str(registry.issues[0])


# --------------------------------------------------------------------------- #
# Cross-manifest checks (FR-11)
# --------------------------------------------------------------------------- #


def test_a_duplicate_id_names_both_manifests(tmp_path):
    _component(tmp_path, "twin")
    _component(tmp_path / "elsewhere", "twin")

    registry = Registry.load(tmp_path)

    assert len(registry.issues) == 1, [str(i) for i in registry.issues]
    issue = registry.issues[0]
    assert issue.component == "twin"
    assert "declared by 2 manifests" in issue.reason
    assert "twin/manifest.json" in issue.reason
    assert "elsewhere/twin/manifest.json" in issue.reason


def test_a_declared_source_that_does_not_exist_is_reported(tmp_path):
    _component(tmp_path, "phantom", source="ghost.py")

    registry = Registry.load(tmp_path)

    assert len(registry.issues) == 1, [str(i) for i in registry.issues]
    assert "source file does not exist: ghost.py" in str(registry.issues[0])


def test_a_dependency_on_an_unloaded_component_is_reported(tmp_path):
    _component(tmp_path, "needy", dependencies=["nowhere"])

    registry = Registry.load(tmp_path)

    assert len(registry.issues) == 1, [str(i) for i in registry.issues]
    message = str(registry.issues[0])
    assert "unknown component 'nowhere'" in message
    assert "loaded ids: needy" in message


def test_a_component_cannot_depend_on_itself(tmp_path):
    _component(tmp_path, "ouroboros", dependencies=["ouroboros"])

    registry = Registry.load(tmp_path)

    assert len(registry.issues) == 1, [str(i) for i in registry.issues]
    assert "depends on itself" in str(registry.issues[0])


# --------------------------------------------------------------------------- #
# Destination safety beyond the schema's pattern (see validator docstring)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "destination,pattern",
    [
        # Forward slashes, so the backslash rule below does not fire first and
        # the drive-qualified rule is what is actually being exercised.
        ("C:/Windows/Temp/component.py", "drive-qualified"),
        ("src\\rag\\component.py", "forward slashes"),
    ],
)
def test_destinations_the_schema_pattern_allows_are_still_rejected(
    tmp_path, destination, pattern
):
    """The schema's pattern passes these; the registry must not.

    Both are accepted by ``schemas/component-manifest.schema.json``'s
    ``destination`` pattern -- measured, not assumed -- so this is the extra
    rule the validator applies on purpose. See KNOWLEDGE-82bf789c.
    """
    _component(tmp_path, "winpath", destination=destination)

    registry = Registry.load(tmp_path)

    assert len(registry.issues) == 1, [str(i) for i in registry.issues]
    assert pattern in str(registry.issues[0])


def test_a_normal_destination_passes_both_layers(tmp_path):
    """The safety rule must not reject anything the real components use."""
    _component(tmp_path, "wellbehaved", destination="src/{project_package}/rag/x.py")

    assert Registry.load(tmp_path).ok


# --------------------------------------------------------------------------- #
# Resolution (FR-11)
# --------------------------------------------------------------------------- #


def test_resolution_joins_declared_paths_onto_the_directory_the_manifest_is_in():
    registry = Registry.load(_COMPONENTS)

    for entry in registry:
        resolved = registry.resolve(entry.manifest.id)
        assert resolved.directory == entry.directory
        assert resolved.source == entry.directory / entry.manifest.source
        assert resolved.source.is_file()


def test_resolution_returns_the_declared_paths_for_a_known_component():
    registry = Registry.load(_COMPONENTS)
    resolved = registry.resolve("core-types")

    assert resolved.source.name == "core_types.py"
    assert resolved.source.is_file()
    assert resolved.documentation is not None and resolved.documentation.is_file()
    assert resolved.tests is not None and resolved.tests.name == "test_component.py"
    # The template is not a path yet: {project_package} is substituted at add
    # time, which is why it is read off the manifest rather than the resolver.
    assert resolved.manifest.destination.endswith("core_types.py")
    assert "{project_package}" in resolved.manifest.destination


def test_resolving_an_unknown_id_names_the_ones_that_do_exist():
    registry = Registry.load(_COMPONENTS)

    with pytest.raises(UnknownComponentError) as excinfo:
        registry.resolve("does-not-exist")

    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "core-types" in message  # a real alternative, not just a refusal


def test_the_registry_answers_whether_an_id_is_loaded():
    registry = Registry.load(_COMPONENTS)
    assert "core-types" in registry
    assert "does-not-exist" not in registry
    assert registry.get("does-not-exist") is None


# --------------------------------------------------------------------------- #
# Drift between the model and the schema (FR-12, NFR-6)
# --------------------------------------------------------------------------- #


def test_model_and_schema_declare_the_same_fields():
    """A field added to one file and not the other is the drift FR-12 forbids."""
    schema_fields = set(load_schema()["properties"])
    model_fields = set(ComponentManifest.model_fields)

    missing = schema_fields - model_fields
    extra = model_fields - schema_fields
    assert not missing, f"model does not model schema fields: {sorted(missing)}"
    assert not extra, f"model requires fields the schema does not: {sorted(extra)}"


def test_model_and_schema_agree_on_which_fields_are_required():
    """Requiredness is half the contract: `optional` here, `required` there."""
    schema_required = set(load_schema()["required"])
    model_required = {
        name for name, field in ComponentManifest.model_fields.items() if field.is_required()
    }

    missing = schema_required - model_required
    extra = model_required - schema_required
    assert not missing, f"schema requires them, model lets them go: {sorted(missing)}"
    assert not extra, f"model requires them, schema lets them go: {sorted(extra)}"


def test_model_and_schema_agree_on_the_enums():
    schema = load_schema()
    model = ComponentManifest.model_json_schema()

    for name in ("category", "status"):
        assert set(_property(model, name)["enum"]) == set(_property(schema, name)["enum"])

    # And the tuples exported for the CLI still match what the model enforces.
    assert set(CATEGORIES) == set(_property(schema, "category")["enum"])
    assert set(STATUSES) == set(_property(schema, "status")["enum"])


def test_the_schema_file_is_where_the_code_says_it_is():
    """Both load_schema() and this test derive one path; keep them honest."""
    assert load_schema()["title"] == "RAG Component Manifest"


# --------------------------------------------------------------------------- #
# validate_components.py itself (FR-24, AC-5)
# --------------------------------------------------------------------------- #


def test_the_validate_script_exits_zero_over_the_real_component_set():
    """AC-5, first half: 0 over the full component set."""
    result = _run_script()

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK:"), result.stdout
    assert result.stderr == ""


def test_the_validate_script_exits_nonzero_naming_each_corrupt_component():
    """AC-5, second half: non-zero with a per-component reason.

    The target is given as an *absolute* path on purpose: output must still be
    repository-relative, or two CI runs could never be diffed.
    """
    result = _run_script(str(_FIXTURES))

    assert result.returncode == 1, result.stdout
    assert str(_ROOT) not in result.stderr, "output leaked an absolute path"
    assert "tests/fixtures/invalid_manifests/" in result.stderr
    assert "malformed_version_format.json" in result.stderr
    assert "version: String should match pattern" in result.stderr
    assert "FAILED:" in result.stderr

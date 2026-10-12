"""Component tests for `document-discovery`.

Runs standalone: ``python -m pytest components/ingestion/document_discovery/tests``
from the repository root. Does not import `rag_code_library`.

The component owns two questions, so the tests are built around both: *what is
in this tree*, and *what changed since last time*. Discovery is allowed to find
nothing; it is not allowed to raise about one broken entry, and it is not
allowed to leave the root it was given.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from document_discovery import (
    DEFAULT_ALLOWED_EXTENSIONS,
    Discovery,
    DiscoveryDelta,
    DiscoveryError,
    DiscoveryReport,
    DiscoveredFile,
    EntryState,
    InvalidScanOption,
    InvalidSourceRoot,
    SkipReason,
    classify,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small tree with a nested document and one deliberately odd file."""
    root = tmp_path / "corpus"
    (root / "guide").mkdir(parents=True)
    (root / "guide" / "intro.md").write_text("# Intro\n", encoding="utf-8")
    (root / "guide" / "setup.md").write_text("# Setup\n", encoding="utf-8")
    (root / "notes.txt").write_text("hello\n", encoding="utf-8")
    (root / "scratch.bin").write_bytes(b"\x00\x01\x02")
    return root


@pytest.fixture
def scan(corpus: Path) -> Discovery:
    """A scan accepting the document formats and nothing else."""
    return Discovery(corpus, allowed_extensions={".md", ".txt"})


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_documents_are_found_recursively_and_in_declared_order(scan, corpus):
    report = scan.discover()
    assert report.paths() == ("guide/intro.md", "guide/setup.md", "notes.txt")
    assert report.source_root == corpus.resolve()


def test_each_record_carries_the_cheap_facts_discovery_owns(scan, corpus):
    record = next(r for r in scan.discover().files if r.relative_path == "notes.txt")
    on_disk = corpus / "notes.txt"
    assert record.size_bytes == on_disk.stat().st_size
    assert record.modified_ns == on_disk.stat().st_mtime_ns
    assert record.path == on_disk.resolve()
    assert record.stamp() == (record.size_bytes, record.modified_ns)


def test_a_non_recursive_scan_does_not_descend(corpus):
    report = Discovery(corpus, recursive=False, allowed_extensions={".md", ".txt"}).discover()
    assert report.paths() == ("notes.txt",)
    assert report.recursive is False


def test_an_extension_allow_list_filters_and_counts(corpus):
    report = Discovery(corpus, allowed_extensions={".md"}).discover()
    assert report.paths() == ("guide/intro.md", "guide/setup.md")
    # notes.txt and scratch.bin are both counted, neither is returned.
    assert report.filtered_by_extension == 2


def test_an_extension_is_matched_case_insensitively(tmp_path):
    (tmp_path / "UPPER.MD").write_text("x", encoding="utf-8")
    report = Discovery(tmp_path, allowed_extensions={".md"}).discover()
    assert report.paths() == ("UPPER.MD",)


def test_extensions_may_be_given_without_a_leading_dot(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    report = Discovery(tmp_path, allowed_extensions={"md"}).discover()
    assert report.paths() == ("a.md",)


def test_an_empty_allow_list_matches_nothing_without_raising(corpus):
    report = Discovery(corpus, allowed_extensions=set()).discover()
    assert report.files == ()
    assert report.filtered_by_extension == 4  # every file in the corpus


def test_no_allow_list_accepts_everything(corpus):
    scan = Discovery(corpus)
    report = scan.discover()
    assert report.paths() == ("guide/intro.md", "guide/setup.md", "notes.txt", "scratch.bin")
    assert scan.allowed_extensions is None
    assert report.filtered_by_extension == 0


def test_the_default_allow_list_is_a_document_set():
    assert ".pdf" in DEFAULT_ALLOWED_EXTENSIONS
    assert ".md" in DEFAULT_ALLOWED_EXTENSIONS


def test_an_exclusion_pattern_drops_a_whole_subtree(corpus):
    report = Discovery(corpus, exclude_patterns=["guide"]).discover()
    assert report.paths() == ("notes.txt", "scratch.bin")
    assert report.filtered_by_pattern == 1


def test_an_exclusion_pattern_can_match_the_relative_path(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "drafts").mkdir()
    (tmp_path / "drafts" / "b.md").write_text("b", encoding="utf-8")
    report = Discovery(tmp_path, exclude_patterns=["drafts/*"]).discover()
    assert report.paths() == ("keep/a.md",)


def test_ordering_is_identical_across_repeated_scans(scan):
    assert scan.discover().paths() == scan.discover().paths()


def test_ordering_breaks_ties_on_relative_path(tmp_path):
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    report = Discovery(tmp_path).discover()
    assert report.paths() == ("a.md", "b.md")


def test_sorting_by_size_orders_ascending_with_a_stable_tiebreak(tmp_path):
    (tmp_path / "small.md").write_text("x", encoding="utf-8")
    (tmp_path / "large.md").write_text("x" * 100, encoding="utf-8")
    report = Discovery(tmp_path, sort_by="size_bytes").discover()
    assert report.paths() == ("small.md", "large.md")


def test_total_entries_seen_adds_up(scan):
    report = scan.discover()
    assert report.total_entries_seen == len(report.files) + len(report.skipped) + 1


def test_relative_paths_are_posix_form(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.md").write_text("x", encoding="utf-8")
    report = Discovery(tmp_path).discover()
    assert report.paths() == ("nested/a.md",)


def test_the_snapshot_round_trips_through_plain_data(scan):
    stamps = scan.discover().stamps()
    assert all(isinstance(key, str) and len(value) == 2 for key, value in stamps.items())
    assert all(isinstance(value[0], int) and isinstance(value[1], int) for value in stamps.values())


def test_the_source_root_property_is_resolved(corpus):
    assert Discovery(corpus).source_root == corpus.resolve()


# ---------------------------------------------------------------------------
# Change detection -- AC-23
# ---------------------------------------------------------------------------


def test_editing_adding_and_deleting_are_each_classified(corpus, scan):
    previous = scan.discover().stamps()

    (corpus / "notes.txt").write_text("hello, edited\n", encoding="utf-8")
    (corpus / "new.md").write_text("new\n", encoding="utf-8")
    (corpus / "guide" / "setup.md").unlink()

    delta = scan.delta(previous)
    assert delta.added == ("new.md",)
    assert delta.modified == ("notes.txt",)
    assert delta.deleted == ("guide/setup.md",)
    assert delta.unchanged == ("guide/intro.md",)
    assert not delta.is_empty


def test_state_reports_each_path_including_one_in_neither_scan(corpus, scan):
    previous = scan.discover().stamps()
    (corpus / "notes.txt").write_text("changed\n", encoding="utf-8")
    (corpus / "new.md").write_text("new\n", encoding="utf-8")
    delta = scan.delta(previous)

    assert delta.state("notes.txt") is EntryState.MODIFIED
    assert delta.state("new.md") is EntryState.ADDED
    assert delta.state("guide/setup.md") is EntryState.UNCHANGED
    assert delta.state("never-seen.md") is EntryState.UNKNOWN


def test_changed_lists_every_path_requiring_work(corpus, scan):
    previous = scan.discover().stamps()
    (corpus / "notes.txt").write_text("changed\n", encoding="utf-8")
    (corpus / "a.md").write_text("a\n", encoding="utf-8")
    delta = scan.delta(previous)
    assert delta.changed() == ("a.md", "notes.txt")


def test_scanning_an_unchanged_tree_twice_is_an_empty_delta(scan):
    previous = scan.discover().stamps()
    assert scan.delta(previous).is_empty


def test_an_empty_delta_from_nothing_is_empty():
    assert classify({}, {}).is_empty


def test_classify_is_pure_and_usable_on_plain_mappings():
    previous = {"a.md": (10, 100), "b.md": (20, 200)}
    current = {"a.md": (11, 100), "c.md": (30, 300)}
    delta = classify(previous, current)
    assert delta.added == ("c.md",)
    assert delta.modified == ("a.md",)
    assert delta.deleted == ("b.md",)
    assert delta.unchanged == ()


def test_a_snapshot_can_be_read_back_and_still_classify(scan, tmp_path):
    """A snapshot is plain data, so it survives being written to disk."""
    import json

    snapshot_file = tmp_path / "snapshot.json"
    snapshot_file.write_text(json.dumps(scan.discover().stamps()), encoding="utf-8")
    previous = {key: tuple(value) for key, value in json.loads(snapshot_file.read_text()).items()}
    assert isinstance(scan.delta(previous), DiscoveryDelta)


# ---------------------------------------------------------------------------
# Failure paths -- an entry must never abort the scan
# ---------------------------------------------------------------------------


def test_a_missing_source_root_is_refused_at_construction(tmp_path):
    with pytest.raises(InvalidSourceRoot) as excinfo:
        Discovery(tmp_path / "nope")
    assert "nope" in str(excinfo.value)


def test_a_source_root_that_is_a_file_is_refused(tmp_path):
    target = tmp_path / "a-file.md"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(InvalidSourceRoot):
        Discovery(target)


def test_a_non_positive_sort_is_refused(corpus):
    with pytest.raises(InvalidScanOption) as excinfo:
        Discovery(corpus, sort_by="invented")
    assert "invented" in str(excinfo.value)


def test_a_non_boolean_recursive_is_refused(corpus):
    with pytest.raises(InvalidScanOption):
        Discovery(corpus, recursive=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_a_blank_exclusion_pattern_is_refused(corpus, bad):
    with pytest.raises(InvalidScanOption):
        Discovery(corpus, exclude_patterns=[bad])  # type: ignore[list-item]


def test_an_unreadable_directory_is_reported_and_the_scan_continues(
    corpus, monkeypatch
):
    """One broken directory must cost one record, not the whole corpus."""
    blocked = corpus / "blocked"
    blocked.mkdir()
    (blocked / "hidden.md").write_text("nope", encoding="utf-8")

    real_scandir = os.scandir

    def guarded(path):
        if Path(path) == blocked:
            raise PermissionError(13, "Permission denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded)
    report = Discovery(corpus).discover()

    assert "guide/intro.md" in report.paths()
    assert "notes.txt" in report.paths()
    unreadable = [s for s in report.skipped if s.reason is SkipReason.UNREADABLE]
    assert unreadable and unreadable[0].relative_path == "blocked"
    assert "Permission denied" in unreadable[0].detail


def test_an_unreadable_file_is_reported_and_the_scan_continues(corpus, monkeypatch):
    """A file whose stat() raises must cost one record, not the scan.

    ``DirEntry.stat`` is implemented in C and does not route through the
    Python-level ``os.stat``, so patching that would test nothing. The entry
    itself is stood in for instead.
    """

    class UnreadableEntry:
        """A DirEntry whose stat raises, exactly as a denied file does."""

        name = "notes.txt"

        def __init__(self, root: Path) -> None:
            self.path = str(root / "notes.txt")

        def is_symlink(self) -> bool:
            return False

        def is_dir(self, *, follow_symlinks: bool = True) -> bool:
            return False

        def is_file(self, *, follow_symlinks: bool = True) -> bool:
            return True

        def stat(self, *, follow_symlinks: bool = True):
            raise PermissionError(13, "Permission denied")

    real_scandir = os.scandir

    def guarded(path):
        entries = list(real_scandir(path))
        if Path(path) == corpus:
            entries = [e for e in entries if e.name != "notes.txt"]
            entries.append(UnreadableEntry(corpus))
        return iter(entries)

    monkeypatch.setattr(os, "scandir", guarded)
    report = Discovery(corpus).discover()

    assert "guide/intro.md" in report.paths()
    assert not any(r.relative_path == "notes.txt" for r in report.files)
    unreadable = [s for s in report.skipped if s.reason is SkipReason.UNREADABLE]
    assert unreadable and unreadable[0].relative_path == "notes.txt"


def test_a_link_out_of_the_root_is_reported_not_followed(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "in.md").write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("not ours\n", encoding="utf-8")

    link = root / "escape.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - host dependent
        pytest.skip(f"this host does not permit symlinks: {exc.strerror or exc}")

    report = Discovery(root).discover()
    assert report.paths() == ("in.md",)
    escaped = [s for s in report.skipped if s.reason is SkipReason.SYMLINK_ESCAPE]
    assert escaped and escaped[0].relative_path == "escape.md"
    assert "outside.md" in escaped[0].detail


def test_a_directory_link_out_of_the_root_is_reported_not_walked(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "in.md").write_text("inside\n", encoding="utf-8")
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "secret.md").write_text("not ours\n", encoding="utf-8")

    link = root / "away"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host dependent
        pytest.skip(f"this host does not permit symlinks: {exc.strerror or exc}")

    report = Discovery(root).discover()
    assert report.paths() == ("in.md",)
    assert any(s.reason is SkipReason.SYMLINK_ESCAPE for s in report.skipped)


def test_a_link_loop_terminates_and_is_walked_once(tmp_path):
    """The regression a naive walk has: following a loop never ends."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("a\n", encoding="utf-8")
    try:
        (root / "loop").symlink_to(root, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host dependent
        pytest.skip(f"this host does not permit symlinks: {exc.strerror or exc}")

    report = Discovery(root).discover()
    assert report.paths() == ("a.md",)


def test_a_link_inside_the_root_is_reported_once_not_twice(tmp_path):
    """The same physical file must not be handed to ingestion twice."""
    root = tmp_path / "corpus"
    (root / "real").mkdir(parents=True)
    (root / "real" / "doc.md").write_text("x", encoding="utf-8")
    try:
        (root / "alias.md").symlink_to(root / "real" / "doc.md")
    except OSError as exc:  # pragma: no cover - host dependent
        pytest.skip(f"this host does not permit symlinks: {exc.strerror or exc}")

    report = Discovery(root).discover()
    assert len(report.files) == 1
    assert {s.reason for s in report.skipped} == {SkipReason.DUPLICATE}
    duplicate = report.skipped[0]
    assert "same file is already reported as" in duplicate.detail


def _make_junction(link: Path, target: Path) -> str | None:
    """Create a directory junction; return an OS reason string if refused.

    ``mklink /J`` needs no elevation on Windows, unlike ``os.symlink``, so a
    junction is what actually gets exercised on a host that refuses symlinks.
    """
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    return f"mklink /J refused: {(result.stderr or result.stdout).strip()}"


def test_a_junction_out_of_the_root_is_reported_not_walked(tmp_path):
    """The containment rule, verified on a host that may refuse symlinks."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "in.md").write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.md").write_text("not ours\n", encoding="utf-8")

    reason = _make_junction(root / "away", outside)
    if reason:  # pragma: no cover - host dependent
        pytest.skip(reason)

    report = Discovery(root).discover()
    assert report.paths() == ("in.md",)
    assert not any("secret.md" in record.relative_path for record in report.files)
    escaped = [s for s in report.skipped if s.reason is SkipReason.SYMLINK_ESCAPE]
    assert escaped and escaped[0].relative_path == "away"
    assert "elsewhere" in escaped[0].detail


def test_a_junction_inside_the_root_is_walked_once(tmp_path):
    """A junction is only refused when it would leave the root."""
    root = tmp_path / "corpus"
    (root / "real").mkdir(parents=True)
    (root / "real" / "doc.md").write_text("x", encoding="utf-8")

    reason = _make_junction(root / "alias", root / "real")
    if reason:  # pragma: no cover - host dependent
        pytest.skip(reason)

    report = Discovery(root).discover()
    assert report.paths() == ("real/doc.md",)
    assert report.skipped == ()


def test_a_non_regular_file_is_reported_not_opened(tmp_path):
    if not hasattr(os, "mkfifo"):  # pragma: no cover - Windows
        pytest.skip("os.mkfifo is POSIX-only; Windows cannot make a FIFO with the stdlib")
    os.mkfifo(tmp_path / "pipe.md")
    report = Discovery(tmp_path).discover()
    assert report.files == ()
    assert any(s.reason is SkipReason.NOT_A_FILE for s in report.skipped)


# ---------------------------------------------------------------------------
# Results are values, not mutable bags
# ---------------------------------------------------------------------------


def test_a_discovered_file_is_immutable(scan):
    record = scan.discover().files[0]
    with pytest.raises(AttributeError):
        record.size_bytes = 0  # type: ignore[misc]


def test_a_report_is_immutable(scan):
    report = scan.discover()
    with pytest.raises(AttributeError):
        report.files = ()  # type: ignore[misc]


def test_the_error_hierarchy_is_a_single_root():
    assert issubclass(InvalidSourceRoot, DiscoveryError)
    assert issubclass(InvalidScanOption, DiscoveryError)


def test_the_component_imports_with_no_third_party_packages():
    """FR-3: nothing here may need a package the library does not declare."""
    import document_discovery

    assert not document_discovery.__dict__.get("__third_party__", False)
    source = Path(document_discovery.__file__).read_text(encoding="utf-8")
    for forbidden in ("pydantic", "numpy", "pandas", "requests"):
        assert f"import {forbidden}" not in source

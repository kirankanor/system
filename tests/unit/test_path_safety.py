"""Hostile-path tests run against the real Windows filesystem of this host.

AC-11 / FR-34 require actual symlinks and actual junctions, created for real
and then exercised through :class:`SafeDestination`. Where the host refuses to
create one, the test skips with the OS reason recorded in the skip message
rather than pretending to have covered the case.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import from the working tree rather than from an installed copy.
#
# The gate command is literally `python -m pytest`, and `python` on this host is
# the system interpreter (3.14.5) where the package is NOT installed -- the
# installed copy lives in the uv venv (3.12.14). Prepending `src` makes the
# suite runnable under any interpreter without a second install, which is what
# AC-3 / NFR-5 require when the same suite must pass on 3.12 and 3.14. It also
# guarantees we test the working tree, never a stale installed snapshot.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_code_library.security import SafeDestination, UnsafePathError  # noqa: E402

# AC-11 names these explicitly; each must raise and write nothing.
TRAVERSAL = "../outside.txt"
NESTED_TRAVERSAL = "pkg/../../outside.txt"
ABSOLUTE_OUTSIDE = None  # filled per-test with a real absolute path
DRIVE_LETTER = r"C:\Windows\Temp\ragkit-hostile.txt"
DRIVE_RELATIVE = "C:ragkit-hostile.txt"
UNC = r"\\server\share\ragkit-hostile.txt"
EXTENDED = r"\\?\C:\ragkit-hostile.txt"
ADS = "payload.py:ragkit-stream"
RESERVED = ["CON.txt", "com1", "LPT9.py", "nested/AUX"]


def _codes(excinfo) -> str:
    return excinfo.value.code


def _assert_wrote_nothing(root: Path, outside: Path) -> None:
    """Prove a rejection performed zero writes, anywhere."""
    forbidden = [
        outside / "outside.txt",
        Path(DRIVE_LETTER),
        Path(r"C:\Windows\Temp\ragkit-hostile.txt").parent / "ragkit-hostile.txt",
    ]
    for path in forbidden:
        assert not path.exists(), f"rejected path wrote {path}"
    # No file anywhere under the root may be new except the ones the test made.
    for candidate in root.rglob("*"):
        assert candidate.name != "ragkit-hostile.txt", f"wrote {candidate}"


@pytest.fixture()
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    return root, outside


# --------------------------------------------------------------------------- #
# AC-11: every listed hostile form fails with a typed error and writes nothing
# --------------------------------------------------------------------------- #


def test_relative_traversal_rejected(sandbox):
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(TRAVERSAL, "pwned")
    assert _codes(exc) == "traversal"
    _assert_wrote_nothing(root, outside)


def test_traversal_at_depth_rejected(sandbox):
    """`..` must be caught at any depth, not only at the start."""
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(NESTED_TRAVERSAL, "pwned")
    assert _codes(exc) == "traversal"
    _assert_wrote_nothing(root, outside)


def test_absolute_path_outside_project_rejected(sandbox, tmp_path):
    root, _ = sandbox
    outside_file = tmp_path / "absolute-escape.txt"
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(str(outside_file), "pwned")
    # A POSIX-style absolute path has no drive; a Windows one is caught earlier.
    assert _codes(exc) in {"absolute", "drive-letter"}
    assert not outside_file.exists()


def test_drive_letter_path_rejected(sandbox):
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(DRIVE_LETTER, "pwned")
    assert _codes(exc) == "drive-letter"
    _assert_wrote_nothing(root, outside)


def test_drive_relative_path_rejected(sandbox):
    """`C:file` resolves against that drive's CWD, not ours."""
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(DRIVE_RELATIVE, "pwned")
    assert _codes(exc) == "drive-relative"
    _assert_wrote_nothing(root, outside)


def test_unc_path_rejected(sandbox):
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(UNC, "pwned")
    assert _codes(exc) == "unc"
    _assert_wrote_nothing(root, outside)


def test_extended_length_prefix_rejected(sandbox):
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(EXTENDED, "pwned")
    assert _codes(exc) == "extended-prefix"
    _assert_wrote_nothing(root, outside)


def test_alternate_data_stream_rejected(sandbox):
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(ADS, "pwned")
    assert _codes(exc) == "ads"
    # The visible file beside the stream must not appear either.
    assert not (root / "payload.py").exists()
    _assert_wrote_nothing(root, outside)


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_device_names_rejected(sandbox, name):
    """Reserved device names are rejected in ANY segment, case-insensitively."""
    root, outside = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(name, "pwned")
    assert _codes(exc) == "reserved-name"
    _assert_wrote_nothing(root, outside)


def test_error_is_typed_and_actionable(sandbox):
    """FR-17 requires a typed error; NFR-7 requires it to name the fix."""
    root, _ = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text(TRAVERSAL, "pwned")
    err = exc.value
    assert isinstance(err, ValueError)
    assert err.code and err.path and err.root
    assert "outside" in str(err) or "escape" in str(err) or "'..'" in str(err)
    assert err.details["code"] == err.code


# --------------------------------------------------------------------------- #
# FR-34: real symlinks and real junctions on this filesystem
# --------------------------------------------------------------------------- #


def _make_symlink(link: Path, target: Path) -> str | None:
    """Create a directory symlink; return an OS skip reason if refused."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        return f"host refused directory symlink: {exc}"
    return None


def _make_junction(link: Path, target: Path) -> str | None:
    """Create a junction via mklink /J, which needs no elevation; return reason if refused."""
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        shell=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return f"host refused junction (mklink /J exit {result.returncode}): {detail}"
    if not link.exists() and not link.is_symlink():
        return "mklink /J reported success but no link was created"
    return None


def test_symlink_pointing_outside_is_rejected(sandbox):
    root, outside = sandbox
    victim = outside / "victim"
    victim.mkdir()
    reason = _make_symlink(root / "escape", victim)
    if reason:
        pytest.skip(f"FR-34 skip, symlink not creatable here: {reason}")

    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text("escape/pwned.txt", "pwned")

    assert _codes(exc) in {"reparse-point", "not-contained"}
    assert not (victim / "pwned.txt").exists(), "wrote through an escaping symlink"


def test_junction_pointing_outside_is_rejected(sandbox):
    root, outside = sandbox
    victim = outside / "victim"
    victim.mkdir()
    reason = _make_junction(root / "escape", victim)
    if reason:
        pytest.skip(f"FR-34 skip, junction not creatable here: {reason}")

    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text("escape/pwned.txt", "pwned")

    assert _codes(exc) in {"reparse-point", "not-contained"}
    assert not (victim / "pwned.txt").exists(), "wrote through an escaping junction"


def test_reparse_branch_catches_mid_operation_substitution(sandbox):
    """Cover the `reparse-point` branch directly.

    Through ``resolve()`` this branch is normally unreachable: ``realpath``
    resolves an escaping junction first, so ``not-contained`` fires (which the
    two tests above prove on the real filesystem). The branch exists for the
    TOCTOU window inside ``_commit``, where the chain is re-checked *before*
    the second resolution. Exercising it directly keeps that path honest
    instead of shipping an untested branch.
    """
    root, outside = sandbox
    victim = outside / "victim"
    victim.mkdir()
    reason = _make_junction(root / "escape", victim)
    if reason:
        pytest.skip(f"FR-34 skip, junction not creatable here: {reason}")

    from rag_code_library.security import path_safety

    with pytest.raises(UnsafePathError) as exc:
        path_safety._check_reparse_points(root, root / "escape" / "pwned.txt")

    assert _codes(exc) == "reparse-point"
    assert str(victim) in str(exc.value)


def test_link_pointing_inside_root_is_allowed(sandbox):
    """Only escape is forbidden; an internal link must still work."""
    root, _ = sandbox
    real = root / "real"
    real.mkdir()
    reason = _make_symlink(root / "alias", real)
    if reason:
        pytest.skip(f"FR-34 skip, symlink not creatable here: {reason}")

    dest = SafeDestination(root)
    written = dest.write_text("alias/ok.txt", "fine")
    assert written.exists()
    assert (real / "ok.txt").read_text() == "fine"


# --------------------------------------------------------------------------- #
# FR-17 / FR-33: overwrite only with --force, and force widens nothing
# --------------------------------------------------------------------------- #


def test_existing_file_requires_force(sandbox):
    root, _ = sandbox
    dest = SafeDestination(root)
    dest.write_text("pkg/comp.py", "first")
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text("pkg/comp.py", "second")
    assert _codes(exc) == "exists"
    assert (root / "pkg" / "comp.py").read_text() == "first"


def test_force_overwrites_inside_root(sandbox):
    root, _ = sandbox
    dest = SafeDestination(root)
    dest.write_text("pkg/comp.py", "first")
    dest.write_text("pkg/comp.py", "second", force=True)
    assert (root / "pkg" / "comp.py").read_text() == "second"


def test_force_cannot_widen_the_root(sandbox):
    """FR-33: --force must never convert a rejected path into a permitted one."""
    root, outside = sandbox
    dest = SafeDestination(root)
    for hostile in (TRAVERSAL, DRIVE_LETTER, UNC, EXTENDED, ADS):
        with pytest.raises(UnsafePathError) as exc:
            dest.write_text(hostile, "pwned", force=True)
        assert _codes(exc) not in {"exists"}, f"force bypassed containment for {hostile!r}"
    _assert_wrote_nothing(root, outside)


def test_case_variant_counts_as_conflict(sandbox):
    """Windows compares names case-insensitively, so a case variant conflicts."""
    root, _ = sandbox
    dest = SafeDestination(root)
    dest.write_text("pkg/Component.py", "original")
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text("pkg/component.py", "shadow")
    assert _codes(exc) == "exists"
    assert (root / "pkg" / "Component.py").read_text() == "original"


# --------------------------------------------------------------------------- #
# Happy path + long-path form
# --------------------------------------------------------------------------- #


def test_normal_write_round_trips(sandbox):
    root, _ = sandbox
    dest = SafeDestination(root)
    written = dest.write_text("src/app/mod.py", "print('ok')")
    assert written == root / "src" / "app" / "mod.py"
    assert written.read_text() == "print('ok')"


def test_root_is_resolved_once(sandbox):
    """NFR-3: the root is resolved and cached for the operation."""
    root, _ = sandbox
    dest = SafeDestination(root)
    first = dest.root
    assert dest.root is first, "root was re-resolved after construction"
    assert not os.path.islink(dest.root)


def test_long_paths_use_supported_long_path_form(sandbox, monkeypatch):
    """Above MAX_PATH we write via the extended form; beyond the ceiling we refuse."""
    from rag_code_library.security import path_safety

    root, _ = sandbox
    deep = "a/" * 140 + "file.txt"  # > 260 characters once rooted
    dest = SafeDestination(root)
    resolved = dest.resolve(deep)
    assert len(str(resolved)) > path_safety.MAX_PATH
    assert path_safety._long_form(resolved).startswith("\\\\?\\")

    written = dest.write_text(deep, "long")
    # Plain Path.exists() cannot see past MAX_PATH; go through the long form.
    assert path_safety._exists(written), "long-path write did not land"
    with open(path_safety._long_form(written), "rb") as handle:
        assert handle.read() == b"long"

    # Now prove the hard ceiling rejects: patch the limit below this path's length.
    monkeypatch.setattr(path_safety, "MAX_NATIVE_PATH", 10)
    with pytest.raises(UnsafePathError) as exc:
        dest.resolve(deep)
    assert _codes(exc) == "too-long"


def test_empty_target_rejected(sandbox):
    root, _ = sandbox
    dest = SafeDestination(root)
    with pytest.raises(UnsafePathError) as exc:
        dest.write_text("", "x")
    assert _codes(exc) == "empty"

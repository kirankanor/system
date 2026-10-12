"""The single choke point for every filesystem write reachable from the CLI.

Why one module: ``ragkit add`` copies files into a project the user names, so a
hostile component manifest or a mistyped destination must never be able to write
outside that project. NFR-3 requires that *no* CLI command, component, or script
opens a file for writing except through this module, so containment is decided
in exactly one place rather than re-derived (and eventually mis-derived) by each
caller.

What is rejected, and why each case exists
------------------------------------------
============================  ==========================================
``..`` at any depth           traversal out of the destination root
absolute path / rooted path   targets are always relative to the root
drive letter ``C:\\``          escapes to another volume
drive-relative ``C:foo``      resolves against a *per-drive* CWD (Windows)
UNC ``\\\\server\\share``      escapes to another machine/share
extended prefix ``\\\\?\\``     bypasses normal path normalisation
alternate data stream ``x:y`` hidden stream beside a visible file
reserved device name ``CON``  opens a device instead of a file
reparse point outside root    symlink/junction redirecting the write
============================  ==========================================

Containment is decided on the **resolved real path**, and the root is resolved
once and cached by :class:`SafeDestination` for the lifetime of the operation.

Residual risk (stated, not hidden): substituting a reparse point between the
final resolution and ``open()`` is a TOCTOU race. It is *mitigated* -- the
parent chain is re-verified immediately before each write and the operation
aborts on any mismatch -- but it is **not** claimed as prevented. See NFR-3.
"""

from __future__ import annotations

import ntpath
import os
from pathlib import Path

__all__ = [
    "MAX_PATH",
    "MAX_NATIVE_PATH",
    "RESERVED_NAMES",
    "SafeDestination",
    "UnsafePathError",
]

#: Windows ``MAX_PATH``. Above this a write needs the extended-length form.
MAX_PATH = 260

#: Hard ceiling for the extended-length (``\\?\\``) form itself.
MAX_NATIVE_PATH = 32767

#: Windows device names that open a device rather than creating a file.
#: Matched against the basename *without* extension, case-insensitively, so
#: ``con.txt`` and ``CON.txt`` are caught too.
RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class UnsafePathError(ValueError):
    """A path failed containment or platform safety validation.

    Carries a machine-readable ``code`` so callers can branch on the reason and
    tests can assert on it without parsing prose (NFR-7 wants a readable
    message; the code is for the machine).

    ``details`` holds non-sensitive values only: the rejected path, the root it
    was measured against, and the rule that fired.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.path = None if path is None else os.fspath(path)
        self.root = None if root is None else os.fspath(root)

    @property
    def details(self) -> dict[str, str | None]:
        """Non-sensitive structured form; safe to log."""
        return {"code": self.code, "path": self.path, "root": self.root}


# --------------------------------------------------------------------------- #
# Validation primitives
# --------------------------------------------------------------------------- #


def _split_segments(target: str) -> list[str]:
    """Split on both separators; Windows accepts ``/`` and ``\\`` alike."""
    normalized = target.replace("/", "\\")
    return [seg for seg in normalized.split("\\") if seg != ""]


def _reject_unsafe_input(target: str, root: Path) -> None:
    """Reject hostile *input* forms before any filesystem call is made.

    Deliberately pure string work: these checks must not need the path to
    exist, because a non-existent hostile path is exactly the case we must
    refuse rather than discover by having written it.
    """
    if not target or not target.strip():
        raise UnsafePathError(
            "destination path is empty; give a path relative to the destination root",
            code="empty",
            path=target,
            root=root,
        )
    if "\x00" in target:
        raise UnsafePathError(
            "destination path contains a NUL byte",
            code="nul-byte",
            path=target,
            root=root,
        )

    if target.startswith("\\\\?\\") or target.startswith("//?/"):
        raise UnsafePathError(
            f"destination path {target!r} uses the extended-length prefix \\\\?\\, "
            "which bypasses normal path validation; give a plain relative path",
            code="extended-prefix",
            path=target,
            root=root,
        )
    if target.startswith("\\\\") or target.startswith("//"):
        raise UnsafePathError(
            f"destination path {target!r} is a UNC path; writes must stay inside "
            f"the destination root {root}",
            code="unc",
            path=target,
            root=root,
        )

    drive = ntpath.splitdrive(target)[0]
    if drive:
        # ``C:\\x`` is absolute; ``C:x`` is drive-relative and resolves against
        # that drive's current directory. Both escape the root, so both go.
        code = "drive-letter" if ntpath.isabs(target) else "drive-relative"
        raise UnsafePathError(
            f"destination path {target!r} is on drive {drive!r} rather than inside "
            f"the destination root {root}; give a path relative to the root",
            code=code,
            path=target,
            root=root,
        )
    if target[0] in "/\\":
        raise UnsafePathError(
            f"destination path {target!r} is absolute; writes must be relative to "
            f"the destination root {root}",
            code="absolute",
            path=target,
            root=root,
        )

    segments = _split_segments(target)
    if any(seg == ".." for seg in segments):
        raise UnsafePathError(
            f"destination path {target!r} contains a '..' segment, which could "
            f"escape the destination root {root}; remove it",
            code="traversal",
            path=target,
            root=root,
        )

    # After the absolute/UNC/drive checks no legitimate ':' can remain, so any
    # colon is an alternate data stream.
    if ":" in target:
        raise UnsafePathError(
            f"destination path {target!r} names an alternate data stream; write a "
            "regular file instead",
            code="ads",
            path=target,
            root=root,
        )

    for seg in segments:
        stem = seg.split(".", 1)[0]
        if stem.upper() in RESERVED_NAMES:
            raise UnsafePathError(
                f"destination path {target!r} uses the reserved device name "
                f"{seg!r}; choose a different file name",
                code="reserved-name",
                path=target,
                root=root,
            )


def _contained(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` is strictly inside ``root``.

    Comparison is case-insensitive on Windows so that a case variant of an
    existing file cannot slip past a containment check written on a
    case-sensitive assumption (NFR-3).
    """
    left = os.path.normcase(str(root))
    right = os.path.normcase(str(candidate))
    return right.startswith(left + os.sep)


def _check_reparse_points(root: Path, candidate: Path) -> None:
    """Reject a reparse point on the parent chain that resolves outside root.

    Walks each existing ancestor between the root and the target's parent. A
    symlink or junction pointing *inside* the root is allowed -- only escape is
    forbidden, exactly as FR-17 and NFR-3 state.

    A candidate that is no longer under ``root`` is rejected rather than
    skipped: that is precisely the outcome a just-substituted junction
    produces, and returning quietly would bypass the check that exists to
    catch it.
    """
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise UnsafePathError(
            f"path {candidate} is no longer inside the destination root {root}; "
            "the path changed after validation",
            code="not-contained",
            path=str(candidate),
            root=str(root),
        ) from None
    parts = relative.parts[:-1]  # parents only; the leaf is created, not followed
    current = root
    for part in parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError:
            continue  # not yet created, so it cannot be a redirect today
        if info.st_file_attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            real = Path(os.path.realpath(current))
            # Resolving to the root itself is still contained, so allow it.
            still_inside = _contained(root, real) or (
                os.path.normcase(str(real)) == os.path.normcase(str(root))
            )
            if not still_inside:
                raise UnsafePathError(
                    f"path {current} is a symlink or junction resolving to "
                    f"{real}, outside the destination root {root}; refusing "
                    "to write through it",
                    code="reparse-point",
                    path=str(current),
                    root=str(root),
                )


def _long_form(path: Path) -> str:
    """Extended-length form, needed above ``MAX_PATH`` on Windows.

    This is the *supported* long-path form NFR-3 refers to. It is constructed
    internally only: user-supplied ``\\?\\`` input is rejected above, because an
    input we did not build is one we have not validated.
    """
    text = str(path)
    if os.name != "nt" or len(text) < MAX_PATH or text.startswith("\\\\?\\"):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def _mkdirs(path: Path) -> None:
    """Create ``path`` and its parents using the supported long-path form.

    ``Path.mkdir(parents=True)`` does *not* apply the ``\\?\\`` prefix, so a
    nested target past ``MAX_PATH`` fails with WinError 206 before we ever
    reach the write. ``os.makedirs`` threads the prefixed head through each
    recursive step, so it is the one that works.
    """
    os.makedirs(_long_form(path), exist_ok=True)


def _case_variant(directory: Path, name: str) -> str | None:
    """Existing entry in ``directory`` equal to ``name`` ignoring case, else None.

    Windows cannot have two names differing only by case, so a case variant is
    a conflict with the file the user believes is new. Treating it as one is
    what makes ``--force`` semantics honest on this platform (NFR-3).
    """
    try:
        entries = os.listdir(_long_form(directory))
    except OSError:
        return None
    target = os.path.normcase(name)
    for entry in entries:
        if os.path.normcase(entry) == target and entry != name:
            return entry
    return None


def _exists(path: Path) -> bool:
    """``Path.exists()`` without the long-path form silently reports False."""
    try:
        return os.path.exists(_long_form(path))
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class SafeDestination:
    """A resolved destination root plus the only supported way to write into it.

    The root is resolved **once**, in ``__init__``, and cached: containment for
    every subsequent write is measured against that one value rather than
    re-resolving per call, which is what NFR-3 means by "resolved once and
    cached for the operation".

    All write methods validate first and open second, so a rejected path
    performs zero writes.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw = os.fspath(root)
        if not raw or not raw.strip():
            raise UnsafePathError(
                "destination root is empty; pass a directory with --destination",
                code="empty",
                path=raw,
            )
        # Resolve the root itself through the real filesystem so that a symlinked
        # destination is measured by where it actually lands.
        self._root = Path(os.path.realpath(os.path.abspath(raw)))

    # -- introspection ----------------------------------------------------- #

    @property
    def root(self) -> Path:
        """The resolved real path every write is contained by."""
        return self._root

    def resolve(self, target: str | os.PathLike[str]) -> Path:
        """Validate ``target`` and return its resolved real path inside the root.

        Raises :class:`UnsafePathError` with a specific ``code`` for every
        hostile form listed in the module docstring.
        """
        text = os.fspath(target)
        _reject_unsafe_input(text, self._root)

        joined = Path(self._root, *(_split_segments(text)))
        resolved = Path(os.path.realpath(str(joined)))

        if len(str(resolved)) > MAX_NATIVE_PATH:
            raise UnsafePathError(
                f"resolved path exceeds the Windows maximum of "
                f"{MAX_NATIVE_PATH} characters: {len(str(resolved))}",
                code="too-long",
                path=str(resolved),
                root=str(self._root),
            )
        if not _contained(self._root, resolved):
            raise UnsafePathError(
                f"path {text!r} resolves to {resolved}, outside the destination "
                f"root {self._root}; refusing to write there",
                code="not-contained",
                path=text,
                root=str(self._root),
            )
        _check_reparse_points(self._root, resolved)
        return resolved

    def _conflict(self, resolved: Path, *, force: bool) -> None:
        """Enforce FR-17/FR-33: no overwrite without explicit ``--force``.

        ``--force`` is checked *after* containment, never before it, so force
        can widen nothing -- it only permits replacing a file that already
        exists inside the root.
        """
        variant = _case_variant(resolved.parent, resolved.name)
        exists = _exists(resolved) or variant is not None
        if not exists:
            return
        if not force:
            shown = resolved.name if variant is None else variant
            raise UnsafePathError(
                f"{resolved.parent / shown} already exists inside the destination "
                "root; re-run with --force to overwrite it, or remove it first",
                code="exists",
                path=str(resolved.parent / shown),
                root=str(self._root),
            )

    # -- writes ------------------------------------------------------------ #

    def _prepare(self, target: str, *, force: bool) -> Path:
        resolved = self.resolve(target)
        self._conflict(resolved, force=force)
        return resolved

    def _commit(self, resolved: Path, payload: bytes) -> Path:
        """Resolve-then-write, re-verifying the parent chain immediately before.

        This is the TOCTOU mitigation: the chain is checked as late as
        possible and the operation aborts on mismatch. It reduces the window;
        it does not close it (NFR-3 states this residual risk explicitly).
        """
        _check_reparse_points(self._root, resolved)
        recheck = Path(os.path.realpath(str(resolved)))
        if not _contained(self._root, recheck):
            raise UnsafePathError(
                f"path changed between validation and write: {recheck} is outside "
                f"the destination root {self._root}",
                code="not-contained",
                path=str(recheck),
                root=str(self._root),
            )
        _mkdirs(resolved.parent)
        # mkdir can itself be redirected by a junction created just now.
        _check_reparse_points(self._root, resolved)
        with open(_long_form(resolved), "wb") as handle:
            handle.write(payload)
        return resolved

    def write_bytes(self, target: str, payload: bytes, *, force: bool = False) -> Path:
        """Write raw bytes. Returns the resolved path written."""
        resolved = self._prepare(target, force=force)
        return self._commit(resolved, payload)

    def write_text(
        self, target: str, text: str, *, force: bool = False, encoding: str = "utf-8"
    ) -> Path:
        """Write ``text`` encoded as ``encoding``. Returns the resolved path."""
        return self.write_bytes(target, text.encode(encoding), force=force)

    def copy_file(
        self, source: str | os.PathLike[str], target: str, *, force: bool = False
    ) -> Path:
        """Copy an existing file into the destination root.

        The *destination* is validated here; ``source`` is read from the local
        component tree, not from user input, so it needs no containment check.
        """
        resolved = self._prepare(target, force=force)
        payload = Path(source).read_bytes()
        return self._commit(resolved, payload)

    # -- queries ----------------------------------------------------------- #

    def exists(self, target: str) -> bool:
        """True when ``target`` resolves inside the root and is present.

        Case variants count as present, matching :meth:`_conflict`.
        """
        resolved = self.resolve(target)
        return _exists(resolved) or _case_variant(resolved.parent, resolved.name) is not None

    def ensure_root(self) -> Path:
        """Create the destination root if needed and return it."""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

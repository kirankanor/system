"""Component tests for `document-identity-resolver`.

Runs standalone: ``python -m pytest components/ingestion/document_identity_resolver/tests``
from the repository root. Does not import `rag_code_library` (AC-13).

The component is built around one distinction -- content identity versus file
identity -- so the tests are weighted towards the cases where those two come
apart: the same bytes under two names, and one name holding two different
bytes. The id-rule test is the one that matters most for the library, because a
silent divergence there would desynchronise every stored vector.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import pytest
from document_identity_resolver import (
    DOCUMENT_ID_LENGTH,
    ContentIdentity,
    DocumentIdentityResolver,
    FileIdentity,
    IdentityError,
    InvalidDigest,
    MissingFile,
    Resolution,
    UnreadableFile,
    make_document_id,
)

CONTENT = b"the quick brown fox\n"
OTHER = b"a completely different document\n"


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


@pytest.fixture
def resolver():
    return DocumentIdentityResolver()


# --- the shared id rule -----------------------------------------------------


def test_the_document_id_matches_the_library_rule():
    """``sha256[:32]`` -- the same value config/models.py and core-types produce.

    This component may not import either of them (AC-13), so the rule is
    restated and pinned here. If this test ever fails, three components have
    started disagreeing about what a document id *is*, and every vector already
    stored under the old rule is orphaned.
    """
    digest = hashlib.sha256(CONTENT).hexdigest()

    assert make_document_id(digest) == digest[:DOCUMENT_ID_LENGTH]
    assert len(make_document_id(digest)) == 32


def test_the_id_is_a_prefix_of_the_digest_not_a_rehash():
    """Truncation, so the full digest remains recoverable from the id's source."""
    digest = hashlib.sha256(CONTENT).hexdigest()

    assert digest.startswith(make_document_id(digest))


@pytest.mark.parametrize(
    "bad",
    ["", "not-hex", "A" * 64, "0" * 63, "0" * 65, "g" * 64],
)
def test_a_digest_that_is_not_64_lowercase_hex_is_refused(bad):
    """An uppercase or short digest would silently mint a different id."""
    with pytest.raises(InvalidDigest):
        make_document_id(bad)


def test_an_uppercase_digest_is_refused_rather_than_folded():
    digest = hashlib.sha256(CONTENT).hexdigest()

    with pytest.raises(InvalidDigest):
        make_document_id(digest.upper())


# --- content identity -------------------------------------------------------


def test_hashing_is_deterministic(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)

    assert resolver.hash_file(path).sha256 == resolver.hash_file(path).sha256


def test_content_identity_carries_the_size_alongside_the_digest(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)

    identity = resolver.hash_file(path)

    assert isinstance(identity, ContentIdentity)
    assert identity.size_bytes == len(CONTENT)


def test_two_files_with_identical_bytes_share_a_document_id(resolver, tmp_path):
    """The deduplication case: same content, different names."""
    first = _write(tmp_path / "original.txt", CONTENT)
    second = _write(tmp_path / "copy.txt", CONTENT)

    assert resolver.hash_file(first).document_id == resolver.hash_file(second).document_id


def test_a_renamed_file_keeps_its_identity(resolver, tmp_path):
    """A filename is not an identifier, so renaming must not re-identify."""
    path = _write(tmp_path / "before.txt", CONTENT)
    before = resolver.hash_file(path).document_id
    path.rename(tmp_path / "after.txt")

    assert resolver.hash_file(tmp_path / "after.txt").document_id == before


def test_one_byte_of_difference_produces_a_different_id(resolver, tmp_path):
    first = _write(tmp_path / "a.txt", CONTENT)
    second = _write(tmp_path / "b.txt", CONTENT[:-1] + b"x")

    assert resolver.hash_file(first).document_id != resolver.hash_file(second).document_id


# --- resolving --------------------------------------------------------------


def test_a_first_resolution_reports_version_one_and_no_history(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)

    resolution = resolver.resolve(path)

    assert isinstance(resolution, Resolution)
    assert resolution.version == 1
    assert resolution.previous_sha256 is None
    assert resolution.identical_content_at == ()


def test_the_resolved_identity_is_a_file_identity(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)

    identity = resolver.resolve(path).identity

    assert isinstance(identity, FileIdentity)
    assert identity.path == str(path.resolve())
    assert identity.document_id == make_document_id(hashlib.sha256(CONTENT).hexdigest())


def test_identical_content_elsewhere_is_reported(resolver, tmp_path):
    """The whole reason file identity and content identity are separated."""
    original = _write(tmp_path / "original.txt", CONTENT)
    _write(tmp_path / "elsewhere.txt", CONTENT)

    resolver.resolve(original)
    resolution = resolver.resolve(tmp_path / "elsewhere.txt")

    assert resolution.identical_content_at == (str(original.resolve()),)


def test_duplicates_are_symmetric(resolver, tmp_path):
    a = _write(tmp_path / "a.txt", CONTENT)
    b = _write(tmp_path / "b.txt", CONTENT)

    resolver.resolve(a)
    resolver.resolve(b)

    assert str(b.resolve()) in resolver.resolve(a).identical_content_at
    assert str(a.resolve()) in resolver.resolve(b).identical_content_at


def test_resolving_the_same_file_twice_does_not_report_it_as_its_own_duplicate(
    resolver, tmp_path
):
    path = _write(tmp_path / "a.txt", CONTENT)

    resolver.resolve(path)

    assert resolver.resolve(path).identical_content_at == ()


# --- modification and versioning --------------------------------------------


def test_editing_a_file_produces_a_new_identity_and_a_new_version(resolver, tmp_path):
    """A modified document is a new document, and the resolver says so.

    The id is content-derived, so it cannot survive an edit. What the caller
    needs instead is the version number and the digest being retired.
    """
    path = _write(tmp_path / "a.txt", CONTENT)
    first = resolver.resolve(path)

    _write(tmp_path / "a.txt", OTHER)
    second = resolver.resolve(path)

    assert second.identity.document_id != first.identity.document_id
    assert second.version == 2
    assert second.previous_sha256 == first.identity.content.sha256


def test_resolving_unchanged_content_does_not_bump_the_version(resolver, tmp_path):
    """``version`` counts states, not lookups."""
    path = _write(tmp_path / "a.txt", CONTENT)

    first = resolver.resolve(path)

    assert resolver.resolve(path).version == first.version == 1


def test_the_version_counts_distinct_states_not_the_order_of_changes(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)
    resolver.resolve(path)

    _write(tmp_path / "a.txt", OTHER)
    assert resolver.resolve(path).version == 2

    _write(tmp_path / "a.txt", CONTENT)
    assert resolver.resolve(path).version == 2  # already a state this file has been in


def test_a_changed_file_is_not_reported_as_a_duplicate_of_itself(resolver, tmp_path):
    path = _write(tmp_path / "a.txt", CONTENT)
    resolver.resolve(path)
    _write(tmp_path / "a.txt", OTHER)

    assert resolver.resolve(path).identical_content_at == ()


def test_a_rename_starts_the_version_history_over_but_keeps_the_identity(resolver, tmp_path):
    """History is keyed by path, so a rename is a new file identity.

    Documented rather than worked around: file identity is (path, size, mtime),
    and pretending otherwise would mean deriving identity from something other
    than the bytes. What *does* survive is the content identity, which is the
    part that identifies the document -- so both halves of the split are
    exercised here, and neither is allowed to quietly stand in for the other.
    """
    path = _write(tmp_path / "a.txt", CONTENT)
    first = resolver.resolve(path)
    path.rename(tmp_path / "renamed.txt")

    renamed = resolver.resolve(tmp_path / "renamed.txt")

    assert renamed.identity.document_id == first.identity.document_id
    assert renamed.version == 1
    assert renamed.previous_sha256 is None


def test_paths_for_document_finds_every_copy(resolver, tmp_path):
    a = _write(tmp_path / "a.txt", CONTENT)
    b = _write(tmp_path / "b.txt", CONTENT)
    _write(tmp_path / "c.txt", OTHER)

    for path in (a, b, tmp_path / "c.txt"):
        resolver.resolve(path)

    found = resolver.paths_for_document(resolver.hash_file(a).document_id)

    assert found == (str(a.resolve()), str(b.resolve()))


# --- failure paths ----------------------------------------------------------


def test_a_missing_file_is_reported_with_its_path(resolver, tmp_path):
    with pytest.raises(MissingFile) as excinfo:
        resolver.hash_file(tmp_path / "absent.txt")

    assert "absent.txt" in str(excinfo.value)


def test_a_directory_is_not_a_document(resolver, tmp_path):
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(IdentityError):
        resolver.hash_file(directory)


def test_a_file_that_cannot_be_read_to_the_end_is_reported(resolver, tmp_path, monkeypatch):
    """A short read is not a valid digest, and must not silently become one."""
    path = _write(tmp_path / "a.txt", CONTENT)
    original = Path.open

    def failing_open(self, *args, **kwargs):
        handle = original(self, *args, **kwargs)
        if self == path:
            handle.read = lambda *_: (_ for _ in ()).throw(OSError("device error"))
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(UnreadableFile):
        resolver.hash_file(path)


# --- immutability -----------------------------------------------------------


def test_a_content_identity_cannot_be_rewritten_after_use(resolver, tmp_path):
    identity = resolver.hash_file(_write(tmp_path / "a.txt", CONTENT))

    with pytest.raises(AttributeError):
        identity.sha256 = "0" * 64


def test_a_document_id_cannot_be_set_independently_of_the_digest():
    """The id is derived, so there is no way to state one without the other."""
    identity = ContentIdentity(sha256="0" * 64, size_bytes=1)

    with pytest.raises(AttributeError):
        identity.document_id = "somewhere-else"

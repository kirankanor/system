"""Runnable example for `document-identity-resolver`.

    python components/ingestion/document_identity_resolver/example.py

Four demonstrations in a temporary directory, in the order they depend on each
other: identical bytes under two names, a rename that must not change identity,
an edit at the same path that must, and the id rule itself. The rename comes
before the edit deliberately -- identity is tracked per path, so editing after
a rename would be editing a file with no history to report.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_identity_resolver import (
    DocumentIdentityResolver,
    make_document_id,
)

DOC = b"Ingestion runs before retrieval.\n"
EDITED = b"Ingestion runs before retrieval. Updated.\n"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resolver = DocumentIdentityResolver()

        original = root / "report.txt"
        original.write_bytes(DOC)
        copy = root / "report (copy).txt"
        copy.write_bytes(DOC)

        print("== the same bytes under two names ==")
        first = resolver.resolve(original)
        print(f"  {original.name:<20} id={first.identity.document_id[:16]}...")
        print(f"  {copy.name:<20} id={resolver.hash_file(copy).document_id[:16]}...")
        duplicate = resolver.resolve(copy)
        print("  -> reported, though the names have nothing in common:")
        print(f"     identical_content_at = {duplicate.identical_content_at}")

        print("\n== a rename does not change identity ==")
        renamed = root / "final-report.txt"
        original.rename(renamed)
        after_rename = resolver.resolve(renamed)
        print(f"  renamed to {renamed.name}")
        print(
            "  -> document id unchanged: "
            f"{after_rename.identity.document_id[:16]}... "
            f"(was {first.identity.document_id[:16]}...)"
        )

        print("\n== editing at the same path produces a new document ==")
        renamed.write_bytes(EDITED)
        changed = resolver.resolve(renamed)
        print(f"  version: 1 -> {changed.version}")
        print(f"  retired digest: {str(changed.previous_sha256)[:16]}...")
        print(f"  old document id: {first.identity.document_id[:16]}...")
        print(f"  new document id: {changed.identity.document_id[:16]}...")
        assert changed.identity.document_id != first.identity.document_id
        assert changed.previous_sha256 == first.identity.content.sha256

        print("\n== the id rule is the library's rule ==")
        digest = resolver.hash_file(renamed).sha256
        print(f"  sha256 : {digest[:20]}...")
        print(f"  id     : {make_document_id(digest)}")
        print("  -> first 32 hex of the digest, as in settings and core-types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

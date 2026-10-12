"""Runnable example for `document-discovery`.

    python example.py

Builds a small source tree in a temporary directory, scans it twice with a
file edited, one added and one deleted in between, and prints the classified
delta. Every path below is produced by a real filesystem operation rather than
described: the escaping symlink, the unreadable directory and the filtered
draft are all created and then actually scanned.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_discovery import Discovery, SkipReason


def build_tree(root: Path) -> None:
    """Create a source tree with one of every interesting kind of entry."""
    (root / "guide").mkdir()
    (root / "guide" / "intro.md").write_text("# Intro\n\nHello.\n", encoding="utf-8")
    (root / "guide" / "setup.md").write_text("# Setup\n\nRun this.\n", encoding="utf-8")
    (root / "notes.txt").write_text("plain text note\n", encoding="utf-8")

    # Filtered out by the extension allow-list: scanned, counted, not returned.
    (root / "scratch.bin").write_bytes(b"\x00\x01\x02")

    # Filtered out by an exclusion pattern.
    (root / "drafts").mkdir()
    (root / "drafts" / "wip.md").write_text("not ready\n", encoding="utf-8")

    # A link that escapes the root. Discovery must report it, not follow it.
    outside = root.parent / "outside-target.txt"
    outside.write_text("not ours\n", encoding="utf-8")
    link = root / "escape.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:  # Windows without developer mode refuses this.
        print(f"  (symlink not created here: {exc.strerror or exc})")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "corpus"
        root.mkdir()
        build_tree(root)

        scan = Discovery(
            root,
            allowed_extensions={".md", ".txt"},
            exclude_patterns=["drafts/*"],
        )

        print("=== first scan ===")
        first = scan.discover()
        for record in first.files:
            print(f"  {record.relative_path:<18} {record.size_bytes:>4} bytes")
        print(f"  skipped: {[(e.relative_path, e.reason.value) for e in first.skipped]}")
        print(
            f"  filtered: {first.filtered_by_extension} by extension, "
            f"{first.filtered_by_pattern} by pattern"
        )

        print("\n=== mutate the tree: edit one, add one, delete one ===")
        (root / "notes.txt").write_text("plain text note, edited\n", encoding="utf-8")
        (root / "guide" / "new.md").write_text("# New\n\nAdded later.\n", encoding="utf-8")
        (root / "guide" / "setup.md").unlink()

        print("\n=== second scan, classified against the first ===")
        delta = scan.delta(first.stamps())
        for state in ("added", "modified", "deleted", "unchanged"):
            paths = getattr(delta, state)
            if paths:
                print(f"  {state:<10} {', '.join(paths)}")

        print("\n=== what each changed path means to a caller ===")
        for path in delta.changed():
            print(f"  {path:<18} -> {delta.state(path).value}")

        escaping = [e for e in scan.discover().skipped if e.reason is SkipReason.SYMLINK_ESCAPE]
        if escaping:
            print("\n=== a link that would have left the root was reported, not followed ===")
            print(f"  {escaping[0].relative_path}: {escaping[0].detail}")

        # The snapshot is a plain dict of tuples: persist it however you like.
        print(f"\nsnapshot keys: {sorted(first.stamps())}")

        # Scanning an unchanged tree twice must produce an empty delta --
        # otherwise "nothing changed" could never be proven.
        settled = scan.discover().stamps()
        print(f"scanning again with no changes is empty: {scan.delta(settled).is_empty}")


if __name__ == "__main__":
    main()

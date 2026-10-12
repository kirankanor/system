#!/usr/bin/env python
"""Compare per-section component counts in `cookbook/data.js` against the registry (AC-15).

Why a count check at all
------------------------
AC-6 already proves `data.js` is byte-identical to freshly generated output, so
this looks redundant. It is not: the two fail differently. A byte comparison
tells you the file drifted; it does not tell you the site would *show* the
right thing. If a generator bug dropped every component from one category while
keeping the file self-consistent, `--check` would pass and the sidebar would
have an empty section nobody noticed. This asks the narrower question the site
actually depends on: does each section list what the registry has?

This is AC-15's scripted half. The other half -- opening `cookbook/index.html`
from disk and checking that the sidebar expands, search narrows, breadcrumbs
navigate, and the copy button reaches the clipboard -- is recorded human
inspection and is deliberately not asserted here.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_code_library.registry import Registry

DATA_JS = REPO / "cookbook" / "data.js"


def load_cookbook(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text.split("=", 1)[1].strip().rstrip(";").strip())


def main() -> int:
    if not DATA_JS.is_file():
        print(f"ERROR: {DATA_JS} is missing; run scripts/build_cookbook.py", file=sys.stderr)
        return 1

    registry = Registry.load(REPO / "components")
    if not registry.ok:
        for issue in registry.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1

    cookbook = load_cookbook(DATA_JS)
    sections = Counter(record["category"] for record in cookbook["components"])
    declared = Counter(entry.manifest.category for entry in registry)

    print(f"{'section':<20}{'registry':>9}{'cookbook':>10}   status")
    print("-" * 56)
    problems: list[str] = []
    for category in sorted(set(declared) | set(sections)):
        want, got = declared.get(category, 0), sections.get(category, 0)
        ok = want == got
        if not ok:
            problems.append(f"{category}: registry has {want}, data.js lists {got}")
        print(f"{category:<20}{want:>9}{got:>10}   {'ok' if ok else 'MISMATCH'}")

    print("-" * 56)
    print(f"{'TOTAL':<20}{sum(declared.values()):>9}{sum(sections.values()):>10}")

    if len(cookbook["components"]) != len(registry):
        problems.append(
            f"total: registry has {len(registry)}, data.js has "
            f"{len(cookbook['components'])}"
        )

    if problems:
        print(file=sys.stderr)
        for problem in problems:
            print(f"MISMATCH: {problem}", file=sys.stderr)
        print(
            "Regenerate with: python scripts/build_cookbook.py",
            file=sys.stderr,
        )
        return 1

    print("\nOK: every section lists the components the registry declares")
    return 0


if __name__ == "__main__":
    sys.exit(main())

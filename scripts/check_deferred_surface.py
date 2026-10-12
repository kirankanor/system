"""Prove the v2-deferred categories ship no implementation code (AC-19, NFR-13).

``observability`` and ``integrations`` are declared in the category enum so
that the deferral is a *recorded decision* rather than an absence nobody
noticed. Declaring them costs nothing and lets a reader see the shape of what
was deliberately left out. What must not exist is code pretending to be v1
support for them.

The rule, and why it is checkable rather than aspirational
---------------------------------------------------------
AC-19 requires that no file under a deferred category exists as implementation
code at v1 release. That is a statement about the filesystem, so it is decided
by looking at the filesystem:

* no ``.py`` file anywhere under ``components/<deferred>/``
* no deferred category directory in the source tree
* any manifest that *does* sit in a deferred category must carry status
  ``v2-deferred`` **and** must not point at a source, example or test file that
  actually exists

That last case is the one worth defending against. A manifest alone looks
harmless, but a manifest naming a real ``source`` file is a promise that
``ragkit add`` would try to keep -- so a deferred component pointing at existing
code would be deferred in name only.

This check reads; it never writes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from rag_code_library.registry.manifest import CATEGORIES

#: The categories the specification defers to v2. Named here rather than
#: inferred, because inference would silently follow a mistake: if someone
#: removed a category from the enum, an inferred check would stop looking for
#: it instead of failing.
DEFERRED_CATEGORIES: tuple[str, ...] = ("observability", "integrations")


@dataclass
class Finding:
    """One thing that would make a deferred category not actually deferred."""

    path: str
    problem: str

    def __str__(self) -> str:
        return f"{self.path}: {self.problem}"


def _iter_python_files(directory: Path) -> list[Path]:
    """Every ``.py`` under ``directory``, skipping bytecode caches."""
    if not directory.is_dir():
        return []
    return [
        p
        for p in sorted(directory.rglob("*.py"))
        if "__pycache__" not in p.parts and p.is_file()
    ]


def _deferred_manifests() -> list[tuple[Path, dict]]:
    """Every manifest declaring a component in a deferred category."""
    found: list[tuple[Path, dict]] = []
    for manifest_path in sorted((_ROOT / "components").rglob("manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Not this check's failure to report: the registry and
            # validate_components.py own manifest parseability.
            continue
        if data.get("category") in DEFERRED_CATEGORIES:
            found.append((manifest_path, data))
    return found


def check() -> list[Finding]:
    findings: list[Finding] = []

    # 1. The enum has to still know these categories exist. If one is dropped,
    #    the deferral stops being visible anywhere, and so does this check.
    for category in DEFERRED_CATEGORIES:
        if category not in CATEGORIES:
            findings.append(
                Finding(
                    "src/rag_code_library/registry/manifest.py",
                    f"category {category!r} is deferred but is absent from the "
                    "category enum, so nothing records that it was deferred "
                    "rather than forgotten",
                )
            )

    # 2. No implementation code under a deferred category in components/.
    for category in DEFERRED_CATEGORIES:
        directory = _ROOT / "components" / category
        for path in _iter_python_files(directory):
            findings.append(
                Finding(
                    path.relative_to(_ROOT).as_posix(),
                    f"implementation file exists under the deferred category "
                    f"{category!r}",
                )
            )

    # 3. Nothing in the source tree named for a deferred category.
    source_root = _ROOT / "src" / "rag_code_library"
    for category in DEFERRED_CATEGORIES:
        for path in _iter_python_files(source_root / category):
            findings.append(
                Finding(
                    path.relative_to(_ROOT).as_posix(),
                    f"source module implements the deferred category {category!r}",
                )
            )

    # 4. A manifest may *declare* a deferred component, but must not point at
    #    code that exists -- that would make it addable, hence not deferred.
    for manifest_path, data in _deferred_manifests():
        relative = manifest_path.relative_to(_ROOT).as_posix()
        category = data["category"]

        status = data.get("status")
        if status != "v2-deferred":
            findings.append(
                Finding(
                    relative,
                    f"component {data.get('id')!r} is in the deferred category "
                    f"{category!r} but declares status {status!r}; a deferred "
                    "component must carry status 'v2-deferred'",
                )
            )

        base = manifest_path.parent
        for field in ("source", "documentation", "tests"):
            declared = data.get(field)
            if not declared:
                continue
            target = base / str(declared)
            # The documentation may exist -- a manifest explaining the deferral
            # is exactly what should exist. Implementation, example and test
            # may not.
            if field == "documentation":
                continue
            if target.exists():
                findings.append(
                    Finding(
                        target.relative_to(_ROOT).as_posix(),
                        f"deferred component {data.get('id')!r} declares "
                        f"{field}={declared!r} and that file exists; a deferred "
                        "component must ship no implementation, example or test",
                    )
                )

    return findings


def main() -> int:
    print("check_deferred_surface: verifying no implementation ships in a")
    print(f"  v2-deferred category: {', '.join(DEFERRED_CATEGORIES)}")

    categories_in_enum = [c for c in DEFERRED_CATEGORIES if c in CATEGORIES]
    print(f"  categories present in the enum: {len(categories_in_enum)}/"
          f"{len(DEFERRED_CATEGORIES)}")

    manifests = _deferred_manifests()
    print(f"  manifests declaring a deferred component: {len(manifests)}")

    for category in DEFERRED_CATEGORIES:
        files = _iter_python_files(_ROOT / "components" / category)
        print(f"  implementation files under components/{category}/: "
              f"{len(files)}")

    findings = check()

    if findings:
        print(f"\n{len(findings)} problem(s):")
        for finding in findings:
            print(f"  FAIL {finding}")
        print(
            "\nA v2-deferred category may be declared and listed, but it may not\n"
            "ship code. Remove the implementation, or promote the category out\n"
            "of deferral deliberately -- do not weaken this check."
        )
        return 1

    print(
        "\nOK: the deferred categories are declared in the enum and ship no\n"
        "    implementation, example or test."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

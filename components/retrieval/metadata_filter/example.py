"""Runnable example for `metadata-filter`.

    python example.py

Shows the criteria language every filtering caller owes themselves, and the
three ways a naive filter silently loses recall. Runs standalone with
`core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

The four guarantees this example demonstrates:

1. a literal criterion means equality, and criteria combine with AND;
2. an unknown operator raises, naming the operator and listing the valid
   ones -- instead of matching nothing and returning a correct-looking empty
   list;
3. an incomparable operand raises a typed error naming both types, instead
   of leaking a TypeError traceback;
4. the caller's ranking survives the filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory under components/storage.
# `ragkit add` copies it beside this file; here we reach across to the sibling
# so the example runs from the component tree.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "storage" / "core_types")
)

import core_types as core  # noqa: E402

from metadata_filter import (  # noqa: E402
    OPERATORS,
    MetadataFilterError,
    MetadataFilterInputError,
    apply_to_chunks,
    filter_documents,
)

DOCUMENTS = [
    core.ScoredDocument(
        chunk_id="c0", score=0.9, text="Installing the toolkit",
        chunk_index=0, source_name="setup.md",
        metadata={"lang": "en", "page": 1, "tags": "install setup"},
    ),
    core.ScoredDocument(
        chunk_id="c1", score=0.7, text="Installation en franais",
        chunk_index=1, source_name="setup-fr.md",
        metadata={"lang": "fr", "page": 4, "tags": "install"},
    ),
    core.ScoredDocument(
        chunk_id="c2", score=0.5, text="Troubleshooting",
        chunk_index=2, source_name="help.md",
        metadata={"lang": "en", "page": 9, "tags": "help"},
    ),
]


def ids(documents) -> str:
    return ", ".join(d.chunk_id for d in documents) or "(none)"


def main() -> None:
    print("A literal criterion means equality; several combine with AND:")
    print(f"  lang == 'en'          -> {ids(filter_documents(DOCUMENTS, {'lang': 'en'}))}")
    print(f"  lang == 'en' AND page >= 9 -> "
          f"{ids(filter_documents(DOCUMENTS, {'lang': 'en', 'page': {'gte': 9}}))}")

    print("\nThe operators:")
    for label, criteria in [
        ("gte  page >= 4", {"page": {"gte": 4}}),
        ("lt   page < 4", {"page": {"lt": 4}}),
        ("ne   lang != en", {"lang": {"ne": "en"}}),
        ("in   lang in [fr, de]", {"lang": {"in": ["fr", "de"]}}),
        ("contains  tags contains 'install'", {"tags": {"contains": "install"}}),
        ("startswith  source starts 'setup'", {"source_name": {"startswith": "setup"}}),
        ("exists  'lang' is present", {"lang": {"exists": True}}),
        ("exists  'author' is absent", {"author": {"exists": False}}),
    ]:
        print(f"  {label:<42} -> {ids(filter_documents(DOCUMENTS, criteria))}")

    print("\nA document missing the field simply does not match:")
    print(f"  author == 'kiran' -> {ids(filter_documents(DOCUMENTS, {'author': 'kiran'}))}")

    print("\nThe three ways recall disappears, and what happens instead:")

    try:
        filter_documents(DOCUMENTS, {"page": {"gtr": 3}})
    except MetadataFilterInputError as exc:
        print(f"  1. a typo'd operator raises:")
        print(f"       {exc}")

    print(f"\n     by contrast, a genuine no-match is an empty list:")
    print(f"       page > 100 -> {ids(filter_documents(DOCUMENTS, {'page': {'gt': 100}}))}")

    try:
        filter_documents(DOCUMENTS, {"page": {"gt": "three"}})
    except MetadataFilterInputError as exc:
        print(f"\n  2. an incomparable operand raises, naming both types:")
        print(f"       {exc}")

    try:
        filter_documents(DOCUMENTS, {"lang": {"in": "en"}})
    except MetadataFilterInputError as exc:
        print(f"\n  3. 'in' on a bare string raises rather than matching per character:")
        print(f"       {exc}")

    print(f"\nEvery failure is catchable under one base class: "
          f"{issubclass(MetadataFilterInputError, MetadataFilterError)}")

    print("\nThe same criteria serve plain chunks, for pre-filtering a corpus:")
    chunks = [
        core.Chunk(chunk_index=0, text="a", chunk_id="k0", metadata={"lang": "en"}),
        core.Chunk(chunk_index=1, text="b", chunk_id="k1", metadata={"lang": "fr"}),
    ]
    print(f"  lang == 'en' -> {', '.join(c.chunk_id for c in apply_to_chunks(chunks, {'lang': 'en'}))}")

    print(f"\noperators: {', '.join(OPERATORS)}")


if __name__ == "__main__":
    main()

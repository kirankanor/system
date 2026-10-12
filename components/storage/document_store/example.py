"""Runnable example for `document-store`.

    python example.py

Shows the input checks a document-store implementation owes its callers, and
the one behaviour the contract states out loud. Runs standalone with
`core_types.py` beside it -- the manifest declares that component a
dependency, so `ragkit add` copies it along.

The three guarantees this example demonstrates:

1. a document is checked before it is written, and the message names this
   component rather than surfacing a pydantic traceback;
2. filters are flat, and a nested value is rejected rather than silently
   dropped;
3. a bare string of ids is rejected, because a str *is* a sequence of
   characters and would mean one lookup per letter.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory. `ragkit add` copies it
# beside this file; here we reach across to the sibling so the example runs
# from the component tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_types"))

import core_types as core  # noqa: E402

from document_store import (  # noqa: E402
    DocumentStoreError,
    DocumentStoreInputError,
    DocumentStoreProtocol,
    validate_filter,
    validate_ids,
    validate_source_document,
)

DOCUMENT = core.SourceDocument(
    document_id="doc-1",
    source_name="handbook.md",
    sha256="a" * 64,
    page_count=12,
)


def main() -> None:
    print("validate_source_document accepts a well-formed document:")
    accepted = validate_source_document(DOCUMENT)
    print(f"  {accepted.document_id!r} from {accepted.source_name!r}, "
          f"{accepted.page_count} pages")

    print("\nvalidate_source_document rejects, naming this component:")
    try:
        validate_source_document({"document_id": "doc-1"})
    except DocumentStoreInputError as exc:
        print(f"  a plain dict -> {exc}")

    print("\nvalidate_filter accepts:")
    print(f"  None                 -> {validate_filter(None)}")
    print(f"  {{'source': 'a.md'}}   -> {validate_filter({'source': 'a.md'})}")

    print("\nvalidate_filter rejects, rather than dropping:")
    try:
        validate_filter({"doc": {"id": "doc-1"}})
    except DocumentStoreInputError as exc:
        print(f"  a nested mapping -> {exc}")

    print("\nvalidate_ids accepts:")
    print(f"  ['doc-1', 'doc-2']  -> {validate_ids(['doc-1', 'doc-2'])}")

    print("\nvalidate_ids rejects:")
    try:
        validate_ids("doc-1")
    except DocumentStoreInputError as exc:
        print(f"  a bare string -> {exc}")

    print("\nevery failure is catchable under one base class:")
    print(f"  DocumentStoreInputError is a DocumentStoreError: "
          f"{issubclass(DocumentStoreInputError, DocumentStoreError)}")
    print(f"  the protocol is re-exported so an implementation needs only this "
          f"file: {DocumentStoreProtocol.__name__}")

    # The behaviour the contract states out loud, and the reason it is not an
    # exception: a chunk can refer to a document that was never ingested.
    print("\nget() on an unknown id returns None rather than raising -- a gap "
          "is an expected condition, and handling it belongs at the call site.")


if __name__ == "__main__":
    main()

"""Runnable example for `vector-store-protocol`.

    python example.py

Shows the input checks every vector-store implementation owes its callers.
Runs standalone with `core_types.py` beside it -- the manifest declares that
component a dependency, so `ragkit add` copies it along.

The four guarantees this example demonstrates:

1. filters are flat, and a nested value is rejected rather than silently
   dropped (which would narrow the caller's result set invisibly);
2. a non-finite vector value is rejected, because NaN in an index poisons a
   nearest-neighbour search without failing visibly;
3. a bare string of ids is rejected, because a str *is* a sequence of
   characters and would mean one delete per letter;
4. every failure is catchable under one base class.
"""

from __future__ import annotations

import sys
from pathlib import Path

# core-types sits in its own component directory. `ragkit add` copies it
# beside this file; here we reach across to the sibling so the example runs
# from the component tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_types"))

from vector_store_protocol import (  # noqa: E402
    StorageBackendError,
    StorageError,
    StorageInputError,
    VectorStoreProtocol,
    validate_filter,
    validate_ids,
    validate_k,
    validate_vector,
)


def main() -> None:
    print("validate_filter accepts:")
    print(f"  None                  -> {validate_filter(None)}")
    print(f"  {{'source': 'a.txt'}}   -> {validate_filter({'source': 'a.txt'})}")

    print("\nvalidate_filter rejects, rather than dropping:")
    for label, bad in [
        ("a nested mapping", {"doc": {"id": "d1"}}),
        ("a list of values", {"source": ["a.txt", "b.txt"]}),
        ("not a mapping at all", "source=a.txt"),
    ]:
        try:
            validate_filter(bad)
        except StorageInputError as exc:
            print(f"  {label}\n    -> {str(exc)[:74]}...")

    print("\nvalidate_vector accepts:")
    print(f"  [1, 2.5]  -> {validate_vector([1, 2.5])}   (normalised to floats)")

    print("\nvalidate_vector rejects:")
    for label, bad in [
        ("NaN", [float('nan'), 1.0]),
        ("infinity", [1.0, float('inf')]),
        ("an empty vector", []),
        ("a string", "1,2,3"),
    ]:
        try:
            validate_vector(bad)
        except StorageInputError as exc:
            print(f"  {label:<16} -> {str(exc)[:64]}...")

    print("\nvalidate_ids accepts:")
    print(f"  ['doc-1', 'doc-2']  -> {validate_ids(['doc-1', 'doc-2'])}")

    print("\nvalidate_ids rejects:")
    try:
        validate_ids("doc-1")
    except StorageInputError as exc:
        print(f"  a bare string -> {str(exc)[:70]}...")
    try:
        validate_ids(["ok", ""])
    except StorageInputError as exc:
        print(f"  an empty id   -> {str(exc)[:70]}...")

    print("\nvalidate_k:")
    print(f"  5   -> {validate_k(5)}")
    try:
        validate_k(0)
    except StorageInputError as exc:
        print(f"  0   -> {str(exc)[:70]}...")

    print("\nevery failure in the area is catchable together:")
    print(f"  StorageInputError   is a StorageError: "
          f"{issubclass(StorageInputError, StorageError)}")
    print(f"  StorageBackendError is a StorageError: "
          f"{issubclass(StorageBackendError, StorageError)}")
    print(f"  the protocol is re-exported so an implementation needs only this "
          f"file: {VectorStoreProtocol.__name__}")

    try:
        validate_filter({"nested": {"deep": 1}})
    except StorageError as exc:
        print(f"\ncaught under the one base class: {type(exc).__name__}")


if __name__ == "__main__":
    main()

"""Runnable example for `citation-formatter`.

    python example.py

Shows citation spans being resolved, and an orphan span being refused. Runs
standalone with `core_types.py` beside it -- the manifest declares that
component a dependency, so `ragkit add` copies it along.

Nothing is installed to make this run.

The five guarantees this example demonstrates:

1. valid spans resolve, and text is returned unchanged;
2. an orphan span -- one naming a passage never shown -- raises a typed error;
3. every orphan is reported at once, not one per run;
4. the error says what IS valid, not only what failed;
5. the bibliography lists cited sources only, and never renders blank.
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

from citation_formatter import (  # noqa: E402
    CitationFormatter,
    CitationInputError,
    CitationOrphanSpanError,
)


def chunk(cid, source, index):
    return core.ScoredDocument(chunk_id=cid, source_name=source, text=cid,
                               score=1.0, chunk_index=index, metadata={})


def ctx(text, chunks, budget=100):
    return core.Context(text=text, chunks=tuple(chunks), token_count=5,
                        budget=budget, truncated=False)


def main() -> None:
    formatter = CitationFormatter()
    print(f"{formatter}")

    evidence = [chunk("c0", "python/errors.md", 0), chunk("c1", "cooking/pasta.md", 1)]
    good = ctx("Error handling [1] matters. Cooking [2] does not. Again, [1] is key.", evidence)

    print("\n1. Valid spans resolve, and the text is returned UNCHANGED:")
    print(f"    spans      : {formatter.spans(good.text)}")
    print(f"    references : {formatter.references(good)}")
    print(f"    format     : {formatter.format(good)!r}")
    print("    (unchanged on purpose: appending a bibliography would blow the")
    print("     budget context-selector just enforced)")

    print("\n   The bibliography lists cited sources only:")
    for label in formatter.bibliography(good):
        print(f"     - {label}")
    print("    -> both were cited, so both appear")

    print("\n2. An orphan span raises a typed error:")
    ghost = ctx("Error handling [1]. See also [3].", evidence)
    try:
        formatter.validate(ghost)
    except CitationOrphanSpanError as exc:
        print(f"    caught    : {type(exc).__name__}")
        print(f"    message   : {exc}")

    print("\n3. Every orphan is reported at once, not one per run:")
    many = ctx("[1] [9] [2] [8] [7]", evidence)
    try:
        formatter.validate(many)
    except CitationOrphanSpanError as exc:
        print(f"    spans     : {[n for n, _ in exc.spans]}")
        print("    -> fixing one per run would be a long loop")

    print("\n4. The error says what IS valid:")
    print(f"    -> the message above names '[1] through [{len(evidence)}]'")
    print("    -> a span with no chunks at all says 'none at all'")

    print("\n5. The bibliography never renders blank:")
    bare = core.ScoredDocument(chunk_id="c9", text="x", score=1.0, chunk_index=0, metadata={})
    print(f"    source_name missing -> {formatter.bibliography(ctx('[1]', [bare]))}")
    print(f"    nothing at all cited -> {formatter.bibliography(ctx('no citations', evidence))}")
    print(f"    an orphan is omitted -> {formatter.bibliography(ctx('[9]', evidence))}")

    print("\nNumbering is 1-based, because nobody writes [0] in prose:")
    try:
        formatter.validate(ctx("zero [0]", evidence))
    except CitationOrphanSpanError as exc:
        print(f"    [0] -> {type(exc).__name__}")
    print(f"    one_based=False accepts it: "
          f"{CitationFormatter(one_based=False).spans('[0]')}")

    print("\nBracketed non-numbers are ordinary text, left alone:")
    print(f"    {formatter.spans('see [note], [1.5] and [1]')}")

    print("\nText with no citations at all is valid:")
    print(f"    {formatter.format(ctx('a plain answer', []))!r}")
    print("    -> a model may answer without citing; inventing citations would be worse")

    print("\nCustom styles are a constructor argument, not a fork:")
    parentheses = CitationFormatter(pattern=r"\((\d+)\)")
    section = CitationFormatter(pattern=r"§\s*(\d+)")
    print(f"    parentheses : {parentheses.spans('see (1) here')}")
    print(f"    section sign: {section.spans('see §1 here')}")

    print("\nBad input is rejected with a typed error:")
    for label, call in [
        ("a broken regex", lambda: CitationFormatter(pattern="[")),
        ("two capture groups", lambda: CitationFormatter(pattern=r"\[(\d+)\]-(\d+)")),
        ("a non-Context", lambda: formatter.validate("not a context")),
        ("a non-string name", lambda: CitationFormatter(name=5)),
    ]:
        try:
            call()
        except CitationInputError as exc:
            print(f"    {label:<20} -> {str(exc)[:56]}...")


if __name__ == "__main__":
    main()

"""Runnable example for `text-normalizer`.

    python example.py

Shows every rule firing on extracted-text mess, and the property the component
is built around: normalizing is idempotent, so a re-run never changes what a
first run produced. This component has NO dependencies -- no core-types, no
packages, nothing installed -- so it runs anywhere Python 3.12 does.

The six guarantees this example demonstrates:

1. unicode normalization is idempotent (three passes, every form);
2. control characters are stripped -- but a page break stays visible;
3. line endings, tabs and blank-line floods each get their own rule;
4. the result reports what changed rather than hiding it;
5. NFC preserves compatibility forms, NFKC folds them;
6. leading indentation survives, because in a code corpus it is meaning.
"""

from __future__ import annotations

from text_normalizer import (
    FORMS,
    NormalizerInputError,
    TextNormalizer,
    normalize_text,
)

MESS = [
    ("crlf from a Windows toolchain", "line one\r\nline two\rline three"),
    ("a page break at every form feed", "page one text\x0cpage two text"),
    ("a vertical tab", "a\x0bb"),
    ("invisible controls", "a\x00b\x1bc\x85d"),
    ("runs of spaces and tabs", "a    b\t\tc"),
    ("trailing whitespace", "text   \nmore\t"),
    ("a flood of blank lines", "a\n\n\n\n\n\n\nb"),
    ("preserved indentation", "def f():\n        return 1"),
    ("an fi ligature", "the \ufb01le"),
    ("full-width latin", "\uff50\uff59\uff54"),
]


def main() -> None:
    normalizer = TextNormalizer()
    print(f"{normalizer}\n")

    print("1. Unicode normalization is idempotent -- a re-run never changes")
    print("   what a first run produced, because ingestion stages get retried")
    print("   and resumed, and a resumed job must not index different text:")
    everything_ok = True
    for label, text in MESS:
        once = normalizer.clean(text)
        twice = normalizer.clean(once)
        thrice = normalizer.clean(twice)
        ok = once == twice == thrice
        everything_ok = ok
        print(f"   {label:<34} once==twice==thrice: {ok}")
    print(f"   ALL IDEMPOTENT: {everything_ok}")

    print("\n   ...in every Unicode form, over the same corpus:")
    for form in FORMS:
        nn = TextNormalizer(form=form)
        good = all(
            nn.is_idempotent(text) for _, text in MESS
        )
        print(f"     {form:<5} idempotent over all {len(MESS)} samples: {good}")

    print("\n   is_idempotent() is public, so a pipeline can assert this")
    print(f"   about ITS OWN corpus: {normalizer.is_idempotent('a\\r\\n\\x00 b    c')}")

    print("\n2. Control characters are stripped -- but a page break stays")
    print("   visible, because in extracted text a form feed IS the only")
    print("   signal that a new page began:")
    for label, text, in [
        ("a NUL", "a\x00b"),
        ("an ESC", "a\x1bb"),
        ("a C1 control", "a\x85b"),
        ("DEL", "a\x7fb"),
        ("a form feed", "page one\x0cpage two"),
        ("a vertical tab", "a\x0bb"),
    ]:
        print(f"   {label:<16} {text!r:<24} -> {normalizer.clean(text)!r}")

    print("\n   Tab and newline are content, not commands -- but a tab")
    print("   MID-LINE is horizontal whitespace and does collapse. Only")
    print("   LEADING indentation is kept, because that is where it means")
    print("   something:")
    print(f"     'a\\tb'  -> {normalizer.clean('a\tb')!r}   (mid-line tab collapses)")
    print(f"     '\\t\\tx'  -> {normalizer.clean('\t\tx')!r}   (leading tab survives)")
    print(f"     'a\\nb'  -> {normalizer.clean('a\nb')!r}   (newline survives)")

    print("\n3. Each transformation gets its own rule, and reports itself")
    print("   only if it actually changed something:")
    for label, text in MESS[:8]:
        result = normalizer.normalize(text)
        rules = ", ".join(result.rules) if result.rules else "(none)"
        print(f"   {label:<34} {rules}")

    print("\n4. The result reports what changed rather than hiding it:")
    result = normalizer.normalize("a\r\n\x00b    c")
    print(f"     report : {result.report()}")
    print(f"     original kept: {result.original!r}")
    print(f"     changed      : {result.changed}")
    print("     a component that rewrote your corpus without saying so would")
    print("     be a component you could not debug.")
    print(f"     nothing to do: {normalizer.normalize('already clean').report()}")

    print("\n5. NFC preserves compatibility forms; NFKC folds them:")
    samples = ["\ufb01le", "\uff50\uff59\uff54"]
    for form in ("NFC", "NFKC"):
        nn = TextNormalizer(form=form)
        for sample in samples:
            out = nn.clean(sample)
            # ascii(), not !r: repr() of a PRINTABLE char keeps it raw, and a
            # Windows console in cp1252 cannot encode the ligature.
            print(f"     {form:<5} {ascii(sample):<18} -> {ascii(out)}")
    print("     NFC is the default because it is the LEAST destructive")
    print("     choice. But if you index for matching, NFKC is likely what")
    print("     you want -- without it, full-width and half-width spellings")
    print("     of the same term never match, and every query containing one")
    print("     silently returns nothing. That trade-off is yours to make.")

    print("\n   The rules can also be turned off individually:")
    off = TextNormalizer(collapse_whitespace=False, collapse_blank_lines=False)
    r = off.normalize("a    b\n\n\n\n\nc")
    print(f"     rules off -> {r.rules or '(none)'}  out -> {r.text!r}")

    print("\n6. Bad input is a typed error, not a surprise:")
    for label, fn in [
        ("form='NFX'", lambda: TextNormalizer(form="NFX")),
        ("collapse_whitespace=1", lambda: TextNormalizer(collapse_whitespace=1)),
        ("max_length=True", lambda: TextNormalizer(max_length=True)),
        ("text=None", lambda: normalizer.normalize(None)),
        ("over max_length", lambda: TextNormalizer(max_length=5).clean("x" * 10)),
    ]:
        try:
            fn()
            print(f"   {label:<24} NO ERROR")
        except NormalizerInputError as exc:
            print(f"   {label:<24} {type(exc).__name__}: {str(exc)[:46]}...")

    print("\n   The convenience function does the common case in one call:")
    print(f"     normalize_text('a\\r\\n\\x00b') -> {normalize_text('a\r\n\x00b')!r}")
    print(f"     normalize_text(..., form='NFKC') -> {normalize_text('fi', form='NFKC')!r}")

    print("\nWhat this component will not do:")
    print("   no stemming, no case folding, no accent stripping. Making text")
    print("   structurally comparable is not the same as deciding that")
    print("   'Resume' and 'resume' are one term -- that is a linguistic")
    print("   decision belonging to the analyzer, and doing it here would")
    print("   silently change what is stored.")
    print("   zero-width characters survive too: stripping U+200B would")
    print("   also strip U+200D, which is how emoji sequences are built.")


if __name__ == "__main__":
    main()

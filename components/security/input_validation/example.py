"""Runnable example for `input-validation`.

    python example.py

Shows the four hostile character families, what each one really does, and the
size limit. This component has NO dependencies -- no core-types, no packages,
nothing installed -- so it runs anywhere Python 3.12 does.

The five guarantees this example demonstrates:

1. every invisible, meaning-changing character is refused;
2. each offender is reported with its codepoint and offset;
3. the input comes back unchanged -- nothing is silently stripped;
4. oversized input raises a typed error, checked before the character scan;
5. recovery is explicit, and nothing is ever repaired behind your back.
"""

from __future__ import annotations

from input_validation import (
    CONTROLS,
    BIDI_OVERRIDES,
    DEFAULT_MAX_TEXT_LENGTH,
    ForbiddenCharacter,
    InputError,
    InputValidator,
    OversizedInputError,
    contains_forbidden,
    normalise,
    what_is_forbidden,
)

FAMILIES = [
    ("NUL byte", "report.pdf\x00.jpg",
     "one string to Python, two filenames to anything crossing into C -- "
     "which is how an extension check gets bypassed"),
    ("terminal escape", "window title\x1b]0;pwned\x07",
     "\\x1b[0]...\\x07 is a command: it silently rewrites a terminal's "
     "window title"),
    ("screen clear", "before\x1b[2Jafter",
     "\\x1b[2J clears the screen -- text as a destructive command"),
    ("RTLO override", "malware exe\u202etxt",
     "U+202E renders everything after it right-to-left, so this DISPLAYS as "
     "'txt.exe' -- the CVE-2017-1000112 filename spoof"),
    ("bidi isolate", "a\u2066b\u2069c",
     "U+2066/U+2069 open and close an isolated RTL run"),
    ("line separator", "line one\u2028line two",
     "a newline to some parsers, an ordinary character to others -- so a "
     "string can hide a 'line break' a downstream check never sees"),
    ("C1 control", "text\x85more",
     "NEL, a C1 control: invisible, and treated inconsistently across "
     "encodings"),
]


def main() -> None:
    validator = InputValidator(max_text_length=120)
    print(f"{validator}\n")

    print("A clean string passes, and comes back as the SAME object:")
    clean = "The json module serialises Python objects into JSON text."
    returned = validator.validate_text(clean)
    print(f"     {returned[:52]!r}...")
    print(f"     returned is the input: {returned is clean}")

    print("\n1. Every invisible, meaning-changing character is refused:")
    for label, text, why in FAMILIES:
        try:
            validator.validate_text(text, field="query")
        except ForbiddenCharacter as exc:
            print(f"\n   {label}")
            print(f"     refused : {', '.join(exc.characters[:3])}")
            print(f"     why     : {why}")

    print("\n2. Each offender is reported with its codepoint and offset:")
    try:
        validator.validate_text("a\x00b\u202ec", field="document")
    except ForbiddenCharacter as exc:
        print(f"     characters : {exc.characters}")
        print(f"     message    :")
        for line in str(exc).split(". "):
            print(f"       {line.strip()}")
    print("     every one at once -- discovering them one per request is a")
    print("     slow way to find out what is in an input.")

    print("\n3. The input comes back unchanged -- nothing is silently stripped:")
    hostile = "keep\x00this"
    try:
        validator.validate_text(hostile, field="value")
    except ForbiddenCharacter:
        print("     refused rather than repaired, because a filter that")
        print("     quietly removed the NUL WOULD BE THE BYPASS:")
        print(f"       'a\\x00b' repaired -> {normalise('a\x00b')!r}")
        print("     the caller never learns the input was malformed at all.")

    print("\n   what_is_forbidden inspects without raising:")
    for code, name, offset in what_is_forbidden("a\x00b\u202e"):
        print(f"     {code} at offset {offset}: {name}")
    print("     (NUL is unnamed in UnicodeData.txt, so it is described as a")
    print("      control character -- it is assigned, not unassigned)")

    print(f"   contains_forbidden: {contains_forbidden('clean')} -> clean")
    print(f"                       {contains_forbidden('bad\x00')} -> has a NUL")

    print("\n4. Oversized input raises a typed error:")
    print(f"     limit = {validator.max_text_length:,} characters "
          f"(default is {DEFAULT_MAX_TEXT_LENGTH:,})")
    try:
        validator.validate_text("x" * 500, field="document")
    except OversizedInputError as exc:
        print(f"     {type(exc).__name__}")
        print(f"       {str(exc)[:74]}...")
        print("       ...the limit is a CEILING meaning 'this is not a real")
        print("       document', not a formatting policy. Set your own.")
    try:
        validator.validate_text("x" * 200, field="note", max_length=150)
    except OversizedInputError:
        print("     a per-call max_length= overrides the validator's limit")

    print("   Size is checked BEFORE the character scan, so the real bug is")
    print("   what gets reported:")
    try:
        InputValidator(max_text_length=4).validate_text("a\x00" * 100)
    except OversizedInputError as exc:
        print(f"     {str(exc)[:40]}...  <- the size, not U+0000")
    print(f"   Exactly the limit passes: {len(validator.validate_text('x' * 120)) == 120}")

    print("\n5. Recovery is explicit:")
    print(f"     normalise(U+FB01 ligature)  -> {normalise('\ufb01')!r}   "
          "(NFKC unifies compatibility forms)")
    print(f"     normalise(U+FF21 fullwidth) -> {normalise('\uff21')!r}   "
          "(fullwidth A collapses)")
    print(f"     normalise('a\\x00b') -> {normalise('a\x00b')!r}   "
          "(and DROPS NOTHING -- this is not a cleaning function)")

    print("\n   validate_non_empty catches the blank case, which is a")
    print("   different bug from a hostile one:")
    try:
        validator.validate_non_empty("   ", field="query")
    except InputError as exc:
        print(f"     {str(exc)[:66]}...")

    print("\n   validate_identifier is a narrower problem: identifiers are")
    print("   interpolated into paths, filenames and queries, so the")
    print("   character set is RESTRICTED rather than merely screened:")
    for value, ok in [("my-doc_1.2", True), ("a/../b", False),
                      ("a b", False), ("café", False)]:
        try:
            validator.validate_identifier(value)
            print(f"     {value!r:<14} accepted")
        except InputError:
            print(f"     {value!r:<14} refused")

    print("\nThe four families, as sets you can check yourself:")
    print(f"   CONTROLS      {len(CONTROLS)} chars  "
          f"(NUL={chr(0) in CONTROLS}, ESC={chr(0x1b) in CONTROLS}, "
          f"TAB={chr(9) in CONTROLS}, LF={chr(10) in CONTROLS})")
    print(f"   BIDI_OVERRIDES {len(BIDI_OVERRIDES)} chars  "
          f"(U+202E={chr(0x202e) in BIDI_OVERRIDES})")
    print("   TAB and LF are ordinary whitespace and are allowed; CR is not,")
    print("   because it rewrites the start of a line in a log.")

    print("\nWhy this refuses rather than sanitises:")
    print("   everything here has a legitimate form in some corpus, and every")
    print("   attempt to keep the useful version while blocking the abuse is")
    print("   another bypass to find. Rejecting leaves the decision with the")
    print("   caller, who can see what arrived.")


if __name__ == "__main__":
    main()

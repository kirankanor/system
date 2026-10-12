# index-verifier

**A successful write is never reported as verification.** That sentence is
the whole component.

A write returning success proves the write returned success. It says nothing
about whether the rows are there now, whether they are the rows that were
expected, whether their vectors are the width the index claims, or whether a
query against them returns anything.

This component takes an *expectation* — the records that should be in the
index — and reads the index back until it can say, per record, whether that
expectation holds.

## The checks

| Check | Question it answers |
| --- | --- |
| `count` | Does the index hold as many records as were expected? |
| `presence` | Is every expected record actually retrievable? |
| `embedding` | Does each stored record have a usable vector? |
| `dimension` | Is every vector the width the index is built for? |
| `metadata` | Is the metadata we declared compatible with what is stored? |
| `relationship` | Does document → chunk → source still hold? |
| `retrieval` | Does querying this index return the probe record? (FR-50) |

**A count match is not verification.** Some backends report a count that
includes stale rows, so the count can be exactly right while a record has
quietly gone. `presence` names the missing record; `count` notices the total
moved. Both run because neither alone is sufficient.

## Fatal versus warning

A **fatal** finding means the index cannot be trusted: a missing record, a
dimension mismatch, a disagreeing count, a broken relationship at the root, a
failed smoke check. A **warning** means it will answer, but degraded: no
source file to cite, no location within the document, declared metadata that
is empty.

```python
verdict = IndexVerifier(store, expected=items, dimensions=1536).verify()
verdict.ok        # no fatal finding
verdict.ready      # ok AND every check ran
verdict.fatal      # tuple[Finding]
verdict.warnings   # tuple[Finding]
verdict.named("c-7")          # everything about one record (AC-27)
verdict.checks_not_run()      # what could not be examined
print(verdict.report())
verdict.raise_if_failed()     # opt-in; verify() itself never raises
```

`ready` is deliberately unreachable while `checks_not_run()` is non-empty: a
verdict missing evidence is not a verdict that passed. The writer is not
reachable from this object at all — there is no `add`, `put` or `delete` — so
no configuration turns a green write into a green verification.

## Eventual consistency

A write may not be visible the instant it returns. `wait_ms` bounds how long
verification keeps re-checking **only** the findings consistency can fix:
`count`, `presence`, `retrieval`. A dimension mismatch will not resolve
itself, so waiting for it would only delay an inevitable failure.

The bound is an attempt count derived from `wait_ms / interval_ms`, not just
the clock, so a clock that does not advance cannot cause an infinite loop.

```python
verifier = IndexVerifier(
    store,
    expected=items,
    dimensions=1536,
    wait_ms=2_000,      # bounded
    interval_ms=100,
)
verdict = verifier.verify()   # verdict.waited_ms, verdict.attempts
```

## Evidence: readback or query

`VectorStoreProtocol` offers `count` and `query`, not `get`. When the store
*does* expose a `get(chunk_id)` (used automatically, or passed as `fetch=`),
presence and dimension are confirmed by reading each record directly, which
is stronger evidence and names the record exactly.

Without a readback both are established by a filtered vector query: a record
that cannot be found still gets a fatal finding with its id and the reason,
but a stored vector's width cannot be inspected directly — in that case
`dimension` is established by the probe query succeeding at the expected
width. `verifier.has_readback` reports which path was available.

## Errors

| Exception | Raised when |
| --- | --- |
| `VerificationError` | Bad construction (no store, non-positive dimensions, unusable `wait_ms`/`probe_k`), or by `raise_if_failed()` on a fatal verdict. |

Verification itself **never raises for a finding** — a failed verification is
a result.

## Example

```
python example.py
```

## Not implemented

No repair, no automatic re-index, no metric collection (recall@k against a
labelled set). This component answers "is the index what we said it would
be?", which is a different question from "is retrieval any good?".

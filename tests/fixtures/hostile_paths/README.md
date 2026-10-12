# Hostile path fixtures

These fixtures are **created on the real Windows filesystem at test time**, not
committed, because their whole point is to be real reparse points rather than
descriptions of one (FR-34, AC-11).

Covered by `tests/unit/test_path_safety.py`:

| Case | How it is created | Result on this host |
|---|---|---|
| `../` at any depth | plain string, no filesystem needed | rejected, `traversal` |
| absolute path outside the project | plain string | rejected, `absolute` / `drive-letter` |
| drive letter `C:\...` | plain string | rejected, `drive-letter` |
| drive-relative `C:file` | plain string | rejected, `drive-relative` |
| UNC `\\server\share` | plain string | rejected, `unc` |
| extended prefix `\\?\` | plain string | rejected, `extended-prefix` |
| alternate data stream `f.py:s` | plain string | rejected, `ads` |
| reserved device name `CON`, `com1`, `LPT9`, `nested/AUX` | plain string | rejected, `reserved-name` |
| **junction** pointing outside | `cmd /c mklink /J` (needs no elevation) | created for real, rejected |
| **dangling junction** branch | `cmd /c mklink /J` to a non-existent target | created for real, rejected |
| symlink pointing outside | `os.symlink` | **skipped**, see below |
| symlink pointing inside root | `os.symlink` | **skipped**, see below |

## Recorded skip reason (FR-34)

Two tests skip because this host refuses directory symlinks:

```
[WinError 1314] A required privilege is not held by the client
```

`os.symlink` for directories requires `SeCreateSymbolicLinkPrivilege`, which is
granted to administrators only unless **Developer Mode** is enabled. The tests
do not fabricate a pass for this case; they call `pytest.skip` with the OS
reason verbatim so the gap is visible in every run.

What this means for coverage:

* **Reparse-point escape is still covered on the real filesystem** — by the two
  junction tests, which create actual junctions and confirm no byte is written
  past the destination root.
* The **symlink**-specific path is not exercised here. Symlinks and junctions
  are both reparse points and hit the same containment code, so the logic is
  covered; only the syscall that creates them differs.
* Enabling Developer Mode would make both symlink tests run. No assertion
  depends on that, and no test passes merely because creation was refused.

Junctions are used in preference to symlinks precisely because `mklink /J` works
without elevation, so the majority of the reparse coverage runs everywhere.

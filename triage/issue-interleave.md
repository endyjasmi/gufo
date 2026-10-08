While triaging #6 we found that the final interleaved-pair check of `qwen38_flash_next_session_test --sampling-only` fails on **pristine Windows builds**. It is not related to the #6 fix.

## Signature

The last lines of the suite (all 25 serial cases pass with `replay_exact=1` before this):

```text
interleave mismatch: backend=ar case=<case-name> actual_tokens=8 expected=8 first_diff=7 drafts 0/0 accepted 0/0
FAIL: interleaving changed serving sampling or acceptance
```

## Reproduces on

- the pre-#6-fix base (verified by stashing the fix during the fix session, 2026-10-07),
- the v0.9.0 base (`c1028db1`, re-confirmed 2026-10-08).

## Interpretation

A pre-existing batch-path AR-vs-AR token flip under interleaved execution: the same deterministic case produces different tokens when another session's work interleaves with it. The serial-vs-interleaved contract this check asserts does not hold on Windows builds today. Likely batch geometry, graph capture, or scheduling under concurrency — not yet localized.

## Ops note

The test exe must be copied into the build root before running: from `tests/…/` the Windows DLL search picks up the driver's `C:\Windows\System32\amdhip64_7.dll`, whose comgr cannot find matching device bitcode, and the process segfaults with a 0-byte log. `run-session-v090.bat` on the `triage/issue-6-empty-turns` branch does copy/run/delete.

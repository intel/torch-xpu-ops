# XPU UT Refactor Review Checklist

Evaluate every changed line against these patterns. Each row gives a code
pattern to look for, what it means (the defect and the correct form), and a
severity. Severities follow this scale:

- **Blocker** — XPU gets wrong or zero test execution (the test never
  instantiates, runs on the wrong device, silently runs the wrong dtype/skip
  gating, risks an unguarded OOM/crash), or the change breaks another backend.
  Must fix before merge.
- **Major** — coverage or scope is silently wrong (a dropped/weakened assertion
  or case, a mismatched or missing skip/xfail, a behavior-changing "refactor").
  Fix before merge.
- **Minor** — tuning- or convention-level issue (tolerance, over-broad skip
  scope, naming/classification mismatch, stale skip). Fix or justify.
- **Info** — no direct action required; guidance, or a pattern with no XPU
  equivalent. Note and move on.

Rows are grouped by area and roughly ordered by how often they surfaced as real
review comments across the studied PRs.

This checklist focuses on the XPU-specific and distributed concerns of a test
refactor. Generic test-refactoring checks (classification, naming, API
replacement, `HardwareClassification` tagging, instantiation mechanism) are out
of scope here.

## 1. Device gating precision (blanket-skip anti-pattern)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `if device_type == "xpu": self.skipTest(...)` inside a test the PR enables on XPU | May be an over-skip: disables the whole XPU test to dodge one CUDA-specific check, hiding the real predicate. Human review required. | Info |
| `@unittest.skipIf(not SM70OrLater, ...)` | Decorators like `@unittest.skipIf(not SM70OrLater, ...)` used in non-CUDA specific test classes will erroneously skip XPU and other active devices. | Blocker |

## 2. Skips, xfails, and tolerances

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `@skipXPU` / `@skipIfXpu` / `@xfailIf(TEST_XPU)` / `DecorateInfo(unittest.skip("Skipped"))` without tracking issue link | Every skip/xfail needs a tracking issue. A bare skip/xfail is untraceable. | Major |
| `device_type='xpu'` skip with no `dtypes=(...)` for a single-dtype failure | May be an over-skip beyond the actual failure. Narrow the scope to the failing dtype/device. | Info |
| Old `@skipIfXpu` / skip left in place when the test already passes on XPU | Stale skips must be removed. | Minor |

## 3. Test intent and coverage preservation

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| Helper moved into a mixin/base class with an altered body | Helper extraction must be behavior-preserving; confirm no method body changed and both classes still reach it. | Major |
| `if TEST_CUDA: < code block >` widened to `TEST_CUDA or TEST_XPU` | Assumes XPU exposes the same stats key as CUDA. Confirm the key exists for XPU rather than assuming parity, or it `KeyError`s. | Blocker |
| Test exercises an op with only a CUDA/Meta registration, no XPU one | An op is mistakenly identified as being supported on XPU, which makes the "enabled" test fail or silently no-op. | Blocker |

## 4. Cross-device blast radius

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| Reordered `skips=`/`decorators=` tuples or changed `active_if` | Can silently alter another backend (MPS/HPU); a real reviewer concern. | Blocker |
| An iterable feeding `@parametrize` converted `tuple -> set` (or `set -> tuple`) | Introduces nondeterministic ordering; has caused real breakage needing a follow-up fix. Flag any such conversion of a parametrization source. | Blocker |

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

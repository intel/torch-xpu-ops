---
name: ut-fixing
description: >
  Fix XPU unit test failures — covers removing stale skip/xfail decorators,
  enabling newly added tests for XPU, and verifying fixes locally.
  Read this skill when the issue involves UT failures in the pytorch repo.
---

# UT Fixing for XPU

## When to Use
- The issue involves unit test failures (test_type is `ut`)
- The fix requires removing XPU skip/xfail decorators
- A newly added upstream test needs XPU enablement
- The triage category is "Skip decorator stale"

## Step 1: Identify the Target Test

1. Check the issue record or the PR that originally introduced the skip to locate the target UT.
2. Verify the UT still exists in the current codebase.
3. If not found by name, check modification history — tests may have been renamed or moved.
4. **Dynamic test names:** Many test names are constructed dynamically (e.g., function names from `op_db` concatenated with device suffixes). If simple text search fails, search for the constructed case patterns:
   - `TestCommonXPU` is generated from base class `TestCommon` via `instantiate_device_type_tests`
   - Pattern: `<BaseClass>` + device suffix (`XPU`, `CUDA`, `CPU`)

## Step 2: Find XPU Skip Markers

Scan the test definition and surrounding context for XPU-specific skip or xfail markers:

| Pattern | Location |
|---------|----------|
| `@skipXPU` or `@unittest.skipIf(..., "XPU ...")` | Decorator on test method |
| `@expectedFailureXPU` or `@xfailIfXPU` | Decorator on test method |
| `DecorateInfo(unittest.skip("..."), device_type='xpu')` | Inside `OpInfo` definitions |
| Skip dictionaries: `skip_xpu`, `xfail_xpu` | Used in `instantiate_device_type_tests` |
| `skipIfXpu` in conditional blocks | Inline skip logic |

### Search Commands
```bash
# Find skip decorators in a specific test file
grep -n "skipXPU\|skipIfXpu\|xfailIfXPU\|expectedFailureXPU\|device_type='xpu'" test/<test_file>.py

# Find OpInfo skip entries for a specific op
grep -n -A2 "DecorateInfo.*skip.*xpu" torch/testing/_internal/common_methods_invocations.py
```

## Step 3: Remove the Skip Marker

**Proceed only if an XPU-specific skip or xfail marker is found.**

- **Decorator:** Delete the decorator line. Remove unused imports if the decorator was the last usage.
- **List/dictionary entry:** Remove the entry from the XPU skip collection. If it was the only entry, remove the entire collection variable if unused.
- **OpInfo DecorateInfo:** Remove the `DecorateInfo(...)` entry from the `decorators` list.

### Cleanup Checklist
- [ ] Removed the skip/xfail marker
- [ ] Removed unused imports (e.g., `from torch.testing._internal.common_utils import skipIfXpu`)
- [ ] No other tests in the file rely on the removed import
- [ ] `git diff` shows only skip removal changes

## Step 4: Local Verification on XPU

Run the test on the local XPU device to confirm it passes after the skip is removed.

```bash
# For pytest-style tests
source .env && python test/<test_file>.py -v -k "<test_name> and xpu"

# For specific class::method
source .env && pytest -xvs "test/<test_file>.py::<TestClass>::<test_method>"
```

### Expected Results
- ✅ **PASSED** — test reports `PASSED`, not `SKIPPED` or `XPASS`
- ❌ **XPASS** — means expected failure but it passed; the xfail decorator should be removed
- ❌ **SKIPPED** — another skip decorator may still be active; investigate
- ❌ **FAILED** — the underlying bug is not fixed yet; the skip should NOT be removed

**If the test FAILS after removing the skip:** The underlying issue still exists. Do NOT remove the skip — report that the test still fails and the issue needs a code fix first.

## Step 5: Verify and Report

1. Run `git diff` — verify only skip removal changes are present
2. Run the full test file to check for regressions:
   ```bash
   source .env && python test/<test_file>.py -v 2>&1 | tail -30
   ```
3. Report to the issue:
   - Modified file path(s) and removed skip logic
   - Terminal output confirming the test passed
   - Completion message: "UT unskip is complete and locally verified."

## HARD RULES
- NEVER remove a skip decorator if the underlying test still FAILS.
- NEVER add new skip decorators — the goal is to REMOVE them.
- ALWAYS verify locally before reporting success.
- ALWAYS clean up unused imports created by the removal.

---
name: issue-fix
description: >
  Fix a triaged CI failure for pytorch or torch-xpu-ops. Use when an issue
  already contains a root-cause analysis and a proposed fix strategy, and you
  need to implement the fix, verify it, and prepare a summary. Handles both
  unit-test (UT) and end-to-end (E2E) failures, including stale XPU skip-marker
  removal.
---

# Fix CI Failure

This skill takes a triaged issue and turns it into a verified code fix. It does
not open the PR itself — see the cross-references at the end.

> **Execution mode:** this skill behaves differently in interactive (default)
> vs pipeline mode. See [../references/execution-modes.md](../references/execution-modes.md).
> For environment activation, build commands, the torch-xpu-ops `xpu.txt` pin,
> and rebuild pitfalls, see
> [../references/environment-setup.md](../references/environment-setup.md).

## Inputs

You need:

- **A triaged CI failure issue** whose body contains a root-cause analysis and
  a proposed fix strategy. Follow them. (If the issue is not yet triaged, triage
  it first with the `xpu-issues-triaging` skill.)
- **The target repo**: `pytorch` or `torch-xpu-ops`. Confirm with
  `basename $(git rev-parse --show-toplevel)`.
- **A reproducer command** (pytest invocation, python script, or bash command),
  or enough information (failed test name + error context) to construct one.

## Step 0: Verify Environment
1. Check which repo you're in: `basename $(git rev-parse --show-toplevel)`
   - If `torch-xpu-ops`: you're fixing XPU kernel/operator code (files under `src/`)
   - If `pytorch`: you're fixing PyTorch core code (files under `torch/`, `aten/`, `test/`, `c10/`)
2. Verify the worktree is clean: `git status` should show no uncommitted changes
3. Start from a clean base: checkout the main branch (or your reproducer branch)

## Step 1: Reproduce
Reproduce the failure with the `test-verification` skill — pass it the
reproducer command from the issue (it activates the env and runs it). If the
issue has no command, construct one from the failed test name and error context
first. Confirm the bug actually reproduces before changing any code.

If you modified C++/CUDA/SYCL code (not just Python), rebuild before re-running.
See [../references/environment-setup.md](../references/environment-setup.md) for
the environment activation, build commands, and the torch-xpu-ops `xpu.txt`
submodule-pin workflow (how to test a torch-xpu-ops fix inside pytorch without
triggering a full rebuild).

## Step 2: Implement the Fix

### Key Rules
- **Minimal changes** — fix only what's broken
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`.
- **Stay in your repo** — if in pytorch, don't modify `third_party/*`.
  - Exception: you may edit `third_party/torch-xpu-ops/` files when the
    triage output explicitly says `target_repo == "torch-xpu-ops"`.
- **Never modify unrelated files**
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk, rebase (`git rebase origin/main`) instead.

### Fix Strategies by Category
- **Unit tests (non-E2E):** For UT failures (not end-to-end models), see the **UT Skip Removal** section below.
- **Newly added test:** Try to enable it for XPU. If XPU support is genuinely missing and out of scope for this fix, do NOT add skip decorators — instead, try to loop and fix, only add "NEEDS_HUMAN" or report to the user if you really can't solve.
- **Regression:** Find the guilty commit by reviewing recent commit history. Apply an XPU-specific fix if necessary. If you can't identify the guilty commit, compare with cuda/rocm backend to find the root cause.
- **Tolerance:** Match upstream CUDA tolerances when adjusting XPU tolerances.
- **Skip decorator stale:** See the **UT Skip Removal** section below.

For deeper XPU-operator or CUDA-UT-porting fixes (locating the original CUDA
kernel/test, diffing CUDA vs XPU behavior, deciding pytorch vs torch-xpu-ops as
the fix repo), use the `xpu-issues-triaging` skill.

### UT Skip Removal
This section states about how to remove the `@skipIfXpu` / `@xfailIfXPU` / `@expectedFailureXPU` decorator in unit-tests. If not needed, directly go to the next section.

**1. Find skip markers** — scan for these patterns:

| Pattern | Location |
|---------|----------|
| `@skipXPU`, `@unittest.skipIf(..., "XPU ...")` | Decorator on test method |
| `@expectedFailureXPU`, `@xfailIfXPU` | Decorator on test method |
| `DecorateInfo(unittest.skip("..."), device_type='xpu')` | Inside `OpInfo` definitions |
| Skip dictionaries: `skip_xpu`, `xfail_xpu` | Used in `instantiate_device_type_tests` |
| `skipIfXpu` in conditional blocks | Inline skip logic |

```bash
grep -n "skipXPU\|skipIfXpu\|xfailIfXPU\|expectedFailureXPU\|device_type='xpu'" test/<test_file>.py
grep -n -A2 "DecorateInfo.*skip.*xpu" torch/testing/_internal/common_methods_invocations.py
```

**2. Remove the marker** — delete the decorator/entry. Clean up unused imports if the decorator was the last usage. For `OpInfo` `DecorateInfo` entries, remove the entry from the `decorators` list.


**3. Dynamic test names** — many test classes are dynamically generated via `instantiate_device_type_tests` (e.g., `TestCommonXPU` from `TestCommon`). If simple search fails, check for the base class + device suffix pattern.

## Step 3: Verify
After the fix is proposed, re-run via the `test-verification` skill to confirm
the fix works, and run related tests to check for regressions.

**Run EVERY failing test case from the report individually.** Do not skip any
case or assume verifying one representative case is sufficient.

If you modified C++/CUDA/SYCL code, rebuild pytorch before verifying (see Step 1).

## Step 4: Clean Up
```bash
# Stage only your changes (exclude third_party, submodules)
git add <your_files>
git diff --cached --stat  # verify only intended files
```

## Step 5: Update Issue Description
After fixing, summarize what was changed and why, plus the test results
(pass/fail). In **interactive mode** (default), report this to the user. In
**pipeline mode**, write it into the issue body (see below).

### Issue-body status (Pipeline mode only)
**Pipeline mode only.** This stage corresponds to legacy status `IMPLEMENTING`.
A Python driver script formerly wrote this status; in pipeline mode the agent
does it directly. This stage owns the `<!-- agent:fix-log -->` slot and the
"Fix implemented" Action Item, and advances `<!-- agent:status:IMPLEMENTING -->`.
Keep the markers from the templates under `.github/ISSUE_TEMPLATE/agent/` intact.
In interactive mode, report to the user instead. See `issue-handler/SKILL.md`
for the overall stage/label contract and "Execution modes".

## Step 6: Open the PR
To open the PR (branch naming, reproducer test, PR body, lint, push), use the
`xpu-ops-pr-creation` skill. Do not duplicate that workflow here.

## HARD RULES
- NEVER modify files outside your repo scope (`torch-xpu-ops` vs `pytorch`).
  The only exception is editing `third_party/torch-xpu-ops/` files from inside
  the pytorch repo when the triage output explicitly says `target_repo == "torch-xpu-ops"`.
- NEVER modify unrelated files — every changed line must trace directly to the issue.
- NEVER add skip decorators (`@skipIfXpu`, `@skip`, `unittest.skip`). Fix the test.
- NEVER submit a torch-xpu-ops PR for a bug whose root cause is in pytorch core.
- NEVER cherry-pick upstream commits. Rebase instead.

## Output
At the end, output:
```
### Agent Summary
- **What I found:** <root cause in one sentence>
- **What I changed:** <bullet list of files>
- **Test result:** <PASS/FAIL with test command>
- **Open questions / risks:** <concerns or "None">
```

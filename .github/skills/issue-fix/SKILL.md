---
name: issue-fix
description: >
  Fix a triaged CI failure. The issue body contains root cause analysis
  and proposed fix strategy — follow them. Works for both pytorch and
  torch-xpu-ops repos, UT and E2E failures.
---

# Fix CI Failure

## Step 0: Verify Environment
1. Verify torch is importable: `python -c "import torch; print(torch.__version__)"`
2. Verify the worktree is clean: `git status` should show no uncommitted changes
3. Start from a clean base: checkout the main branch (or your reproducer branch)

## Step 1: Reproduce
Extract the reproducer command from the issue description. It may be:
- A pytest command
- A python script
- A bash command
- Any other test invocation

Run it exactly as specified. If the issue has no reproducer, construct one
from the failed test name and error context.

If you modified C++/CUDA/SYCL code (not just Python), rebuild pytorch first:
```bash
python setup.py develop 2>&1 | tail -20
```

## Step 2: Implement the Fix
Follow the **Proposed Fix Strategy** from the issue. Key principles:
- **Minimal changes** — fix only what's broken
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`
- **Never modify submodules** — no changes to `third_party/*`

## Step 3: Verify
Run the reproducer command again to confirm the fix works.
Also run related tests to check for regressions.

If you modified C++/CUDA/SYCL code, rebuild pytorch before verifying.

## Step 4: Clean Up
```bash
# Stage only your changes (exclude third_party, submodules)
git add <your_files>
git diff --cached --stat  # verify only intended files
```

## Step 5: Update Issue Description
After fixing, update the issue body with:
- What was changed and why
- Test results (pass/fail)

## HARD RULES
- NEVER add skip decorators. FIX the test.
- NEVER commit submodule changes (`third_party/*`).
- NEVER modify unrelated files.
- NEVER force-push. This makes commits un-trackable.
- Use `git add` on specific files only.

## Output
At the end, output:
```
### Agent Summary
- **What I found:** <root cause in one sentence>
- **What I changed:** <bullet list of files>
- **Test result:** <PASS/FAIL with test command>
- **Open questions / risks:** <concerns or "None">
```

---
name: issue-fix
description: >
  Fix a triaged CI failure. The issue body contains root cause analysis
  and proposed fix strategy — follow them. Works for both pytorch and
  torch-xpu-ops repos, UT and E2E failures.
---

# Fix CI Failure

## Step 0: Verify Environment
1. Check which repo you're in: `basename $(git rev-parse --show-toplevel)`
   - If `torch-xpu-ops`: you're fixing XPU kernel/operator code (files under `src/`)
   - If `pytorch`: you're fixing PyTorch core code (files under `torch/`, `aten/`, `test/`, `c10/`)
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

If you modified C++/CUDA/SYCL code (not just Python), rebuild first:
- **pytorch repo**: `python setup.py develop 2>&1 | tail -20`
- **torch-xpu-ops repo**: No separate build step needed (built as part of pytorch)

### torch-xpu-ops submodule pin (xpu.txt)
pytorch pins torch-xpu-ops at a specific commit via `third_party/xpu.txt`.
During CMake build, pytorch reads this SHA, fetches the torch-xpu-ops repo,
and checks out that exact commit into `third_party/torch-xpu-ops/`.

**To test a torch-xpu-ops fix inside pytorch:**
1. Copy only the changed files into `~/pytorch/third_party/torch-xpu-ops/`
2. Run `ninja -C ~/pytorch/build` for an incremental rebuild (only recompiles changed files)
3. Run tests from the pytorch root directory
4. After testing, restore: `cd ~/pytorch/third_party/torch-xpu-ops && git checkout .`

**Do NOT** do a full `git checkout <branch>` in `third_party/torch-xpu-ops/` —
this changes mtimes on all files and triggers a massive ninja rebuild.
Copy only the changed files to keep incremental builds fast.
**Do NOT** modify `third_party/xpu.txt` — changing the pin triggers CMake
reconfiguration and a full rebuild from scratch (~hours).

## Step 2: Implement the Fix
Follow the **Proposed Fix Strategy** from the issue. Key principles:
- **Minimal changes** — fix only what's broken
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`
- **Stay in your repo** — if in pytorch, don't modify `third_party/*`; if in torch-xpu-ops, only modify files under `src/`
- **Never modify unrelated files**

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
- NEVER modify files outside your repo scope. Exception: if you're in the pytorch repo and the issue targets `third_party/torch-xpu-ops/` files, you may edit those (they are torch-xpu-ops source bundled as a submodule). Do NOT modify other `third_party/*` submodules. If in torch-xpu-ops: only modify files under `src/`.
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

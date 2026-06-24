---
name: fix/implement
description: >
  Implement a fix for a triaged XPU/PyTorch failure. Takes triage output and
  produces a verified code change. Used by both issue-handler and
  nightly-ci-fix orchestrators via the allow_skip parameter.
---

# Implement — Apply the Fix

Takes triage output and makes the code change. Does not run tests (that is
`fix/verify`'s job) and does not open PRs or commit (that is the
orchestrator's job).

## Inputs

- `triage_result` — JSON output from `fix/triage` (root_cause, fix_strategy,
  target_repo).
- `pytorch_dir` — path to local PyTorch checkout.
- `allow_skip` — controls skip decorator strategy:
  - `false` (**issue-handler**): never add skip decorators; must unskip and
    really fix.
  - `true` (**nightly-ci-fix**): may add `@skipIfXpu` with a tracking issue
    when the fix requires significant implementation work beyond the current
    scope. Stale skips must still be removed.
- `commit_message_template` (optional) — orchestrator-provided format. If
  absent, use a concise imperative message.

## Step 0: Verify environment

```bash
basename $(git rev-parse --show-toplevel)  # confirm which repo you're in
git status                                  # confirm clean worktree
```

- `torch-xpu-ops` → fix XPU kernel/operator code (files under `src/`)
- `pytorch` → fix PyTorch core code (files under `torch/`, `aten/`, `test/`,
  `c10/`)

## Step 1: Read the triage output

Read `triage_result` carefully before touching any file. Understand:
- Which files/functions need to change
- Why CUDA works but XPU fails (if applicable)
- Whether the fix is in pytorch or torch-xpu-ops

If the issue is not yet triaged, run `fix/triage` first.

## Step 2: Implement the fix

### Key rules

- **Minimal changes** — fix only what's broken; every changed line must trace
  to the triage output.
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk,
  rebase (`git rebase origin/main`) instead.
- **Stay in your repo** — if in pytorch, do not modify `third_party/*` unless
  `triage_result.target_repo == "torch-xpu-ops"`, in which case editing
  `third_party/torch-xpu-ops/` is the correct action.
- **Never modify unrelated files.**

### Fix strategies by category

See [../references/failure-categories.md](../references/failure-categories.md)
for the full taxonomy. Common strategies:

- **Tolerance:** match upstream CUDA `atol`/`rtol` values exactly.
- **Regression:** find the guilty commit (`git log --oneline -20 -- <file>`),
  apply an XPU-side fix aligned with upstream intent; document any CUDA
  divergence in comments.
- **Newly added test:** enable XPU support. If `allow_skip=false` and support
  is genuinely missing, report `NEEDS_HUMAN` — do not add a skip. If
  `allow_skip=true`, add `@skipIfXpu` with a tracking issue.
- **Unknown root cause:** compare with CUDA/ROCm backend behavior.

### UT Skip Removal

When the fix is to remove a stale `@skipIfXpu` or `@expectedFailureXPU`:

**1. Find skip markers:**

```bash
grep -n "skipXPU\|skipIfXpu\|xfailIfXPU\|expectedFailureXPU\|device_type='xpu'" \
  <test_file>
grep -n -A2 "DecorateInfo.*skip.*xpu" \
  torch/testing/_internal/common_methods_invocations.py
```

Patterns to look for:

| Pattern | Location |
|---------|----------|
| `@skipXPU`, `@unittest.skipIf(..., "XPU ...")` | Decorator on test method |
| `@expectedFailureXPU`, `@xfailIfXPU` | Decorator on test method |
| `DecorateInfo(unittest.skip("..."), device_type='xpu')` | Inside `OpInfo` definitions |
| `skip_xpu`, `xfail_xpu` dict entries | Used in `instantiate_device_type_tests` |
| `skipIfXpu` in conditional blocks | Inline skip logic |

**2. Remove the marker** — delete the decorator/entry. Clean up unused imports
if this was the last usage. For `OpInfo` `DecorateInfo` entries, remove the
entry from the `decorators` list.

**3. Dynamic test names** — many test classes are generated via
`instantiate_device_type_tests` (e.g. `TestCommonXPU` from `TestCommon`). If a
simple search fails, look for the base class + device suffix pattern.

### allow_skip=true: adding a new skip (nightly-ci-fix only)

Only when `allow_skip=true` AND the fix genuinely requires significant
implementation work (missing kernel, blocked feature):

```python
@skipIfXpu  # TODO: <short reason>. Tracking: <repo>#<issue_number>
def test_something(self):
    ...
```

Requirements:
- Must have a tracking issue number before adding the skip.
- Record in the orchestrator's summary report.
- This is a temporary measure — the tracking issue owns the real fix.

## Step 3: Stage changes

```bash
git add <your_files>
git diff --cached --stat   # verify only intended files are staged
```

Never stage `third_party/xpu.txt` or unrelated files.

## Output

Return to the orchestrator:

```
### Implement Result
- **What I changed:** <bullet list of files and what changed in each>
- **Why:** <one sentence connecting each change to the triage root cause>
- **CUDA alignment:** <how this aligns with CUDA behavior, or "N/A">
- **Skip added:** <yes (tracking: #N) / no>
- **Ready for verify:** yes
```

## HARD RULES
- NEVER add skip decorators when `allow_skip=false`.
- NEVER modify files outside your repo scope.
- NEVER modify unrelated files.
- NEVER cherry-pick upstream commits. Rebase instead.
- NEVER submit a torch-xpu-ops PR for a pytorch-core bug.

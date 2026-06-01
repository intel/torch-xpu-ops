---
name: fix-issues-identified-by-comments
description: Fix code issues identified by PR review comments (e.g., Copilot AI reviews). Use when fixing XPU compatibility, CUDA-to-XPU migration, or addressing specific review feedback. Performs deep semantic analysis of comment issues before applying fixes.
---

# Fix Issues Identified by Comments

## Description

Fix issues identified by PR review comments such as Copilot AI code reviews. This skill performs deep semantic analysis of comment feedback, identifies the root cause of issues, and applies targeted fixes while following PyTorch conventions.

Use when:
- Addressing Copilot AI review comments on PRs
- Fixing XPU compatibility issues found in code review
- Migrating CUDA-specific code to support Intel XPU
- Applying fixes based on specific PR feedback

## Skill Integration

**This skill follows agent-guidelines AND extends it with specific constraints.**

Always apply agent-guidelines rules including:
- Mandatory post-write commit protocol (ask user before committing)
- Deep semantic analysis instead of pattern matching
- Atomic commits for each fix
- All constraints defined in agent-guidelines

## Preconditions

Before starting, ensure:

1. **Working directory**: PyTorch repo root `/home/daisydeng/daisy_pytorch`
2. **Git remotes configured**: Both `intel` and `daisyden` remotes for torch-xpu-ops
3. **GitHub authentication**: Password-less tokens configured for push operations
4. **Environment**: Miniforge environment available at `~/miniforge3/bin/activate`
5. **PR context**: URL or PR number provided to understand context of comments

## Used Tools

The following tools are available and MUST be used appropriately:

### Read Tool
- **Purpose**: Load files at specific line numbers for context
- **Usage**: Always read 15-30 lines around each issue location
- **Constraint**: Use targeted reads, not full file scans

### Edit Tool
- **Purpose**: Apply targeted code fixes
- **Usage**: Make surgical changes, preserve surrounding code
- **Constraint**: Include same-line `# noqa:` comments when appropriate

### Glob/Grep Tools
- **Purpose**: Find similar patterns in codebase for consistency
- **Usage**: Use AFTER analysis to FIND similar code, NOT for analysis itself
- **Constraint**: DO NOT use for analysis - only for pattern discovery

### Bash Tool
- **Purpose**: Run git commands, checkouts, commits, and pushes
- **Functions**:
  - `git status -s`: Check modified files
  - `git diff`: Show changes before commit
  - `git add/commit`: Apply atomic commits
  - `git push`: Push to correct branch

### Task Tool (explore subagent)
- **Purpose**: Deep semantic analysis of complex code patterns
- **Usage**: When issue requires understanding cross-file dependencies
- **Constraint**: Used for ANALYSIS only, not for code generation

### todowrite Tool
- **Purpose**: Track multi-step fix progress
- **Usage**: List all issues from comment, track completion

### question Tool
- **Purpose**: Ask user approval before commits
- **Usage**: MANDATORY before each commit per agent-guidelines
- **Constraint**: Must ask, wait for response, then act

## Working Workflow

### Step 1: Setup Environment

```bash
# Activate correct environment
source ~/miniforge3/bin/activate ~/miniforge3/envs/pytorch_opencode_env

# Navigate to torch-xpu-ops
cd third_party/torch-xpu-ops
```

### Step 2: Identify Target Branch (CRITICAL — use exact PR head ref)

**Never guess the branch name.** Always query `gh pr view` for the PR's
actual `headRefName` and `headRepository`, and remember the exact string
(it may contain slashes, e.g. `daisyden/dynamo_xpu`). You will need this
exact name for both the fetch and the push.

```bash
PR_NUM=3383
PR_REPO=intel/torch-xpu-ops

# Get the head branch metadata. headRefName is the literal branch name on
# the head repo; it may contain slashes.
gh pr view "$PR_NUM" --repo "$PR_REPO" \
    --json headRefName,headRepositoryOwner,headRepository,commits \
    --jq '{branch: .headRefName, owner: .headRepositoryOwner.login, repo: .headRepository.name, lastCommit: .commits[-1].oid}'

# Example output:
# {"branch":"daisyden/dynamo_xpu","owner":"daisyden","repo":"torch-xpu-ops","lastCommit":"da9f..."}

PR_BRANCH="daisyden/dynamo_xpu"   # use the exact headRefName above
PR_REMOTE="daisyden"               # the remote that points to head repo

# Fetch using fully-qualified refspec so a slash-containing branch name
# is not misinterpreted. Map it to a unique local tracking name.
git fetch "$PR_REMOTE" "refs/heads/${PR_BRANCH}:refs/remotes/${PR_REMOTE}/pr-${PR_NUM}-head"

# Verify your local tracking ref tip matches lastCommit from gh pr view.
git log -1 --oneline "${PR_REMOTE}/pr-${PR_NUM}-head"

# Create/reset a local working branch from the tracking ref.
git checkout -B "pr-${PR_NUM}-fix" "${PR_REMOTE}/pr-${PR_NUM}-head"
```

**Common pitfall**: `git fetch daisyden dynamo_xpu` (short form) will
match any branch with a trailing component `dynamo_xpu` — it can pick
up the wrong branch silently. Always use the fully-qualified refspec
`refs/heads/<exact-name>:...` when the head ref contains slashes.

**Sanity-check before continuing**: confirm `git rev-parse HEAD`
matches the `lastCommit` reported by `gh pr view`. If they differ, you
fetched the wrong branch — stop and re-check the head ref.

### Step 3: Analyze Comment Issues

For each issue from the comment:

1. **Extract issue details**:
   - File path and line numbers from comment
   - Description of what needs to be fixed
   - Suggested fix (if provided)

2. **Read context** (MANDATORY):
   ```bash
   # Read 15-30 lines around each issue location
   read file.py offset=N-10 limit=40
   ```

3. **Understand semantic purpose**:
   - What is the code trying to do?
   - Why was it written this way?
   - What would break if changed incorrectly?

4. **Check similar patterns**:
   ```bash
   # Find similar code for consistency reference
   grep -n "pattern_type" related/files/*.py
   ```

### Step 4: Apply Deep Analysis

DO NOT use pattern matching or regex substitution for analysis. Instead:

#### Analysis Protocol

```
FOR EACH ISSUE:

1. INTENT RECOGNITION
   - Read surrounding code (15-30 lines)
   - Understand what function/module does
   - Identify requirements (triton, CUDA, XPU, etc.)

2. ERROR/ISSUE CLASSIFICATION
   - Is this a XPU vs CUDA compatibility issue?
   - Is this a hard-coded backend assumption?
   - Is this a missing import/decorator?
   - Is this an expected test that needs modification?

3. SOLUTION VALIDATION
   - Design minimal fix that maintains behavior
   - Check if similar fixes exist in codebase
   - Verify fix works for both CUDA and XPU if applicable

4. APPLY TARGETED FIX
   - Use Edit tool with exact string match
   - Include noqa comments on same line when appropriate
```

### Step 5: Validate Fix

After applying each fix:

```bash
# Check consistency with similar patterns
grep -n "new_pattern" *.py | head -10

# Verify no regressions in related tests
git diff path/to/fixed.file

# Check for new lint issues
lintrunner -a path/to/fixed.file
```

### Step 6: Track Progress

```bash
# Track each fix
todowrite --add "Fix issue #1: XPU device context" --status pending
todowrite --add "Fix issue #2: CUDA Event/Stream" --status completed
```

### Step 7: User Approval Before Each Commit (MANDATORY)

Following agent-guidelines, you MUST ask for approval:

```
**Issue fixed**: [brief description]
**File modified**: [file path]
**Line changed**: [line numbers]
**Change summary**: [what was modified]

Should I commit this fix?
```

Wait for user response before committing.

### Step 8: Push Changes (CRITICAL — push to the PR's exact head ref)

After user approves, push **to the same `headRefName`** captured in
Step 2, using a fully-qualified destination refspec. Do NOT use the
short form (`git push remote local-name`) — if your local branch name
differs from the PR head ref (and it usually will when the PR head
contains slashes), the short form silently creates a new branch on the
remote instead of updating the PR.

```bash
# Push using fully-qualified refspec to the EXACT headRefName.
git push "$PR_REMOTE" "HEAD:refs/heads/${PR_BRANCH}"

# Example for PR #3383 whose headRefName is "daisyden/dynamo_xpu":
# git push daisyden HEAD:refs/heads/daisyden/dynamo_xpu
```

**Mandatory post-push verification** — confirm the PR now points at
your new commit. If it does not, you pushed to the wrong branch.

```bash
gh pr view "$PR_NUM" --repo "$PR_REPO" --json commits \
    --jq '.commits[-1] | {oid, messageHeadline}'

# The oid must equal `git rev-parse HEAD` locally.
```

If the PR's last commit does not match your local HEAD:

1. Re-run `gh pr view ... --json headRefName` to recover the correct ref.
2. Inspect `git ls-remote "$PR_REMOTE"` to see which branch you actually
   updated.
3. Push again with the corrected fully-qualified refspec.
4. Optionally delete the stray branch you created accidentally:
   `git push "$PR_REMOTE" --delete <stray-branch-name>` (only after
   confirming with the user).

## Common XPU Compatibility Patterns

When fixing CUDA-to-XPU issues, use these patterns:

### Pattern 1: Import Changes
```python
# BEFORE (CUDA-only)
from torch.testing._internal.triton_utils import requires_cuda_and_triton
from torch.testing._internal.common_cuda import requires_cuda

# AFTER (XPU-compatible)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton
```

### Pattern 2: Hard-coded Device Strings
```python
# BEFORE
torch.Stream(device="cuda")
x = torch.ones(2, 2, device="cuda")

# AFTER
torch.Stream(device=GPU_TYPE)
x = torch.ones(2, 2, device=GPU_TYPE)
```

### Pattern 3: Decorator Changes
```python
# BEFORE
@requires_cuda
@requires_cuda_and_triton

# AFTER
@requires_gpu_and_triton
```

### Pattern 4: Device Module Selection
```python
# BEFORE
with torch.cuda.stream(s):
    x = torch.sin(x)

# AFTER
device_module = torch.get_device_module(GPU_TYPE)
with device_module.stream(s):
    x = torch.sin(x)
```

### Pattern 5: Backend Capability Checks
```python
# BEFORE
if not torch.cuda.is_bf16_supported():
    raise unittest.SkipTest("requires bf16")

# AFTER
if GPU_TYPE == "cuda":
    bf16_supported = torch.cuda.is_bf16_supported()
elif GPU_TYPE == "xpu":
    bf16_supported = getattr(torch.xpu, "is_bf16_supported", lambda: False)()
else:
    bf16_supported = False
if not bf16_supported:
    raise unittest.SkipTest("requires bf16")
```

### Pattern 6: Event/Class Instantiation
```python
# BEFORE
event = torch.cuda.Event() if torch.cuda.is_available() else torch.xpu.Event()

# AFTER
Event = torch.cuda.Event if torch.cuda.is_available() else torch.xpu.Event
event = Event()
```

## Constraints

1. **PR branch first**: Always get correct PR branch before making changes
2. **Use exact headRefName**: Capture `headRefName` via `gh pr view` and
   use it verbatim for both `git fetch refs/heads/<name>:...` and
   `git push <remote> HEAD:refs/heads/<name>`. Never use the short
   `git push remote branch` form for PR pushes — it can create a new
   branch when the local name differs from the head ref (especially
   when the head ref contains slashes).
3. **Verify push landed on the PR**: After pushing, run `gh pr view
   --json commits` and confirm the PR's last commit oid equals your
   local HEAD. If not, fix the push before reporting success.
4. **Context required**: ALWAYS read 15-30 lines before analysis
5. **No pattern matching for analysis**: Use Read tool for semantic understanding
6. **Deep analysis over shortcuts**: Understand code intent before fixing
7. **Minimal fixes**: Only change what is necessary
8. **Preserve test logic**: Fixes should not change test behavior
9. **Ask before commit**: MANDATORY per agent-guidelines
10. **Atomic commits**: One issue per commit
11. **No force push to main**: Only to feature/PR branches
12. **Verify consistency**: Check similar patterns before applying fix

## Validation Checklist

Before considering a fix complete:

- [ ] PR `headRefName` captured via `gh pr view` and used verbatim for
      fetch and push
- [ ] Local working branch is based on the exact `headRefName` tip
      (verified `git rev-parse HEAD` against gh's `lastCommit` before
      starting work)
- [ ] Issue analyzed with context read (15-30 lines)
- [ ] Root cause identified
- [ ] Solution validated against similar patterns
- [ ] Fix applied with targeted Edit
- [ ] Changes verified with git diff
- [ ] User approval obtained (MANDATORY)
- [ ] Committed with descriptive message
- [ ] Pushed using `HEAD:refs/heads/<headRefName>` (fully-qualified)
- [ ] Post-push: `gh pr view --json commits` shows the new commit as
      the PR's latest commit
- [ ] Progress tracked in todowrite

## Example Complete Workflow

### Input
PR #3383 Copilot AI comment:
> `test_cuda_device` runs when either CUDA or XPU is available, but the implementation unconditionally uses `torch.cuda.device(...)`. This will raise on XPU-only test runs.

### Analysis
1. Read context at line 673 of test_ctx_manager_xpu.py
2. Understand: Function sets device context using CUDA API
3. Classify: XPU compatibility issue
4. Solution: Use torch.get_device_module(GPU_TYPE).device(...)

### Fix Applied
```python
# BEFORE
with torch.cuda.device(x.device.index - 1):
    x = torch.sin(x + 1)

# AFTER  
with torch.get_device_module(GPU_TYPE).device(x.device.index - 1):
    x = torch.sin(x + 1)
```

### Validation
```bash
git diff test/xpu/dynamo/test_ctx_manager_xpu.py | grep -A2 "test_cuda_device"
```

### Approval Request
```
**Issue fixed**: test_cuda_device XPU compatibility
**File modified**: test/xpu/dynamo/test_ctx_manager_xpu.py  
**Line changed**: 673
**Change summary**: Changed torch.cuda.device() to torch.get_device_module(GPU_TYPE).device()

Should I commit this fix?
```

### Push
```bash
# headRefName for PR #3383 is "daisyden/dynamo_xpu" (literal, contains slash)
git push daisyden HEAD:refs/heads/daisyden/dynamo_xpu

# Then verify the PR picked it up:
gh pr view 3383 --repo intel/torch-xpu-ops --json commits \
    --jq '.commits[-1].oid'
# must equal: git rev-parse HEAD
```
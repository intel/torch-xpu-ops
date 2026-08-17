---
name: xpu-nightly-ci-fix
description: >
  Orchestrator for fixing XPU nightly CI failures in batch. Takes a CI failure
  report, reproduces each failure, triages root cause, applies fixes, and
  generates a summary report. Uses fix/ leaf skills for all core logic.
---

# Nightly CI Fix — Orchestrator

Processes a batch of nightly CI failures. Each failure runs through the same
`fix/reproduce` → `fix/root-cause` → `fix/implement` → `fix/verify` pipeline
independently. All detailed fix logic lives in the `fix/` leaf skills — this
skill owns the batch scheduling, branch strategy, commit format, and progress
tracking.

## Execution modes

- **Interactive (default):** human present. Report progress conversationally.
  Ask when blocked.
- **Pipeline:** automated. No human to ask. Write progress to
  `agent_space_xpu/runs/<report_date>/` and stop on blockers.

## Progress tracking

All runs write to `agent_space_xpu/runs/<report_date>/`:

```
runs/<report_date>/
  progress.md          # per-failure status table, updated after each stage
  logs/<failure_id>.md # per-failure stage log, append-only
  summary.md           # generated at the end
```

**`progress.md`:**
```markdown
# Progress — <report_date>

ci_commit: <hash>

| failure_id | status | stage | commit |
|---|---|---|---|
| test_ops_xpu::TestFooXPU::test_bar | done | verify | abc1234 |
| test_nn_xpu::TestNNXPU::test_conv3d | in_progress | implement | — |
| test_sparse_xpu::TestSparseXPU::test_mm | pending | — | — |
```

`status` values: `pending` / `in_progress` / `done` / `needs_human`.

**`logs/<failure_id>.md`** — append one section per stage as it completes:
```markdown
## Reproduce
result: REPRODUCED
command: pytest ...

## Triage
domain: xpu-kernel
root_cause: ...
fix_strategy: ...

## Implement (attempt 1)
files_changed: ...

## Verify (attempt 1)
result: FAILED
output: ...

## Verify (attempt 2)
result: PASSED
commit: abc1234
```

**Recovery:** on restart, read `progress.md`. Skip `done`/`needs_human` rows.
Resume `in_progress` from its current `stage`. Start `pending` from scratch.

> **Before starting:** Read the `## Working Principles` section of `AGENTS.md`.
> State which principles apply to this task before proceeding.

## Inputs

- A nightly failure report: email, test list, or log snippet containing:
  - Failing test names (file, class, method)
  - PyTorch commit hash (`ci_commit`)
  - Report date

## Required: Initialize Todo List

Immediately after parsing the failure report, create a TodoWrite list:

```
- [ ] Step 0: Ensure PyTorch checkout
- [ ] Step 1: Parse report — extract ci_commit, date, failing test list
- [ ] Step 2: Reproduce <failure_1> with `fix/reproduce` (ci_commit, test command)
- [ ] Step 3: Reproduce <failure_2> with `fix/reproduce` (ci_commit, test command)
      ... (one entry per failure)
- [ ] Step 4-6: Fix <failure_1> (checkout fix branch -> triage → load domain → implement → verify → commit)
- [ ] Step 4-6: Fix <failure_2> (checkout fix branch -> triage → load domain → implement → verify → commit)
      ... (one entry per failure)
- [ ] Step 7: Generate summary report
```

Mark a fix item `completed` only after the test actually passes and is
committed. Never skip to Step 7.

## Step 0: Preflight — gh auth + PyTorch checkout

**Shell helpers.** The recipes below use `abort` and `log_warn`
helpers. Define them once at the top of your shell (or source a
shared file):

```bash
abort()    { echo "ABORT: $*" >&2; exit 1; }
log_warn() { echo "WARN: $*"  >&2; }
```

**gh auth preflight** — this skill will `gh issue create` in
`fix/skip-management` (via `fix/implement` with `allow_skip=true`),
so a write-capable token is mandatory. Fail early if not authenticated,
so partial runs don't strand fixes mid-loop:

```bash
gh auth status 2>&1 | grep -q "Logged in to github.com" \
  || abort "gh not authenticated; run: gh auth login"

scopes=$(gh auth status -t 2>&1 | grep -oE "'[a-z:_-]+'" | tr -d "'")
echo "$scopes" | grep -qE '^(repo|write:issues)$' \
  || abort "nightly-ci-fix requires 'repo' or 'write:issues' scope; got: $scopes"
```

**PyTorch checkout** — check `agent_space_xpu/pytorch/`:
```bash
ls agent_space_xpu/pytorch/ 2>/dev/null || echo "NOT FOUND"
```

If not found, clone:
```bash
git clone --filter=blob:none https://github.com/pytorch/pytorch.git \
  agent_space_xpu/pytorch
git -C agent_space_xpu/pytorch submodule update --init --recursive
```

If found, fetch latest:
```bash
git -C agent_space_xpu/pytorch fetch origin
```

## Step 1: Parse the failure report

Extract:
- `report_date` (e.g. `20260608`)
- `ci_commit` — pytorch commit hash. `fix/reproduce` reproduces on
  `origin/main` first and only falls back to `ci_commit` if trunk
  itself fails to build; this field is optional but should be
  extracted when the report supplies it.
- Failing test list: group by test file/module


## Step 2-3: Reproduce each failure

For each failure, call `fix/reproduce` with:
- `reproducer_command` — the CI test command
- `ci_commit` — from Step 1 (used only as build fallback by reproduce)
- `pytorch_dir` — `agent_space_xpu/pytorch/`

Interpret output per failure:

| Output | Action |
|--------|--------|
| `REPRODUCED` | Continue to Step 4 for this failure. Record `base` from the output. `fix/reproduce` only emits `base=<ci_commit_sha>` when the `stage=source_build` fallback path was used; for `stage=nightly` REPRODUCED (or when `base` is absent for any other reason), default `base=origin/main`. |
| `NOT_REPRODUCED` | Mark in summary: "already fixed on trunk"; skip to next failure |
| `NO_REPRODUCER` | Mark in summary: "no reproducer command available"; skip to next failure |
| `CANNOT_VERIFY` | Mark in summary: "cannot verify (+ blocker)"; skip to next failure |

## Step 4: Triage each reproduced failure

Call `fix/root-cause` with the failure description and error log.
Branch creation is deferred to Step 4.6 (below), because it depends
on `target_repo` from triage — a torch-xpu-ops fix needs its branch
inside `third_party/torch-xpu-ops/`, not in the pytorch checkout.

| Verdict | Action |
|---------|--------|
| `IMPLEMENTING` | Continue to Step 4.5 (load domain skill), then Step 4.6 (create branch), then Step 5 |
| `NEEDS_HUMAN` | Mark in summary: "needs human (+ reason)"; skip to next failure |

## Step 4.5: Load the domain skill (via registry)

Consult `.claude/skills/fix/domains/README.md` — the domain registry —
before loading anything:

1. Read the `domain` field from the triage output (Step 4).
2. Look it up in the registry's JSON list. If not present → **mark the
   failure `NEEDS_HUMAN`** in the summary, reason: `"fix/root-cause emitted
   domain not in fix/domains/README.md: <domain>"`; skip to next
   failure.
3. Check the registry row: `skill_path` directory must exist. If not →
   **`NEEDS_HUMAN`**, reason: `"registry lists <domain> but
   <skill_path> is missing"`.
4. Compare the row's `target_repo` with triage's `target_repo` output.
   Mismatch → **`NEEDS_HUMAN`**, reason: `"triage target_repo=<x>
   conflicts with registry <y> for domain <domain>"`.
5. Only then, use the skill tool to load the `skill_path`.

Do NOT fall back to "proceed without a domain skill" — that silent
no-op is the bug the registry exists to prevent. Mirrors
`issue-handler` Stage 3.5.

## Step 4.6: Create the fix branch in the target checkout

Now that `target_repo` is known, create the per-failure branch **in
the checkout that will actually hold the diff**:

```bash
# Pick the checkout by target_repo.
case "$target_repo" in
  pytorch)        target_repo_dir=agent_space_xpu/pytorch ;;
  torch-xpu-ops)  target_repo_dir=agent_space_xpu/pytorch/third_party/torch-xpu-ops ;;
  *)              abort "unknown target_repo='$target_repo'" ;;
esac

# One branch per failure; name it fix-<report_date>-<short_test_name>
# where short_test_name is the last component of the test method
# name (e.g. test_add from TestBinaryUfuncsXPU::test_add_xpu).
git -C "$target_repo_dir" checkout -b fix-<report_date>-<short_test_name> <base>
# e.g. git -C agent_space_xpu/pytorch checkout -b fix-20260608-test_add origin/main

# For torch-xpu-ops fixes, also apply the dev-override pin so the
# pytorch build sees this branch's HEAD (Critical rules "Fix in
# torch-xpu-ops?" below). Do NOT commit third_party/xpu.txt.
if [ "$target_repo" = "torch-xpu-ops" ]; then
    git -C "$target_repo_dir" rev-parse HEAD \
      > agent_space_xpu/pytorch/third_party/xpu.txt
fi
```

Pass `target_repo_dir` (and `target_repo`) into Step 5 and Step 6.

## Step 5: Implement each fix

Call `fix/implement` with:
- `triage_result` from Step 4
- `pytorch_dir` — `agent_space_xpu/pytorch/`
- `target_repo_dir` — from Step 4.6
- `allow_skip=true` — nightly-ci-fix may add `@skipIfXpu` with tracking issue
  when implementation is out of scope for a nightly fix
- `patch_proposal_mode=false` — nightly-ci-fix always commits its own
  fixes; patch-proposal is issue-handler's mode.

`fix/implement` returns the machine-readable block described in its
"Output" section. Step 6 reads `changed_files`, `skip_added`, and
`tracking_issue` from it.

## Step 6: Verify and commit each fix

If `fix/implement` returned `ready_for_verify: false`, do NOT call
`fix/verify`. Mark in summary: "needs human (implement bailed after
Step 3.5 rejected the diff)"; skip to next failure.

Otherwise call `fix/verify` with:
- `refined_command` from Step 3 (`fix/reproduce` output)
- `pytorch_dir` — `agent_space_xpu/pytorch/`
- `target_repo_dir` — from Step 4.6
- `changed_files` from Step 5
- `run_before_after_diff=true`
- `run_lint=true`

| Output | Action |
|--------|--------|
| `PASSED` | Commit (one fix per commit); mark in summary: "fixed (commit: <hash>)" |
| `FAILED` | Re-triage with failure context, then re-implement (see fix loop below) |
| `CANNOT_VERIFY` | Mark in summary: "cannot verify after fix"; skip to next failure |

**Fix loop** (max 3 attempts total, counting the first verify):

```
attempt N (starting at 1 after the first fix/verify returns FAILED):
  1. Call fix/root-cause again, passing:
     - original failure description
     - fix/verify failure_output from this attempt
     - fix/verify suggestion from this attempt
     - prior fix strategy (so triage knows what was already tried)
     - prior target_repo (so triage can flip decisions consciously)
  1a. If the new target_repo differs from the previous attempt's,
      the previous attempt's fix branch is orphaned in the wrong
      checkout. Do NOT continue the loop with mixed state — the
      pipeline cannot silently migrate a staged diff from
      third_party/torch-xpu-ops to pytorch (or vice versa) and
      preserve reviewability. Exit the loop, mark in summary:
      "needs human (triage flipped target_repo between attempts:
      <prev> -> <new>; both diffs preserved on their branches for
      manual inspection)". Do NOT reset either branch — leave both
      staged diffs on disk for human review.
  2. Otherwise, call fix/implement with the new triage_result and
     the SAME target_repo_dir as attempt N-1. fix/implement's Step 0
     loop-back rule handles the staged-from-previous-attempt state.
  3. Call fix/verify again.
      - PASSED → commit, exit loop.
      - FAILED and attempt < 3 → increment attempt, repeat from step 1.
      - FAILED and attempt == 3 → exit loop.
      - CANNOT_VERIFY → exit loop.
```

If loop exits without `PASSED`, mark in summary: "needs human (fix loop exhausted after 3 attempts)"; record each attempt's `failure_output` and `suggestion` in the summary under a "Fix Attempts" subsection; skip to next failure.

Commit after each verified fix. The orchestrator (not `fix/implement`)
owns the commit; use this template for the commit message:

```
[xpu][fix] <short description>

## Motivation
<why this fix is needed>

## Solution
<what was changed and CUDA alignment if applicable>

## Test plan
<how it was verified>

Note: This commit was authored with AI assistance.
```

```bash
# fix/implement leaves changes already staged; commit directly.
# The commit MUST happen in the checkout that holds the staged diff,
# which depends on triage's target_repo:
#   target_repo == "pytorch"        -> agent_space_xpu/pytorch/
#   target_repo == "torch-xpu-ops"  -> agent_space_xpu/pytorch/third_party/torch-xpu-ops/
# A pytorch-only commit against a torch-xpu-ops fix would either be
# empty or capture only a submodule-pointer bump, losing the real
# kernel diff. Always commit in the repo that owns the changes.
case "$target_repo" in
  pytorch)
    commit_dir=agent_space_xpu/pytorch
    ;;
  torch-xpu-ops)
    commit_dir=agent_space_xpu/pytorch/third_party/torch-xpu-ops
    ;;
  *)
    abort "unknown target_repo='$target_repo'"
    ;;
esac
git -C "$commit_dir" commit -m "<commit_message>"

# After committing in torch-xpu-ops, do NOT commit the resulting
# third_party/xpu.txt pointer change in pytorch — the dev-override
# pin is local-only per AGENTS.md.
```

If the fix was a skip (`fix/implement` output has `skip_added: true`):
- Record the `tracking_issue` URL in the summary under "Needs Human" (skip added).
- The tracking issue already contains the root cause and fix strategy from triage.

Each fix is one commit. Do not batch multiple fixes into one commit.

### Reset between failures

After Step 6 completes (or after a NEEDS_HUMAN / CANNOT_VERIFY exit),
reset shared state before starting the next failure. Otherwise the
next failure inherits this failure's xpu.txt override, stale build
artifacts, and per-failure fix branch — the "before" run of its
verify loop then mixes the two fixes.

```bash
# Reset the pytorch checkout to <base>. This drops the per-failure
# fix branch pointer from HEAD (the branch itself remains for audit).
git -C agent_space_xpu/pytorch checkout <base>
# Restore third_party/xpu.txt to its pinned commit so the next
# rebuild starts from a clean pin.
git -C agent_space_xpu/pytorch checkout -- third_party/xpu.txt
# Drop stale build outputs; nested submodules preserved automatically.
git -C agent_space_xpu/pytorch clean -fdx

# Reset the torch-xpu-ops override checkout to its own base, if this
# failure targeted torch-xpu-ops.
if [ "$target_repo" = "torch-xpu-ops" ] && \
   [ -d agent_space_xpu/pytorch/third_party/torch-xpu-ops/.git ]; then
    git -C agent_space_xpu/pytorch/third_party/torch-xpu-ops \
        checkout <xpu_ops_base>
    git -C agent_space_xpu/pytorch/third_party/torch-xpu-ops clean -fdx
fi
```

Where `<base>` is the pytorch base commit (`origin/main` normally, or
the CI commit sha if reproduce fell back) and `<xpu_ops_base>` is the
torch-xpu-ops working branch's base. Both were established at the
start of this run and remain constant across all failures.

## Step 7: Generate summary report

Write to `agent_space_xpu/runs/<report_date>/summary.md` (the path
declared in "Progress tracking" above):

```markdown
# Nightly CI Fix Summary — <report_date>

PyTorch commit: <ci_commit>
Total failures: N | Fixed: X | Skipped (already fixed): Y | Needs human: Z | Cannot verify: W

## Status at a Glance

| Failure | Status | Commit | Notes |
|---------|--------|--------|-------|
| test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu | Fixed | abc1234 | tolerance: 1e-5→1e-4 |
| test_nn_xpu.py::TestNNXPU::test_conv3d_groups | Needs human | — | missing kernel, tracking: #1234 |
| test_sparse_xpu.py::TestSparseXPU::test_mm | Already fixed | — | passes on nightly |

## Fixed

### test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu
- Root cause: tolerance too tight
- Fix: increased atol 1e-5 → 1e-4 to match CUDA
- Commit: abc1234
- AR: submit PR to pytorch/pytorch

## Needs Human

### test_nn_xpu.py::TestNNXPU::test_conv3d_groups
- Root cause: missing XPU kernel for grouped conv3d
- Decision: skip added with tracking issue intel/torch-xpu-ops#1234
- AR: prioritize kernel implementation

### test_foo_xpu.py::TestFooXPU::test_bar (fix loop exhausted)
- Fix Attempts:
  - Attempt 1: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
  - Attempt 2: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
  - Attempt 3: failure: <output> | suggestion: <suggestion> | fix tried: <what implement changed>
- AR: manual investigation required

## Already Fixed / Cannot Verify

...
```

## Critical rules

- **Never cherry-pick** upstream fixes. Rebase (`git rebase origin/main`) instead.
- **Always rebuild after rebase or branch switch** before running tests.
- **Fix in torch-xpu-ops?** Use the dev override from `AGENTS.md` "Commit Pin
  & Development Override": clone your torch-xpu-ops branch into
  `agent_space_xpu/pytorch/third_party/torch-xpu-ops/`, then update the pin
  so CMake's checkout becomes a no-op:
  ```bash
  cd agent_space_xpu/pytorch/third_party/torch-xpu-ops
  git checkout <your-fix-branch>
  git rev-parse HEAD > ../xpu.txt
  ```
  Do NOT commit `xpu.txt`. Then rebuild from the pytorch checkout root:
  ```bash
  cd <repo_root>/agent_space_xpu/pytorch
  pip install -e . -v --no-build-isolation
  ```
  For pure C++ changes that do not touch codegen (no new ops, no dispatch
  registration changes), `ninja -C agent_space_xpu/pytorch/build` can be used
  to speed up incremental C++ compilation, but `pip install -e .` is always
  safe and required whenever in doubt.
- Each failure is independent — one failure's `CANNOT_VERIFY` does not block
  others.
- One fix per commit.

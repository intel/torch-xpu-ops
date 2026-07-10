---
name: fix/implement
description: >
  Implement a fix for a triaged failure. Takes triage output and produces a
  verified code change. Used by both issue-handler and nightly-ci-fix
  orchestrators via the allow_skip parameter.
---

# Implement — Apply the Fix

Takes triage output and makes the code change. Does not run tests (that is
`fix/verify`'s job) and does not open PRs or commit (that is the
orchestrator's job).

## Inputs

- `triage_result` — JSON output from `fix/triage` (root_cause, fix_strategy,
  target_repo, domain).
- `pytorch_dir` — path to local PyTorch checkout.
- `allow_skip` — controls skip decorator strategy:
  - `false` (**issue-handler**): never add skip decorators; must unskip and
    really fix.
  - `true` (**nightly-ci-fix**): may add a skip with tracking issue when the
    fix requires significant implementation work beyond the current scope.
    Stale skips must still be removed.
- `patch_proposal_mode` (optional, default `false`) — set by the
  orchestrator when the fix must land in a repo the current run is not
  allowed to open a PR against. See "Patch-proposal mode" below.
- `commit_message_template` (optional) — orchestrator-provided format. If
  absent, use a concise imperative message.

## Step 0: Verify environment

```bash
basename $(git rev-parse --show-toplevel)  # confirm which repo you're in
git status                                  # confirm clean worktree
```

## Step 1: Read the triage output

Read `triage_result` carefully before touching any file. Understand:
- Which files/functions need to change
- Why the failure occurs (if applicable, why CUDA works but XPU fails)
- Whether the fix is in pytorch or the backend repo

If the issue is not yet triaged, run `fix/triage` first.

## Step 2: Implement the fix

### Key rules

- **Minimal changes** — fix only what's broken; every changed line must trace
  to the triage output.
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk,
  rebase (`git rebase origin/main`) instead.
- **Stay in your repo** — see domain skill (loaded by orchestrator) for path
  conventions.
- **Never modify unrelated files.**

### Fix strategies by category

See `fix/triage` Step 1 for domain routing. Common strategies:

- **Tolerance:** match upstream `atol`/`rtol` values exactly.
- **Regression:** find the guilty commit (`git log --oneline -20 -- <file>`),
  apply a fix aligned with upstream intent; document any divergence in comments.
- **Newly added test:** enable backend support. If `allow_skip=false` and
  support is genuinely missing, report `NEEDS_HUMAN` — do not add a skip. If
  `allow_skip=true`, load `fix/pytorch-skip` to add a skip with tracking issue.
- **Unknown root cause:** compare with upstream backend behavior.

### Skip operations

For removing stale skip decorators or adding new skips, load `fix/pytorch-skip`.

When **adding** a new skip (`allow_skip=true`), load `fix/pytorch-skip` and
follow its "Add a new skip" procedure. It handles filing the tracking issue
and returning the issue URL. Include that URL in the implement output
(`tracking_issue` field).

## Step 3: Stage changes

```bash
git add <your_files>
git diff --cached --stat   # verify only intended files are staged
```

Never stage `third_party/xpu.txt` or unrelated files.

## Patch-proposal mode

When `patch_proposal_mode=true`, the fix must land in a repo the current
run is not allowed to open a PR against (usually `pytorch` when the issue
is on `torch-xpu-ops`, or vice versa). In this mode:

- Apply the fix in the `target_repo`'s local checkout exactly as normal
  (Step 1 through Step 3).
- **Do NOT commit.** Leave the change staged only. The orchestrator's
  Stage 6 will read it back via `git -C <target_repo_dir> diff --cached`
  and post the diff as a comment on the issue.
- Do NOT branch, tag, or push anything.
- Do NOT invoke any PR-creation skill downstream.
- The default "leave staged but uncommitted" contract already matches this
  requirement; `patch_proposal_mode` just reinforces it and forbids any
  future PR handoff.

## Output

Return to the orchestrator:

```
### Implement Result
- **What I changed:** <bullet list of files and what changed in each>
- **Why:** <one sentence connecting each change to the triage root cause>
- **Skip added:** <yes (tracking: intel/torch-xpu-ops#N, url: <url>) / no>
- **Ready for verify:** yes
```

**Contract:** changes are left staged (`git add`) but NOT committed. The
orchestrator commits only after `fix/verify` returns `PASSED`. `fix/verify`
relies on `git stash` to record a before-state, which requires uncommitted
changes to be present when verify is called.

## HARD RULES
- NEVER add skip decorators when `allow_skip=false`.
- NEVER modify files outside your repo scope.
- NEVER modify unrelated files.
- NEVER cherry-pick upstream commits. Rebase instead.
- NEVER submit a torch-xpu-ops PR for a pytorch-core bug.
- NEVER commit when `patch_proposal_mode=true`.

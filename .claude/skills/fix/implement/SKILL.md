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

- `triage_result` — JSON output from `fix/root-cause` (root_cause, fix_strategy,
  target_repo, domain).
- `pytorch_dir` — path to local PyTorch checkout.
- `target_repo_dir` — path to the checkout that will be edited. Derived
  from `pytorch_dir` and `triage_result.target_repo`:
  - `target_repo == "pytorch"` → `target_repo_dir = pytorch_dir`.
  - `target_repo == "torch-xpu-ops"` →
    `target_repo_dir = <pytorch_dir>/third_party/torch-xpu-ops`
    (per AGENTS.md "Commit Pin & Development Override"; the caller
    is expected to have cloned the working branch there ahead of
    time).
  All git and edit operations in this skill run against
  `target_repo_dir`, never against `pytorch_dir` when the two differ.
- `allow_skip` — controls skip decorator strategy:
  - `false` (**issue-handler**): never add skip decorators; must unskip and
    really fix.
  - `true` (**nightly-ci-fix**): may add a skip with tracking issue when the
    fix requires significant implementation work beyond the current scope.
    Stale skips must still be removed.
- `patch_proposal_mode` (optional, default `false`) — set by the
  orchestrator when the fix must land in a repo the current run is not
  allowed to open a PR against. See "Patch-proposal mode" below.

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

If the issue is not yet triaged, run `fix/root-cause` first.

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

See `fix/root-cause` Step 1 for domain routing. Common strategies:

- **Tolerance:** match upstream `atol`/`rtol` values exactly.
- **Regression:** find the guilty commit (`git log --oneline -20 -- <file>`),
  apply a fix aligned with upstream intent; document any divergence in comments.
- **Newly added test:** enable backend support. If `allow_skip=false` and
  support is genuinely missing, report `NEEDS_HUMAN` — do not add a skip. If
  `allow_skip=true`, load `fix/skip-management` and use its "Add a new skip"
  procedure to add a skip with tracking issue.
- **Unknown root cause:** compare with upstream backend behavior.

### Skip operations

XPU skip decorators live in the pytorch test tree regardless of the triaged
domain. For removing stale skip decorators or adding new skips, load
`fix/skip-management`.

When **adding** a new skip (`allow_skip=true`), follow the "Add a new skip"
procedure in `fix/skip-management`. It handles filing the tracking issue and
returning the issue URL. Include that URL in the implement output
(`tracking_issue` field).

## Step 3: Stage changes

```bash
git add <your_files>
git diff --cached --stat   # verify only intended files are staged
```

Never stage unrelated files. In particular, **never stage
`third_party/xpu.txt`** — it is a submodule pin managed by build
tooling, not part of any bug fix (see HARD RULES).

## Step 3.5: Skip-guard review (only when `allow_skip=false`)

Skip this step entirely when `allow_skip=true` (nightly-ci-fix flow).

When `allow_skip=false`, spawn a fresh-context subagent via the `Task`
tool (`subagent_type=general`) to inspect the staged diff for
skip-shaped workarounds before returning to the orchestrator. The
implementer must not review its own diff for this specific bias — the
gatekeeper here is a separate agent with no memory of the reasoning
that produced the diff.

Pass the reviewer:

- The full staged diff: `git -C <repo_dir> diff --cached`.
- The `triage_result` (so the reviewer knows what the root cause is
  supposed to be).
- The value of `allow_skip` (always `false` when this step runs).

Instruct the reviewer to reject the diff (return `REQUEST_CHANGES`) if
any of the following appears anywhere in added lines:

- A new `@skipIfXpu` / `@skipXPU` / `@skipCUDAIf` / `@skipMPS` /
  `@unittest.skip` / `@unittest.skipIf` / `@pytest.mark.skip` /
  `@pytest.mark.skipif` / `@pytest.mark.xfail` /
  `@expectedFailureXPU` / `@expectedFailureCUDA` / `@expectedFailureMPS`
  / `@expectedFailure` decorator on a test.
- A new `DecorateInfo(unittest.skip, ...)` / `DecorateInfo(skipIfXpu, ...)`
  / `DecorateInfo(unittest.expectedFailure, ...)` entry in an
  `OpInfo` / `ModuleInfo` skips/decorators list.
- A new `xfail(...)` / `skip(...)` entry in an
  `instantiate_device_type_tests` skip dict for XPU.
- Bare `raise unittest.SkipTest(...)` / `self.skipTest(...)` inserted
  into a previously-running test to short-circuit it on XPU.
- Loosening `atol` / `rtol` on `assertEqual` (or any tolerance-carrying
  assertion) by more than an order of magnitude, when the diff has no
  quantitative justification for the new value in a comment.
- Hardcoded `set_rng_seed(...)` / `torch.manual_seed(...)` /
  `random.seed(...)` inserted into a previously-random test purely to
  dodge a failure region.
- Deleting or commenting out the failing assertion / the failing test
  function.
- A broad `try / except Exception: pass` (or equivalent) wrapping the
  call that used to fail.

Existing skips being **removed** by the diff are fine — that is a
legitimate root-cause fix in the "stale test expectation" category.
The rule only fires on *added* skip-shaped constructs.

The reviewer returns one of:

- `APPROVE` — no skip-shaped workaround found. Continue to Step 4
  (output).
- `REQUEST_CHANGES` — cite each offending hunk (file + line + which
  rule it matched). The implementer MUST address every citation
  (either replace the workaround with a real root-cause fix or, if no
  root-cause fix is possible within this run's scope, unstage the
  offending change and return `NEEDS_HUMAN` to the orchestrator with
  the reviewer's citations attached).

**Do not loop this step more than once.** If a second run of Step 3.5
still returns `REQUEST_CHANGES`, unstage the offending change and
return `NEEDS_HUMAN` — that is a signal the fix cannot be produced
without a workaround and a human should take it.

This step is intentionally narrower than the orchestrator's Stage 5.5
review. Stage 5.5 checks the entire diff for correctness, minimalism,
and root-cause alignment; Step 3.5 checks only for the specific class
of "hide the failure instead of fixing it" workarounds that
`allow_skip=false` is meant to forbid. Both run; they do not replace
each other.

## Patch-proposal mode

When `patch_proposal_mode=true`, the fix must land in a repo the current
run is not allowed to open a PR against (usually `pytorch` when the issue
is on `torch-xpu-ops`, or vice versa). In this mode:

- Apply the fix in the `target_repo`'s local checkout exactly as normal
  (Step 1 through Step 3.5). Step 3.5 still runs when `allow_skip=false`.
- **Do NOT commit.** Leave the change staged only. The orchestrator's
  Stage 6 will read it back via `git -C <target_repo_dir> diff --cached`
  and post the diff as a comment on the issue.
- Do NOT branch, tag, or push anything.
- Do NOT invoke any PR-creation skill downstream.
- The default "leave staged but uncommitted" contract already matches this
  requirement; `patch_proposal_mode` just reinforces it and forbids any
  future PR handoff.

## Output

Return to the orchestrator BOTH a human-readable summary and a
machine-readable block. The machine-readable block is authoritative;
`fix/verify` and both orchestrators read specific fields from it.

Human-readable:

```
### Implement Result
- **What I changed:** <bullet list of files and what changed in each>
- **Why:** <one sentence connecting each change to the triage root cause>
- **Skip added:** <yes (tracking: intel/torch-xpu-ops#N, url: <url>) / no>
- **Ready for verify:** <yes / no>
```

Machine-readable (must appear once, exactly as shown, at the end of the
response):

```json
{
  "changed_files": ["path/to/file1.py", "src/ATen/native/xpu/Foo.cpp"],
  "skip_added": false,
  "tracking_issue": null,
  "patch_proposal_mode": false,
  "ready_for_verify": true
}
```

Field contract:

- `changed_files` — list of paths (relative to `target_repo` root) that
  are staged. `fix/verify` reads this to decide whether a C++/SYCL
  rebuild is required. Must equal `git -C <target_repo> diff --cached
  --name-only` **at output time**. Re-run that command immediately
  before emitting the JSON block; do not cache a pre-Step-3.5 file
  list, because Step 3.5's reviewer may have unstaged offending files.
- `skip_added` — `true` only when this run added a new skip decorator
  under `allow_skip=true`. Removing a stale skip is NOT `skip_added`.
- `tracking_issue` — issue URL returned by `fix/skip-management` when
  `skip_added=true`; `null` otherwise. `xpu-nightly-ci-fix` reads this
  to populate the "Needs Human (skip added)" section of its summary.
- `patch_proposal_mode` — echo the input flag verbatim. The orchestrator
  is the source of truth for the branching decision (issue-handler
  Stage 6 branches on `target_repo == pr_repo`; nightly always passes
  `false`). This echo lets a reviewer or a post-hoc log check confirm
  which mode the implementer actually ran under.
- `ready_for_verify` — `true` when Step 3.5 (if it ran) returned
  `APPROVE`; `false` if the implementer decided to bail out with
  `NEEDS_HUMAN` (in which case the orchestrator should not call
  `fix/verify`).

**Contract:** changes are left staged (`git add`) but NOT committed. The
orchestrator commits only after `fix/verify` returns `PASSED`. `fix/verify`
relies on `git stash` to record a before-state, which requires uncommitted
changes to be present when verify is called.

### HARD RULE: every external claim in "Why" must have a source

The "Why" line and any patch-proposal description text written here will be
read by reviewers as factual statements. Before writing any of the following
phrases, you MUST look up the specific file and line number that backs it up:

- "consistent with upstream"
- "upstream does X" / "upstream already handles this"
- "same as CUDA/MPS/ROCm"
- "mirrors upstream behavior"
- "aligned with upstream"

If you cannot find a specific `file:line` to cite, **do not write the phrase**.
Replace it with a direct statement of the observable fact, e.g.:

| Instead of... | Write... |
|---|---|
| "consistent with upstream's bcomplex32 handling" | "BComplex32 comparison (`isclose`/`mul`) raises `NotImplementedError` in the nightly wheel (pytorch warns 'BComplex32 support is experimental')" |
| "upstream already skips this" | "upstream `test_ops.py:595` skips `{bfloat16, bcomplex32}` inputs in `_ref_test_helper`" |

A "Why" that contains an unsubstantiated upstream comparison is worse than one
that only cites what you directly observed — it inflates reviewer confidence in
a claim that was never verified.

## HARD RULES
- NEVER add skip decorators when `allow_skip=false`.
- When `allow_skip=false`, Step 3.5 (skip-guard reviewer subagent) is
  MANDATORY before returning to the orchestrator. Do not skip it, do
  not run it inline in your own context.
- NEVER stage `third_party/xpu.txt`. It is a submodule pin managed by
  build tooling; staging it is never part of a bug fix, regardless of
  domain (xpu-kernel / inductor / upstream-pytorch).
- NEVER modify files outside your repo scope.
- NEVER modify unrelated files.
- NEVER cherry-pick upstream commits. Rebase instead.
- NEVER submit a torch-xpu-ops PR for a pytorch-core bug.
- NEVER commit when `patch_proposal_mode=true`.

---
name: fix-implement
description: >
  Use when asked to implement a fix for a root-caused failure, apply a
  proposed patch, or produce a staged code change from triage output.
  Takes fix-root-cause's output and edits code; leaves the change staged
  (uncommitted) for fix-verify to check. Does NOT run tests, does NOT
  commit, does NOT open PRs. Called by both issue-handler
  (`allow_skip=false`) and xpu-nightly-ci-fix (`allow_skip=true`).
---

# Implement — Apply the Fix

Takes triage output and makes the code change. Does not run tests (that
is `fix-verify`'s job); staging-only, see HARD RULES.

## Contents

- [Inputs](#inputs)
- Step 0: [Verify environment](#step-0-verify-environment)
- Step 1: [Read the triage output](#step-1-read-the-triage-output)
- Step 2: [Implement the fix](#step-2-implement-the-fix)
- Step 3: [Stage changes](#step-3-stage-changes)
- Step 3.5: [Skip-guard review (allow_skip=false)](#step-35-skip-guard-review-only-when-allow_skipfalse)
- [Output](#output)
- [HARD RULES](#hard-rules)

## Inputs

- `triage_result` — JSON output from `fix-root-cause` (`root_cause`,
  `fix_strategy`, `target_repo`, `domain`, `analyzed_sha`).
- `PYTORCH_DIR` — path to local PyTorch checkout.
- `target_repo_dir` — path to the checkout that will be edited. Derived
  from `PYTORCH_DIR` and `triage_result.target_repo`:
  - `target_repo == "pytorch"` → `target_repo_dir = PYTORCH_DIR`.
  - `target_repo == "torch-xpu-ops"` →
    `target_repo_dir = <PYTORCH_DIR>/third_party/torch-xpu-ops`
    (per AGENTS.md "Commit Pin & Development Override"; the caller
    is expected to have cloned the working branch there ahead of
    time).
  All git and edit operations in this skill run against
  `target_repo_dir`, never against `PYTORCH_DIR` when the two differ.
- `allow_skip` — controls skip decorator strategy:
  - `false` (**issue-handler**): never add skip decorators; must unskip
    and really fix.
  - `true` (**xpu-nightly-ci-fix**): may add a skip with tracking issue
    when the fix requires significant implementation work beyond the
    current scope. Stale skips must still be removed.

## Step 0: Verify environment

```bash
basename $(git -C $target_repo_dir rev-parse --show-toplevel)  # confirm which repo
git -C $target_repo_dir status                                 # inspect current state
```

**On the first attempt** (fresh branch just created by the
orchestrator), the worktree is clean.

**On a loop-back attempt** (orchestrator returned here after
`fix-verify` FAILED or the reviewer requested changes), staged
changes from the previous attempt are still present by design. Do
NOT abort. Refine those changes on top; do not `git reset` or
`git clean` — the orchestrator would have done it if a fresh start
were intended.

## Step 1: Read the triage output

Read `triage_result` carefully before touching any file. Understand:

- Which files/functions need to change
- Why the failure occurs (if applicable, why CUDA works but XPU fails)
- Whether the fix is in pytorch or the backend repo
- Which `analyzed_sha` the triage was against — your edits go on top
  of that base

If the issue is not yet triaged, run `fix-root-cause` first.

## Step 2: Implement the fix

### Key rules

- **Minimal changes** — fix only what's broken; every changed line must
  trace to the triage output.
- **Stay in your repo** — see the domain reference file loaded during
  triage (`../domain-knowledge/domain-<name>.md`, per the domain
  registry) for path conventions.
- **Never modify unrelated files.**

### Fix strategies by category

See `fix-root-cause` Step 1 for domain routing. Common strategies:

- **Tolerance:** match upstream `atol`/`rtol` values exactly.
- **Regression:** find the guilty commit (`git log --oneline -20 -- <file>`),
  apply a fix aligned with upstream intent; document any divergence in
  comments.
- **Newly added test:** enable backend support. If `allow_skip=false`
  and support is genuinely missing, report `NEEDS_HUMAN` — do not add
  a skip. If `allow_skip=true`, follow the "Add a new skip" recipe in
  "Skip operations" below.
- **Unknown root cause:** compare with upstream backend behavior.

### Skip operations

Skip decorators for a failing test live wherever that test lives: the
pytorch test tree for upstream tests, `test/xpu/` inside torch-xpu-ops
for its own tests.

**The skip must live inside `target_repo_dir`.** This skill only ever
produces a single-repo diff, and the orchestrator commits (or reads)
only `target_repo_dir`. If the skip would have to be added to a file
outside `target_repo_dir` — e.g. `target_repo == "torch-xpu-ops"` but
the failing test is in pytorch's `test/` tree — do NOT edit it: that
change would be left uncommitted and then wiped by the orchestrator's
reset between failures. Return `NEEDS_HUMAN(reason=skip_outside_target_repo)`
naming the file that would need the skip.

**Removing a stale skip** (a `@skipIfXpu` / `xfail` decorator whose
underlying failure this run is fixing): `read` the test file, delete
the decorator lines, save. Nothing else to do — the change goes
through the normal `git add` in Step 3.

**Adding a new skip** (`allow_skip=true` only). File a tracking issue
first so the skip has a follow-up owner, then edit:

```bash
# 1. Create the tracking issue and capture the URL.
issue_url=$(gh issue create \
  --repo intel/torch-xpu-ops \
  --title "[skip-added] <test_id> on XPU" \
  --body "Auto-added by fix-implement (allow_skip=true).

  Test: <test_id>
  Original failure: <one-line failure summary from triage_result>
  Root cause: <root_cause from triage_result>
  Reason for skip: <why the actual fix requires human follow-up>
  Base analyzed: <target_repo>@<short_sha from analyzed_sha>
  " \
  --label "agent-added,module: xpu" \
  | tail -1)

# 2. Add the decorator with a comment citing $issue_url so a human
#    can find the tracking issue by grepping the skip in place.
#    Example additions (choose the one matching the test's shape):
#
#      @skipIfXpu(f"see {issue_url}")
#      def test_foo(self): ...
#
#      # or in an OpInfo skips tuple:
#      DecorateInfo(unittest.skip(f"see {issue_url}"), 'TestFoo', 'test_bar', ...)
```

Emit the resulting `issue_url` as `tracking_issue` in the JSON output
(see Output section). `skip_added` becomes `true`. The issue label
`agent-added` lets a human filter for automated-triage tracking issues.

## Step 3: Stage changes

```bash
git -C $target_repo_dir add <your_files>
git -C $target_repo_dir diff --cached --stat   # verify only intended files are staged
```

Never stage unrelated files. In particular, **never stage
`third_party/xpu.txt`** — it is a submodule pin managed by build
tooling, not part of any bug fix (see HARD RULES).

## Step 3.5: Skip-guard review (only when `allow_skip=false`)

Skip this step entirely when `allow_skip=true` (xpu-nightly-ci-fix flow).

When `allow_skip=false`, spawn a fresh-context subagent via the `Task`
tool (`subagent_type=general-purpose`) to inspect the staged diff for
skip-shaped workarounds before returning to the orchestrator. The
implementer must not review its own diff for this specific bias — the
gatekeeper here is a separate agent with no memory of the reasoning
that produced the diff.

Pass the reviewer:

- The full staged diff: `git -C $target_repo_dir diff --cached`.
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
legitimate root-cause fix in the "stale test expectation" category. The
rule only fires on *added* skip-shaped constructs.

The reviewer returns one of:

- `APPROVE` — no skip-shaped workaround found. Continue to Step 4
  (output).
- `REQUEST_CHANGES` — cite each offending hunk (file + line + which
  rule it matched). The implementer MUST address every citation
  (either replace the workaround with a real root-cause fix or, if no
  root-cause fix is possible within this run's scope, unstage the
  offending change and return `NEEDS_HUMAN(reason=skip_guard_rejected)`
  to the orchestrator with the reviewer's citations attached).

**Do not loop this step more than once.** If a second run of Step 3.5
still returns `REQUEST_CHANGES`, unstage the offending change and
return `NEEDS_HUMAN(reason=skip_guard_rejected)` — that is a signal the
fix cannot be produced without a workaround and a human should take it.

This step is intentionally narrower than the orchestrator's Stage 5.5
review. Stage 5.5 checks the entire diff for correctness, minimalism,
and root-cause alignment; Step 3.5 checks only for the specific class of
"hide the failure instead of fixing it" workarounds that
`allow_skip=false` is meant to forbid. Both run; they do not replace
each other.

## Output

Return to the orchestrator a report (markdown block plus JSON block).
The skill does not commit, push, or open PRs — the caller consumes
stdout and decides what to do, per the pattern established by
`issue-triage`, `fix-reproduce`, and `fix-root-cause`.

Include the `<!-- agent:implement -->` marker on the first line of the
markdown block so a downstream caller can locate its own previous
implement comment (if any) and update it in place. Comment location and
update is the **caller's** responsibility.

```
<!-- agent:implement -->

## Implement Result

- **Target repo:** <pytorch | torch-xpu-ops>
- **Analyzed at:** <target_repo>@<short_sha>
- **What I changed:** <bullet list of files and what changed in each>
- **Why:** <one sentence connecting each change to the triage root cause; must cite file:line for every upstream/CUDA comparison — see the hard rule below>
- **Skip added:** <yes (tracking: intel/torch-xpu-ops#N, url: <url>) | no>
- **Ready for verify:** <yes | no>

*Automated by fix-implement.*
```

```json
{
  "target_repo": "pytorch or torch-xpu-ops",
  "analyzed_sha": "<full 40-char sha inherited from triage_result>",
  "changed_files": ["path/to/file1.py", "src/ATen/native/xpu/Foo.cpp"],
  "skip_added": false,
  "tracking_issue": null,
  "allow_skip": false,
  "ready_for_verify": true,
  "verdict": "READY or NEEDS_HUMAN",
  "reason": "<enumerated reason code, see below>",
  "reason_detail": "one-line human-readable detail"
}
```

### Field contract

- `target_repo` / `analyzed_sha` — echo from `triage_result` so
  downstream stages have a self-contained record without re-reading
  triage output.
- `changed_files` — list of paths (relative to `target_repo_dir`) that
  are staged. `fix-verify` reads this to decide whether a C++/SYCL
  rebuild is required. Must equal `git -C $target_repo_dir diff --cached
  --name-only` **at output time**. Re-run that command immediately
  before emitting the JSON block; do not cache a pre-Step-3.5 file list,
  because Step 3.5's reviewer may have unstaged offending files.
- `skip_added` — `true` only when this run added a new skip decorator
  under `allow_skip=true`. Removing a stale skip is NOT `skip_added`.
- `tracking_issue` — issue URL from the "Add a new skip" recipe in
  Step 2 when `skip_added=true`; `null` otherwise. `xpu-nightly-ci-fix`
  reads this to populate the "Needs Human (skip added)" section of
  its summary.
- `allow_skip` — echo the input flag verbatim, so a reviewer can tell
  by looking at the output alone whether Step 3.5 ran.
- `ready_for_verify` — `true` when Step 3.5 (if it ran) returned
  `APPROVE` and staged changes exist; `false` if the implementer
  decided to bail out with `NEEDS_HUMAN` (in which case the
  orchestrator should not call `fix-verify`).
- `verdict` — `READY` (staged diff exists, fix-verify may run) or
  `NEEDS_HUMAN` (see reason).

### `reason` values

On `verdict=READY`: `ok`.

On `verdict=NEEDS_HUMAN`:

- `skip_outside_target_repo` — the skip decorator that would need to be
  added lives outside `target_repo_dir`; see Step 2's "Skip operations".
- `skip_guard_rejected` — Step 3.5's subagent reviewer returned
  `REQUEST_CHANGES` twice, or the implementer chose to bail out rather
  than fight the reviewer.
- `no_fix_possible` — the fix strategy in `triage_result` is not
  implementable without either widening the diff outside `target_repo`
  or adding a skip when `allow_skip=false`.
- `other` — fallback; put full explanation in `reason_detail`.

**Contract:** this leaf leaves the changes **staged** (`git add`) and
does not itself commit. The orchestrator (or the invoking workflow) may
then either verify the staged diff directly or commit it to the fix
branch before verifying — `fix-verify` accepts the fix staged *or*
committed, so it no longer depends on `git stash` / an uncommitted
working tree.

### HARD RULE: every upstream/CUDA claim in "Why" must cite a source

The "Why" line written here will be read by reviewers as factual
statements. Before writing any of the following phrases, you MUST
look up the specific file and line number that backs it up:

- "consistent with upstream"
- "upstream does X" / "upstream already handles this"
- "same as CUDA/MPS/ROCm"
- "mirrors upstream behavior"
- "aligned with upstream"

If you cannot find a specific `file:line` to cite, **do not write the
phrase**. Replace it with a direct statement of the observable fact,
e.g.:

| Instead of... | Write... |
|---|---|
| "consistent with upstream's bcomplex32 handling" | "BComplex32 comparison (`isclose`/`mul`) raises `NotImplementedError` in the nightly wheel (pytorch warns 'BComplex32 support is experimental')" |
| "upstream already skips this" | "upstream `test_ops.py:595` skips `{bfloat16, bcomplex32}` inputs in `_ref_test_helper`" |

A "Why" that contains an unsubstantiated upstream comparison is worse
than one that only cites what you directly observed — it inflates
reviewer confidence in a claim that was never verified.

## HARD RULES

- NEVER add skip decorators when `allow_skip=false`.
- NEVER edit a file outside `target_repo_dir` — including when the skip
  or fix "belongs" in the other repo. That diff cannot be committed or
  read back by the orchestrator; return `NEEDS_HUMAN` instead.
- When `allow_skip=false`, Step 3.5 (skip-guard reviewer subagent) is
  MANDATORY before returning to the orchestrator. Do not skip it, do
  not run it inline in your own context.
- NEVER stage `third_party/xpu.txt`. It is a submodule pin managed by
  build tooling; staging it is never part of a bug fix, regardless of
  domain (xpu-kernel / inductor / upstream-pytorch).
- NEVER cherry-pick upstream commits. If a fix already landed on trunk,
  rebase (`git rebase origin/main`) instead.
- NEVER commit, push, tag, or open a PR. This skill only stages the
  diff (`git add`); the workflow that invoked it takes over after
  `fix-verify` passes and drives its own PR-creation path.

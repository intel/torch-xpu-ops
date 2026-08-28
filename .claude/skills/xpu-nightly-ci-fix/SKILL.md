---
name: xpu-nightly-ci-fix
description: >
  Use when asked to fix nightly CI failures, analyze a nightly failure
  report, debug XPU tests from a CI run, or process a batch of failing
  tests emailed from the nightly job. Runs the same leaf pipeline as
  `issue-handler` but batches multiple failures using a two-phase
  sweep-then-fix loop: reproduce all failures first (sweep summary),
  then deep-fix each STILL_FAILING entry with `allow_skip=true` so
  unfixable failures can be skip-listed with a tracking issue and
  nightly CI is unblocked while the deep fix is pursued
  asynchronously.
---

# XPU Nightly CI Fix — Batch Orchestrator

Analyzes a nightly XPU CI failure report and produces staged fixes
(or tracked skips) for the failing tests. Runs the same leaf skills
as `issue-handler`; the differences are:

1. Input is a **list of failing tests** (from a nightly report /
   email / log excerpt), not a single GitHub issue.
2. Uses a **two-phase sweep-then-fix loop** structurally (the same
   shape as `issue-handler`'s Stage 1u batch fan-out): Phase 1
   sweeps every failure through `fix-reproduce(stage=auto)`,
   Phase 2 deep-fixes STILL_FAILING entries.
3. Runs `fix-implement` with `allow_skip=true` — a nightly failure
   the agent cannot deep-fix in this run can be skip-listed against
   a tracking issue so CI unblocks now and the deep fix is followed
   up async. `issue-handler` runs with `allow_skip=false` and never
   skips.

Every agent-produced diff is a **proposal**. This orchestrator and
its leaves never commit, push, tag, or open a PR — the invoking
workflow takes the staged diff after `fix-verify` passes and drives
its own PR-creation path with human review.

## Contents

- [Pipeline overview](#pipeline-overview)
- [Inputs](#inputs)
- [Execution modes](#execution-modes)
- Step 1: [Parse the failure report](#step-1-parse-the-failure-report)
- Step 2: [Preflight — install nightly wheel once](#step-2-preflight--install-nightly-wheel-once)
- Step 3: [Phase 1 — reproduce sweep](#step-3-phase-1--reproduce-sweep)
- Step 4: [Phase 1 report](#step-4-phase-1-report)
- Step 5: [Phase 2 — deep-fix each STILL_FAILING](#step-5-phase-2--deep-fix-each-still_failing)
- Step 6: [Final summary](#step-6-final-summary)

## Pipeline overview

```
parse report → install nightly wheel once → for entry in failures:
                                              fix-reproduce(stage=auto)
                                                             │
                                                → sweep summary
                                                             │
                                            for entry in STILL_FAILING:
                                              reset checkouts →
                                              fix-root-cause →
                                              fix-implement(allow_skip=true) →
                                              fix-verify
                                                             │
                                                → final summary
```

Leaves are identical to `issue-handler`'s Stage 2-5; only the
flags differ:

| Leaf | Flag in this orchestrator | Flag in issue-handler |
|---|---|---|
| `fix-reproduce` | `stage=auto` (Phase 1 sweep only; Phase 2 does not call it) | `stage=nightly` |
| `fix-implement` | `allow_skip=true` | `allow_skip=false` |

`fix-verify` takes no flags — it always runs the before/after table
and lint — so the leaf call is identical for both orchestrators.
`fix-verify` accepts the fix staged *or* committed, but this
orchestrator consumes the **staged** diff (`git diff --cached`, see
Output), so keep the fix staged here: do not commit it, or the pickup
below sees an empty diff. (Committing is only for the `issue-handler`
bot path that exports a patch from a branch.)

## Inputs

- A nightly failure report — email, log excerpt, list of pytest
  node ids, or an on-disk file. May include the base commit sha
  (`report_commit`), the report date (`report_date`), and per-test
  failure output; only the test list is strictly required.
- `pytorch_dir` — path to a local pytorch checkout, resolved as
  described in `fix-reproduce` Prepare. If absent, leaves clone
  into `$XPU_OPS_ROOT/agent_space_xpu/pytorch/`.
- Mode (interactive / pipeline; see below).

## Execution modes

Same contract as `issue-handler`. See
[../issue-handler/references/execution-modes.md](../issue-handler/references/execution-modes.md).

- **Interactive:** ask the user when blocked; do not post
  comments unless asked.
- **Pipeline:** post the sweep summary + final summary as
  comments on a tracking issue (created if absent — the caller
  usually passes an `$ISSUE_NUMBER` for the current nightly
  cycle's rollup issue).

## Step 1: Parse the failure report

Extract from the report:

- **Failing tests** — pytest node ids or `Class::method` shorthand.
  Normalize each into a node id; skip empty lines, comments, section
  headers. Group duplicates.
- **Report date** — used to name the eventual commit-message context
  (`fix-<report_date>`, e.g. `fix-20260819`); the workflow uses this
  for the branch it creates after `fix-verify` passes.
- **Report commit** (optional) — the pytorch sha the nightly failed
  against. **Do not use this as the analysis base.** Downstream
  leaves analyze against `origin/main` unless the trunk fails to
  build (see `fix-reproduce` Stage 2 fallback logic). The
  report_commit is recorded in the final summary only.

If the report contains a mix of "test failed" and "infra failed"
(runner crashed, docker pull timeout, disk full), split them: infra
failures go straight to the final summary as `NEEDS_HUMAN(infra)`,
never enter the pipeline.

## Step 2: Preflight — install nightly wheel once

Same rationale as `issue-handler`'s skip-list preflight.
`fix-reproduce` Stage 1 always issues `pip install --pre --upgrade`
(it refuses to reuse a stale wheel), so running the upgrade once here
front-loads the one real install; each per-entry
`fix-reproduce(stage=auto)` afterwards issues the same `--upgrade`
command as its first stage and pip returns quickly against the
already-current environment (only falling through to source_build /
ci_env when the nightly wheel does not reproduce). There is no flag to
pass:

```bash
pip3 install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
python -c "import torch; print('nightly:', torch.__version__)"
```

Record the installed nightly version in the final summary.

## Step 3: Phase 1 — reproduce sweep

For each failing test, call `fix-reproduce` with `stage=auto` (full
three-stage fallback nightly → source_build → ci_env, so a failure that
only reproduces on a source build is still caught in the sweep):

```
for entry in failing_tests:
    result = fix-reproduce(
      reproducer_command=entry,
      stage=auto,
      ci_repo=<inferred from entry path>,
    )
    record (entry, result.verdict, result.refined_command, result.reason)
```

Categorize into four buckets:

- `REPRODUCED` → **STILL_FAILING**. Deep fix in Phase 2.
- `NOT_REPRODUCED` → **ALREADY_FIXED**. The nightly report was
  correct at report_commit but the failure no longer reproduces on
  latest nightly; either upstream fixed it, or it was flaky. Record
  and move on — no fix needed.
- `NO_REPRODUCER` → **INVALID_ENTRY**. Node id does not collect.
  Test was renamed, moved, or removed. Needs human.
- `CANNOT_VERIFY` → **UNVERIFIED**. Environmental — e.g. wheel
  install failed, XPU device unavailable. Needs human.

Do not abort on any single entry. Continue sweeping so the summary
is complete.

## Step 4: Phase 1 report

Post one sweep summary comment (or surface it interactively). Marker
`<!-- agent:nightly-sweep -->`:

```
<!-- agent:nightly-sweep -->

## Nightly failure reproduce sweep — <report_date>

Report commit: <report_commit or "not provided">
Nightly wheel tested: <torch.__version__>

| Test | Sweep verdict | Category |
|---|---|---|
| test/xpu/test_ops_xpu.py::TestBinaryUfuncsXPU::test_add_xpu_float32 | REPRODUCED | STILL_FAILING |
| test/xpu/test_nn_xpu.py::TestNNXPU::test_relu_xpu | NOT_REPRODUCED | ALREADY_FIXED |
| test/xpu/gone.py::TestX::test_removed | NO_REPRODUCER | INVALID_ENTRY |

- **STILL_FAILING:** N tests — Phase 2 will attempt to fix.
- **ALREADY_FIXED:** M tests — no action.
- **INVALID_ENTRY:** P tests — needs human review.
- **UNVERIFIED:** Q tests — environmental issue during sweep.

*Automated by xpu-nightly-ci-fix.*
```

Stop here if the caller asked for sweep-only, or if STILL_FAILING
is empty.

## Step 5: Phase 2 — deep-fix each STILL_FAILING

Capture the two base SHAs and reset both checkouts between entries per
the shared
[reset-between-entries recipe](../issue-handler/references/execution-modes.md#reset-between-entries-recipe-batched-fan-out)
(identical to `issue-handler`'s Phase 2, including the 3-attempt cap).

For each STILL_FAILING entry:

1. Reset both checkouts (see the shared recipe above) so the prior
   entry's staged diff does not bleed into this one.
2. Call `fix-root-cause` on the entry's failure signature. If it
   returns `NEEDS_HUMAN`, log the entry outcome and move on.
3. On `IMPLEMENTING`, call `fix-implement` with `allow_skip=true`.
   If it returns `READY`, call `fix-verify` (no flags — it always
   runs the before/after table and lint).
4. Each leaf posts its own per-entry
   `<!-- agent:root-cause -->` / `<!-- agent:implement -->` /
   `<!-- agent:verify -->` comment. Track each entry's outcome for
   the final summary.
5. On any leaf returning `NEEDS_HUMAN` / `CANNOT_VERIFY` /
   `FAILED`, log the entry outcome and **move on** to the next
   entry — do NOT abort the loop. On attempts exhausted (3-attempt
   cap from the shared recipe), record
   `NEEDS_HUMAN(attempts_exhausted)` for that entry and continue.

### Skip-with-tracking-issue path

`allow_skip=true` means `fix-implement` may add a skip decorator
plus create a tracking issue when the deep fix is out of scope for
this run (missing kernel, complex redesign, etc.). The leaf's
inline "Add a new skip" recipe (see `fix-implement` Step 2) handles
the `gh issue create` + edit + `tracking_issue` field itself.

When Phase 2 encounters this outcome:

- The entry's per-entry `<!-- agent:implement -->` comment records
  the tracking issue URL.
- The final summary (Step 6) lists it under "Skipped (with tracking
  issue)".
- The staged diff — a skip decorator addition, nothing else — is
  still valid for the workflow to commit after `fix-verify` passes
  (the skill re-runs the test and confirms it is now skipped
  rather than failing).

## Step 6: Final summary

After Phase 2, post the final summary. Marker
`<!-- agent:nightly-summary -->`:

```
<!-- agent:nightly-summary -->

## Nightly CI fix summary — <report_date>

Report commit: <report_commit>
Nightly wheel: <torch.__version__>
Total failures: <n> | Fixed: <k> | Skipped-with-tracking: <s> | Needs human: <h> | Already fixed: <m>

### Fixed (staged diffs ready for PR)

| Test | target_repo | analyzed_sha | Verify verdict |
|---|---|---|---|
| test_add_xpu_float32 | torch-xpu-ops | abc1234 | PASSED |

### Skipped with tracking issue

| Test | Tracking issue | Reason |
|---|---|---|
| test_conv3d_groups | intel/torch-xpu-ops#1234 | missing kernel |

### Needs human

| Test | Reason | Blocker |
|---|---|---|
| test_flaky_bar | CANNOT_VERIFY | test_timeout |
| test_removed | NO_REPRODUCER | INVALID_ENTRY (renamed/removed upstream) |

### Already fixed on latest nightly

- test_relu_xpu

*Automated by xpu-nightly-ci-fix. The workflow that invoked this
skill picks up the staged diffs (`git diff --cached`) and drives
its own PR-creation path.*
```

Advance the issue's status marker per the shared
[execution-modes.md](../issue-handler/references/execution-modes.md)
contract — final stage is `DONE` if all STILL_FAILING resolved,
`NEEDS_HUMAN` if any entry remains unfixed, `SKIPPED` if all
entries were `ALREADY_FIXED` / `INVALID_ENTRY`.

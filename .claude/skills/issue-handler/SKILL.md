---
name: issue-handler
description: >
  Use when asked to fix a GitHub issue end-to-end, run the agent pipeline
  on an issue, or process an `agent:active` / skip-list tracking issue.
  Orchestrates the full pipeline: `issue-triage` → `fix-reproduce` →
  `fix-root-cause` → `fix-implement` → `fix-verify` → report. Handles
  both single-bug issues and skip-list tracking issues (two-phase loop:
  reproduce all entries first, then fix the ones that still fail).
---

# Issue Handler — End-to-End Orchestrator

This is the **high-level scenario skill** for handling a single GitHub
issue. It does not do the detailed work itself; it sequences the leaf
skills into one iterative pipeline and reports the result. Each stage's
mechanics live in its own skill — read and follow that skill when you
reach its stage.

Every agent-produced diff is a **proposal**. This skill and the leaves
it calls never commit, push, tag, or open a PR — the invoking workflow
takes the staged diff after `fix-verify` passes and drives its own
PR-creation path with human review.

## Contents

- [Pipeline overview](#pipeline-overview)
- [Inputs](#inputs)
- [Execution modes](#execution-modes)
- Stage 1: [Triage](#stage-1--triage-issue-triage)
- Stage 2: [Reproduce](#stage-2--reproduce-fix-reproduce)
- [Stage 2b: Skip-list two-phase loop](#stage-2b--skip-list-two-phase-loop)
- Stage 3: [Root cause](#stage-3--root-cause-fix-root-cause)
- Stage 4: [Implement](#stage-4--implement-fix-implement)
- Stage 5: [Verify](#stage-5--verify-fix-verify)
- Stage 6: [Report](#stage-6--report)
- [Iterative loop bounds](#iterative-loop-bounds)
- [Issue-body status contract](#issue-body-status-contract)

## Pipeline overview

Single-bug path (default for `issue_type=bug`):

```
triage → reproduce → root-cause → implement → verify → report
                       ↑                          |
                       └─── loop up to 3 times ───┘
```

Skip-list path (for `issue_type=skip-list`):

```
                    ┌── Phase 1 (fast reproduce sweep) ──┐
triage → install nightly wheel once → for entry in list: fix-reproduce
                                                      │
                                          → report reproduce summary
                                                      │
                    ┌── Phase 2 (deep fix per STILL_FAILING) ──┐
                    for entry in STILL_FAILING:
                       reset checkouts →
                       root-cause → implement → verify →
                       leaf skills post their own per-entry comments
                                                      │
                                          → final report
```

| Stage | Leaf skill | Purpose |
|-------|-----------|---------|
| 1. Triage | `issue-triage` | Text-only classification: bug / skip-list / nonbug, `scope`, `runtime_dependencies`, preliminary verdict |
| 2. Reproduce | `fix-reproduce` | Verify the failure still reproduces (three-stage fallback: nightly → source_build → ci_env) |
| 3. Root cause | `fix-root-cause` | Deep source analysis, `target_repo`, `domain`, `IMPLEMENTING`/`NEEDS_HUMAN` |
| 4. Implement | `fix-implement` | Edit code, stage the diff (never commit) |
| 5. Verify | `fix-verify` | Run the refined command against source build, PASSED/FAILED/CANNOT_VERIFY |
| 6. Report | this skill | Summarize outcome to the user (or into the issue in pipeline mode) |

## Inputs

- A GitHub issue on `intel/torch-xpu-ops` or `pytorch/pytorch` (URL,
  number, or raw body).
- `pytorch_dir` — path to a local pytorch checkout, resolved as
  described in `fix-reproduce` Prepare. If absent, this skill lets
  `fix-reproduce` / `fix-root-cause` clone it into
  `$XPU_OPS_ROOT/agent_space_xpu/pytorch/`.
- Mode (see below).

## Execution modes

The pipeline runs in one of two modes — **interactive (default)** or
**pipeline** — which changes how every stage reports results and
whether it writes to the GitHub issue. Decide the mode at the start
and pass it to every leaf. See
[references/execution-modes.md](references/execution-modes.md) for
the full contract.

- **Interactive (default):** ask the user when blocked; report
  conversationally; do not touch the issue body / labels / comments
  unless the user asks.
- **Pipeline (explicit):** no human to interrupt — advance the
  issue's `agent:status` marker, update stage labels, let leaf
  skills leave their `<!-- agent:<name> -->` comments, and stop when
  the pipeline settles on a terminal verdict.

## Stage 1 — Triage (`issue-triage`)

Call `issue-triage` on the issue body + comments. It emits
`issue_type` (`bug` / `skip-list` / `nonbug`), `reproduction_missing`
(`yes` / `no`), `scope`, `runtime_dependencies`, and a preliminary
`handling` (`agent-fixable` / `needs-human`).

Branch on `issue_type`:

- `nonbug` → stop the fix pipeline; skip to Stage 6 Report with
  `SKIPPED(reason=nonbug)`.
- `bug` with `reproduction_missing=yes` → stop; Stage 6 Report with
  `NEEDS_HUMAN(reason=reproduction_missing)`. `issue-triage`'s
  own comment already asks the reporter for a reproducer.
- `bug` with `reproduction_missing=no` → continue to Stage 2.
- `skip-list` → go to Stage 2b (two-phase loop).

## Stage 2 — Reproduce (`fix-reproduce`)

Only for single-bug path (`issue_type=bug`). Call `fix-reproduce`
with:

- `reproducer_command` — extracted by `issue-triage` from the issue
  body.
- `stage=auto` — full three-stage fallback.
- `ci_repo` — inferred from repo (`torch-xpu-ops` for issues on
  intel/torch-xpu-ops, `pytorch` for pytorch/pytorch), or the value
  the bot passes explicitly.

Branch on its verdict:

- `REPRODUCED` → continue to Stage 3. Record the `refined_command`
  and `base` for downstream stages.
- `NOT_REPRODUCED` → the issue is stale; Stage 6 Report with
  `SKIPPED(reason=no_longer_reproduces)`.
- `NO_REPRODUCER` → Stage 6 Report with
  `NEEDS_HUMAN(reason=no_reproducer)`.
- `CANNOT_VERIFY` → Stage 6 Report with
  `NEEDS_HUMAN(reason=cannot_verify)` and the `blocker` field.

## Stage 2b — Skip-list two-phase loop

Only for `issue_type=skip-list`. A skip-list tracking issue lists
many failing tests that were skipped in CI; the orchestrator sweeps
them all first (Phase 1), reports the sweep result, then only
attempts to fix the ones that still fail (Phase 2).

### Preflight: install nightly wheel once

Skip-list issues can list dozens of entries. Each `fix-reproduce`
Stage 1 would re-run `pip install --pre --upgrade` on the nightly
wheel — an expensive no-op after the first. Install once here,
before the loop:

```bash
pip3 install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
python -c "import torch; print('nightly:', torch.__version__)"
```

Pass `skip_wheel_install=true` (or the equivalent flag; see
`fix-reproduce` Inputs if that skill grows one — for now, the leaf
detects an already-current nightly and no-ops on reinstall) so
Phase 1's per-entry `fix-reproduce` calls reuse the same install.

### Extract the skip-list entries

Parse the issue body — typical shape is a checklist of tests, e.g.

```markdown
- [ ] test/xpu/test_ops_xpu.py::TestFooXPU::test_bar_xpu_float32
- [ ] test/xpu/test_ops_xpu.py::TestFooXPU::test_baz_xpu_bfloat16
- [ ] TestQuxXPU::test_quux
```

Normalize each into a pytest node id (a bare `Class::method` gets
resolved during `fix-reproduce`'s Prepare step). Skip empty lines,
comments, and headers.

### Phase 1: reproduce sweep

For each entry, call `fix-reproduce` with `stage=nightly` (Stage 1
only — skip-list sweeps do not need source builds; source build is
Phase 2's concern):

```
for entry in entries:
    result = fix-reproduce(reproducer_command=entry, stage=nightly, ci_repo=<from repo>)
    record: (entry, result.verdict, result.refined_command, result.reason)
```

Categorize:

- `REPRODUCED` → **STILL_FAILING**. Keep the `refined_command` and
  `stage=source_build` needed base for Phase 2.
- `NOT_REPRODUCED` → **STALE_SKIP**. Candidate for skip removal.
- `NO_REPRODUCER` → **INVALID_ENTRY**. The entry doesn't collect;
  either the test was renamed / removed upstream, or the entry is
  malformed. Human decides.
- `CANNOT_VERIFY` → **UNVERIFIED**. Environmental — human decides.

Do **not** abort on any single entry's `CANNOT_VERIFY` or
`NO_REPRODUCER` — continue sweeping so the summary is complete.

### Phase 1 report

Post a single Phase-1 comment on the issue (or surface it to the user
in interactive mode) with a summary table:

```
<!-- agent:skip-list-sweep -->

## Skip-list reproduce sweep

Base: <torch nightly version>

| Test | Verdict | Category |
|---|---|---|
| test_bar_xpu_float32 | NOT_REPRODUCED | STALE_SKIP |
| test_baz_xpu_bfloat16 | REPRODUCED (FAILED) | STILL_FAILING |
| test_quux | NO_REPRODUCER | INVALID_ENTRY |

- **STALE_SKIP:** N tests — remove the skip decorator.
- **STILL_FAILING:** M tests — Phase 2 will attempt to fix.
- **INVALID_ENTRY:** P tests — needs human review.
- **UNVERIFIED:** Q tests — environmental issue during sweep.

*Automated by issue-handler.*
```

The `<!-- agent:skip-list-sweep -->` marker is unique to this
orchestrator so a re-run can locate and update the same comment in
place instead of duplicating it.

**Stop here** if the caller asked for reproduce-only, or if
STILL_FAILING is empty (nothing to fix — the STALE_SKIP entries can
be cleared by a single follow-up `fix-implement` batch or by a
human).

### Phase 2: deep-fix each STILL_FAILING entry

For each STILL_FAILING entry, run the full pipeline. Each entry is
an independent sub-bug — the diffs must not bleed across entries.

Before entering the loop, capture the two independent base SHAs
you will reset to between entries:

```bash
pytorch_base=$(git -C $pytorch_dir rev-parse HEAD)
xpu_ops_base=$(git -C $pytorch_dir/third_party/torch-xpu-ops rev-parse HEAD)
```

Track them as two separate variables. A torch-xpu-ops fix uses
`xpu_ops_base` as its `target_repo_dir`'s base while `pytorch_base`
stays pinned for the pytorch tree; a pytorch fix does the reverse.
Do not conflate them.

Before each entry:

1. **Reset both candidate checkouts.** Different entries can triage
   to different `target_repo`; a prior entry's staged diff must not
   pollute the next. Reset both bases so the next entry starts
   clean:

   ```bash
   git -C $pytorch_dir reset --hard $pytorch_base
   git -C $pytorch_dir clean -fdx
   if [ -d "$pytorch_dir/third_party/torch-xpu-ops/.git" ]; then
       git -C $pytorch_dir/third_party/torch-xpu-ops reset --hard $xpu_ops_base
       git -C $pytorch_dir/third_party/torch-xpu-ops clean -fdx
   fi
   ```

2. Call `fix-root-cause` on the entry's failure signature.
3. If `IMPLEMENTING`: call `fix-implement`. If `READY`: call
   `fix-verify`. If any of those return `NEEDS_HUMAN` /
   `CANNOT_VERIFY` / `FAILED`, log and move on to the next entry —
   do **not** stop the loop.
4. Each leaf posts its own `<!-- agent:root-cause -->` /
   `<!-- agent:implement -->` / `<!-- agent:verify -->` comment per
   entry (leaves handle their own comment location; nothing to do
   here beyond passing the issue number).

After the loop, go to Stage 6 Report for the final skip-list
outcome.

## Stage 3 — Root cause (`fix-root-cause`)

Single-bug path only. Call `fix-root-cause` with the failure
description and the `refined_command` from Stage 2.

Branch on its `verdict`:

- `IMPLEMENTING(reason=ok)` → continue to Stage 4. Record
  `target_repo`, `domain`, `analyzed_sha`, `root_cause`,
  `fix_strategy`.
- `NEEDS_HUMAN` → Stage 6 Report with the specific
  `reason` (`umbrella_task` / `feature_gap` / `hardware_specific` /
  `cross_repo_coordinated` / `no_registered_domain` / etc.). Each
  reason maps to a different final `agent:status` value; see
  [execution-modes.md](references/execution-modes.md).

## Stage 4 — Implement (`fix-implement`)

Call `fix-implement` with `triage_result`, `pytorch_dir`,
`target_repo_dir` (derived from `target_repo`), and `allow_skip`:

- `allow_skip=false` for the standard issue-handler pipeline —
  never add skip decorators, must actually fix.
- `allow_skip=true` only when the caller explicitly opts in
  (e.g. `xpu-nightly-ci-fix` orchestrator with a nightly-CI issue).

Branch on the verdict:

- `READY(reason=ok)` → continue to Stage 5.
- `NEEDS_HUMAN` → Stage 6 Report. The specific `reason`
  (`skip_outside_target_repo` / `skip_guard_rejected` /
  `no_fix_possible` / etc.) drives the final label.

## Stage 5 — Verify (`fix-verify`)

Call `fix-verify` with `refined_command` (from Stage 2),
`target_repo_dir`, `changed_files` (from Stage 4),
`run_before_after_diff=false`, `run_lint=false` (issue-handler does
not need the before/after table or lint auto-fix; those are for
xpu-nightly-ci-fix).

Branch on the verdict:

- `PASSED(reason=ok)` → Stage 6 Report with
  `IMPLEMENTING(fix_verified)`. The staged diff is ready for the
  workflow to open a PR (with human review).
- `FAILED` → **loop back to Stage 4** with the failure output as
  additional context. See "Iterative loop bounds" below.
- `CANNOT_VERIFY` → Stage 6 Report with
  `NEEDS_HUMAN(reason=<verify's reason>)` and the blocker. Do not
  loop on CANNOT_VERIFY — the environment problem will not fix
  itself.

## Stage 6 — Report

At the end, summarize the outcome. In **interactive mode** present
this to the user in plain language. In **pipeline mode** advance
the issue's `agent:status` to the terminal stage
(`DONE` / `NEEDS_HUMAN` / `SKIPPED`) and update the checklist per
[execution-modes.md](references/execution-modes.md); the leaf skills
already left their own `<!-- agent:<name> -->` comments so no extra
summary comment is needed unless the pipeline is a skip-list run
(which posts the Phase-1 sweep summary as a distinct comment).

Always include in the summary:

- **Issue:** link/number and one-line title.
- **Path:** `single-bug` / `skip-list`.
- **Outcome:** `IMPLEMENTING(fix_verified)` / `NEEDS_HUMAN(<reason>)`
  / `SKIPPED(<reason>)`.
- **Root cause** (from Stage 3, if reached).
- **Files changed** (from Stage 4, if reached).
- **Verification** (from Stage 5, if reached).
- For skip-list: the four category counts (`STALE_SKIP`,
  `STILL_FAILING`, `INVALID_ENTRY`, `UNVERIFIED`) plus per-entry
  Phase-2 verdicts.

If the outcome is `IMPLEMENTING(fix_verified)`, the invoking
workflow reads the staged diff (`git -C $target_repo_dir diff
--cached`) and drives its own PR-creation path. **Do not open the
PR from this skill.**

## Iterative loop bounds

The pipeline is not strictly linear. Loop when a later stage
invalidates an earlier assumption:

- Stage 5 `FAILED` → return to Stage 4 (refine the fix).
- Stage 4's Step 3.5 skip-guard rejects → the leaf itself re-runs
  once; a second rejection returns `NEEDS_HUMAN` and the
  orchestrator does not retry.

Bound: **maximum 3 fix attempts** (Stage 4 → Stage 5 → Stage 4 …).
This matches the legacy pipeline's `max_agent_attempts`. When you
stop without success, report `NEEDS_HUMAN(reason=attempts_exhausted)`
with the last `fix-verify` failure output in `reason_detail`.

Do **not** loop on:

- `CANNOT_VERIFY` at any stage (environment problem, not fix
  problem).
- `NEEDS_HUMAN` from any leaf (contract: the leaf already decided
  it needs a human).
- Stage 3 `no_registered_domain` (domain registry is a fixed set,
  looping won't unstick it).

## Issue-body status contract

**Pipeline mode only.** In interactive mode, do not touch the issue
body/markers/labels unless the user asks — report to the user
instead.

This orchestrator owns advancing the overall `<!-- agent:status:X -->`
marker through:

```
DISCOVERED → TRIAGING → REPRODUCING → TRIAGED → IMPLEMENTING →
VERIFYING → DONE
```

with terminal alternates `NEEDS_HUMAN` and `SKIPPED`. Stage-by-stage
mapping to labels is in
[references/execution-modes.md](references/execution-modes.md); each
leaf skill owns its own `<!-- agent:<name> -->` comment/log slot,
this orchestrator owns the overall `agent:status` marker + the
Action Items checklist.

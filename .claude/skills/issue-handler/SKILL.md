---
name: issue-handler
description: >
  End-to-end orchestrator for handling a single GitHub issue on pytorch or
  torch-xpu-ops. Use when asked to fix the issue — formatting the issue,
  verifying it reproduces, triaging the root cause, proposing and verifying a
  fix, and reporting back to the user. Coordinates the issue-format,
  test-verification, xpu-issues-triaging, and issue-fix sub-skills.
---

# Issue Handler — End-to-End Orchestrator

This is the **high-level scenario skill**. It does not do the detailed work
itself; it sequences four leaf skills into one iterative pipeline and reports
the result to the user. Each stage's mechanics live in its own skill — read and
follow that skill when you reach the stage or you are asked to do one specific task.

## Pipeline overview

```
format  ->  verify-exists  ->  triage  ->  propose-fix  ->  verify-fix  ->  report
```

| Stage | Sub-skill | Purpose |
|-------|-----------|---------|
| 1. Format | `issue-format` | Classify bug/nonbug, extract metadata, normalize the issue body |
| 2. Verify exists | `test-verification` | Confirm the failure still reproduces locally |
| 3. Triage | `xpu-issues-triaging` | Root cause, fix strategy, `IMPLEMENTING`/`NEEDS_HUMAN` verdict |
| 4. Propose fix | `issue-fix` | Implement the fix and re-verify the reproducer |
| 5. Verify fix | `test-verification` | Re-run to confirm the fix resolves the failure |
| 6. Report | this skill | Summarize outcome for the user (and update issue body) |

## Inputs

- A GitHub issue (URL, number, or raw body) on `pytorch` or `torch-xpu-ops`.
- For local reproduction/fix stages: a local checkout and Python environment
  (see `test-verification` and `issue-fix` for environment setup).

## Execution modes

This pipeline runs in one of two modes — **interactive (default)** or
**pipeline** — which changes how every stage reports results and whether it
writes to the GitHub issue. Decide the mode at the start of the run and pass it
down to every leaf skill. See the shared reference for the full rules:
[references/execution-modes.md](references/execution-modes.md).

- **Interactive (default):** ask the user when blocked; report conversationally;
  do not touch the issue body/labels unless asked.
- **Pipeline (explicit):** no human to ask — write status into the issue body,
  advance the `agent:status` marker and labels, leave a comment, and stop.

## How to run the pipeline

Work the stages in order. After each stage, decide whether to continue, loop,
or stop based on that stage's output.

### Stage 1 — Format (`issue-format`)
Classify the issue and extract metadata. If `issue_type` is `nonbug` (task,
feature request, enhancement, performance, question), **stop the fix pipeline**:
record the classification, mark the issue accordingly, and skip to Report. Only
`bug` issues continue to Stage 2.

### Stage 2 — Verify it reproduces (`test-verification`)
Resolve the reproducer to a local command and run it.
- `FAILED` (bug reproduces) -> continue to Stage 3.
- `PASSED` (no longer reproduces) -> the issue may be already fixed; record that
  and skip to Report.
- `CANNOT_VERIFY` -> note why; you may still triage statically (Stage 3 does not
  require a successful local run), but flag the uncertainty in the report.

### Stage 3 — Triage (`xpu-issues-triaging`)
Determine root cause, fix strategy, target repo, and verdict.
- Verdict `IMPLEMENTING` -> continue to Stage 4.
- Verdict `NEEDS_HUMAN` -> stop the fix pipeline and skip to Report with the
  reason.

### Stage 4 — Propose the fix (`issue-fix`)
Implement the fix following the triage strategy. `issue-fix` re-runs the
reproducer as part of its own verification.

### Stage 5 — Verify the fix (`test-verification`)
Re-run the reproducer (and related tests) to confirm the failure is resolved
and nothing regressed. If it still fails, **loop back to Stage 4** with the new
information. Stop looping when the fix verifies or you hit a genuine blocker
(then report `NEEDS_HUMAN`).

### Stage 6 — Report
See "Reporting to the user" below.

## Iterative loop

The pipeline is not strictly linear. Loop when a later stage invalidates an
earlier assumption:

- Stage 5 fails -> return to Stage 4 (refine the fix).
- Stage 4 reveals the triage was wrong -> return to Stage 3 (re-triage).
- Stage 3 finds the reproducer is wrong/incomplete -> return to Stage 2.

Bound the loop: stop after the fix verifies, or when you have exhausted a
reasonable number of attempts (treat 3 fix attempts as the soft cap, matching
the legacy pipeline's `max_agent_attempts`). When you stop without success,
report `NEEDS_HUMAN` with the blocker.

## Issue-body status (backward compatible)

**Pipeline mode only.** In interactive mode (default), do not touch the issue
body, markers, or labels unless the user asks — report to the user instead.

The full backward-compatible contract (status markers, stage->label map, Action
Items checklist, canonical headings) lives in the shared reference:
[references/execution-modes.md](references/execution-modes.md). This orchestrator owns
advancing the overall `agent:status` stage and the checklist; each leaf skill
owns its own marker/log slot.

## Reporting to the user

At the end of every run, summarize the outcome. In **interactive mode** present
this to the user in plain language; in **pipeline mode** write the same summary
into the GitHub issue (comment + body) and stop. Always include:

- **Issue:** link/number and one-line title.
- **Classification:** bug / nonbug (+ category).
- **Reproduced:** yes / no / cannot-verify (with the command used).
- **Root cause:** one sentence (from triage).
- **Action taken:** files changed, or "none" with the reason.
- **Fix verified:** PASS / FAIL / not-attempted (with the command).
- **Outcome:** one of `IMPLEMENTING` (fix ready / PR next), `NEEDS_HUMAN`
  (with blocker), `SKIPPED` (nonbug or already-fixed), and any open risks.

If the outcome is `IMPLEMENTING` and a PR should be opened, hand off to the
`xpu-ops-pr-creation` skill — do not open the PR from this skill.

## See also

> **Sub-skill discoverability:** The leaf skills under `issue-handler/` (e.g.
> `issue-fix`, `issue-format`, `test-verification`) are nested two levels deep
> under `.claude/skills/` and may not be auto-discovered by the skill system.
> They are intended to be reached via explicit orchestrator delegation (this
> skill calls them by name). Load them directly if needed by invoking
> `issue-handler/issue-fix`, etc.

- `issue-format` — Stage 1 classification + metadata.
- `test-verification` — Stages 2 and 5 local reproduction.
- `xpu-issues-triaging` — Stage 3 root cause + verdict (canonical triage skill).
- `issue-fix` — Stage 4 implementation.
- `xpu-ops-pr-creation` — opens the PR once a fix is verified.

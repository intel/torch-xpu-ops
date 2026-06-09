---
name: issue-handler
description: >
  End-to-end orchestrator for handling a single GitHub issue on pytorch or
  torch-xpu-ops. Use when asked to "handle", "process", "work through", or
  "drive" an issue from raw report to a proposed fix — formatting the issue,
  verifying it reproduces, triaging the root cause, proposing and verifying a
  fix, and reporting back to the user. Coordinates the issue-format,
  test-verification, xpu-issues-triaging, and issue-fix sub-skills.
---

# Issue Handler — End-to-End Orchestrator

This is the **high-level scenario skill**. It does not do the detailed work
itself; it sequences four leaf skills into one iterative pipeline and reports
the result to the user. Each stage's mechanics live in its own skill — read and
follow that skill when you reach the stage.

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

## How to run the pipeline

Work the stages in order. After each stage, decide whether to continue, loop,
or stop based on that stage's output. Do **not** spawn subagents for triage
(see `xpu-issues-triaging`) — it can hang on large repos.

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

Formerly a Python driver script advanced the issue through stages and wrote
status into the GitHub issue body. That orchestration is now done by **this
skill** directly — no script. To stay compatible with any tooling that still
parses issue bodies, preserve the markers defined in the agent body templates:

- Bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body.yml`
- Non-bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body-nonbug.yml`

When you update an issue body, keep these contracts intact:

1. **Status marker** at the top of the body:
   `<!-- agent:status:STAGE -->`. Advance `STAGE` through:
   `DISCOVERED -> UPSTREAM_VERIFYING -> WAITING_UPSTREAM -> TRIAGING ->
   TRIAGED -> IMPLEMENTING -> IN_REVIEW -> PUBLIC_PR -> CI_WATCH -> MERGED`,
   with terminal stages `DONE`, `SKIPPED`, or `NEEDS_HUMAN`.

2. **Stage -> label mapping** (apply the matching GitHub label when you can):

   | Stage(s) | Label |
   |----------|-------|
   | DISCOVERED, UPSTREAM_VERIFYING, TRIAGING, IMPLEMENTING, IN_REVIEW, PUBLIC_PR, CI_WATCH, MERGED | `agent:active` |
   | WAITING_UPSTREAM | `agent:waiting-upstream` |
   | TRIAGED | `agent:triaged` |
   | DONE, SKIPPED | `agent:done` |
   | NEEDS_HUMAN | `agent:needs-human` |

3. **Action Items checklist** — check off `- [ ]` items as stages complete and
   fill the matching log placeholders in the template:
   `<!-- agent:discovery-log -->` (format),
   `<!-- agent:env-log -->` (environment setup),
   `<!-- agent:upstream-log -->` and `<!-- agent:triage-log -->` (triage),
   `<!-- agent:fix-log -->` (fix),
   `<!-- agent:verification-log -->` (verification).

4. **Canonical section headings** — the format stage lays out the skeleton
   headings; their content is filled across later stages. Bug issues use
   exactly: `Description, Reproducer, Error Log, Environment, Test Info,
   Root Cause Analysis, Proposed Fix Strategy, Target Repository,
   Additional Context` — where `Root Cause Analysis`, `Proposed Fix Strategy`,
   and `Target Repository` are filled at the **triage** stage, not by format.
   Non-bug issues use: `Description, Objective, Current Status`.

Each leaf skill notes which marker/label/log slot it owns; this skill owns
advancing the overall `agent:status` stage and the checklist.

## Reporting to the user

At the end of every run, summarize the outcome for the user in plain language.
Always include:

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

- `issue-format` — Stage 1 classification + metadata.
- `test-verification` — Stages 2 and 5 local reproduction.
- `xpu-issues-triaging` — Stage 3 root cause + verdict (canonical triage skill).
- `issue-fix` — Stage 4 implementation.
- `xpu-ops-pr-creation` — opens the PR once a fix is verified.

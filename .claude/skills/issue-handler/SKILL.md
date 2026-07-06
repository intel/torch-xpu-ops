---
name: issue-handler
description: >
  End-to-end orchestrator for fixing a single GitHub issue on pytorch or
  torch-xpu-ops. Sequences fix/ leaf skills into a pipeline and reports the
  result to the user or writes to the GitHub issue body.
---

# Issue Handler — Orchestrator

Sequences `fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify` into a
single pipeline for one GitHub issue. All the detailed logic lives in those
leaf skills — this skill owns the scheduling, mode handling, and reporting.

## Execution modes

Decide mode once at the start and keep it for every stage.

- **Interactive (default):** human present. Ask when blocked. Report
  conversationally. Do not write to the GitHub issue body unless the user asks.
- **Pipeline:** automated. No human to ask. Write status into the issue body,
  advance `agent:status` marker and labels, leave a comment, and stop.

### Pipeline mode: issue body contract

Status marker at top of body: `<!-- agent:status:STAGE -->`. Advance through:
```
DISCOVERED -> UPSTREAM_VERIFYING -> WAITING_UPSTREAM -> TRIAGING ->
TRIAGED -> IMPLEMENTING -> IN_REVIEW -> PUBLIC_PR -> CI_WATCH -> MERGED
```
Terminal stages: `DONE`, `SKIPPED`, `NEEDS_HUMAN`.

Stage → label mapping:

| Stage(s) | Label |
|---|---|
| DISCOVERED, UPSTREAM_VERIFYING, TRIAGING, IMPLEMENTING, IN_REVIEW, PUBLIC_PR, CI_WATCH, MERGED | `agent:active` |
| WAITING_UPSTREAM | `agent:waiting-upstream` |
| TRIAGED | `agent:triaged` |
| DONE, SKIPPED | `agent:done` |
| NEEDS_HUMAN | `agent:needs-human` |

Log slots (fill as stages complete):

| Slot | Stage |
|---|---|
| `<!-- agent:discovery-log -->` | issue-format |
| `<!-- agent:env-log -->` | reproduce |
| `<!-- agent:upstream-log -->`, `<!-- agent:triage-log -->` | triage |
| `<!-- agent:fix-log -->` | implement |
| `<!-- agent:verification-log -->` | verify |

Templates: `.github/ISSUE_TEMPLATE/agent/agent-issue-body.yml` (bug),
`agent-issue-body-nonbug.yml` (non-bug).

## Inputs

- A GitHub issue URL, number, or raw body on `pytorch` or `torch-xpu-ops`.
- Local checkout and Python environment for reproduction/fix stages.

## Pipeline

```
issue-format → reproduce → triage → implement → verify → report
```

### Stage 1 — issue-format

Classify as `bug` or `nonbug` and extract metadata.

- `nonbug` → record classification, report to user, **stop**.
- `bug` → continue to Stage 2.

### Stage 2 — fix/reproduce

Call `fix/reproduce` with:
- `reproducer_command` from the issue body (if present)
- `ci_commit` if the issue references a specific CI run
- `pytorch_dir` if available; otherwise `fix/reproduce` clones to
  `agent_space_xpu/pytorch/`

Interpret the output:

| Output | Action |
|--------|--------|
| `REPRODUCED` | Continue to Stage 3 with `refined_command` |
| `NOT_REPRODUCED` | Triage to collect why; report to user; stop |
| `NO_REPRODUCER` | Continue to Stage 3 (static triage only) |
| `CANNOT_VERIFY` | Report blocker to user; stop |

### Stage 3 — fix/triage

Call `fix/triage` with the failure description (error log, context, and
`refined_command` if available from Stage 2).

| Verdict | Action |
|---------|--------|
| `IMPLEMENTING` | Continue to Stage 3.5 |
| `NEEDS_HUMAN` | Report reason to user; stop |

### Stage 3.5 — Load domain skill

Read the `domain` field from the triage output. Use the skill tool to load
`fix/domains/<domain>` (e.g. `fix/domains/xpu-kernel`). If no domain skill
exists for the reported domain, proceed without it.

### Stage 4 — fix/implement

Call `fix/implement` with:
- `triage_result` from Stage 3
- `pytorch_dir`
- `allow_skip=false` — issue-handler never allows adding skip decorators
- no `commit_message_template` (use standard format)

### Stage 5 — fix/verify

Call `fix/verify` with:
- `refined_command` from Stage 2
- `pytorch_dir`
- `changed_files` from Stage 4
- `run_before_after_diff=false`
- `run_lint=false`

| Output | Action |
|--------|--------|
| `PASSED` | Continue to Stage 6 |
| `FAILED` | Loop back to Stage 4 with failure output (max 3 attempts) |
| `CANNOT_VERIFY` | Report to user; stop |

If 3 attempts exhausted without `PASSED`, report `NEEDS_HUMAN`.

### Stage 6 — Report and hand off

Summarize the outcome. In **interactive mode**, report to the user. In
**pipeline mode**, write to the issue body and leave a comment.

Always include:
- Issue link and one-line title
- Classification (bug/nonbug + category)
- Reproduced: yes / no / cannot-verify (+ command used)
- Root cause (one sentence)
- Files changed (or "none" + reason)
- Fix verified: PASS / FAIL / not-attempted (+ command)
- Outcome: `IMPLEMENTING` / `NEEDS_HUMAN` / `SKIPPED` / `NOT_REPRODUCED`

If outcome is `IMPLEMENTING` (fix verified), hand off to `xpu-ops-pr-creation`
to open the PR. Do not open it from this skill.

## Iterative loop

The pipeline is not strictly linear. Loop when a later stage invalidates an
earlier assumption:

- Stage 5 FAILED → return to Stage 4 (refine the fix)
- Stage 4 reveals triage was wrong → return to Stage 3
- Stage 3 finds reproducer is wrong → return to Stage 2

Soft cap: 3 fix attempts (Stages 4–5). Stop with `NEEDS_HUMAN` if exhausted.

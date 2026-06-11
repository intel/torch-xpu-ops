# Execution Modes (shared reference)

This doc applies to shared contract for the `issue-handler` pipeline and its leaf skills
(`issue-format`, `test-verification`, `xpu-issues-triaging`, `issue-fix`).
Every skill that reports results or touches a GitHub issue follows the mode
rules below. Decide the mode once at the start of a run and keep it for every
stage.

## The two modes

- **Interactive mode (default).** The skill is loaded in a chat session with a
  human present. In this mode, human could be looped in to give hints.
  When a stage hits a blocker, a `NEEDS_HUMAN` verdict, an
  ambiguous classification, a failure you cannot reproduce, or a fix that will
  not verify, **ask the user and wait for their input** instead of stopping
  silently. Report progress and results conversationally.
  Do **not** write status markers/labels into the GitHub issue body or
  leave GitHub comments unless the user explicitly asks you to.
- **Pipeline mode (explicit).** Selected only when the caller states the run is
  automated / non-interactive / "in the pipeline". There is no human to ask, so
  follow the legacy behavior: write status into the issue body, advance the
  `agent:status` marker and labels, leave a GitHub comment explaining the
  outcome, and stop.

Whenever a skill says "leave a comment and stop" or "update the issue body",
that is the **pipeline-mode** action. The **interactive-mode** equivalent is to
surface the same information to the user and ask how to proceed.

## Issue-body status contract (pipeline mode only)

In interactive mode, do not touch the issue body, markers, or labels unless the
user asks — report to the user instead.

In pipeline mode the `issue-handler` orchestrator does this directly — no script. To stay compatible with any tooling that still parses issue bodies, preserve the markers defined in the agent body templates:

- Bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body.yml`
- Non-bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body-nonbug.yml`

When updating an issue body, keep these contracts intact:

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

Per-stage ownership of each marker/log slot is noted in the owning skill; the
`issue-handler` orchestrator owns advancing the overall `agent:status` stage and
the checklist.

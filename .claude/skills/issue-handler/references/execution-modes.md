# Issue Body Status Contract (issue-handler pipeline mode only)

In interactive mode, do not touch the issue body, markers, or labels — report
to the user instead.

In pipeline mode, `issue-handler` writes status directly into the GitHub issue
body. Keep these contracts intact:

## Status marker

Top of body: `<!-- agent:status:STAGE -->`. Advance `STAGE` through:
```
DISCOVERED -> UPSTREAM_VERIFYING -> WAITING_UPSTREAM -> TRIAGING ->
TRIAGED -> IMPLEMENTING -> IN_REVIEW -> PUBLIC_PR -> CI_WATCH -> MERGED
```
Terminal stages: `DONE`, `SKIPPED`, `NEEDS_HUMAN`.

## Stage → label mapping

| Stage(s) | Label |
|---|---|
| DISCOVERED, UPSTREAM_VERIFYING, TRIAGING, IMPLEMENTING, IN_REVIEW, PUBLIC_PR, CI_WATCH, MERGED | `agent:active` |
| WAITING_UPSTREAM | `agent:waiting-upstream` |
| TRIAGED | `agent:triaged` |
| DONE, SKIPPED | `agent:done` |
| NEEDS_HUMAN | `agent:needs-human` |

## Log slots and checklist

Check off `- [ ]` items as stages complete and fill the matching slots:

| Slot | Stage |
|---|---|
| `<!-- agent:discovery-log -->` | issue-format |
| `<!-- agent:env-log -->` | reproduce |
| `<!-- agent:upstream-log -->`, `<!-- agent:triage-log -->` | triage |
| `<!-- agent:fix-log -->` | implement |
| `<!-- agent:verification-log -->` | verify |

Templates: `.github/ISSUE_TEMPLATE/agent/agent-issue-body.yml` (bug),
`agent-issue-body-nonbug.yml` (non-bug).

The `issue-handler` orchestrator owns advancing the overall `agent:status`
stage and the checklist.

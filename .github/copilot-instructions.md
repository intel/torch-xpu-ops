# Copilot Instructions — torch-xpu-ops

## Required reading (mandatory)

All agent instructions live in `CLAUDE.md` at the repository root. You MUST read the linked file in
full before proceeding. Do not skip this step. Do not paraphrase from
memory. The contents of these files are authoritative.

| When you are about to... | Read this file first |
|--------------------------|---------------------|
| All agent instructions | `CLAUDE.md` |
| Review a pull request | `.claude/skills/pr-review/SKILL.md` |

## Review mode is read-only

When invoked to review a PR — the comment contains "review", "/pr-review",
"/skill-writer", or any similar analysis request — the response is a
**single markdown comment on the PR**. No file edits, no commits, no
new branches. Findings that could be fixed directly must be described in
the comment for the author to apply.

The user can request code changes in a separate follow-up comment with an
explicit action verb ("fix", "apply", "commit"). Do not preemptively
combine review with modification, even when the fix looks trivial —
splitting the two keeps the review reproducible and lets the author
audit diffs before they land.

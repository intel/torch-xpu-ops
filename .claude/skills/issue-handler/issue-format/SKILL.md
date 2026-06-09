---
name: issue-format
description: >
  Classify a GitHub issue as bug or nonbug and extract metadata
  (test_type, category, dependency, platform, context). Use as the first
  stage of issue handling, when you need a structured classification of an
  issue's type and key attributes returned as JSON.
---

# Issue Format — Classify & Extract Metadata

## Inputs

You receive a GitHub issue: its title, body, and labels.

If the body is already split into sections (description, reproducer,
error_log, environment, additional_context), use them directly. If it is
not pre-sectioned, identify those portions yourself from the raw body
before classifying.

## Your Task

1. **Classify** the issue as `"bug"` or `"nonbug"`.
2. **Extract** metadata fields.
3. Return a JSON object.

## Classification Rules

**Bug** — test failures, runtime errors, assertion errors, incorrect output, crashes.
Indicators: error tracebacks, failing test names, "RuntimeError", "AssertionError",
"fails with", `### 🐛 Describe the bug`, test logs.

**Non-bug** — feature requests, tasks, performance issues, questions, discussions,
tracking issues, enhancement proposals, feature gaps.
Indicators: "Enable", "[Task]", "Consider", "Align", "feature gap", "clarification",
checklists of work items, "implement", `enhancement` label, `performance` label,
no failing tests, discussion-style content.

## Output Format
Return ONLY this JSON object, no markdown fences, no explanation:

```json
{
  "metadata": {
    "test_type": "ut | e2e | ...",
    "category": "category if identifiable",
    "dependency": "upstream | ...",
    "platform": "xpu | BMG | ...",
    "context": "upstream links, version info, brief notes"
  },
  "issue_type": "bug"
}
```

## Rules

1. **issue_type**: `"bug"` for test failures, runtime errors, crashes, incorrect output.
   `"nonbug"` for feature requests, tasks, enhancements, performance, questions.
2. **test_type**: `"ut"` for unit tests, `"e2e"` for end-to-end tests, `""` if unclear.
3. **dependency**: `"upstream"` if fix exists or is needed in pytorch/pytorch, `""` otherwise.
4. **platform**: `"xpu"` unless a specific GPU is mentioned (BMG, etc.).
5. **context**: Brief one-line summary with upstream links if available.
6. **Labels are authoritative** — if labels say `agent_test: ut`, test_type is `ut`.
7. **Do NOT hallucinate** — if info isn't in the issue, use `""`.
8. Keep the output small. Do NOT echo back the issue body.

## Issue-body status (backward compatible)

This is the first pipeline stage (legacy status `DISCOVERED`). The JSON above
was formerly consumed by a Python driver script; now the agent applies it
directly when building the issue body from the templates under
`.github/ISSUE_TEMPLATE/agent/` (`agent-issue-body.yml` for bug,
`agent-issue-body-nonbug.yml` for non-bug).

When you write the formatted body, this stage owns:
- the top status marker `<!-- agent:status:DISCOVERED -->`,
- the canonical section headings (`Description, Reproducer, Error Log,
  Environment` for bug; `Description, Objective, Current Status` for non-bug),
- the `<!-- agent:discovery-log -->` slot,
- checking the "Issue formatted" Action Item.

The overall stage sequencing is owned by `issue-handler/SKILL.md`.

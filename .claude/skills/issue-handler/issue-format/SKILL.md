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
error_log, environment, additional_context), use them directly. If not,
identify those portions yourself from the raw body before classifying.

## Classification

**Bug** — test failures, runtime errors, assertion errors, incorrect output, crashes.
Indicators: error tracebacks, failing test names, "RuntimeError", "AssertionError",
"fails with", `### 🐛 Describe the bug`, test logs.

**Non-bug** — feature requests, tasks, performance issues, questions, discussions,
tracking issues, enhancement proposals, feature gaps.
Indicators: "Enable", "[Task]", "Consider", "Align", "feature gap", "clarification",
checklists of work items, `enhancement` label, `performance` label, no failing tests.

**Labels are authoritative** — if labels say `agent_test: ut`, test_type is `ut`.

## Output

Return ONLY this JSON object, no markdown fences, no explanation:

```json
{
  "issue_type": "bug | nonbug",
  "test_type": "ut | e2e | \"\"",
  "dependency": "upstream | \"\"",
  "platform": "xpu | <specific GPU model>",
  "category": "<category string>",
  "related_components": "<components string>",
  "context": "<one-line summary with upstream links if available>",
  "formatted_body": "<pipeline mode only; empty string in interactive mode>"
}
```

- `test_type`: `"ut"` for unit tests, `"e2e"` for end-to-end, `""` if unclear.
- `dependency`: `"upstream"` if fix exists or is needed in pytorch/pytorch, `""` otherwise.
- `platform`: `"xpu"` unless a specific GPU is mentioned (e.g. BMG).
- Do NOT hallucinate — if info isn't in the issue, use `""`.

In pipeline mode, populate `formatted_body` using the templates under
`.github/ISSUE_TEMPLATE/agent/` (`agent-issue-body.yml` for bug,
`agent-issue-body-nonbug.yml` for non-bug).

## Pipeline mode: issue body

In interactive mode (default), do not touch the issue body. In pipeline mode,
this stage owns:
- the top status marker `<!-- agent:status:DISCOVERED -->`,
- canonical section headings,
- the `<!-- agent:discovery-log -->` slot,
- checking the "Issue formatted" Action Item.

See [../references/execution-modes.md](../references/execution-modes.md) for
the full pipeline mode contract.

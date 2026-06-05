---
name: issue-format
description: >
  Classify an issue and extract metadata.
  The format agent has already split the body into sections.
  The LLM classifies as bug/nonbug and extracts metadata
  (test_type, category, dependency, platform, context).
---

# Issue Format — Classify & Extract Metadata

## Your Task

Given a GitHub issue that has already been split into sections (description,
reproducer, error_log, environment, additional_context):

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
Return ONLY valid JSON, no markdown fences, no explanation:

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

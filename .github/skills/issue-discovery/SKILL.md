---
name: issue-discovery
description: >
  Format a raw issue into a structured template.
  Classifies as bug or non-bug, then extracts fields accordingly.
---

# Issue Discovery — Format Raw Issues

## Your Task
1. **Classify** the issue as `"bug"` or `"nonbug"`.
2. **Extract** structured fields based on the classification.
3. Return a JSON object with the extracted data.

## Classification Rules

**Bug** — test failures, runtime errors, assertion errors, incorrect output, crashes.
Indicators: error tracebacks, failing test names, "RuntimeError", "AssertionError",
"fails with", `### 🐛 Describe the bug`, test logs.

**Non-bug** — feature requests, tasks, performance issues, questions, discussions,
tracking issues, enhancement proposals, feature gaps.
Indicators: "Enable", "[Task]", "Consider", "Align", "feature gap", "clarification",
checklists of work items, "implement", `enhancement` label, `performance` label,
no failing tests, discussion-style content.

## Bug Output Format
Return JSON with these fields:
```json
{
  "issue_type": "bug",
  "summary": "one-paragraph description of the failure",
  "test_type": "ut | e2e | ...",
  "category": "category if identifiable",
  "dependency": "upstream | ... if applicable",
  "platform": "xpu | BMG | ...",
  "failed_tests": "- `test_file.py::TestClass::test_method`\n- ...",
  "error_log": "the COMPLETE error traceback verbatim — include full stack trace, do NOT truncate or summarize",
  "reproducer": "verbatim reproducer from the issue body",
  "commit_scope": "SHA hashes, compare links, or last pass/first fail",
  "context": "additional context: links, version info, notes"
}
```

## Non-Bug Output Format
Return JSON with these fields:
```json
{
  "issue_type": "nonbug",
  "summary": "one-paragraph description",
  "category": "enhancement | performance | feature-gap | task | question",
  "platform": "xpu | BMG | ...",
  "related_components": "kernels, modules, files involved",
  "objective": "what needs to be done",
  "current_status": "VERBATIM tables, checklists, and status indicators from the issue — do NOT summarize or rewrite",
  "context": "relevant background, links, discussion"
}
```

## Extraction Guidelines

1. **Extract verbatim** — preserve original formatting, URLs, code blocks.
2. **If a field is not found**, leave it as an empty string `""`.
3. **Do NOT hallucinate** — if info isn't in the issue, leave it empty.
4. **Failed Tests** — format as `- \`test_file.py::TestClass::test_method\`` (one per line).
5. **Error Log** — extract the COMPLETE error traceback/assertion VERBATIM from the issue.
   Include the full stack trace. Do NOT truncate, summarize, or shorten.
6. **Reproducer** — copy **verbatim**. Look for code blocks, "Reproducer", "How to reproduce".
   Do NOT fabricate. The reproducer could be a pytest command, a script, or bash commands.
7. **Labels are authoritative** — if labels say `agent_test: ut`, test_type is `ut`.
8. **Do NOT extract environment/versions** — handled programmatically.
9. **current_status** — copy markdown tables, checklists (🟢🟡❌), and progress notes
   VERBATIM from the issue body. Do NOT summarize tables into prose.
10. **For bug issues with `### 🐛 Describe the bug`** — the description IS the summary, the
   code blocks ARE the reproducer/error log. Parse them carefully.

## Output
Return ONLY valid JSON. No markdown fences, no explanation.

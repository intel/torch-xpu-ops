---
name: issue-discovery
description: >
  Format a raw CI failure issue into a structured template.
  Used by the discovery agent when a raw issue has labels
  (agent_test, agent_category, agent_dependency) but the body
  is sparse or unstructured.
---

# Issue Discovery — Format Raw Issues

## Your Task
1. Read the template file at `.github/ISSUE_TEMPLATE/agent-issue-body.yml`.
2. Identify all `{placeholder}` fields in the template's `body` value.
3. Given the raw issue body + labels, **search** for a value for each placeholder.
4. Return a JSON object with one key per placeholder.

## Search Strategy

Issues may range from well-structured (reproducer, logs, context) to minimal
(just a title with no body). Handle the worst case gracefully:

1. **Search** the issue body for each piece of information.
2. **If found**, extract it verbatim (preserve formatting, URLs, code blocks).
3. **If not found**, leave the field as an empty string `""`.
4. The triage agent will fill in blanks later — do NOT hallucinate content.

## Extraction Guidelines

1. **Failed Tests** — look for test names in backticks, bullet lists, or log output.
   Format as `- \`test_file.py::TestClass::test_method\`` (one per line).
   If none found, leave empty.

2. **Error Log** — extract the actual error traceback/assertion.
   Truncate to last ~50 lines. Remove CI infrastructure noise.
   If none found, leave empty.

3. **Reproducer** — copy the reproducer **verbatim** from the issue body.
   Look for code blocks with `python` invocations, bash commands, or sections titled
   "Reproducer" / "How to reproduce". Do NOT rewrite or simplify.
   If no reproducer exists, leave empty (do NOT fabricate one).

4. **Context** — copy relevant additional context verbatim: upstream PR/issue links,
   commit references, version info, env details. Preserve original URLs and formatting.

5. **Commit Scope** — look for SHA hashes, compare links, or "last pass / first fail" tables.

6. **Labels are authoritative** — if labels say `agent_test: ut`, the test_type is `ut`
   regardless of what the body says.

7. **Do NOT extract environment/versions** — that is handled programmatically, not by the LLM.

## Output
Return ONLY valid JSON matching the template placeholders. No markdown fences, no explanation.

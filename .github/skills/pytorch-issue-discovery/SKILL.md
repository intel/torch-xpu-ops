---
name: pytorch-issue-discovery
description: >
  Format a raw CI failure issue into a structured template.
  Used by the discovery agent to enrich sparse issues with
  reproducer, error log, test info, and context.
---

# Issue Discovery — Format Raw CI Failures

## When to Use
When a raw issue has labels (agent_test, agent_category, agent_dependency)
but the body is sparse or unstructured.

## Your Task
Given a raw issue body + labels, extract and return structured JSON with these fields:

```json
{
  "summary": "one-line description of the failure",
  "test_type": "ut or e2e",
  "category": "from agent_category label (Torch Ops, Inductor, Distributed, etc.)",
  "dependency": "from agent_dependency label (upstream-pytorch, oneDNN, triton, etc.)",
  "platform": "PVC, ATS-M, DG2, BMG, etc. if mentioned",
  "failed_tests": "list of failing tests, one per line, backtick-wrapped",
  "error_log": "relevant error output (last ~50 lines)",
  "reproducer": "commands to reproduce the failure",
  "commit_scope": "commit range if mentioned (last pass SHA → first fail SHA)",
  "context": "any additional context useful for triage"
}
```

## Extraction Rules

1. **Failed Tests** — look for test names in backticks, bullet lists, or log output.
   Format as `- \`test_file.py::TestClass::test_method\`` (one per line).

2. **Error Log** — extract the actual error traceback/assertion.
   Truncate to last ~50 lines. Remove CI infrastructure noise.

3. **Reproducer** — copy the reproducer **verbatim** from the issue body.
   Look for code blocks with `python` invocations, bash commands, or sections titled
   "Reproducer" / "How to reproduce". Do NOT rewrite or simplify.
   If no reproducer exists at all, construct from the test name:
   ```bash
   python -m pytest test/<test_file>.py -k <test_name> -x
   ```

4. **Context** — copy relevant additional context verbatim: upstream PR/issue links,
   commit references, version info, env details. Preserve original URLs and formatting.

5. **Commit Scope** — look for SHA hashes, compare links, or "last pass / first fail" tables.

6. **Labels are authoritative** — if labels say `agent_test: ut`, the test_type is `ut`
   regardless of what the body says.

## Output
Return ONLY valid JSON. No markdown fences, no explanation.

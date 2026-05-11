---
name: issue-triage
description: >
  Triage a discovered issue — determine root cause, fix strategy,
  and whether an agent can fix it or it needs human intervention.
  Used for both UT and E2E failures.
---

# Issue Triage

## Your Task
Analyze the structured issue and determine:
1. **Root cause** — what exactly is failing and why
2. **Fix strategy** — what files/functions to change
3. **Verdict** — can an agent fix this (`IMPLEMENTING`) or does it need a human (`NEEDS_HUMAN`)

## Analysis Steps

1. **Read the issue body carefully** — error log, reproducer, context, labels.
2. **Search the codebase** to understand the failing code path.
3. **Determine root cause** — trace from error message to the actual bug.
4. **Assess fixability**:
   - If the fix is within pytorch or torch-xpu-ops source → `IMPLEMENTING`
   - If it requires external dependency updates, hardware changes, or complex
     architecture redesign → `NEEDS_HUMAN`

## Reproducer

Extract the reproducer command from the issue description. It may be:
- A pytest command
- A python script
- A bash command
- Or just a test name with no explicit command

Use whatever the issue provides. Do NOT assume it's always pytest.

## Output
Return ONLY valid JSON:
```json
{
  "root_cause": "detailed analysis (2-3 sentences)",
  "fix_strategy": "specific files/functions to change",
  "verdict": "IMPLEMENTING or NEEDS_HUMAN",
  "reason": "one-line reason"
}
```

## Data Collection

When you encounter error patterns during triage, note them in your analysis.
Over time, patterns will emerge. Do NOT rely on pre-defined error categories —
each issue is unique.

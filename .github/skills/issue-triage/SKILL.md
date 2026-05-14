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
2. **Quick classification (BEFORE deep analysis):**
   - If labeled as `[Task]`, `[Feature]`, or describes broad alignment/enablement work → immediately return `NEEDS_HUMAN` with reason "Umbrella/task issue, not a single fixable bug"
   - If it describes a "feature gap" or "blocked by missing feature" → `NEEDS_HUMAN`
   - If category is "performance" and there's no specific failing test → `NEEDS_HUMAN` with reason "Performance optimization requires human design decision"
   - If it has a clear error message/stack trace → proceed to step 3
3. **Search the codebase** to understand the failing code path. **Limit to 3 file reads** — if you haven't found the root cause after reading 3 files, output your best analysis with what you have.
4. **Determine root cause** — trace from error message to the actual bug.
5. **Assess fixability**:
   - If the fix is within pytorch or torch-xpu-ops source → `IMPLEMENTING`
   - If it requires external dependency updates, hardware changes, or complex
     architecture redesign → `NEEDS_HUMAN`

**TIME BUDGET: You MUST output your JSON within 5 minutes. If unsure, output IMPLEMENTING with your best guess — don't keep searching.**

**IMPORTANT: Do NOT use the `task` tool or spawn subagents. Do all analysis yourself with `read` and `grep` only. Subagents hang in large repos.**

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
  "target_repo": "pytorch or torch-xpu-ops",
  "verdict": "IMPLEMENTING or NEEDS_HUMAN",
  "reason": "one-line reason"
}
```

### target_repo rules:
- `"torch-xpu-ops"` — if the fix is in `src/ATen/native/xpu/sycl/`, `src/ATen/native/xpu/`, or any path relative to torch-xpu-ops root. **This includes files under `third_party/torch-xpu-ops/` in the pytorch tree** — those are torch-xpu-ops source files bundled as a submodule; fixes belong in torch-xpu-ops, not pytorch.
- `"pytorch"` — if the fix is in `torch/`, `aten/src/ATen/`, `test/`, `c10/`, `torch/_dynamo/`, `torch/_inductor/`, or any top-level pytorch path (but NOT `third_party/torch-xpu-ops/`)

## Data Collection

When you encounter error patterns during triage, note them in your analysis.
Over time, patterns will emerge. Do NOT rely on pre-defined error categories —
each issue is unique.

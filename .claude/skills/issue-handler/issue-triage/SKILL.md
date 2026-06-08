---
name: issue-triage
description: >
  Triage a single GitHub issue: determine root cause, fix strategy, and a
  verdict on whether an agent can implement the fix (IMPLEMENTING) or it
  needs human intervention (NEEDS_HUMAN). Works for both UT and E2E failures.
---

# Issue Triage

## Inputs
You receive one GitHub issue — structured or raw — including its error log,
reproducer (if any), surrounding context, and labels. You also have read-only
access to the codebase (`read` and `grep`). You analyze; you do not execute.

## Your Task
Analyze the issue and determine:
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
3. **Identify what changed** — if the issue describes a regression (worked on version X, broke on version Y), always ask: *which component changed between those versions?* The root cause belongs to **the thing that changed**, not just where the error happens to fire. If an external library broke because PyTorch changed its behavior, the fix lives in PyTorch — not in the external library.
4. **Search the codebase** to trace the failing code path. Stop when you have enough to make a call, not after counting files.
5. **Determine root cause** — trace from error message to the actual bug. You cannot execute code, so you must exhaust static analysis. Read the full call chain. If the root cause and a specific fix are identifiable from reading code alone, output `IMPLEMENTING` — even without running the reproducer. Only conclude "needs hardware to reproduce" if static analysis genuinely cannot determine what's wrong.
6. **Skip/xfail decorators are NOT fixes:**
   - If the issue describes tests with `@skipIfXpu`, `@xfailIfXpu`, or similar skip decorators, and the issue wants to remove them and make the tests actually pass on XPU — the presence of those decorators in the codebase **confirms the issue EXISTS**. The decorator IS the problem, not a fix.
   - Do NOT conclude "already fixed" just because skip decorators exist. The goal is to remove the skip and fix the underlying failure.
7. **Assess fixability**:
   - If the fix is within pytorch or torch-xpu-ops source → `IMPLEMENTING`
   - If it requires hardware changes, complex architecture redesign, or the root cause is genuinely unresolvable without running code → `NEEDS_HUMAN`

## Guardrails (keep triage cheap)

- **Stay time-boxed.** Aim to produce your JSON within ~5 minutes. If unsure, output `IMPLEMENTING` with your best guess rather than searching indefinitely.
- **Do the analysis yourself.** Do NOT use the `task` tool or spawn subagents — use `read` and `grep` only. Subagents hang in large repos.

## Reproducer

Extract the reproducer command from the issue description. It may be:
- A pytest command
- A python script
- A bash command
- Or just a test name with no explicit command

You cannot run the reproducer (no execution environment). Use it to understand the code path, not to verify the failure.

## Output
Return ONLY this JSON:
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

## Failure Categories

Use these categories to classify the root cause. They help structure the fix strategy:

| Category | Description | Typical fix location |
|----------|-------------|---------------------|
| **XPU backend bug** | Bug in XPU kernel or backend code | `torch/_inductor/` or `third_party/torch-xpu-ops/` |
| **Tolerance too tight** | Numerical precision mismatch vs CUDA | Adjust `atol`/`rtol` to match CUDA tolerances |
| **Skip decorator stale** | `@skipIfXpu` / `@expectedFailure` but test now passes | Remove decorator (see `issue-fix` skill) |
| **Upstream regression** | New upstream code broke XPU; needs XPU-specific workaround | `torch/`, `aten/`, `test/` |
| **Test infrastructure** | Environment, import, or setup issue | Test file or CI config |

When the issue describes a newly added test, check the commit/PR that introduced it to see if XPU support is expected — this affects the fix strategy.

## See also

For XPU kernel/operator bugs and CUDA-UT-porting specifics, consult the
`xpu-issues-triaging` skill for deeper triage guidance.

## Data Collection

When you encounter error patterns during triage, note them in your analysis.
Over time, patterns will emerge. These categories are starting points, not
exhaustive — each issue is unique.

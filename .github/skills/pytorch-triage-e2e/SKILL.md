---
name: pytorch-triage-e2e
description: >
  Triage an end-to-end test CI failure — determine root cause and whether
  an agent can fix it or it needs human intervention.
---

# End-to-End Test Triage

## When to Use
When triaging an issue with `agent_test: e2e` label.

## Your Task
Same as UT triage but with E2E-specific considerations.

## Key Differences from UT Triage

1. **E2E tests run full model workloads** — failures may be in:
   - Model compilation (Inductor/Triton)
   - Multi-operator interaction
   - Memory management
   - Distributed communication

2. **Reproduction is harder** — E2E tests may require:
   - Specific model downloads
   - Large memory
   - Multiple GPUs (for distributed tests)
   - Specific dataset availability

3. **Root causes tend to be more complex**:
   - Inductor codegen for XPU
   - Triton kernel compilation failures
   - Memory fragmentation under load
   - Collective communication (XCCL) issues

## E2E-Specific Fix Patterns

| Error Pattern | Root Cause | Fix Location |
|---|---|---|
| `Compilation failed` | Inductor codegen issue | `torch/_inductor/codegen/` |
| `triton.CompilationError` | Triton XPU backend | Usually NEEDS_HUMAN (triton repo) |
| `OutOfMemoryError` | Memory management | May need operator optimization |
| `NCCL/XCCL error` | Collective comm failure | Usually NEEDS_HUMAN (xccl repo) |
| `torch.compile` failure | Graph break or unsupported op | `torch/_dynamo/` or op implementation |
| Model accuracy regression | Numerical issue in op chain | Individual op precision fix |

## Verdict Guidelines
- **IMPLEMENTING** — Inductor codegen fix, missing op fallback, test config issue
- **NEEDS_HUMAN** — Triton backend issue, XCCL issue, complex memory issue, model-specific regression

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

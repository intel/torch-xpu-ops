---
name: pytorch-triage-ut
description: >
  Triage a unit test CI failure — determine root cause and whether
  an agent can fix it or it needs human intervention.
---

# Unit Test Triage

## When to Use
When triaging an issue with `agent_test: ut` label.

## Your Task
Analyze the structured issue and determine:
1. **Root cause** — what exactly is failing and why
2. **Verdict** — can an agent fix this (`IMPLEMENTING`) or does it need a human (`NEEDS_HUMAN`)

## Analysis Steps

1. **Read the error log** — identify the exception type and message:
   - `RuntimeError: ... not implemented for 'XPU'` → missing XPU kernel dispatch
   - `AssertionError: ... not close enough` → tolerance/precision issue
   - `AttributeError: ... has no attribute` → API change in upstream pytorch
   - `ImportError` / `ModuleNotFoundError` → missing dependency
   - `SYCL error` / `PI_ERROR` → driver/runtime issue

2. **Check the dependency label**:
   - `upstream-pytorch` → likely fixable in pytorch/pytorch
   - `oneDNN` → needs oneDNN update, usually NEEDS_HUMAN
   - `triton` → needs triton-xpu update, usually NEEDS_HUMAN
   - `driver` / `oneAPI` → environment issue, NEEDS_HUMAN
   - `CPU fallback` → workaround exists but may need proper XPU impl

3. **Determine fixability**:
   - **IMPLEMENTING** — missing XPU dispatch, tolerance fix, dtype support, test adaptation
   - **NEEDS_HUMAN** — third-party dependency, hardware issue, complex architecture change

## Common Fix Patterns (UT)

| Error Pattern | Root Cause | Fix Location |
|---|---|---|
| `not implemented for 'XPU'` | Missing kernel dispatch | `aten/src/ATen/xpu/` or `torch/xpu/` |
| `not close enough` / tolerance | Precision difference | Adjust `atol`/`rtol` in test or fix kernel |
| `expected XPU but got CPU` | Missing device propagation | Op implementation |
| `XFAIL` removed upstream | Test was expected to fail, now needs fixing | Test file or kernel |
| `shape mismatch` | Kernel output shape bug | Kernel implementation |

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

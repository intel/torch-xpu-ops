---
name: pytorch-triage
description: Triage ai_generated issues — decide if fix belongs in pytorch/pytorch or intel/torch-xpu-ops
---

# PyTorch Issue Triage

## When to Use
When analyzing an `ai_generated` issue from `intel/torch-xpu-ops` to decide where the fix belongs.

## Decision Criteria

### Fix belongs in **pytorch/pytorch** if:
- The issue is in core PyTorch code (torch/, aten/, c10/)
- The bug is in a device-agnostic path that affects XPU
- The stack trace points to pytorch source files
- The fix would benefit all backends, not just XPU

### Fix belongs in **intel/torch-xpu-ops** (skip) if:
- The issue is XPU-specific kernel code
- The fix is in torch-xpu-ops' own registration or kernels
- The issue is a configuration/build problem specific to XPU toolchain
- The issue is about missing XPU op coverage (needs new kernel, not pytorch fix)

## Output Format
Respond with EXACTLY one line:
```
VERDICT: pytorch — <one-sentence reason>
```
or
```
VERDICT: skip — <one-sentence reason>
```

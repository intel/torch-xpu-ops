# Failure Categories — Domain Routing Guide

Used by `fix/triage` to select the right domain skill. Load the indicated
domain skill after triage returns its verdict.

| Failure type | Domain skill to load |
|---|---|
| Bug in XPU kernel, operator, or dispatch code | `fix/domains/xpu-kernel` |
| Test ported from CUDA fails due to porting gaps | `fix/domains/cuda-porting` |
| Bug in device-agnostic PyTorch framework code | `fix/domains/upstream-pytorch` (when available) |

For the detailed category taxonomy within each domain (e.g. tolerance issue,
stale skip, upstream regression), see the loaded domain skill.

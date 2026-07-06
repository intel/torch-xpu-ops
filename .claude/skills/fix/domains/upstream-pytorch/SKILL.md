---
name: fix/domains/upstream-pytorch
description: >
  Domain knowledge pack for device-agnostic PyTorch framework bugs that surface
  on XPU. Loaded by orchestrators after fix/triage returns
  domain=upstream-pytorch. Not loaded directly by users.
---

# Domain: Upstream PyTorch Framework

- `target_repo` must be `"pytorch"`. Never `third_party/torch-xpu-ops/`.
- Never stage `third_party/xpu.txt`.

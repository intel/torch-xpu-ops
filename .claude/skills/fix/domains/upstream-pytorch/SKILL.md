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

## Common signatures that belong here

Recurring failure patterns whose root cause is in device-agnostic pytorch
code, not in any backend kernel. Match on signature; if the failure fits,
route to this domain and stop searching the backend kernel.

- **`RuntimeError: Expected X.is_contiguous(memory_format)` under vmap /
  functorch.** Origin:
  `aten/src/ATen/native/*.cpp` (`group_norm.cpp`, `layer_norm.cpp`,
  similar norms) has `TORCH_CHECK(X.is_contiguous(memory_format))` where
  `memory_format` is `at::MemoryFormat::Contiguous` on non-CPU devices
  and `X.suggest_memory_format()` on CPU. vmap produces batched inputs
  that are not contiguous in `Contiguous` layout, so any non-CPU device
  trips the check. Fix location: relax the check in pytorch (allow
  vmap-batched inputs), add a batching rule, or add a decomposition.
  **Not fixable in the backend kernel** — the kernel never sees the
  tensor because the check fires in the framework wrapper.

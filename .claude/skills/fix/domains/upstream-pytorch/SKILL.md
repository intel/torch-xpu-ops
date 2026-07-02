---
name: fix/domains/upstream-pytorch
description: >
  Domain knowledge pack for device-agnostic PyTorch framework bugs that surface
  on XPU. Loaded by orchestrators after fix/triage returns
  domain=upstream-pytorch. Covers target repo path conventions, upstream
  cross-reference, and test path resolution for core PyTorch code. Not loaded
  directly by users.
---

# Domain: Upstream PyTorch Framework

Loaded by the orchestrator after `fix/triage` returns
`"domain": "upstream-pytorch"`. Applies when the root cause is in
device-agnostic framework code (e.g. `torch/`, `aten/src/ATen/native/`,
`c10/`, `torch/_dynamo/`, `torch/_inductor/`) rather than in XPU backend code.

## Environment and build

Load `xpu-build-pytorch` for the full env/build workflow (oneAPI sourcing,
build command, arch flags, rebuild pitfalls).

## Target repo path conventions

- **Fix location:** top-level pytorch paths — `torch/`, `aten/src/ATen/`,
  `c10/`, `test/`, `torch/_dynamo/`, `torch/_inductor/`, etc.
- **Not here:** `third_party/torch-xpu-ops/`. If the root cause is in that
  subtree, the domain should be `xpu-kernel`, not `upstream-pytorch`.
- `target_repo` must be `"pytorch"`.

Never stage `third_party/xpu.txt`.

## Upstream cross-reference

The fix is in pytorch itself. Use `git log`, `git blame`, and `gh` search to
find the relevant commit or existing fix:

```bash
# Check recent changes to the affected file
git -C agent_space_xpu/pytorch log --oneline -20 -- <file>

# Search for related fixes already merged
gh search commits --repo pytorch/pytorch "<keyword>" --limit 10
```

If the bug is already fixed on `origin/main`, rebase rather than duplicating
the fix:

```bash
git -C agent_space_xpu/pytorch rebase origin/main
```

## Failure categories

| Category | Description | Typical fix location |
|---|---|---|
| **Framework regression** | A pytorch core change broke XPU (or all backends) | `torch/`, `aten/src/ATen/native/`, `c10/` |
| **Inductor / Dynamo** | Compiler or graph-capture issue surfacing on XPU | `torch/_inductor/`, `torch/_dynamo/` |
| **Test infrastructure** | Incorrect device-type assumptions in test utilities | `torch/testing/`, `test/` |

## Skip decorators

For all skip decorator operations (finding, removing, adding), load
`fix/pytorch-skip`.

## Test path resolution

Working directory is `$PYTORCH_DIR/`. Upstream pytorch test paths are used
as-is (e.g. `test/test_torch.py`, `test/nn/test_embedding.py`). No submodule
remapping needed.

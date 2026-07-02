---
name: fix/domains/cuda-porting
description: >
  Domain knowledge pack for CUDA unit test porting issues on XPU. Loaded by
  orchestrators after fix/triage returns domain=cuda-porting. Covers how to
  locate the original CUDA test, construct an XPU reproducer, diff CUDA vs XPU
  behavior, and determine root cause. Not loaded directly by users.
---

# Domain: CUDA UT Porting

Loaded by the orchestrator after `fix/triage` returns `"domain": "cuda-porting"`.
Applies when a test ported from CUDA to XPU fails due to porting gaps.

## Environment and build

Load `xpu-build-pytorch` for the full env/build workflow.

## Analysis steps

### Step 1: Locate the original CUDA test

Find the upstream CUDA test in the PyTorch repo that was ported to XPU.
Reuse the checkout in `agent_space_xpu/pytorch/` if it already exists:

```bash
if [[ ! -d agent_space_xpu/pytorch/.git ]]; then
    git clone --depth 1 https://github.com/pytorch/pytorch.git agent_space_xpu/pytorch
fi
```

Search for the test by name under `agent_space_xpu/pytorch/test/`. Identify any
CUDA-specific decorators, assertions, tolerances, dtypes, or device
assumptions in the original.

### Step 2: Sketch an XPU reproducer

Mirror the CUDA test's logic but target `device="xpu"`. Keep it minimal —
just enough to reproduce the failure.

### Step 3: Diff CUDA vs XPU behavior

Compare across these axes:

| Axis | What to check |
|---|---|
| **dtypes** | Does the CUDA test use dtypes not supported on XPU? |
| **Tolerances** | Does `atol`/`rtol` need adjustment for XPU numerical behavior? |
| **Dispatch path** | Does the XPU kernel exist and dispatch correctly? |
| **Device decorators** | Are there `@skipIf` or `@onlyCUDA` decorators in the original that were incorrectly ported? |

### Step 4: Determine root cause

| Root cause | Fix location |
|---|---|
| Missing XPU kernel | `src/ATen/native/xpu/sycl/` in torch-xpu-ops |
| Incorrect test tolerance | Test file |
| Wrong device assumption in ported test | Test file |
| Genuine behavioral difference (expected) | Document; load `fix/pytorch-skip` to add skip with tracking issue |

## Skip decorators

For all skip decorator operations (finding, removing, adding), load
`fix/pytorch-skip`.

## Target repo path conventions

Same as `fix/domains/xpu-kernel` — torch-xpu-ops paths under
`third_party/torch-xpu-ops/`; pytorch framework paths under `torch/`,
`aten/`, `test/`.

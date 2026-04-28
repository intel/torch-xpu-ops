---
name: xpu-backend-scan
description: Determine whether a PyTorch operator has correct XPU support — analyze dispatch coverage, CUDA-XPU parity, fallback implications, waiver applicability, and missing implementation evidence.
---

# XPU Backend Scan

Determine whether a PyTorch operator has correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

Scope: all XPU-related code in both `pytorch/pytorch` (upstream XPU backend, dispatch, codegen, tests) and `intel/torch-xpu-ops` (XPU kernels, SYCL implementations, fallback, backend YAML).

References:
- [references/dispatch-coverage.md](references/dispatch-coverage.md) — how to determine XPU coverage and where to look
- [references/triage-patterns.md](references/triage-patterns.md) — common false positives and decision heuristics

## Workflow

### Step 1: Identify the operator surface

Pin the exact schema/overload from `native_functions.yaml` (torch-xpu-ops `yaml/native/` or upstream `aten/src/ATen/native/`). Do not compare by base op name or filename alone.

### Step 2: Check waivers

Use the inline waiver categories in this step as the waiver authority. If the op belongs to a known non-actionable category (NVIDIA-specific infra like cuDNN/cuBLAS/NCCL, hardware-only features like FP8/Tensor Core, documented unsupported families like flash/efficient attention, or vendor-specific backends like Triton/MIOpen), stop — it is not an XPU bug.

### Step 3: Determine existing XPU coverage

Check in this order; if any source confirms coverage, do not conclude "missing implementation":
1. `XPUFallback.template` — explicit per-op fallback registration is callable coverage. The global backend fallback only becomes callable when `PYTORCH_ENABLE_XPU_FALLBACK=1`; otherwise the default path errors with not-implemented. CPU fallback on a GPU op may still be a defect.
2. Backend YAML with explicit XPU dispatch keys
3. Structured delegate or codegen path resolving to XPU
4. Source-backed registration (`TORCH_IMPL_FUNC`, landed implementation)
5. Composite (`CompositeImplicitAutograd`/`CompositeExplicitAutograd`) or decomposition

### Step 4: Classify the finding

- **XPU defect** — intrinsic problem in XPU code (broken dispatch, silent CPU fallback, missing validation, backward gap, race condition)
- **Parity gap** — both CUDA and XPU have coverage, but user-visible contract differs (input space, parameter semantics, dtype support, backward behavior, error paths)
- **Missing implementation** — CUDA has usable support, XPU has no callable path after all exclusions checked
- **Waived** — op belongs to a known non-actionable category identified in Step 2
- **OK** — coverage exists via any valid path; differences are stylistic or optimization-only
- **Needs review** — mixed evidence, cannot conclude without runtime validation

### Step 5: Validate before concluding

- Require user-visible contract difference, not just implementation shape difference.
- Read helper definitions before comparing call sites.
- Distinguish family-level truth from row-level truth.
- Do not call a finding runtime-confirmed from static review alone.

## Hard Rules

- If CUDA has a feature that XPU lacks, treat as XPU bug unless hardware-justified.
- If XPU silently falls back to CPU, treat as XPU defect unless explicitly waived.
- SYCL vs CUDA style differences are not bugs.
- Vendor library choice (oneDNN vs cuDNN) is not itself a bug.
- Missing local XPU kernel file is not evidence of missing support — check delegates, composites, shared paths first.
- Explicit per-op fallback, or global fallback with `PYTORCH_ENABLE_XPU_FALLBACK=1`, blocks "missing implementation" but may indicate a defect (CPU fallback on GPU op).
- Test skip/xfail metadata is supporting evidence only, never sufficient alone.

## Output

Return findings grounded in inspected code:
- Exact schema/overload under review
- Verdict with confidence level
- XPU-side evidence (files, code paths)
- Peer evidence (CUDA, upstream, shared paths)
- Exclusion checks performed (waiver, fallback, composite, delegate)
- Next action (stop, downgrade, hand off for repro)

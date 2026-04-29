---
name: xpu-backend-scan
description: Analyze whether a PyTorch operator has correct XPU support. Inspects dispatch coverage, CUDA-XPU parity, fallback status, and missing implementation evidence. Reports all findings with priority — does not dismiss or waive anything.
---

# XPU Backend Scan

Analyze whether a PyTorch operator has correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

Scope: all XPU-related code in both `pytorch/pytorch` (upstream XPU backend, dispatch, codegen, tests) and `intel/torch-xpu-ops` (XPU kernels, SYCL implementations, fallback, backend YAML).

References:
- [references/dispatch-coverage.md](references/dispatch-coverage.md) — how to determine XPU coverage and where to look
- [references/triage-patterns.md](references/triage-patterns.md) — pattern context for prioritization

## Input / Output

**Input**: One or more operator names, schemas, or overloads to analyze.

**Output**: Per-operator finding with:
- Verdict and priority (high / medium / low)
- Evidence from inspected code
- All coverage signals found (do not stop at the first one)

**Important**: Report everything. Do not dismiss or waive any finding. Assign priority instead.

## Workflow

### Step 1: Identify the operator surface

Pin the exact schema/overload from `native_functions.yaml` (torch-xpu-ops `yaml/native/` or upstream `aten/src/ATen/native/`). Do not compare by base op name or filename alone.

### Step 2: Collect ALL XPU coverage signals

Check every source below and record what you find. Do not stop at the first positive signal — collect all of them:
1. Backend YAML with explicit XPU dispatch keys (native XPU path)
2. Source-backed registration (`TORCH_IMPL_FUNC`, landed implementation in `src/ATen/native/xpu/sycl/`)
3. Structured delegate or codegen path resolving to XPU
4. Composite (`CompositeImplicitAutograd`/`CompositeExplicitAutograd`) or decomposition
5. `XPUFallback.template` — explicit per-op fallback (CPU fallback, callable but not native XPU)

See [references/dispatch-coverage.md](references/dispatch-coverage.md) for where to look and how to interpret each signal.

### Step 3: Classify the finding

- **XPU defect** (high) — intrinsic problem in XPU code: broken dispatch, silent CPU fallback, missing validation, backward gap, race condition
- **Parity gap** (high) — both CUDA and XPU have coverage, but user-visible contract differs: input space, parameter semantics, dtype support, backward behavior, error paths
- **Missing native implementation** (high) — CUDA has usable support, XPU has no native path (no backend YAML, no source-backed kernel, no composite/decomp)
- **Fallback only** (low) — op is callable only via CPU fallback in `XPUFallback.template`; XPU lacks a native GPU implementation
- **Needs review** (medium) — mixed evidence, cannot conclude without runtime validation

### Step 4: Validate before concluding

- Require user-visible contract difference, not just implementation shape difference.
- Read helper definitions before comparing call sites.
- Distinguish family-level truth from row-level truth.
- Do not call a finding runtime-confirmed from static review alone.

## Hard Rules

- If CUDA has a feature that XPU lacks, report it.
- If XPU silently falls back to CPU, report as XPU defect.
- If XPU only has CPU fallback coverage, report as "fallback only" (low priority).
- SYCL vs CUDA style differences are not bugs.
- Vendor library choice (oneDNN vs cuDNN) is not itself a bug.
- Missing local XPU kernel file is not evidence of missing support — check delegates, composites, shared paths first.
- Test skip/xfail metadata is a signal worth reporting, not dismissing.

## Output

Return findings grounded in inspected code:
- Exact schema/overload under review
- Verdict with priority (high / medium / low)
- All coverage signals found (list each one)
- XPU-side evidence (files, code paths)
- Peer evidence (CUDA, upstream, shared paths)
- Next action (hand off for repro, needs runtime check, or informational only)

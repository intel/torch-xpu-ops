---
name: xpu-backend-scan
description: Analyze XPU operator coverage, parity, and defects. Supports single-op analysis, full scan of all XPU operators, and daily incremental scan. Use when asked to scan all XPU ops, run a daily XPU scan, or analyze specific operator support.
---

# XPU Backend Scan

Analyze whether PyTorch operators have correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

Scope: all XPU-related code in both `pytorch/pytorch` (upstream XPU backend, dispatch, codegen, tests) and `intel/torch-xpu-ops` (XPU kernels, SYCL implementations, fallback, backend YAML).

References:
- [references/dispatch-coverage.md](references/dispatch-coverage.md) — how to determine XPU coverage and where to look
- [references/triage-patterns.md](references/triage-patterns.md) — pattern context for prioritization
- [references/batch-scan-workflow.md](references/batch-scan-workflow.md) — full scan, daily scan, output format, auto-resume

## Input / Output

**Single op**: operator name/schema → per-operator finding (verdict, priority, evidence, next action)

**Full scan** (`scan all`): → `xpu_scan_full_<date>_<time>.json` + `.md`. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

**Daily scan** (`scan daily` or `scan since <date>`): → `xpu_scan_daily_<date>_<time>.json` + `.md`. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

Report everything. Do not dismiss or waive any finding. Assign priority instead.

## Examples

```
Single op:   "Analyze XPU support for aten::addmm"
Full scan:   "scan all"
Daily scan:  "scan daily" or "scan since 2026-04-28"
Resume:      "resume scan xpu_scan_full_2026-04-29_143052.json"
```

## Per-Operator Analysis

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

### Step 3: Classify and validate

Classify based on user-visible impact, not implementation shape:

- **XPU defect** (high) — broken dispatch, silent CPU fallback, missing validation, backward gap, race condition
- **Parity gap** (high) — user-visible contract differs between CUDA and XPU: input space, dtype, backward, error paths
- **Missing native implementation** (high) — CUDA has support, XPU has no native path
- **Fallback only** (low) — callable only via CPU fallback in `XPUFallback.template`
- **Needs review** (medium) — mixed evidence, needs runtime validation

Before concluding: read helper definitions before comparing call sites. Do not call a finding runtime-confirmed from static review alone. See [references/triage-patterns.md](references/triage-patterns.md) for non-obvious patterns that need extra context.

## Hard Rules

- If CUDA has a feature that XPU lacks, report it.
- If XPU silently falls back to CPU, report as XPU defect.
- If XPU only has CPU fallback coverage, report as "fallback only" (low priority).
- SYCL vs CUDA style differences are not bugs.
- In batch scans: if a single op fails, record `status: error` and continue.

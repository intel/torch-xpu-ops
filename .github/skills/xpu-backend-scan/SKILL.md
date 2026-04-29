---
name: xpu-backend-scan
description: Analyze XPU operator coverage, parity, and defects. Supports single-op analysis, full scan of all XPU operators, and daily incremental scan. Use when asked to scan XPU ops, check if XPU supports an operator, find missing XPU operators, find XPU bugs, or run a daily/full XPU scan.
---

# XPU Backend Scan

Analyze whether PyTorch operators have correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

Scope: all XPU-related code in both `pytorch/pytorch` and `intel/torch-xpu-ops`.

Detailed reference:
- [references/batch-scan-workflow.md](references/batch-scan-workflow.md) — full scan, daily scan, output format, auto-resume
- [references/pitfalls.md](references/pitfalls.md) — non-obvious judgment traps and fallback mechanisms

## Input / Output

**Single op**: operator name/schema → per-operator finding (verdict, priority, evidence, next action)

**Full scan** (`scan all`): → `xpu_scan_full_<date>_<time>.json` + `.md`. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

**Daily scan** (`scan daily` or `scan since <date>`): → `xpu_scan_daily_<date>_<time>.json` + `.md`. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

Report everything. Do not dismiss or waive any finding. Assign priority instead.

## Workflow

### Step 1: Identify the operator surface
Pin the exact schema/overload from `native_functions.yaml`. Do not compare by base op name or filename alone.

### Step 2: Collect ALL XPU coverage signals
Check every source and record what you find. Do not stop at the first positive signal — collect all of them.

Where to look (intel/torch-xpu-ops):
1. `yaml/native/native_functions.yaml` — dispatch key configuration
2. `src/ATen/native/xpu/XPUFallback.template` — fallback list
3. `src/ATen/native/xpu/` and `src/ATen/native/xpu/sycl/` — kernel implementations
4. `yaml/xpu_functions.yaml` — auxiliary metadata

Where to look (pytorch/pytorch):
1. `aten/src/ATen/native/native_functions.yaml` — authoritative dispatch keys
2. `aten/src/ATen/native/xpu/` — upstream XPU native code
3. `aten/src/ATen/native/cuda/` — CUDA peer implementations
4. `tools/autograd/derivatives.yaml` — backward formulas
5. `torch/_decomp/` and `torch/_refs/` — decomposition registration
6. `test/xpu/` — upstream XPU tests (skip/xfail annotations signal known gaps)

### Step 3: Classify and validate
Classify based on user-visible impact, not implementation shape:
- **XPU defect** (high) — broken dispatch, silent CPU fallback, missing validation, backward gap
- **Parity gap** (high) — user-visible contract differs between CUDA and XPU
- **Missing native implementation** (high) — CUDA has support, XPU has no path at all (not even fallback)
- **Fallback only** (low) — callable only via CPU fallback
- **Needs review** (medium) — mixed evidence, needs runtime validation

Read helper definitions before comparing call sites. Do not call a finding runtime-confirmed from static review alone.

## Hard Rules
- If CUDA has a feature that XPU lacks, report it.
- If XPU silently falls back to CPU, report as XPU defect.
- If XPU only has CPU fallback coverage, report as "fallback only" (low priority).
- SYCL vs CUDA style differences are not bugs.
- In batch scans: if a single op fails, record `status: error` and continue.

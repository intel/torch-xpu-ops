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

## Scan Modes

| Mode | Trigger | Description |
|------|---------|-------------|
| **Single op** | Operator name/schema | Analyze one or more specific operators (default) |
| **Full scan** | `scan all` | Deep-analyze ALL XPU operators across both repos |
| **Daily scan** | `scan daily` or `scan since <date>` | Deep-analyze ops affected by recent changes |

## Input / Output

**Mode A — Single op analysis**

- **Input**: One or more operator names, schemas, or overloads to analyze.
- **Output**: Per-operator finding with:
  - Exact schema/overload under review
  - Verdict with priority (high / medium / low)
  - All coverage signals found (list each one)
  - XPU-side evidence (files, code paths)
  - Peer evidence (CUDA, upstream, shared paths)
  - Next action (hand off for repro, needs runtime check, or informational only)

**Mode B — Full scan**

- **Input**: `scan all` (optionally specify output directory)
- **Output**: `xpu_scan_full_<date>_<time>.json` + `.md` — deep analysis of every XPU operator found across both repos. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

**Mode C — Daily scan**

- **Input**: `scan daily` or `scan since <date>` (optionally specify output directory)
- **Output**: `xpu_scan_daily_<date>_<time>.json` + `.md` — deep analysis of operators affected by recent changes. See [references/batch-scan-workflow.md](references/batch-scan-workflow.md).

**Important**: Report everything. Do not dismiss or waive any finding. Assign priority instead.

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
- In batch scans: if a single op analysis fails, record `status: error` and continue — do not abort the scan.

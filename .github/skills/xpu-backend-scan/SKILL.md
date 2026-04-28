---
name: xpu-backend-scan
description: Triage XPU backend coverage — parity gaps, missing implementations, dispatch defects, fallback misuse. Use when analyzing whether an operator is correctly supported on XPU or when reviewing scan findings.
---

# XPU Backend Scan

Determine whether a PyTorch operator has correct, complete XPU support by inspecting dispatch paths, runtime coverage, and user-visible behavior parity with CUDA.

References:
- [references/dispatch-coverage.md](references/dispatch-coverage.md) — how to determine XPU coverage and where to look
- [references/triage-patterns.md](references/triage-patterns.md) — common false positives and decision heuristics

## Workflow

### Step 1: Identify the operator surface

Pin the exact schema/overload from `yaml/native/native_functions.yaml`. Do not compare by base op name or filename alone.

### Step 2: Check waivers

Consult `assets/waivers.yaml`. If the op matches a waived category (NVIDIA-specific infra, hardware-only features, documented unsupported families), stop with `WAIVED`.

### Step 3: Determine existing XPU coverage

Check these in priority order (higher overrides lower):
1. `XPUFallback.template` — runtime coverage exists (blocks "missing impl" conclusion)
2. Backend YAML with explicit XPU dispatch keys
3. Structured delegate or codegen path resolving to XPU
4. Source-backed registration (`TORCH_IMPL_FUNC`, landed implementation)
5. Composite (`CompositeImplicitAutograd`/`CompositeExplicitAutograd`) or decomposition

If any of these provides coverage, do not conclude "missing implementation."

### Step 4: Classify the finding

- **XPU defect** — intrinsic problem in XPU code (broken dispatch, silent CPU fallback, missing validation, backward gap, race condition)
- **Parity gap** — both CUDA and XPU have coverage, but user-visible contract differs (input space, parameter semantics, dtype support, backward behavior, error paths)
- **Missing implementation** — CUDA has usable support, XPU has no callable path after all exclusions checked
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
- Fallback existence blocks "missing implementation" but may indicate a defect (CPU fallback on GPU op).
- Test skip/xfail metadata is supporting evidence only, never sufficient alone.

## Output

Return findings grounded in inspected code:
- Exact schema/overload under review
- Verdict with confidence level
- XPU-side evidence (files, code paths)
- Peer evidence (CUDA, upstream, shared paths)
- Exclusion checks performed (waiver, fallback, composite, delegate)
- Next action (stop, downgrade, hand off for repro)

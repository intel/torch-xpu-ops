# Eager Report Workflow

Use this workflow when the user already has raw OOB eager artifacts and wants per-model T1/T2/R analysis.

This document is the canonical source of truth for per-model report generation, projection-vs-actual analysis, and XPU kernel-to-op attribution.

## Goal

Generate deterministic per-model markdown reports that compare XPU and CUDA software efficiency using T1/T2/R roofline analysis.

## Prerequisites

Per model, the workflow expects data from at least one platform. For cross-platform comparison, it should have both XPU and CUDA.

Minimum useful inputs per platform:

1. T1 calcflops output
2. profiler trace
3. T2 wall-clock value
4. XPU unitrace JSON when available

## Platform Naming Rules

Keep platform naming consistent with the existing methodology:

| User-facing name | Internal ID | Config key | Notes |
|-----------------|-------------|------------|-------|
| B580 | B580 | b580 | XPU platform with unitrace |
| B70 | G31 | b70 | XPU platform with unitrace |
| 4080S | 4080 | 4080s | CUDA platform, no unitrace |

Output can display internal ids (`G31`, `B580`, `4080`), but user-facing descriptions should remain understandable.

See `config/hardware_specs.yaml` for the canonical platform configuration including peak compute, bandwidth, and ridge point values. Config keys are lowercase; use the `label` or `internal_id` fields for display.

## Required Inputs Per Model

For a full B70 vs 4080S eager comparison, expect:

1. `t1/calcflops.txt`
2. `xpu_profiler/trace.json`
3. `cuda_profiler/trace.json`
4. `xpu_t2/rcpi1-ins0.log`
5. `cuda_t2/rcpi1-ins0.log`
6. `unitrace/python.<pid>.json`

## T2 Extraction

Extract T2 from the T2 logs using the line:

```text
GPU Time per batch:  209.353 milliseconds
```

Important rules:

1. T2 is the wall-clock denominator used for R.
2. T2 is not the same as the kernel-sum time from trace or unitrace.
3. T2_device should be shown for reference only.

## Per-Op Actual Time Source

### XPU

Preferred source:

1. unitrace mapped to aten ops using the profiler trace as the bridge

Reason:

1. unitrace avoids profiler overhead inflation
2. it is the preferred source for XPU per-op actual timing

#### Unitrace-to-Op Mapping Algorithm

The profiler trace serves as the bridge between unitrace kernel timings and aten-level ops:

1. Parse the profiler trace to build an op-to-kernel mapping:
   - Each aten op in the trace has child kernel events (GPU kernel launches)
   - Record the mapping: `aten_op → [kernel_name_1, kernel_name_2, ...]`

2. Parse unitrace output to get per-kernel timings:
   - Each unitrace entry has a kernel name and wall-clock duration without profiler overhead
   - Sum durations for duplicate kernel invocations of the same name within one iteration

3. Bridge the two:
   - For each aten op, look up its associated kernel names from step 1
   - Sum the unitrace timings for those kernel names to get the op-level actual time
   - If a kernel is shared across multiple aten ops, distribute proportionally by profiler-reported sub-durations
   - Fallback: if profiler sub-durations are zero or missing for a shared kernel, distribute equally among the sharing ops and flag the attribution as approximate

4. Validate:
   - `sum(all attributed kernel times) ≈ sum(all unitrace kernels)`
   - If total attributed time exceeds T2 by more than ~10%, suspect multi-iteration leakage

This produces per-op actual times that avoid profiler overhead while retaining aten-level attribution.

### CUDA

Use:

1. profiler trace directly

Reason:

1. CUDA does not have the XPU unitrace path in this workflow

## Workflow

### Step 1: Discover Complete Models

Scan the canonical session layout and identify models that have the minimum required files.

If a model is incomplete:

1. skip it for cross-platform reporting
2. keep a note of why it was skipped

### Step 2: Load T1 Inputs

Parse calcflops output and keep only the last benchmark iteration as the T1 input.

Use the following roofline logic:

1. `T1_compute = FLOPs / peak`
2. `T1_memory = bytes / bandwidth`
3. `T1_op = max(T1_compute, T1_memory)`
4. `T1 = sum(T1_op)`

### Step 3: Load Actual Inputs

For each platform:

1. parse profiler trace
2. parse unitrace when available on XPU
3. recover per-op actual timing
4. compute `T2_device` as the kernel sum

### Step 4: Compare Projection vs Actual

For each op:

1. compute `R_op = projected / actual`
2. identify overcounting when `R_op > 1.05`
3. identify undercounting or kernel slowness when `R_op < 0.80`
4. compare against CUDA `R_op` when available to separate projection error from XPU kernel inefficiency

### Step 5: Generate the 5 Report Sections

Per-model reports should follow this exact structure:

1. Summary
2. Projection Quality
3. XPU vs CUDA Consistency
4. XPU vs 4080S Per-Op Efficiency
5. Optimization Targets

See `report-structure-reference.md` for the section contract.

## Section-Level Requirements

### Section 1: Summary

Must include:

1. model metadata
2. cross-platform metrics table
3. cross-platform R ratio
4. hardware specs
5. action items
6. overall assessment

Important rules:

1. `R = T1 / T2`
2. never use `T2_device` as the denominator for R
3. when only some platforms are present, omit missing columns cleanly

### Section 2: Projection Quality

Must include:

1. per-platform flagged-op table
2. `R_op`, actual, `%T2`, projected, gap, and issue classification
3. `T2 Coverage by T1` subsection for uncovered ops

### Section 3: XPU vs CUDA Consistency

Must include:

1. calcflops-based graph consistency comparison
2. trace-based common-op and platform-only-op analysis
3. shape-set differences for meaningful compute ops

### Section 4: XPU vs 4080S Per-Op Efficiency

Must include:

1. per-op `R_xpu`, `R_4080S`, and `R_diff`
2. impact ordering by `% T2`
3. verdict per op

### Section 5: Optimization Targets

Must include:

1. only ops where XPU `R_op < CUDA R_op`
2. target actual time if XPU matched CUDA efficiency
3. saving in ms and `%T2`
4. action category: optimize kernel or fix projection

## Key Interpretation Rules

1. `R` compares software efficiency after normalizing for hardware roofline assumptions.
2. `R_xpu / R_cuda > 1` means XPU software efficiency is better, even if raw T2 may still differ.
3. `R_op < 0.80` on XPU but not on CUDA usually points to XPU kernel work.
4. `R_op < 0.80` on all platforms usually points to projection undercounting.
5. `R_op > 1.05` usually points to projection overcounting.

## Output Location

Per-model reports should be written under a session-scoped reports directory, for example:

```text
reports/<session_name>/models/
```

## Completion Criteria

The eager report workflow is complete when:

1. all complete models in the session have been scanned
2. all valid per-model reports have been emitted
3. skipped models have an explicit reason
4. the report structure matches the reference contract

## References

- [data-contracts.md](data-contracts.md)
- [report-structure-reference.md](report-structure-reference.md)
- [graph-consistency-workflow.md](graph-consistency-workflow.md)
- [troubleshooting.md](troubleshooting.md)

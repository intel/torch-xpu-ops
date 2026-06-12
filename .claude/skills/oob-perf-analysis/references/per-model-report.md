# Per-Model Report

Generate a deterministic markdown report comparing XPU and CUDA software efficiency for one model using T1/T2/R roofline analysis.

See `methodology.md` for T1/T2/R definitions and `inputs.md` for required files.

## Steps

1. Scan session layout; skip models missing required files (note why)
2. Parse `t1/rcpi1-ins0.log` (last iteration only) → compute T1
3. Parse traces and unitrace → compute per-op actual times and T2_device
4. Extract T2 from T2 logs
5. Compute R and per-op R_op; classify issues
6. Write the 5-section report

## Report Structure

### `## 1. Summary`

**1a. Model Info and Metrics Table**

Model metadata: name, batch size, precision, mode, ops per iteration.

Metrics table per available platform:

| Metric | Value |
|--------|-------|
| T2 (wall clock) | |
| T1 (projection) | |
| T1_compute | |
| T1_memory | |
| T2_device (kernel sum) | |
| R = T1/T2 | |
| Actual source | |
| Compute-bound ops | |
| Memory-bound ops | |

Rules: `R = T1 / T2`. Never use T2_device as denominator. Omit missing platform columns cleanly.

**1b. Cross-Platform R Ratio**

Compare `R_xpu / R_cuda`. Value > 1 means XPU software efficiency is better.

**1c. Hardware Specs**

Per platform: peak FP16 TFLOPS, DRAM bandwidth, ridge point.

**1d. Action Items**

Prioritized table with columns: action, target, op, shape, stride, expected impact, priority.

Action categories:
- `Optimize XPU kernel` — when XPU `R_op < 0.80` and CUDA `R_op >= 0.80`
- `Fix projection` — when all platforms are low or all overcount

Show dominant shape and stride; do not truncate.

**1e. Overall Assessment**

One paragraph: health level, high-priority action count, kernel vs projection work, wall-clock comparison to CUDA.

### `## 2. Projection Quality`

Per-platform flagged-op table sorted by actual time descending:

| Op | R_op | Actual (ms) | % T2 | Proj (ms) | Gap (ms) | Perf | Shape | 4080S R_op | Issue |
|----|------|-------------|------|-----------|----------|------|-------|------------|-------|

`Gap = projected - actual`. Positive gap = overcounting. Issue classification per `methodology.md`.

**T2 Coverage by T1** subsection: ops in trace with no calcflops entry.

| Op | Actual (ms) | % T2 | Count |
|----|-------------|------|-------|

### `## 3. XPU vs CUDA Consistency`

**3a. Graph Consistency** (calcflops-based):

- Total FLOPs diff %
- Total memory diff %
- CUDA-only ops, XPU-only ops, common ops with mismatched FLOPs/memory

**3b. Trace Comparison** (runtime-visible):

- Common op count, platform-only op count
- Platform-specific op table when significant
- Shape-set differences for compute ops (not data-movement ops like copy_)

### `## 4. XPU vs 4080S: Per-Op Efficiency`

Sorted by `% T2` descending:

| Op | R_xpu | R_4080S | R_diff | XPU (ms) | 4080S (ms) | % T2 | Verdict |
|----|-------|---------|--------|----------|------------|------|---------|

`R_diff = R_xpu - R_4080S`. Verdict: `XPU wins` (R_diff > +0.05), `XPU behind` (R_diff < -0.05), `~tie` otherwise.

### `## 5. Optimization Targets`

Only ops where `R_xpu < R_4080S`:

| # | Op | R_xpu | R_4080S | Actual (ms) | Target (ms) | Saving (ms) | % T2 | Action |
|---|----|----|---------|-------------|-------------|-------------|------|--------|

`Target (ms) = projected / R_4080S`. `Saving = Actual - Target`.

Actions: `Optimize kernel` or `Fix projection`.

Include a total row and a short note on kernel vs projection split.

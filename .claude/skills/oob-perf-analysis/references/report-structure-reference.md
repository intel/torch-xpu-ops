# Report Structure Reference

This file defines the report-section contract for OOB eager analysis.

The goal is deterministic structure.

1. The same inputs should produce the same section layout.
2. Section meaning should not drift between sessions.
3. Downstream insights generation should be able to rely on these sections.

## Per-Model Report

Per-model eager reports must follow a stable 5-section structure with numbered headers.

Required order:

1. `## 1. Summary`
2. `## 2. Projection Quality`
3. `## 3. XPU vs CUDA Consistency`
4. `## 4. XPU vs 4080S: Per-Op Efficiency`
5. `## 5. Optimization Targets`

## Section 1: Summary

Section 1 should include five logical parts.

### 1a. Model Info And Metrics Table

Must include:

1. model name
2. batch size
3. precision
4. mode
5. ops per iteration

Metrics table must include, per available platform:

1. `T2 (wall clock)`
2. `T1 (projection)`
3. `T1_compute`
4. `T1_memory`
5. `T2_device (kernel sum)`
6. `R = T1/T2`
7. `Actual source`
8. `Compute-bound ops`
9. `Memory-bound ops`

Important rules:

1. `R = T1 / T2`
2. Do not use `T2_device` as the denominator for R
3. `T2_device` is shown only as a reference value
4. If a platform is missing, omit its column cleanly

### 1b. Cross-Platform R Ratio

Must compare XPU platform R against CUDA R.

Interpretation rule:

1. `R_xpu / R_cuda > 1` means XPU software efficiency is better
2. this comparison is about software efficiency, not raw wall time

### 1c. Hardware Specs

Must include, per available platform:

1. peak FP16 TFLOPS
2. DRAM bandwidth
3. ridge point

### 1d. Action Items

Must include a prioritized action table.

Required columns:

1. action
2. target
3. op
4. shape
5. stride
6. expected impact
7. priority

Required action categories:

1. `Optimize XPU kernel`
   - use when XPU `R_op < 0.80` and CUDA `R_op >= 0.80`
2. `Fix projection`
   - use when all platforms are low or all platforms overcount

Shape and stride rule:

1. show the dominant shape and dominant stride
2. do not truncate them if the full value is important to diagnosis

### 1e. Overall Assessment

Must include a concise one-paragraph assessment covering:

1. overall health level
2. number of high-priority actions
3. whether the primary work is kernel optimization or projection fixing
4. high-level wall-clock comparison to CUDA when CUDA exists

Suggested health interpretation:

1. `R >= 0.95` excellent
2. `0.85 <= R < 0.95` good
3. `0.70 <= R < 0.85` fair
4. `R < 0.70` poor

## Section 2: Projection Quality

This section combines overcounting and low-R diagnosis into one view.

Header must be:

`## 2. Projection Quality`

### Per-Platform Flagged-Op Table

Must include, per platform, a table sorted by actual time descending.

Required columns:

1. `Op`
2. `R_op`
3. `Actual (ms)`
4. `% T2`
5. `Proj (ms)`
6. `Gap (ms)`
7. `Perf`
8. `Shape`
9. `4080S R_op`
10. `Issue`

Interpretation rules:

1. `R_op = projected / actual`
2. `Gap = projected - actual`
3. positive gap means overcounting
4. negative gap means undercounting or slow kernel

Required issue classes:

1. `Overcounting`
2. `Kernel slow`
3. `Projection undercounts`
4. `Undercounts or slow`

Issue classification decision logic (R_op thresholds use 0.80 as the boundary because 20% underperformance relative to roofline is the empirically observed point where per-op issues become actionable optimization targets; ops between 0.80 and 1.0 are usually acceptable overheads):

1. If `R_op > 1.05`: classify as `Overcounting`
   - Projection overestimates actual work; likely fusion or calcflops including removed ops
2. If `R_op < 0.80` and CUDA `R_op >= 0.80` for the same op: classify as `Kernel slow`
   - XPU kernel is genuinely slower relative to its roofline; CUDA achieves expected efficiency
3. If `R_op < 0.80` and CUDA `R_op < 0.80` for the same op: classify as `Projection undercounts`
   - Both platforms miss projection; the issue is in the calcflops model, not the kernel
4. If `R_op < 0.80` and no CUDA comparison is available: classify as `Undercounts or slow`
   - Cannot disambiguate without cross-platform evidence

### T2 Coverage By T1

Must include a subsection for ops that appear in actual traces but have no calcflops entry.

Required columns:

1. `Op`
2. `Actual (ms)`
3. `% T2`
4. `Count`

Purpose:

1. show projection blind spots
2. explain why T1 may systematically miss actual GPU time

## Section 3: XPU vs CUDA Consistency

Header must be:

`## 3. XPU vs CUDA Consistency`

This section has two layers.

### 3a. Graph Consistency

Must compare device-independent calcflops output between CUDA and XPU.

Must show:

1. total FLOPs difference percentage
2. total memory difference percentage when available
3. CUDA-only ops
4. XPU-only ops
5. common ops with different FLOPs or memory values

Purpose:

1. detect dispatch divergence
2. distinguish graph mismatch from pure kernel inefficiency

### 3b. Trace Comparison

Must compare runtime-visible ops between CUDA and XPU.

Must include:

1. common op count
2. platform-only op count
3. platform-specific op table when significant
4. shape-set differences for meaningful compute ops

Important rule:

1. do not overemphasize pure data-movement ops like clone or copy_ in shape-set comparison
2. focus on compute ops where graph behavior actually differs

## Section 4: XPU vs 4080S Per-Op Efficiency

Header must be:

`## 4. XPU vs 4080S: Per-Op Efficiency`

Required columns:

1. `Op`
2. `R_xpu`
3. `R_4080S`
4. `R_diff`
5. `XPU (ms)`
6. `4080S (ms)`
7. `% T2`
8. `Verdict`

Important rules:

1. sort by `% T2` descending
2. `R_diff = R_xpu - R_4080S`
3. filter tiny low-impact ops if necessary

Suggested verdict mapping:

1. `XPU wins` when `R_diff > +0.05`
2. `XPU behind` when `R_diff < -0.05`
3. `~tie` otherwise

## Section 5: Optimization Targets

Header must be:

`## 5. Optimization Targets`

Required columns:

1. `#`
2. `Op`
3. `R_xpu`
4. `R_4080S`
5. `Actual (ms)`
6. `Target (ms)`
7. `Saving (ms)`
8. `% T2`
9. `Action`

Inclusion rule:

1. include only ops where `R_xpu < R_4080S`

Computation rule:

1. `Target (ms) = projected / R_4080S`
2. `Saving = Actual - Target`

Required action values:

1. `Optimize kernel`
2. `Fix projection`

Must include:

1. total row for summed saving
2. short note explaining how much of the gap is kernel work vs projection work

## Fleet Summary

Fleet summary reports must follow a stable 7-section structure.

Required order:

1. `## 1. Overall`
2. `## 2. Per-Suite Geomean`
3. `## 3. Model Scorecard`
4. `## 4. Worst Models by R`
5. `## 5. Op Priority Ranking`
6. `## 6. Projection Accuracy`
7. `## 7. Graph Consistency`

## Fleet Section 1: Overall

Must include:

1. number of profiled models
2. geomean R per platform
3. geomean T2 ratio for XPU vs CUDA when CUDA is available

## Fleet Section 2: Per-Suite Geomean

Must include:

1. suite name
2. number of models in the suite
3. geomean R per platform
4. geomean T2 ratio per suite when CUDA is available

If suite metadata is missing:

1. use `unknown`
2. do not block the whole report

## Fleet Section 3: Model Scorecard

Must contain one row per included model.

Recommended columns:

1. model
2. batch size
3. suite
4. R per platform
5. R ratio columns
6. T2 per platform
7. T2 ratio columns
8. top gap op

Sort rule:

1. sort by primary XPU-vs-CUDA R ratio ascending, worst first

## Fleet Section 4: Worst Models by R

Must show the lowest-performing XPU models with a brief diagnosis.

Recommended fields:

1. model
2. suite
3. R per platform
4. top gap op
5. gap magnitude
6. likely issue

## Fleet Section 5: Op Priority Ranking

Must rank ops by fleet-level payoff, not just single-model payoff.

Required logic:

1. aggregate all models where XPU trails CUDA on the same op
2. estimate new fleet geomean if that op matched CUDA efficiency
3. rank by fleet geomean delta or equivalent fleet-level benefit

## Fleet Section 6: Projection Accuracy

Must separate the fleet-wide projection-quality issues into three groups:

1. overcounting
2. undercounting
3. uncovered ops

Purpose:

1. prevent kernel work from being confused with projection-model problems

## Fleet Section 7: Graph Consistency

Must summarize fleet-wide graph differences between CUDA and each XPU platform.

Required outputs:

1. compared model count
2. identical vs different count
3. SDPA-only vs significant vs minor categories
4. top op differences across the fleet

## AI Analysis Layer

If an LLM-generated analysis layer is added after deterministic report generation, it must be treated as additive commentary, not as a replacement for the deterministic section contract.

The deterministic report structure remains the source of truth.

# Fleet Summary Workflow

Use this workflow when the user wants fleet-wide summary analysis across many OOB models.

This document is the canonical source of truth for fleet-level aggregation and summary generation.

## Goal

Aggregate per-model eager analysis into one fleet-level report that shows overall software efficiency, worst models, op-level opportunities, projection accuracy, and graph consistency.

## Inputs

The fleet summary workflow expects:

1. one canonical session layout or equivalent complete per-model inputs
2. complete per-model analysis inputs for enough models to produce meaningful aggregates
3. optional suite metadata if per-suite grouping is desired

## Model Eligibility

Include a model in the fleet summary only when it has the minimum required inputs for the platforms being compared.

Important behavior:

1. if both XPU and CUDA are available, include it in cross-platform ranking
2. if one platform is missing, either skip it from cross-platform sections or mark it incomplete
3. do not silently fabricate cross-platform pairing

## Workflow

### Step 1: Discover Complete Models

Collect all models with complete inputs across the requested platforms.

For each model, recover:

1. model name
2. batch size
3. suite if known
4. platform-level `T1`, `T2`, `R`
5. op-level comparison data needed for priority ranking and projection accuracy sections

### Step 2: Compute Fleet-Level Metrics

Compute:

1. geomean `R` per platform
2. geomean `T2` ratios for XPU vs CUDA
3. suite-level geomean breakdowns when suite metadata is available

#### Geomean Calculation

Use the geometric mean (equal weight per model):

```text
geomean(R) = exp( (1/N) * sum(ln(R_i)) )
```

All models are weighted equally. Do not weight by model size or iteration count.

For T2 ratio geomean:

```text
geomean(T2_xpu / T2_cuda) = exp( (1/N) * sum(ln(T2_xpu_i / T2_cuda_i)) )
```

When estimating fleet-level improvement from fixing one op:

```text
new_geomean = exp( (1/N) * sum(ln(R_i_adjusted)) )
delta = new_geomean - current_geomean
```

Where `R_i_adjusted` uses the corrected per-op time for affected models and keeps R unchanged for unaffected models.

### Step 3: Build Model Ranking

Build a scorecard that sorts models by the primary XPU-vs-CUDA R ratio, worst first.

Per model, include:

1. batch size
2. suite
3. `R` per platform
4. `R_xpu / R_cuda`
5. `T2` per platform
6. `T2_xpu / T2_cuda`
7. top gap op

### Step 4: Rank Fleet-Level Op Opportunities

Across all included models:

1. find ops where XPU `R_op < CUDA R_op`
2. estimate potential saving if XPU matched CUDA efficiency
3. estimate fleet-level geomean improvement
4. rank ops by expected fleet impact

### Step 5: Summarize Projection Accuracy

Aggregate projection-quality issues across the fleet.

Required groupings:

1. overcounting ops
2. undercounting ops
3. uncovered ops

Use these sections to separate projection-model problems from kernel problems.

### Step 6: Summarize Graph Consistency

Aggregate graph differences between CUDA and each XPU platform.

Required outputs:

1. number of compared models
2. identical vs different models
3. SDPA-only vs significant vs minor categories
4. top op differences across the fleet

## Required Fleet Report Structure

Fleet summary reports should follow this exact 7-section structure:

1. Overall
2. Per-Suite Geomean
3. Model Scorecard
4. Worst Models by R
5. Op Priority Ranking
6. Projection Accuracy
7. Graph Consistency

## Section Notes

### Section 1: Overall

Must show:

1. number of models
2. geomean `R` per platform
3. geomean `T2` ratio for XPU vs CUDA

### Section 2: Per-Suite Geomean

Must show:

1. suite name
2. model count
3. geomean `R` per platform
4. geomean `T2` ratio per suite when CUDA is available

### Section 3: Model Scorecard

Must show one row per model, sorted by the primary XPU-vs-CUDA R ratio.

### Section 4: Worst Models by R

Must show a focused subset of the lowest-performing XPU models with a brief diagnosis.

### Section 5: Op Priority Ranking

Must rank ops by fleet-level payoff, not just single-model payoff.

### Section 6: Projection Accuracy

Must separate:

1. overcounting
2. undercounting
3. uncovered ops

### Section 7: Graph Consistency

Must summarize graph-difference categories and the most frequent op-level differences.

## Suite Classification

If suite metadata is available, classify models by suite such as:

1. torchbench
2. timm
3. huggingface

If it is not available:

1. mark suite as `unknown`
2. do not block the full fleet summary on missing suite metadata

## Completion Criteria

The fleet summary workflow is complete when:

1. all eligible models are included
2. all 7 sections are generated
3. the ranking logic is stable and reproducible
4. graph consistency and projection accuracy are included, not deferred

## References

- [report-structure-reference.md](report-structure-reference.md)
- [graph-consistency-workflow.md](graph-consistency-workflow.md)
- [data-contracts.md](data-contracts.md)

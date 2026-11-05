# Fleet Summary

Aggregate per-model results into a fleet-level report showing overall software efficiency, worst models, op-level opportunities, projection accuracy, and graph consistency.

See `methodology.md` for geomean formulas and `inputs.md` for session layout.

## Inputs

- Per-model T1, T2, R, and per-op data from all complete models
- Optional suite metadata for per-suite grouping

Include a model only when it has the minimum required inputs for the platforms being compared. Do not silently fabricate cross-platform pairing.

## Steps

1. Collect all complete models; recover T1, T2, R, suite, and op-level data per model
2. Compute fleet-level geomean R and geomean T2 ratios; break down by suite when available
3. Build model scorecard sorted by XPU-vs-CUDA R ratio, worst first
4. Rank ops by fleet-level payoff (geomean delta if XPU matched CUDA efficiency)
5. Aggregate projection-quality issues across the fleet
6. Aggregate graph consistency results across the fleet
7. Write the 7-section report

## Report Structure

### `## 1. Overall`

- Number of models
- Geomean R per platform
- Geomean T2 ratio XPU vs CUDA

### `## 2. Per-Suite Geomean`

| Suite | Models | R_xpu | R_cuda | T2_xpu/T2_cuda |
|-------|--------|-------|--------|----------------|

Suite: `torchbench`, `timm`, `huggingface`, or `unknown` if metadata missing. Do not block the report on missing suite metadata.

### `## 3. Model Scorecard`

One row per model, sorted by XPU-vs-CUDA R ratio ascending (worst first):

| Model | Batch | Suite | R_xpu | R_cuda | R_xpu/R_cuda | T2_xpu | T2_cuda | T2_xpu/T2_cuda | Top gap op |
|-------|-------|-------|-------|--------|-------------|--------|---------|----------------|------------|

### `## 4. Worst Models by R`

Focused subset of lowest-performing XPU models with brief diagnosis:

| Model | Suite | R_xpu | R_cuda | Top gap op | Gap | Likely issue |
|-------|-------|-------|--------|------------|-----|--------------|

### `## 5. Op Priority Ranking`

Rank ops by fleet-level payoff, not single-model payoff:

1. Aggregate all models where XPU `R_op < CUDA R_op` for the same op
2. Estimate new fleet geomean if that op matched CUDA efficiency
3. Rank by geomean delta

### `## 6. Projection Accuracy`

Separate fleet-wide projection issues into three groups: overcounting, undercounting, uncovered ops. Purpose: prevent kernel work from being confused with projection-model problems.

### `## 7. Graph Consistency`

Summarize fleet-wide graph differences between CUDA and XPU:

- Compared model count
- Identical vs different count
- SDPA-only / Significant / Minor categories
- Top op differences across the fleet

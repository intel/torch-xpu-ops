# Graph Consistency Workflow

Use this workflow when the user wants to compare calcflops-derived computational graphs between CUDA and XPU.

This document is the canonical source of truth for graph-consistency analysis.

## Goal

Detect whether CUDA and XPU run the same computational graph for the same model, independent of runtime kernel performance.

This workflow should answer:

1. Are the two device graphs functionally identical at the calcflops level?
2. If not, is the difference expected and minor, or a meaningful graph divergence?
3. Which ops differ most often across the fleet?

## Why Graph Consistency Matters

Per-op efficiency comparisons are only trustworthy when both platforms run the same or comparable graph.

If the graph differs:

1. a low XPU R may reflect graph divergence, not a slow kernel
2. action items may need to target dispatch or decomposition, not kernel optimization

## Inputs

Required inputs for a single-model comparison:

1. one CUDA calcflops output
2. one XPU calcflops output

For a fleet-level graph consistency report:

1. two directories containing calcflops outputs for matching models on both platforms

## Comparison Principle

The comparison is device-independent.

Use only the calcflops columns that should be identical across devices:

1. cumulative FLOPs
2. cumulative memory bytes

If these differ, the cause is graph or dispatch divergence, not hardware speed.

## Normalization Rules

Before comparing, normalize op names consistently.

Representative normalization examples (not exhaustive; add entries to this list in this document as new backend differences are discovered):

1. `aten::copy_` -> `aten::clone`
2. `aten::convolution_overrideable` -> `aten::convolution`
3. `aten::convolution_backward_overrideable` -> `aten::convolution_backward`
4. `aten::reshape`, `aten::contiguous`, `aten::unbind` -> `__view_noop__`
5. SDPA variants -> common normalized attention op name

Important rule:

1. view-only metadata ops should not pollute graph-difference conclusions
2. normalization must reduce false-positive graph mismatches caused only by naming differences

## Model Eligibility

Only compare a model when:

1. both platforms have calcflops output
2. the calcflops files correspond to the same model, batch size, precision, and test mode

If one side is missing:

1. mark the model unavailable for graph comparison
2. do not infer match or mismatch

## Workflow

### Step 1: Pair Matching Models

Pair models across CUDA and XPU using:

1. model name
2. batch size
3. precision
4. test mode

### Step 2: Parse Calcflops Output

For each platform:

1. parse the calcflops file
2. use the last benchmark iteration
3. recover per-op delta FLOPs and memory
4. aggregate by normalized op name

### Step 3: Compare Aggregates

For each model pair, compute:

1. total FLOPs on CUDA and XPU
2. total memory on CUDA and XPU
3. common ops (matched by exact normalized op name)
4. CUDA-only ops (present in CUDA but absent in XPU after normalization)
5. XPU-only ops (present in XPU but absent in CUDA after normalization)
6. common ops with mismatched FLOPs or memory

Op matching rule: ops are matched by exact normalized op name string after applying the normalization rules above. No fuzzy matching is used.

### Step 4: Classify The Difference

Each model should be classified into one of these buckets:

1. `MATCH`
   - no meaningful difference after normalization
2. `SDPA-only`
   - only SDPA dispatch path differs
3. `Significant`
   - max(FLOPs diff %, memory diff %) > 1%
4. `Minor`
   - mismatch exists but stays below the significant threshold

Threshold rationale:

The 1% threshold separates numerical noise from meaningful graph divergence. In practice, identical graphs may show sub-0.1% differences due to floating-point accumulation order. True dispatch divergence (e.g., decomposition differences, missing fusions) typically shows 5-50% FLOPs differences. The 1% line provides a conservative boundary that catches real divergence while filtering out rounding artifacts. If this threshold proves too sensitive for a specific workload class, it can be adjusted per-session with documentation.

## Output Structure

### Standalone Graph Consistency Report

A standalone graph consistency report should contain:

1. Fleet Summary
2. Difference Categories
3. Significant Divergences
4. Per-Model Details
5. Op Differences Across Fleet

### Embedded Fleet Summary Section

When embedded inside the fleet summary, it should at minimum contain:

1. compared model count
2. identical vs different count
3. difference categories
4. top op differences across the fleet

## Required Fleet Summary Metrics

For the compared model set, report:

1. total models compared
2. identical graph count
3. different graph count
4. percentage of identical vs different models

## Required Difference Categories

Difference category table should include:

1. `SDPA-only`
2. `Significant`
3. `Minor`

Interpretation:

1. `SDPA-only` often reflects known backend path differences in attention
2. `Significant` should be treated as real model-behavior divergence
3. `Minor` may reflect numerically small but real structural differences

## Significant Divergences Table

For significant models, include at least:

1. model name
2. FLOPs diff percentage
3. memory diff percentage
4. CUDA-only op count
5. XPU-only op count
6. mismatched common-op count

Purpose:

1. quickly identify which models are not valid apples-to-apples efficiency comparisons

## Per-Model Detail Expectations

For each model with a mismatch, detail should make it possible to answer:

1. which ops only exist on CUDA
2. which ops only exist on XPU
3. which common ops disagree in FLOPs or memory
4. whether the divergence is likely due to SDPA, overrideable ops, decomposition, or another path difference

## Op Differences Across Fleet

Aggregate differences by op name across all mismatched models.

The table should show:

1. op name
2. number of models where it differs
3. example models

Purpose:

1. identify recurring backend divergence patterns
2. separate one-off anomalies from systemic differences

## Interpretation Rules

1. Graph consistency is a correctness-of-comparison question, not a performance metric by itself.
2. Identical graph does not imply identical efficiency, but it is a prerequisite for meaningful per-op efficiency comparison.
3. A graph mismatch should be considered before claiming XPU kernel inefficiency.
4. SDPA-only mismatch is often less severe than broad FLOPs or memory divergence.

## Completion Criteria

The graph consistency workflow is complete when:

1. all eligible model pairs have been compared
2. each compared model is classified as match, SDPA-only, significant, or minor
3. fleet-level difference categories are summarized
4. recurring op differences are aggregated

## References

- [data-contracts.md](data-contracts.md)
- [report-structure-reference.md](report-structure-reference.md)

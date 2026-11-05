# Insights Workflow

Use this workflow after per-model reports and fleet summary are already available.

This document is the canonical source of truth for the final developer-facing insights summary.

## Goal

Produce a concise developer-facing summary of:

1. XPU vs CUDA software efficiency
2. Which models still trail CUDA in R
3. Which ops should be prioritized for XPU kernel work

## Inputs

1. Fleet summary report
2. Graph consistency report when available
3. Per-model reports
4. Hardware specs reference when platform context is needed

Expected source mapping:

1. Fleet geomean `R`, model scorecard, worst-model summary, and op priority ranking come from the fleet summary.
2. Graph consistency status comes from the graph consistency report when present.
3. Root-cause op details, shapes, and per-op `R_op` gaps come from per-model reports.

## Output

Write one concise markdown summary focused on actionability rather than exhaustive detail.

Recommended output path:

1. `agent_space_xpu/reports/<session>/insights_summary.md`

## Required Structure

The summary should have exactly 3 sections.

### 1. Key Numbers

Provide a compact dashboard table that includes at least:

1. fleet geomean `R` for XPU
2. fleet geomean `R` for CUDA
3. model count where XPU `R >=` CUDA `R`
4. model count or list where XPU `R <` CUDA `R`
5. worst XPU model by `R`
6. best XPU model by `R`
7. graph consistency status when available
8. top fixable op and its expected impact when available

### 2. Insight 1: XPU Software Efficiency vs CUDA

Frame this section from the developer perspective.

Required points:

1. compare platforms using `R`, not raw wall time
2. explain that `R` is the software-efficiency metric because it normalizes away hardware peak differences
3. summarize where XPU already matches or exceeds CUDA in software efficiency
4. list the main models where XPU still trails CUDA

### 3. Insight 2: Specific Fixes to Pull Ahead

Turn the fleet summary and per-model reports into a short prioritized fix list.

For each top opportunity, include when available:

1. op name
2. affected model or models
3. XPU `R_op` vs CUDA `R_op`
4. root-cause hint
5. expected impact on fleet or model-level `R`

## Key Principles

1. Write in English.
2. Keep the framing developer-oriented rather than benchmark-marketing oriented.
3. Compare `R` first; use raw `T2` only as supporting context.
4. Keep the output concise: two insights plus the key-numbers section.
5. Every claim should point back to a concrete source in the fleet summary, graph report, or per-model reports.
6. If graph consistency is poor for a model, avoid presenting kernel optimization as the first conclusion.

## Recommended Workflow

1. Read the fleet summary first and extract the headline metrics.
2. Read the graph consistency report and note whether graph divergence changes the interpretation.
3. Read the per-model reports for the worst underperforming models.
4. Identify the smallest set of recurring high-impact ops that explain most of the XPU gap.
5. Write the final summary as a short action page, not a verbose report.

## Guardrails

1. Do not replace deterministic fleet or per-model analysis with free-form summary text.
2. Do not rank optimization targets only by wall time without checking `R` or `R_op`.
3. Do not treat graph-divergent models as direct kernel-efficiency evidence.
4. If the available inputs are incomplete, say so explicitly in the summary.

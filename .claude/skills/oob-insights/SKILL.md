---
name: insights
description: "Use when generating an insights summary from completed OOB reports. Triggered after per-model reports and fleet summary are ready. Produces a concise developer-facing analysis of XPU vs CUDA software efficiency with actionable kernel fix priorities."
---

# Generate Insights Summary from OOB Reports

Given a completed set of per-model reports and fleet summary, generate a concise
insights summary that highlights XPU software efficiency vs CUDA from a developer's
perspective.

---

## Overview

This skill produces a final insights page after all per-model analysis and fleet summary
reports have been generated. It focuses on actionable developer takeaways:
how well XPU kernels utilize hardware compared to CUDA, and what specific fixes will
close remaining gaps.

---

## Inputs

1. **Fleet summary report** — e.g., `reports/<session>/summary_eager_inference.md`
   - Contains: Geomean R per platform, per-suite breakdown, model scorecard, op priority ranking
2. **Graph consistency report** — e.g., `reports/<session>/graph_consistency_eager_inference.md`
   - Contains: Whether CUDA and XPU run identical computational graphs
3. **Per-model reports** — e.g., `reports/<session>/per_model/eager/<model>_fp16_eval.md`
   - Contains: T1/T2/R per platform, per-op R_op, action items with shapes/strides
4. **Hardware specs** — `config/hardware_specs.yaml`
   - Contains: Peak TFLOPS, BW, ridge point per platform

---

## Output

A single markdown file: `reports/<session>/insights_summary.md`

---

## Structure

The report has exactly 3 sections:

### 1. Key Numbers (dashboard table)

A quick-reference table with the most important metrics:

| Metric to include | Source |
|-------------------|--------|
| Fleet Geomean R — XPU | summary report §1 |
| Fleet Geomean R — CUDA | summary report §1 |
| XPU R ≥ CUDA R (model count) | summary report §3 — count models where R_G31 ≥ R_4080 |
| XPU R < CUDA R (model list) | summary report §3 — list underperforming models |
| Worst XPU R | summary report §4 — lowest R model + root cause op |
| Best XPU R | summary report §4 — highest R model |
| Graph consistency | graph consistency report |
| Fix top-1 op impact | summary report §5 — op priority #1 delta |

### 2. Insight 1: XPU software efficiency vs CUDA

Frame from developer perspective — "how good is our kernel quality?"

- Compare R values (not wall-clock T2) — R normalizes away hardware differences
- List all models in a table: CUDA R, XPU R, who wins
- Emphasize the percentage of models where XPU R ≥ CUDA R
- Key message: R measures software quality, independent of HW specs

### 3. Insight 2: Specific kernel fixes to pull ahead

- Extract the underperforming models (XPU R < CUDA R)
- From per-model reports, identify the exact ops causing the gap (lowest R_op on XPU)
- Present as a prioritized fix table with: op name, affected model(s), XPU R_op vs CUDA R_op, root cause hint, expected fleet R improvement
- Key message: the fix list is short and well-defined

---

## Key Principles

1. **English only** — all text in English
2. **Developer framing** — we are XPU developers; goal is to prove and improve XPU software efficiency
3. **R-based comparison** — always compare roofline efficiency R (not raw wall time) since R isolates software quality from hardware specs
4. **Concise** — max 2 insights, no filler sections
5. **Actionable** — every claim backed by specific op names, shapes, and expected impact numbers

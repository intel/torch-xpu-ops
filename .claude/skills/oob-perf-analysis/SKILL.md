---
name: oob-perf-analysis
description: Generate and analyze T1/T2/R roofline reports for PyTorch OOB workloads comparing Intel XPU and NVIDIA CUDA. Use when working with eager profiling artifacts, per-model reports, fleet summaries, graph consistency, or XPU-vs-CUDA software efficiency analysis.
---

# OOB Perf Analysis Skill

Analyzes OOB eager-mode performance using T1/T2/R roofline methodology to compare XPU and CUDA software efficiency.

Core outputs (all under `agent_space_xpu/`, git-ignored):
- Per-model markdown reports under `agent_space_xpu/reports/<session>/models/`
- Fleet summary under `agent_space_xpu/reports/<session>/summary_eager_inference.md`
- Graph consistency report under `agent_space_xpu/reports/<session>/graph_consistency_eager_inference.md`
- Insights summary under `agent_space_xpu/reports/<session>/insights_summary.md`

## References

| File | Purpose |
|------|---------|
| [methodology.md](references/methodology.md) | T1/T2/R definitions, formulas, classification thresholds |
| [inputs.md](references/inputs.md) | Input file formats, completeness rules, output paths |
| [per-model-report.md](references/per-model-report.md) | Per-model analysis steps and 5-section report structure |
| [fleet-summary.md](references/fleet-summary.md) | Fleet aggregation steps and 7-section report structure |
| [graph-consistency.md](references/graph-consistency.md) | Graph consistency analysis |
| [insights.md](references/insights.md) | Developer-facing insights summary |
| [troubleshooting.md](references/troubleshooting.md) | Diagnosing abnormal R, trace issues, and data problems |

## Usage Modes

### Per-Model Report Mode

User has raw artifacts and wants model-level T1/T2/R analysis.

Follow: `methodology.md` → `inputs.md` → `per-model-report.md`

### Fleet Summary Mode

User wants fleet-wide comparison, model scorecards, op ranking, or projection-quality aggregation.

Follow: `fleet-summary.md` (references `graph-consistency.md` for Section 7)

### Graph Consistency Mode

User wants to compare CUDA vs XPU computational graphs.

Follow: `graph-consistency.md`

### Insights Mode

Per-model reports and fleet summary exist; user wants a concise developer-facing summary.

Follow: `insights.md`

### Troubleshooting Mode

User asks why R is abnormal, why traces disagree, or why results look suspicious.

Follow: `troubleshooting.md`

## Operating Rules

1. Read `methodology.md` before computing any metric.
2. Read `inputs.md` before accessing any artifact.
3. Report structure is deterministic — same inputs produce same section layout.

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: oob-perf-analysis.

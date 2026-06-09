---
name: oob-perf-analysis
description: Generate and analyze T1/T2/R roofline reports for PyTorch OOB workloads comparing Intel XPU and NVIDIA CUDA. Use when working with OOB Jenkins sessions, eager profiling artifacts, per-model reports, fleet summaries, graph consistency, or XPU-vs-CUDA software efficiency analysis.
---

# OOB Perf Analysis Skill

Use this skill for OOB eager performance analysis workflows centered on T1/T2/R roofline methodology, Jenkins session artifacts, per-model reports, fleet summaries, graph consistency, and developer-facing optimization insights.

Detailed reference:
- [overview.md](references/overview.md)
- [eager-session-workflow.md](references/eager-session-workflow.md)
- [eager-report-workflow.md](references/eager-report-workflow.md)
- [fleet-summary-workflow.md](references/fleet-summary-workflow.md)
- [graph-consistency-workflow.md](references/graph-consistency-workflow.md)
- [insights-workflow.md](references/insights-workflow.md)
- [data-contracts.md](references/data-contracts.md)
- [jenkins-pass-reference.md](references/jenkins-pass-reference.md)
- [report-structure-reference.md](references/report-structure-reference.md)
- [troubleshooting.md](references/troubleshooting.md)

## Usage Modes

### Jenkins Session Mode

Use when the user provides one of these inputs:

1. A session YAML with Jenkins trigger-job URLs
2. A request to launch a new OOB eager profiling session
3. Existing Jenkins build URLs or build numbers that need downloading and reporting

Follow:

- [eager-session-workflow.md](references/eager-session-workflow.md)
- [jenkins-pass-reference.md](references/jenkins-pass-reference.md)
- [data-contracts.md](references/data-contracts.md)

### Per-Model Report Mode

Use when the user already has raw OOB eager artifacts and wants model-level T1/T2/R analysis.

Follow:

- [eager-report-workflow.md](references/eager-report-workflow.md)
- [report-structure-reference.md](references/report-structure-reference.md)
- [data-contracts.md](references/data-contracts.md)

### Fleet Summary Mode

Use when the user wants fleet-wide comparison, model scorecards, op ranking, or projection-quality aggregation across many models.

Follow:

- [fleet-summary-workflow.md](references/fleet-summary-workflow.md)
- [report-structure-reference.md](references/report-structure-reference.md)
- [graph-consistency-workflow.md](references/graph-consistency-workflow.md)

### Insights Mode

Use when per-model reports and fleet summary already exist and the user wants a concise developer-facing summary of XPU vs CUDA software efficiency and optimization priorities.

Follow:

- [insights-workflow.md](references/insights-workflow.md)

### Troubleshooting Mode

Use when the user asks why R is abnormal, why XPU trails CUDA, why graph consistency fails, or why traces and unitrace do not match expectations.

Follow:

- [troubleshooting.md](references/troubleshooting.md)
- [data-contracts.md](references/data-contracts.md)
- [jenkins-pass-reference.md](references/jenkins-pass-reference.md)

## Operating Rules

1. Do not rely on local helper scripts under `scripts/` as the workflow contract.
2. Use the canonical session layout defined in `references/data-contracts.md`.
3. Treat `references/report-structure-reference.md` as the source of truth for report section structure.
4. Treat `references/jenkins-pass-reference.md` as the source of truth for the six-pass Jenkins workflow.
5. Prefer direct skill-guided workflow execution over script invocation.

If a calling workflow explicitly requires a skill marker, append this exact literal final line:
Custom skills applied: oob-perf-analysis.

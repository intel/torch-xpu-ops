<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# OOB Performance Analysis

Agent-assisted workflow for analyzing PyTorch OOB model performance on Intel XPU
vs NVIDIA CUDA using T1/T2/R roofline metrics.

## Workflow

```
launch_session.py (trigger 6 Jenkins jobs)
        │
        ▼
download_jenkins_artifacts.py ─► raw_logs/<session>/
        │
        ▼
prepare_flat_views.py ─► flat_views/<session>/
        │
        ├─► generate_all_eager_reports.py ─► per-model T1/T2/R reports
        ├─► generate_fleet_summary.py     ─► fleet summary
        ├─► compare_graphs.py             ─► graph consistency report
        └─► compare_projection_vs_actual.py ─► T1 vs actual analysis
        │
        ▼
oob-insights SKILL.md ─► insights_summary.md
```

## Directory Layout

```
tools/agentic_xpu/oob_perf_analysis/
├── README.md                           # This file
├── download_jenkins_artifacts.py       # Download sub-job artifacts from Jenkins
├── prepare_flat_views.py               # Create per-platform symlink views
├── generate_all_eager_reports.py       # Batch per-model T1/T2/R reports
├── generate_all_reports.py             # Batch report (alternate entry point)
├── generate_fleet_summary.py           # Fleet-level summary across all models
├── generate_report.py                  # Single per-model report
├── compare_projection_vs_actual.py     # T1 projection vs actual per-op comparison
├── compare_graphs.py                   # CUDA vs XPU graph consistency comparison
├── map_kernels_to_ops.py               # Map unitrace kernels to aten ops
├── parse_trace.py                      # Parse Chrome trace JSON
├── parse_unitrace.py                   # Parse unitrace JSON output
├── launch_session.py                   # Trigger 6 Jenkins jobs + full pipeline
└── oob300/                             # Model suite YAML lists
    ├── huggingface_inference.yaml
    ├── huggingface_training.yaml
    ├── timm_inference.yaml
    ├── timm_training.yaml
    ├── torchbench_inference.yaml
    └── torchbench_training.yaml

.github/skills/oob-report/             # OOB report generation skill
.github/skills/oob-insights/           # Insights summary skill
```

## Skills

| Skill | Location | Purpose |
|-------|----------|---------|
| `oob-report` | `.github/skills/oob-report/SKILL.md` | Per-model and fleet report generation |
| `oob-insights` | `.github/skills/oob-insights/SKILL.md` | Final insights summary |

Additional skill guides in `.github/skills/oob-report/`:
- `oob_profile_eager.md` — OOB 300 eager-mode profiling steps
- `oob_report_eager.md` — OOB 300 report generation details
- `oob_llm_profile.md` — HuggingFace LLM profiling steps
- `oob_llm_report.md` — HuggingFace LLM report generation
- `oob_compile_profile.md` — Compile-mode profiling steps
- `oob_compile_report.md` — Compile-mode report generation

## Usage

### 1. Prepare a Session YAML

Create a YAML listing the 6 Jenkins trigger job URLs:

```yaml
t1:           https://jenkins.example.com/job/trigger/<N>/
unitrace:     https://jenkins.example.com/job/trigger/<N>/
xpu_profiler: https://jenkins.example.com/job/trigger/<N>/
cuda_profiler: https://jenkins.example.com/job/trigger/<N>/
xpu_t2:       https://jenkins.example.com/job/trigger/<N>/
cuda_t2:      https://jenkins.example.com/job/trigger/<N>/
```

### 2. Download Artifacts

```bash
python tools/agentic_xpu/oob_perf_analysis/download_jenkins_artifacts.py \
  --session <session.yaml>
```

### 3. Prepare Flat Views

```bash
python tools/agentic_xpu/oob_perf_analysis/prepare_flat_views.py \
  raw_logs/<session> flat_views/<session>
```

### 4. Generate Reports

```bash
# Per-model reports
python tools/agentic_xpu/oob_perf_analysis/generate_all_eager_reports.py \
  --session raw_logs/<session>

# Fleet summary
python tools/agentic_xpu/oob_perf_analysis/generate_fleet_summary.py \
  --b70-dir flat_views/<session>/b70 \
  --4080s-dir flat_views/<session>/4080s \
  --suite-dir tools/agentic_xpu/oob_perf_analysis/oob300 \
  -o reports/<session>/summary_eager_inference.md
```

### 5. Launch Full Session (optional)

```bash
python tools/agentic_xpu/oob_perf_analysis/launch_session.py \
  --session <session.yaml> --dry-run
```

Requires `JENKINS_API_TOKEN` and `HF_TOKEN` environment variables.

## T1/T2/R Metrics

| Metric | Definition |
|--------|-----------|
| T1 | Roofline projection: sum of max(FLOPs/peak, bytes/BW) per op |
| T2 | Actual end-to-end CPU wall clock time |
| R = T1/T2 | Software efficiency (1.0 = perfect roofline utilization) |

## Hardware Specs

Override via `--config hardware_specs.yaml` on report scripts.

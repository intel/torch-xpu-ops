---
name: oob-report
description: "Use when generating T1/T2/R roofline analysis reports for PyTorch OOB models. Triggered by: launching Jenkins profiling jobs, downloading artifacts, generating per-model reports, fleet summary, comparing XPU vs CUDA performance, session YAML config, model list, or any OOB benchmarking analysis task."
---

# OOB Report Generation Skill

Generate T1/T2/R roofline analysis reports for PyTorch OOB (Out-Of-Box) models comparing Intel XPU (B70) vs NVIDIA CUDA (4080 SUPER).

---

## Capabilities

| # | Capability | Status | Trigger |
|---|-----------|--------|---------|
| 1 | **Download + Report** | ✅ Ready | User provides YAML with 6 Jenkins job links |
| 2 | **Launch + Download + Report** | 🔜 Planned | User provides model_list |

---

## Capability 1: Download & Generate Report

### Input

User provides a YAML file (e.g. `config/sessions/my_run.yaml`) containing 6 Jenkins trigger job URLs:

```yaml
# config/sessions/my_run.yaml
t1: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
unitrace: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
xpu_profiler: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
cuda_profiler: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
xpu_t2: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
cuda_t2: $JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/
```

**Session name** = YAML filename without extension (e.g. `my_run.yaml` → `my_run`).
**Models** = auto-parsed from Jenkins `summary.log` during download.

### Workflow

```bash
SESSION_YAML="<yaml file path>"
SESSION_NAME="<filename without .yaml>"

# 1. Download all artifacts
python scripts/download_jenkins_artifacts.py --session "$SESSION_YAML"

# 2. Create flat symlink views
python scripts/prepare_flat_views.py raw_logs/$SESSION_NAME flat_views/$SESSION_NAME

# 3. Generate per-model reports (output to models/ subdir)
python scripts/generate_all_eager_reports.py --session raw_logs/$SESSION_NAME

# 4. Generate fleet summary
python scripts/generate_fleet_summary.py \
  --b70-dir flat_views/$SESSION_NAME/b70 \
  --4080s-dir flat_views/$SESSION_NAME/4080s \
  --config config/hardware_specs.yaml \
  --suite-dir benchmark/oob300 \
  -o reports/$SESSION_NAME/summary_eager_inference.md
```

---

## 6 Jenkins Passes

| Pass | Device | Node | Purpose |
|------|--------|------|---------|
| `t1` | cuda | OOB-RTX4080 | FLOPs/bytes per op (roofline projection) |
| `unitrace` | xpu | OOB-B70 | Kernel-level GPU timing on XPU |
| `xpu_profiler` | xpu | OOB-B70 | Per-op GPU time on XPU |
| `cuda_profiler` | cuda | OOB-RTX4080 | Per-op GPU time on CUDA |
| `xpu_t2` | xpu | OOB-B70 | XPU wall-clock batch latency |
| `cuda_t2` | cuda | OOB-RTX4080 | CUDA wall-clock batch latency |

## T1/T2/R Formula

- **T1_compute** = FLOPs / 10^9 / peak_TFLOPS (ms)
- **T1_memory** = bytes / 10^9 / bandwidth_GBs * 1000 (ms)
- **T1_op** = max(T1_compute, T1_memory) — roofline bound
- **T1** = sum(T1_op) — theoretical minimum
- **T2** = actual measured GPU wall-clock time
- **R = T1 / T2** — roofline efficiency (1.0 = perfect)

## Hardware Specs

Read from `config/hardware_specs.yaml`. Do not hardcode values — always load specs from that file.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `launch_session.py` | Trigger 6 Jenkins jobs + full pipeline |
| `download_jenkins_artifacts.py` | Download artifacts from completed builds |
| `generate_all_eager_reports.py` | Batch per-model reports |
| `generate_report.py` | Single-model T1/T2/R report |
| `generate_fleet_summary.py` | Fleet summary (7 sections) |
| `prepare_flat_views.py` | Symlink views for fleet summary |
| `compare_projection_vs_actual.py` | Core T1 projection + R calc |
| `compare_graphs.py` | CUDA vs XPU graph comparison |

## Jenkins Access

- Set `JENKINS_URL` in your `.env` (see `tools/agentic_xpu/oob_perf_analysis/.env.example`)
- Some servers need `--user <user> --token <token>` (set `JENKINS_API_TOKEN` in `.env`)
- All: `curl -sk --noproxy "*"`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No T2 value" | Model failed at runtime |
| R > 1.0 | Projection overcounts |
| R << 1.0 | Kernel slow or ops missing |
| Build FAILED | Check console: OOM, conda env |
| Empty artifacts | Sub-job timed out |

## Jenkins API Details

### Querying trigger jobs
```bash
curl -sk --noproxy "*" "$JENKINS_URL/job/newOOB_launch_benchmark_trigger/<N>/api/json"
```
Extract: `result`, `actions[].parameters[]`, `artifacts` (contains `summary.log`).

### Distinguishing job types by parameters
| Job Type | Device | `OOB_ADDITION_PARAMS` |
|----------|--------|-----------------------|
| unitrace | xpu | (empty) — has `python.*.json` in artifacts |
| xpu profiler | xpu | `--profile_test` |
| cuda profiler | cuda | `--profile_test` |
| xpu T2 | xpu | (empty) — only `rcpi1-ins0.log` |
| cuda T2 | cuda | (empty) — only `rcpi1-ins0.log` |
| T1 | cuda | (empty) — produces `calcflops.txt` |

### summary.log format
```
framework,model_name,mode_name,compile_mode,precision,batch_size,cores_per_instance,instance,valid_ins,throughput,link,device
```
Note: `link` field has trailing space — must trim.

### T2 extraction from `rcpi1-ins0.log`
```
GPU Time per batch:  209.353 milliseconds
```

### Local directory structure
```
raw_logs/<session_name>/
├── metadata.json
└── <model>/
    ├── unitrace/python.<pid>.json
    ├── xpu_profiler/trace.json, profile_parser.log
    ├── cuda_profiler/trace.json, profile_parser.log
    ├── xpu_t2/rcpi1-ins0.log
    ├── cuda_t2/rcpi1-ins0.log
    └── t1/calcflops.txt
```

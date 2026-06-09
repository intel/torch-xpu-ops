---
name: oob-perf-analysis
description: Generate T1/T2/R roofline performance analysis reports for PyTorch OOB benchmark models comparing Intel XPU vs NVIDIA CUDA. Use when benchmarking models, analyzing performance bottlenecks, downloading Jenkins artifacts, creating roofline reports, comparing XPU/CUDA performance, or working with OOB profiling data (.yaml configs, .json traces, .log files).
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

User provides a YAML file containing 6 Jenkins trigger job URLs:

```yaml
t1: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
unitrace: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
xpu_profiler: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
cuda_profiler: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
xpu_t2: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
cuda_t2: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/
```

**Session name** = YAML filename without extension (e.g. `my_run.yaml` → `my_run`).
**Models** = auto-parsed from Jenkins `summary.log` during download.

### Workflow

```bash
SESSION_YAML="<yaml file path>"
SESSION_NAME="<filename without .yaml>"
SKILL_ROOT=".claude/skills/oob-perf-analysis"

# 1. Download all artifacts
python "$SKILL_ROOT/scripts/download_jenkins_artifacts.py" --session "$SESSION_YAML"

# 2. Create flat symlink views
python "$SKILL_ROOT/scripts/prepare_flat_views.py" raw_logs/$SESSION_NAME flat_views/$SESSION_NAME

# 3. Generate per-model reports (output to models/ subdir)
python "$SKILL_ROOT/scripts/generate_all_eager_reports.py" \
  --session raw_logs/$SESSION_NAME

# 4. Generate fleet summary
# --suite-dir is optional and only needed if you have external OOB suite YAMLs.
python "$SKILL_ROOT/scripts/generate_fleet_summary.py" \
  --b70-dir flat_views/$SESSION_NAME/b70 \
  --4080s-dir flat_views/$SESSION_NAME/4080s \
  -o reports/$SESSION_NAME/summary_eager_inference.md
```

The skill ships its own hardware spec config at
`.claude/skills/oob-perf-analysis/config/hardware_specs.yaml`.
The report scripts auto-detect it, so `--config` is only needed if you want to
override those defaults.

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

| Platform | Peak FP16 (TFLOPS) | DRAM BW (GB/s) |
|----------|-------------------:|---------------:|
| B70 (G31) | 154.0 | 532.0 |
| RTX 4080 SUPER | 100.96 | 716.8 |
| B580 | 93.0 | 410.0 |

## Repository Layout Expectations

- Run the helper scripts from this repository root, or use absolute paths.
- The scripts live under `.claude/skills/oob-perf-analysis/scripts/`, not a
  repo-level `scripts/` directory.
- `--suite-dir` is optional and should point to an external directory that
  contains OOB suite YAML files if you want per-suite grouping in the fleet
  summary.

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

- Most Jenkins servers support anonymous read access
- Private servers may need `--user <username> --token <api-token>`
- Use `curl -sk --noproxy "*"` for SSL/proxy bypass
- Set `JENKINS_SERVER` environment variable to override default server URL

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
curl -sk --noproxy "*" "https://<your-server>/job/newOOB_launch_benchmark_trigger/<BUILD_NUMBER>/api/json"
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

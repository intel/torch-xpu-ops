# Inputs

## Session Layout

The user provides the path to their raw logs directory. At the start of each session, copy it into `agent_space_xpu/` for temporary work:

```bash
cp -r <user_provided_path> agent_space_xpu/raw_logs/<session_name>/
```

Working layout under `agent_space_xpu/raw_logs/<session_name>/`:

```text
agent_space_xpu/raw_logs/<session_name>/
  <model>/
    t1/
      calcflops.txt
    xpu_profiler/
      trace.json
    cuda_profiler/
      trace.json
    unitrace/
      python.<pid>.json
    xpu_t2/
      rcpi1-ins0.log
    cuda_t2/
      rcpi1-ins0.log
```

## File Descriptions

| File | Used for |
|------|----------|
| `t1/calcflops.txt` | T1 projection: cumulative per-op FLOPs and bytes. Use the last benchmark iteration only. |
| `xpu_profiler/trace.json` | XPU per-op actual timing; bridge for unitrace-to-op mapping; graph consistency trace analysis |
| `cuda_profiler/trace.json` | CUDA per-op actual timing; graph consistency trace analysis |
| `unitrace/python.<pid>.json` | XPU kernel-level timing without profiler overhead. Preferred source for XPU per-op time. Single-iteration window. |
| `xpu_t2/rcpi1-ins0.log` | XPU wall-clock T2. Extract from line: `GPU Time per batch:  209.353 milliseconds` |
| `cuda_t2/rcpi1-ins0.log` | CUDA wall-clock T2. Same format. |

## Model Completeness

A model is complete for cross-platform comparison when it has all six files above.

A model is minimally usable for single-platform analysis when it has:
- `t1/calcflops.txt`
- platform-specific `trace.json`
- platform-specific T2 log

If a model is incomplete: mark it and allow downstream workflows to skip it. Do not fabricate values.

## Cross-Platform Pairing

Pair models using: model name + batch size + precision + test mode.

## Report Output Locations

All reports are written under `agent_space_xpu/` (git-ignored).

| Report | Path |
|--------|------|
| Per-model | `agent_space_xpu/reports/<session_name>/models/` |
| Fleet summary | `agent_space_xpu/reports/<session_name>/summary_eager_inference.md` |
| Graph consistency | `agent_space_xpu/reports/<session_name>/graph_consistency_eager_inference.md` |
| Insights | `agent_space_xpu/reports/<session_name>/insights_summary.md` |

## Pre-Analysis Validation

Before generating any report:

1. Session directory exists
2. Required files exist for each included model
3. T2 is extractable from T2 logs
4. calcflops file is parseable
5. trace file is parseable
6. (XPU unitrace) unitrace file exists when expected and kernel sum is not obviously inconsistent with T2

<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# OOB 300 Models — Compile Mode Report Generation

Guide for generating per-model compile-mode T1/T2/R analysis reports from profiling data
collected via `skills/oob_compile_profile.md`.

**Prerequisites**: Profiling data already collected (all 5 passes complete) for target models.
Each model needs: `*_t2_compile.txt` (T2), `*_calcflops.txt` (T1 extern), `*_t1_triton.txt`
(T1 triton), and `*_trace_compile.json` (profiler, for actual kernel times).

**Platforms**: B580 (XPU), B70/G31 (XPU), 4080S (CUDA). Reports compare across platforms
when data is available on multiple platforms.

**Working directory**: All commands run from the repo root (`/home2/jianyizh/pytorch.oob.bkc`).

---

## Per-Model Report

**Script**: `scripts/oob300/generate_report_compile.py`

### Input Files Per Model

| File | Source pass | Contains |
|------|------------|----------|
| `<model>_bs<BS>_t2_compile.txt` | Pass 1 | "GPU Time per batch: X.XXX milliseconds" |
| `<model>_bs<BS>_calcflops.txt` | Pass 2 | Per-op cumulative FLOPs/memory (eager mode) |
| `<model>_bs<BS>_t1_triton.txt` | Pass 3 | `TRITON_KERNEL_BYTES <name> <bytes>` lines |
| `<model>_bs<BS>_trace_compile.json` | Pass 4 | Chrome trace: Triton + extern kernel durations |

### Invocation

```bash
# Generate cross-platform reports for all models with complete data:
python scripts/oob300/generate_report_compile.py \
    --all \
    --b580-dir /home2/jianyizh/results_compile/b580/ \
    --b70-dir /home2/jianyizh/results_compile/b70/ \
    --4080s-dir /home2/jianyizh/results_compile/4080s/ \
    --config config/hardware_specs.yaml \
    --output-dir reports/oob300/per_model/compile/

# Generate single model (all available platforms auto-detected):
python scripts/oob300/generate_report_compile.py \
    --model nanogpt --bs 1024 --suite torchbench \
    --b580-dir /home2/jianyizh/results_compile/b580/ \
    --b70-dir /home2/jianyizh/results_compile/b70/ \
    --4080s-dir /home2/jianyizh/results_compile/4080s/ \
    --config config/hardware_specs.yaml \
    --output-dir reports/oob300/per_model/compile/
```

The script auto-detects which platforms have complete data (`compile_result.txt` present)
and generates cross-platform comparison when multiple platforms are available.

### Adding New Models

With `--all`, the script scans the data directories for any model with `compile_result.txt`
present. No separate model list file is needed — it auto-discovers from the eager YAML
model lists (`benchmark/oob300/`).

---

## Report Structure (5 sections)

Same structure as eager reports (`skills/oob_report_eager.md`), adapted for compile mode.

### Section 1: Summary

- Model metadata (name, suite, BS, precision, mode)
- Cross-platform T1/T2/R metrics table (T2, T1, T1_triton, T1_extern, T2_device, R)
- Hardware specs (peak TFLOPS, BW, ridge point)
- **Action Items**: auto-generated from per-kernel/op R analysis:
  - "Optimize XPU kernel": R_op < 0.80 on XPU but >= 0.80 on CUDA
  - "Fix projection": R_op < 0.80 on ALL platforms (undercounting) or > 1.05 on all (overcounting)
- Overall assessment (excellent/good/fair/poor)

### Section 2: Projection Quality

Per-platform table of kernels/ops where R_op deviates from 1.0 (> 1.05 or < 0.80):
- Sorted by actual time (biggest impact first)
- Columns: Kernel/Op, Type, R_op, Actual (ms), % T2, Proj (ms), Gap, Perf, 4080S R_op, Issue
- Issue diagnosis: "Overcounting", "Kernel slow", "Projection undercounts", "Undercounts or slow"
- **T2 Coverage by T1**: checks for uncaptured trace kernels (type = "other")

### Section 3: XPU vs CUDA Consistency

Checks whether compile produces the same kernel set across platforms:
- **Triton kernel set**: common / platform-only counts. Identical = same inductor graph.
- **Extern op set**: matches or differs (GEMM, SDPA, Conv)
- **Triton kernel bytes**: if bytes differ for same kernel name → different compilation

### Section 4: XPU vs 4080S Per-Op Efficiency

Per-kernel/op R comparison between each XPU platform and 4080S:
- Columns: Kernel/Op, Type, R_xpu, R_4080S, R_diff, XPU (ms), 4080S (ms), % T2, Verdict
- Sorted by % T2 (most impactful first)
- Verdict: **XPU wins** / XPU behind / ~tie (threshold: ±0.05)

### Section 5: Optimization Targets

Ranks kernels/ops by potential T2 saving if XPU matched 4080S efficiency.
Only includes items where R_xpu < R_4080S.

- Columns: Kernel/Op, Type, R_xpu, R_4080S, Actual, Target, Saving, % T2, Action
- Action: "Optimize kernel" or "Fix projection"
- Shows current R → potential R
- Summary: kernel optimization potential (ms and % T2)

---

## Output Location

```
reports/oob300/per_model/compile/<model>_bs<BS>_fp16_compile.md
```

---

## Interpreting Results

### R Values

| R Range | Assessment | Meaning |
|---------|------------|---------|
| >= 0.95 | Excellent | Hardware nearly fully utilized, minimal overhead |
| 0.85-0.95 | Good | Some kernel/scheduling overhead |
| 0.70-0.85 | Fair | Significant gaps — kernel slow or projection undercounts |
| < 0.70 | Poor | Major issues — investigate extern ops and trace |

### Common Gap Sources

| Source | How to Identify | Action |
|--------|----------------|--------|
| SDPA slow (R_kernel << 1) | micro_sdpa actual >> projected | Investigate flash attention backend |
| Conv slow | gen_conv actual >> projected | Check oneDNN kernel selection |
| GEMM slow | gemm_kernel actual >> projected | Check oneMKL GEMM tuning |
| Triton kernel slow | R_kernel < 0.9 | Check Triton autotuning / heuristics |
| Kernel launch overhead | T2 > T2_device | Graph dispatch, sync points |

### Compile vs Eager Comparison

For models with both eager and compile reports, compare:
- **R_compile vs R_eager**: compile should be >= eager (fusion reduces overhead)
- **T2_compile vs T2_eager**: compile should be <= eager (fusion reduces total time)
- **Triton kernel efficiency**: R_kernel close to 1.0 validates the bytes/BW model

---

## Key Rules

- Reports are **deterministic** — re-running with the same data produces identical output
- Reports support **cross-platform comparison** when data available on multiple platforms
- The script reads `config/hardware_specs.yaml` for platform HW specs (peak TFLOPS, BW)
- Trace.json is used for **actual kernel time** (Section 3 projection quality)
- Unitrace is NOT used in report generation (reserved for future per-kernel HW analysis)
- Platform naming: B580, G31 (=B70), 4080 (=4080S) in output; CLI uses --b580-*, --b70-*, --4080s-*
- When a model only has data on 1-2 platforms, omit missing platform columns

## Local Data Locations

| Platform | Path |
|----------|------|
| B580 | `/home2/jianyizh/results_compile/b580/` |
| B70/G31 | `/home2/jianyizh/results_compile/b70/` |
| 4080S | `/home2/jianyizh/results_compile/4080s/` |

Config: `config/hardware_specs.yaml`

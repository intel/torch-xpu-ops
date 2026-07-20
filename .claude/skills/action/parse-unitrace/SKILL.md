---
name: parse-unitrace
description: >
  Parse Intel unitrace chrome-kernel-logging JSON output into per-kernel GPU time
  summaries. Use when analyzing XPU kernel performance without profiler overhead.
---

# Parse Unitrace

Parse Intel unitrace `--chrome-kernel-logging` JSON output and produce per-kernel
GPU time summary tables. Filters out Level Zero runtime events (`ze*`) to show only
actual GPU compute kernels.

## When to use

- After collecting unitrace output on XPU (Intel GPU)
- To get per-kernel GPU time without torch profiler overhead
- To compare against profiler trace results for overhead estimation
- As ground-truth kernel timing for roofline analysis

## Why unitrace over torch profiler

| | torch.profiler | unitrace |
|--|----------------|----------|
| Overhead | Adds ~5-15% inflation | Near-zero overhead |
| Scope | CPU + GPU, full op tree | GPU kernels only |
| Platform | XPU + CUDA | XPU only (Level Zero) |
| Linkage | Has External id (op→kernel) | No op linkage (kernel-only) |

Use unitrace when you need accurate absolute GPU kernel times on XPU.
Use torch profiler when you need op-level attribution or CUDA support.

## Quick start

```bash
python .claude/skills/action/parse-unitrace/scripts/parse_unitrace.py python.1234.json --top 20 --sort-by gpu_time
```

## Usage

```bash
python .claude/skills/action/parse-unitrace/scripts/parse_unitrace.py <unitrace_file> [options]

Options:
  --top N              Show top N kernels (default: 20)
  --sort-by METRIC     Sort by: gpu_time, count (default: gpu_time)
  --timeline           Print kernels in time order
  --timeline-limit N   Max kernels to show in timeline (default: 50)
```

## How it works

1. **Load JSON** — unitrace outputs Chrome trace format (`python.<pid>.json`)
2. **Filter events** — Keep only `ph: "X"` (complete duration) events with `dur` field
3. **Skip `ze*` events** — Level Zero runtime calls (e.g., `zeCommandListAppendLaunchKernel`)
   are infrastructure, not user kernels
4. **Aggregate** — Group by kernel name, sum durations, count invocations

## Output format

### Summary table

```
KERNEL NAME                                                            DUR(ms)    DUR%  COUNT
----------------------------------------------------------------------------------------------------
gen_conv_kernel_xpu<...>                                                 5.234   42.1%     12
gemm_kernel<...>                                                         3.891   31.3%     24
SoftMaxForwardKernel<...>                                                1.234    9.9%      6
...
----------------------------------------------------------------------------------------------------
TOTAL                                                                   12.432    100%     78
```

### Timeline mode (`--timeline`)

Kernels printed in execution order with timestamp, useful for identifying
sequential dependencies and inter-kernel gaps.

## Unitrace output format

Unitrace with `--chrome-kernel-logging` produces a Chrome trace JSON:

```json
{
  "traceEvents": [
    {"ph": "X", "name": "zeCommandListAppendLaunchKernel", "ts": 100, "dur": 5, ...},
    {"ph": "X", "name": "gen_conv_kernel_xpu<float>", "ts": 200, "dur": 1500, ...},
    ...
  ]
}
```

- `ze*` prefixed events = Level Zero runtime (filtered out)
- All other `ph: "X"` events = actual GPU compute kernels

## Validation

Compare against end-to-end measurement:

```
kernel_sum = sum(all kernel durations from unitrace)
T2 = end-to-end wall clock time (from benchmark)

kernel_sum should be close to T2.
If kernel_sum << T2: significant host/launch overhead between kernels.
```

## Integration with other tools

| Tool | Purpose |
|------|---------|
| `.claude/skills/action/parse-pytorch-trace/scripts/parse_trace.py` | Parse torch profiler trace (has op linkage, but with overhead) |
| `.claude/skills/action/map-kernels-to-ops/scripts/map_kernels_to_ops.py` | Map unitrace kernels back to aten:: ops via trace.json External id + time-order matching |

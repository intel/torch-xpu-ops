---
name: map-kernels-to-ops
description: >
  Map unitrace GPU kernel durations to top-level aten:: ops using profiler
  trace.json as the bridge. Combines unitrace accuracy with op-level attribution.
---

# Map Kernels to Ops

Map unitrace GPU kernel durations to top-level aten:: ops by using the profiler
trace.json as a bridge. This gives per-op actual GPU time with unitrace accuracy
(no profiler overhead) while retaining the op-level semantics from the trace.

## When to use

- When you have both trace.json and unitrace output for the same run on XPU
- To get accurate per-op GPU time without profiler overhead inflation
- As input for roofline projection vs actual comparison

## Why this tool exists

| Source | Has op names | Accurate timing |
|--------|-------------|-----------------|
| trace.json | Yes (via External id) | No (profiler overhead) |
| unitrace | No (kernel names only) | Yes (near-zero overhead) |
| **This tool** | **Yes** | **Yes** |

The bridge: trace.json kernel events and unitrace kernel events appear in the
same execution order. Match them 1:1 by position, then use trace.json's
External id to attribute each unitrace duration to its owning aten:: op.

## Quick start

```bash
python .claude/skills/action/map-kernels-to-ops/scripts/map_kernels_to_ops.py trace.json unitrace.json --top 30
```

## Usage

```bash
python .claude/skills/action/map-kernels-to-ops/scripts/map_kernels_to_ops.py <trace_file> <unitrace_file> [options]

Options:
  --top N       Show top N ops by unitrace GPU time (default: 30)
  --detail      Show per-op detail with kernel names
  --csv FILE    Write per-kernel mapping to CSV
  --strict      Fail on kernel count or name mismatch (default: warn)
```

## How it works

1. **Parse trace.json** — Extract `kernel` events (with External id) and `cpu_op` events
2. **Build op index** — Map each External id to the aten:: op that owns it
3. **Parse unitrace** — Extract GPU kernels (filter `ze*` runtime events)
4. **1:1 matching by execution order** — Both kernel lists are sorted by timestamp;
   match them positionally
5. **Name verification** — Strip unitrace's `[SIMDxx {...} {...}]` suffix and compare
   against trace kernel name to catch ordering errors
6. **Aggregate** — Sum unitrace durations per top-level op

## Matching logic

```
trace.json kernels (sorted by ts):    unitrace kernels (sorted by ts):
  [0] gemm_kernel      ext_id=42       [0] gemm_kernel[SIMD32 ...]
  [1] softmax_kernel   ext_id=43       [1] softmax_kernel[SIMD16 ...]
  [2] gemm_kernel      ext_id=44       [2] gemm_kernel[SIMD32 ...]
         ↕ matched by position ↕
```

The `[SIMDxx {a; b; c} {d; e; f}]` suffix in unitrace names is stripped before
comparison. Mismatches are reported as warnings (or errors with `--strict`).

## Output format

### Per-op summary

```
====================================================
   # OP NAME                                  UNITRACE(ms)     UT%    CPU(ms)  Kernels  Input Dims
----------------------------------------------------
   0 aten::addmm                                   12.345   45.2%      0.891       24  [[64, 768], ...]
   3 aten::convolution                              5.678   20.8%      1.234       12  [[1, 3, 224, 224], ...]
...
```

### CSV output (`--csv`)

Two files:
- `output.csv` — Per-kernel row: trace name, unitrace name, durations, owning op
- `output_per_op.csv` — Per-op row: unitrace GPU time, CPU time, kernel count

## Handling mismatches

- **Kernel count mismatch**: trace.json may include `gpu_memcpy` events that unitrace
  does not capture. The script only uses `cat="kernel"` from trace (excludes memcpy).
- **Name mismatch**: Usually caused by kernel reordering between profiled vs
  non-profiled runs. The `--strict` flag makes this a hard error.

## Integration with other tools

| Tool | Role in pipeline |
|------|-----------------|
| `.claude/skills/action/parse-pytorch-trace/scripts/parse_trace.py` | Get op-level view from trace alone (with overhead) |
| `.claude/skills/action/parse-unitrace/scripts/parse_unitrace.py` | Get kernel-level view from unitrace alone (no op linkage) |
| **This tool** | Combine both: op attribution + accurate timing |

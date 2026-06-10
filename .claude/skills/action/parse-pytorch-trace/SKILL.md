---
name: parse-pytorch-trace
description: >
  Parse PyTorch profiler trace.json files into per-op GPU/CPU time summaries.
  Use when analyzing profiler traces, identifying hot ops, or diagnosing
  XPU/CUDA kernel performance from trace.json output.
---

# Parse PyTorch Trace

Parse torch profiler `trace.json` and produce per-op summary tables showing GPU time,
CPU time, kernel count, and kernel names for each top-level aten:: operator.

## When to use

- After collecting a profiler trace (`--profile_test` or `torch.profiler`)
- To identify which aten:: ops consume the most GPU time
- To get per-invocation detail for specific operators
- As input for cross-platform performance comparison

## Quick start

```bash
python .claude/skills/action/parse-pytorch-trace/scripts/parse_trace.py timeline/trace.json --top 30 --sort-by gpu_time
```

## Usage

```bash
python .claude/skills/action/parse-pytorch-trace/scripts/parse_trace.py <trace_file> [options]

Options:
  --top N              Show top N ops (default: 30)
  --sort-by METRIC     Sort by: gpu_time, cpu_time, kernel_count (default: gpu_time)
  --detail             Show per-invocation detail for top ops
```

## How it works

1. **Load trace.json** — Standard Chrome trace format from `torch.profiler`
2. **Extract events** — Separates `cpu_op` events (aten:: host ops) from `kernel`/`gpu_memcpy`
   device events
3. **Link via External id** — Maps device kernels back to their parent host ops using the
   `External id` field in the trace
4. **Merge nested ops** — Identifies top-level aten:: ops and merges nested child ops into
   parents (avoids double-counting)
5. **Aggregate** — Groups by op name, sums GPU/CPU time, counts calls and kernels

## Output format

### Summary table

```
==================================================================
OP NAME                                       GPU(ms)    GPU%    CPU(ms)  Calls  Kernels  Kernel Names
------------------------------------------------------------------
aten::addmm                                    12.345   45.2%     0.891     24       24  gemm_kernel
aten::convolution                               5.678   20.8%     1.234     12       12  gen_conv_kernel
aten::_softmax                                  2.345    8.6%     0.456      6        6  SoftMaxForwardKernel
...
==================================================================
TOTAL                                          27.321  100.0%     8.765
```

### Detail mode (`--detail`)

Per-invocation breakdown showing GPU time, CPU time, kernel count, and input dimensions
for each call of the top ops.

## Understanding the trace

### Event categories in trace.json

| Category | Description |
|----------|-------------|
| `cpu_op` | PyTorch host-side aten:: ops |
| `kernel` | Device compute kernels (SYCL/CUDA) |
| `gpu_memcpy` | Memory transfers (H2D, D2H) |
| `ac2g` | Async CPU-to-GPU flow events (linking host to device) |
| `python_function` | Python call stack frames |
| `xpu_runtime` / `xpu_driver` | Low-level XPU runtime calls (XPU only) |

### Key relationships

- Each host op (`cpu_op`) launches one or more device kernels
- Host ops and device kernels are linked via `External id`
- Nested aten:: ops (e.g., `aten::linear` containing `aten::addmm`) are merged into the
  top-level parent to avoid double-counting GPU time

## Validation

After parsing, verify:

```
sum(all kernel GPU durations) ≈ T2_device (end-to-end device time)
```

- If kernel sum significantly exceeds end-to-end time, the profiler has overhead inflation —
  treat per-op times with caution.
- If kernel sum < end-to-end time, there is inter-kernel gap (host overhead between kernel launches).

## Integration with other tools

| Tool | Purpose |
|------|---------|
| `.claude/skills/action/map-kernels-to-ops/scripts/map_kernels_to_ops.py` | Maps unitrace kernel durations to aten:: ops (XPU, no profiler overhead) |

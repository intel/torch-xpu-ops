# Methodology

## T1/T2/R Roofline

`R = T1 / T2` is the software efficiency metric for one model on one platform.

- `T1` is the roofline projection: the minimum time the hardware could theoretically take given the compute and memory work.
- `T2` is the measured wall-clock batch latency.
- `R` normalizes away hardware peak differences, making XPU and CUDA directly comparable on software efficiency.

`R_xpu / R_cuda > 1` means XPU software efficiency is better, even if raw `T2_xpu > T2_cuda`.

## T1 Calculation

Parse the calcflops file (`t1/rcpi1-ins0.log`; see `inputs.md` for exact column layout and file path) and use the last benchmark iteration only.

Per op:

```
T1_compute = FLOPs / peak_TFLOPS
T1_memory  = bytes / bandwidth_GBps
T1_op      = max(T1_compute, T1_memory)
```

Where:
- `FLOPs` = delta of `cum_flops` (col 1) between consecutive rows
- `bytes` = delta of the **platform-specific cache-adjusted memory column** (col 5 for B580, col 6 for 4080S/CUDA, col 7 for B70/G31) — never use raw memory (col 2)
- For ops in `VECTOR_ENGINE_OPS` (see `inputs.md`), force `FLOPs = 0` so T1 uses memory only

Fleet total:

```
T1 = sum(T1_op) over all ops
```

An op is compute-bound when `T1_compute > T1_memory`, memory-bound otherwise.

## T2 Extraction

Extract from `xpu_t2/rcpi1-ins0.log` or `cuda_t2/rcpi1-ins0.log`:

```
GPU Time per batch:  209.353 milliseconds
```

Rules:

1. `T2` is the wall-clock denominator for `R`. Never use `T2_device` as the denominator.
2. `T2_device` (kernel sum from trace or unitrace) is shown for reference only.

## Per-Op Actual Time

### XPU

Preferred source: unitrace mapped to aten ops via the profiler trace as bridge.

Mapping algorithm:

1. Parse `xpu_profiler/trace.json` to build `aten_op -> [kernel_name, ...]`
2. Parse `unitrace/python.<pid>.json` to get per-kernel wall-clock durations (no profiler overhead)
3. For each aten op, sum unitrace durations for its associated kernels
4. If a kernel is shared across multiple ops, distribute proportionally by profiler sub-durations; if sub-durations are zero, distribute equally and flag as approximate
5. Validate: `sum(attributed kernel times) ≈ sum(all unitrace kernels)`; if total exceeds T2 by >10%, suspect multi-iteration leakage

### CUDA

Use `cuda_profiler/trace.json` directly. No unitrace path for CUDA.

## Per-Op R Classification

`R_op = projected / actual`

| Condition | Classification |
|-----------|---------------|
| `R_op > 1.05` | `Overcounting` — projection overestimates; likely fusion or removed ops |
| `R_op < 0.80` and CUDA `R_op >= 0.80` | `Kernel slow` — XPU kernel genuinely underperforms |
| `R_op < 0.80` and CUDA `R_op < 0.80` | `Projection undercounts` — calcflops model gap, not kernel |
| `R_op < 0.80` and no CUDA available | `Undercounts or slow` — cannot disambiguate |

Threshold rationale: 0.80 is the empirically observed point where per-op issues become actionable optimization targets. Ops between 0.80 and 1.0 are usually acceptable overhead.

## Platform Naming

| User-facing | Internal ID | Config key | Notes |
|-------------|-------------|------------|-------|
| B70 | G31 | b70 | XPU, has unitrace |
| 4080S | 4080 | 4080s | CUDA, no unitrace |

See `config/hardware_specs.yaml` for peak compute, bandwidth, and ridge point per platform.

## Health Interpretation

| R | Health |
|---|--------|
| >= 0.95 | Excellent |
| 0.85 – 0.95 | Good |
| 0.70 – 0.85 | Fair |
| < 0.70 | Poor |

## Geomean (Fleet Level)

Equal weight per model:

```
geomean(R) = exp( (1/N) * sum(ln(R_i)) )
```

Fleet improvement estimate when fixing one op:

```
new_geomean = exp( (1/N) * sum(ln(R_i_adjusted)) )
delta = new_geomean - current_geomean
```

`R_i_adjusted` uses corrected per-op time for affected models; unchanged for unaffected models.

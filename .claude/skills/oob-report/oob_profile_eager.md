<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# OOB 300 Models — Eager Mode Profiling

Step-by-step instructions for eager mode performance profiling of the OOB 300 models
(TorchBench suite) on PyTorch XPU/CUDA. Covers T1 (roofline projection), T2 (wall clock),
per-op analysis, and cross-platform comparison.

**Workload**: OOB 300 models via `run.py` in the `RUIJIEZHONG66166/benchmark` repo (branch `develop`).

**Prerequisites**: Complete `skills/env_setup.md` before starting.

---

## Step 0: Confirm parameters

Before running, confirm with the user:

| Parameter | Options |
|-----------|---------|
| Platform | B580 (xpu), 4080S (cuda), B70 (xpu) |
| Workload | inference (eval, fp16) or training (train, bf16) |
| Scope | all models, specific suite, or specific model names |

---

## Step 1: Batch Run All Models

**Script**: `scripts/oob300/run_all_models.sh`

This is the primary entry point for both testers and the AI agent. The script handles the
full pipeline per model: install → baseline (T2) → unitrace (XPU) → calc_flops (T1) →
profiler trace → per-op analysis report.

### 1.1 Usage

```bash
# Run all inference models (default: reads all *_inference.yaml):
bash scripts/oob300/run_all_models.sh <device> <output_dir> [platform]

# Run specific models only:
bash scripts/oob300/run_all_models.sh <device> <output_dir> [platform] model1,model2,model3

# Run training models:
bash scripts/oob300/run_all_models.sh <device> <output_dir> [platform] "" \
  benchmark/oob300/torchbench_training.yaml \
  benchmark/oob300/timm_training.yaml \
  benchmark/oob300/huggingface_training.yaml

# Skip install step (when models are already installed):
SKIP_INSTALL=1 bash scripts/oob300/run_all_models.sh <device> <output_dir> [platform]
```

Arguments:

- `device`: xpu or cuda
- `output_dir`: directory to store results
- `platform`: G31, B580, or 4080 (for compare_projection_vs_actual.py)
- `model_filter` (4th arg, optional): comma-separated model names to run
- `model_list_yaml` (5th+ args, optional): YAML files from `benchmark/oob300/`
  (default: all `*_inference.yaml`). Precision and test mode are read from the YAML header.
- `SKIP_INSTALL=1`: skip `install.py` step (recommended when models are already installed,
  because `install.py` can hang on pip dependency resolution)

### 1.2 Model lists

Defined in `benchmark/oob300/`:

| File | Suite | Mode | Precision | Models |
|------|-------|------|-----------|--------|
| `torchbench_inference.yaml` | torchbench | eval | fp16 | 60 |
| `timm_inference.yaml` | timm | eval | fp16 | 60 |
| `huggingface_inference.yaml` | huggingface | eval | fp16 | 50 |
| `torchbench_training.yaml` | torchbench | train | bf16 | 40 |
| `timm_training.yaml` | timm | train | bf16 | 60 |
| `huggingface_training.yaml` | huggingface | train | bf16 | 44 |

### 1.3 Behavior

- The script sets proxy env vars (`http_proxy`, `https_proxy`) for pip dependency downloads.
  If behind a different proxy, override via environment before running.
- Skips models that already have a `*_report.txt` file (idempotent).
- If baseline fails (OOM, error), the model is skipped entirely.
- Timeout per step: 600s (10 min). For large models (e.g., hf_BigBird, ~850ms/batch),
  the profiler step may exceed this timeout. In that case, re-run the profiler manually
  with a larger timeout (e.g., `timeout 1200`), copy `timeline/trace.json`, and generate
  the report with `compare_projection_vs_actual.py`.
- **`--channels-last` is always used.** Without it, CNN models use NCHW layout and oneDNN conv
  falls back to a much slower path (e.g., AlexNet 11ms → 25ms, 2.3x regression).
- **NFS is NOT required for OOB 300.** Model weights are self-contained in the benchmark repo
  (installed via `install.py`) or downloaded at runtime by timm/transformers. Do NOT set
  `HF_HOME` to an NFS path — if NFS hangs, model loading will fail. Use local cache instead
  (e.g., `HF_HOME=/root/.cache/huggingface`). Set `HF_HUB_OFFLINE=1` after initial download
  to avoid network dependency on subsequent runs.

### 1.4 Output files per model

```
<output_dir>/<model>_bs<bs>_baseline.txt   # T2 from "CPU Wall Time per batch:" line
<output_dir>/<model>_bs<bs>_unitrace.json  # unitrace conditional trace (XPU only)
<output_dir>/<model>_bs<bs>_calcflops.txt  # calc_flops log (T1 data)
<output_dir>/<model>_bs<bs>_trace.json     # profiler trace
<output_dir>/<model>_bs<bs>_report.txt     # per-op analysis report
```

**Note**: Keep `trace.json` files until cross-platform reports have been generated.
Only delete after all platforms' data has been collected and analyzed.

### 1.5 Post-processing: Generate reports

See `skills/oob_report_eager.md` for per-model and batch report generation.

---

## Manual Single-Model Steps (Reference)

The following steps document what `run_all_models.sh` does internally. Use these when you
need to manually run or debug a specific model.

---

## Step 2: Run Eager Mode Benchmark (T2)

### 2.1 Run the benchmark

```bash
python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train>
```

### 2.2 If model is not installed

```bash
python install.py <model>
```

### 2.3 Output

- **CPU Wall Time per batch** = **T2** (end-to-end wall clock, the real user experience).
  This is what we use for R calculation and reporting.
- T2_device (sum of kernel durations) comes from trace.json (Step 3) or unitrace (Step 4),
  used for per-op breakdown analysis. The difference **T2 - T2_device** reveals host overhead.

---

## Step 3: Profile with PyTorch Profiler

### 3.1 Run benchmark with profiler

```bash
python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --profile_test
```

The profiler configuration is in `context_func.py` (see `benchmark/common/context_func.py`):
- `with_stack=True` — Python call stacks
- `record_shapes=True` — tensor shapes for each op
- `with_modules=True` — nn.Module hierarchy
- `experimental_config(verbose=True)` — extra verbose info

### 3.2 Output files

Saved to `timeline/` directory:
- `trace.json` — Chrome trace format (main file for analysis)
- `profile.pt` — key_averages table sorted by self device time
- `profile_detail.pt` — key_averages grouped by input shape

### 3.3 Visualize and parse trace.json

- **GUI**: Open `chrome://tracing` in Chrome and load `trace.json`
- **CLI**: Use `parse_trace.py` script:
  ```bash
  python parse_trace.py timeline/trace.json --top 30 --sort-by gpu_time
  ```
  Options:
  - `--top N` — Show top N ops (default: 30)
  - `--sort-by {gpu_time,cpu_time,kernel_count}` — Sort metric
  - `--detail` — Show per-invocation detail for top ops

### 3.4 Understanding the trace

- Each **host op** (e.g., `aten::addmm`) corresponds to one or more **device kernels**
- Host ops and device kernels are linked via `External id` in the trace
- Key columns: op name, call count, CPU time, GPU time, GPU time %, kernel names
- Categories in trace.json:
  - `cpu_op` — PyTorch host-side ops
  - `kernel` — Device kernels launched on XPU/CUDA
  - `gpu_memcpy` — Memory transfers (H2D, D2H)
  - `ac2g` — Async CPU-to-GPU flow events (linking host to device)
  - `python_function` — Python call stack frames
  - `xpu_runtime` / `xpu_driver` — Low-level XPU runtime/driver calls (XPU only)

---

## Step 4: Per-Op Kernel Time with Unitrace (XPU only)

### 4.1 Purpose

Unitrace with `--chrome-kernel-logging` collects per-op GPU kernel times. This is the
**preferred method for XPU per-op T2** because:
- **No profiler overhead** — unlike torch profiler which can inflate kernel times
- **Conditional collection** — captures exactly one iteration, avoiding multi-iteration slicing errors
- **Accurate kernel-level timing** — directly from the Level Zero runtime

Torch profiler (Step 3) is still useful for CUDA and for quick visualization, but for XPU per-op
analysis, unitrace conditional collection produces more reliable data.

### 4.2 Running unitrace with conditional collection

```bash
UNITRACE_LAST_ITER=1 \
  <unitrace_bin> --chrome-kernel-logging --start-paused \
  python run.py <model> -d xpu --precision <dtype> --channels-last --bs <batch_size> --test eval
```

Key flags:
- `--chrome-kernel-logging` — output in Chrome trace format (JSON), one event per GPU kernel
- `--start-paused` — start with collection disabled; only collect when `PTI_ENABLE_COLLECTION=1`
- `UNITRACE_LAST_ITER=1` — tells the patched `run.py` to enable collection on the last iteration

`<unitrace_bin>` path is defined in `config/machines.yaml` per machine.

### 4.3 Output

Produces `python.<pid>.json` in the current directory — Chrome trace format with one event per
GPU kernel:
- `name`: kernel function name (e.g., `gen_conv_kernel`, `ClampScalarFunc`, `gemm_kernel`)
- `dur`: duration in microseconds
- `ts`: timestamp in microseconds

### 4.4 Kernel-to-op mapping (using torch profiler as reference)

Unitrace only captures device kernel names and timestamps — it does not have the host-side
aten op names or External id linkage. To map unitrace kernels back to aten ops:

1. **Build the mapping from torch profiler trace.json** (Step 3):
   - In `trace.json`, each host-side `cpu_op` (e.g., `aten::addmm`) is linked to its device
     kernels via the `External id` field.
   - Extract the ordered sequence of (kernel_name, op_name, input_shapes) from the profiler trace.
     This gives the ground-truth mapping and the execution order.

2. **Apply the mapping to unitrace by time order**:
   - Unitrace kernels appear in chronological order (sorted by `ts`).
   - The kernel execution order is the same in both traces (same model, same iteration).
   - Walk both sequences in parallel: for each unitrace kernel (in time order), match it to the
     corresponding kernel from the profiler sequence, inheriting the op name and shapes.
   - Kernel name verification: unitrace appends `[SIMDxx {a; b; c} {d; e; f}]` suffix to kernel
     names; strip this suffix and compare against the trace.json kernel name to catch ordering errors.

This approach is more reliable than keyword-based heuristics because:
- One kernel name can map to different ops in different contexts
- New/unknown kernel names are handled automatically
- Input shapes are preserved for per-invocation analysis

**Prerequisite**: You must have a torch profiler trace (Step 3) before you can analyze unitrace
results at per-op granularity.

**Script**: `scripts/oob300/map_kernels_to_ops.py`

```bash
# Per-op summary (top 30 ops by unitrace GPU time)
python map_kernels_to_ops.py timeline/trace.json python.<pid>.json --top 30

# Per-op detail with kernel names
python map_kernels_to_ops.py timeline/trace.json python.<pid>.json --detail

# Export per-kernel and per-op CSV for further analysis
python map_kernels_to_ops.py timeline/trace.json python.<pid>.json --csv output.csv
```

The script outputs two views:
- **Per-op summary**: each top-level aten:: op with its aggregated unitrace GPU time, percentage,
  CPU time, kernel count, and input dimensions
- **Per-op detail**: execution-order listing of every top-level op with its constituent kernel names

The `--csv` option writes two files: `output.csv` (per-kernel mapping) and `output_per_op.csv`
(per-op aggregation). These are used by Step 6 for projection vs actual comparison.

### 4.5 Why conditional collection matters

Without conditional collection, unitrace captures all iterations. The GPU runs continuously
with no clean gaps between iterations, making it impossible to reliably isolate a single
iteration by timestamp slicing. This caused earlier bugs where "unitrace kernel sum > E2E time"
due to kernels from adjacent iterations leaking into the measurement window.

With `--start-paused` + `PTI_ENABLE_COLLECTION`, only the last iteration's kernels are recorded,
giving a clean single-iteration trace with accurate kernel sums matching E2E time.

### 4.6 Validation

After collecting, verify:

1. **Unitrace kernel sum vs T2**:
   ```
   sum(all kernel durations in unitrace.json) ≈ T2 (E2E baseline)
   ```
   If kernel sum significantly exceeds T2, the collection window is wrong.
   If kernel sum < T2, there is inter-kernel gap (host overhead between kernel launches).

2. **Unitrace kernel sum vs torch profiler kernel sum**:
   ```
   sum(unitrace kernels) ≈ sum(torch profiler device kernels)
   ```
   If torch profiler kernel sum is significantly higher than unitrace, the profiler is adding
   overhead to kernel measurements. In that case, use unitrace as the ground truth for per-op
   timing and treat profiler data with caution.

---

## Step 5: Roofline Model Projection (T1)

### 5.1 Purpose

Use `DispatchLog` (TorchDispatchMode) in `context_func.py` to intercept every aten op and
calculate its FLOPs and memory access bytes for roofline projection.

### 5.2 Run command

```bash
Calculate_Flops=1 python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train>
```

**Important**: DispatchLog prints per-op cumulative data for every iteration (warmup + benchmark).
Only use the **last iteration** for T1 calculation — earlier iterations may have different behavior
due to warmup effects. The iteration boundary markers (`========== ITER x/n ==========`) make it
easy to identify which output belongs to which iteration.

### 5.3 How DispatchLog works

See `benchmark/common/context_func.py` for full implementation.

- Subclasses `TorchDispatchMode`, intercepts every aten op via `__torch_dispatch__`
- For each op, calculates:
  - **memory**: sum of `numel * element_size` for all tensor inputs and outputs (respects stride=0
    for broadcast tensors)
  - **flops**: compute-specific calculation for GEMM/conv/SDPA/layernorm/softmax/batch_norm etc.
- Skips pure-view ops (view, transpose, slice, permute, expand, as_strided, etc.) — no real compute
- Handles reshape: only counts memory if storage pointer changes (actual copy, not just view)
- Decomposes `aten::matmul` and `aten::linear` into lower-level ops

### 5.4 Platform-specific cache modeling

Small ops whose memory access < cache threshold get reduced cost (not zero — scaled by
DRAM_BW / L2_BW ratio). Hardware specs are in `config/hardware_specs.yaml`:

**Note**: `context_func.py` currently hardcodes these values. They should match
`config/hardware_specs.yaml`. Refactoring to read from config is a future TODO.

### 5.5 Per-op roofline classification

For each op, compute arithmetic intensity = `op_flops / op_memory`:
- If `intensity > ridge_point` → **compute-bound** → projected time = flops / peak_TFLOPS
- If `intensity ≤ ridge_point` → **memory-bound** → projected time = memory / bandwidth

Ridge point = peak_TFLOPS / bandwidth (OPs per byte). Can be derived from
`config/hardware_specs.yaml`; do NOT hardcode.

Model T1 = sum of per-op projected times.

### 5.6 Per-op output format

Each op prints a cumulative line (values are cumulative — take consecutive diffs for per-op):
```
{op_name}|{cum_flops}|{cum_memory}|{cum_gemm_conv_flops}|{cum_gemm_conv_memory}|
{mem_B580}|{mem_4080}|{mem_G31}|{gemm_conv_mem_B580}|{gemm_conv_mem_4080}|{gemm_conv_mem_G31}|
{overlapped_flops_B580}|{overlapped_flops_4080}|{overlapped_flops_G31}|
{overlapped_memory_B580}|{overlapped_memory_4080}|{overlapped_memory_G31}|
{overlapped_gemm_conv_flops_B580}|...|{overlapped_gemm_conv_memory_G31}|
args:{input_shapes}|zero:{is_all_zero}
```

### 5.7 Final summary output

At the end, totals are printed:
```
memory: <total_memory_bytes>
flops: <total_flops>
memory_gemm_conv: <total_gemm_conv_memory>
flops_gemm_conv: <total_gemm_conv_flops>
memory_B580 / memory_G31 / memory_4080: <with cache threshold applied>
overlapped_flops_<platform> / overlapped_memory_<platform>: <roofline-classified totals>
```

### 5.8 Notes

- `Calculate_Flops=1` runs through DispatchLog on CPU-side — execution is much slower than normal
  (no actual GPU compute, just accounting)
- Can combine with `SAVE_ARGS=1` to save op metadata for replay
- Can combine with `OP_REPLAY=1` to microbenchmark each op individually

---

## Step 6: Per-Op Projection vs Actual Comparison

### 6.1 Purpose

Compare roofline projection time (T1, from Calculate_Flops) against actual GPU kernel time
at per-op granularity. This identifies:
- **Well-predicted ops** — projection ≈ actual
- **Overcounted ops** — projection > actual (e.g., op fusion in runtime)
- **Undercounted ops** — projection < actual (kernel inefficiency or FLOPs underestimate)

### 6.2 Sources for "actual" per-op GPU time

There are two sources for per-op actual GPU time, with different tradeoffs:

| Source | Script | Pros | Cons |
|--------|--------|------|------|
| **Unitrace** (via `map_kernels_to_ops.py`) | `map_kernels_to_ops.py` | No profiler overhead; kernel sum ≈ T2; most accurate | XPU only; requires both trace.json and unitrace.json |
| **Torch profiler** (direct from trace.json) | `compare_projection_vs_actual.py` | Works on both XPU and CUDA | May have profiler overhead inflation; known M2D memcpy bug on XPU |

**Recommendation**: On XPU, use unitrace mapped through `map_kernels_to_ops.py` (Step 4.4) as
the ground truth for per-op actual time. On CUDA, use torch profiler trace directly (unitrace
is not available).

### 6.3 Prerequisites

- `calcflops.txt` — from Step 5:
  ```bash
  Calculate_Flops=1 python run.py <model> ... > calcflops.txt 2>&1
  ```
- For XPU: `trace.json` (Step 3) + `unitrace.json` (Step 4) → run `map_kernels_to_ops.py` (Step 4.4)
- For CUDA: `trace.json` (Step 3)

### 6.4 Script usage

```bash
python compare_projection_vs_actual.py calcflops.txt trace.json \
  [--top 40] [--sort-by diff_ms|proj|actual] [--platform G31|B580|4080]
```

Options:
- `--sort-by diff_ms`: sort by absolute difference (default)
- `--platform`: select which platform's cache-adjusted memory to use

### 6.5 How the script works

1. **Parse calc_flops**: reads cumulative per-op data, diffs consecutive lines for per-op
   `delta_flops` and `delta_memory`
2. **Per-op roofline classification**: computes arithmetic intensity, classifies as compute/memory
   bound, calculates projected time
3. **Parse trace.json**: extracts `aten::` cpu_ops, links each to device kernels via External id,
   sums GPU duration per op
4. **Name normalization**: strips variant suffixes (`.Tensor`, `.Scalar`, `.self`, `.start`, etc.)
   so calc_flops `aten::add.Tensor` matches trace `aten::add`
5. **Aggregate by op name**: sums projection and actual times per op type, reports ratio and diff

### 6.6 Common patterns

- **SDPA fusion**: DispatchLog sees unfused `bmm` + `softmax` + `bmm`, but runtime fuses into
  `aten::_scaled_dot_product_fused_attention_overrideable` (XPU) or
  `aten::_scaled_dot_product_flash_attention` (CUDA). This causes overcounting in projection.
- **`_to_copy` (dtype cast)**: Memory counted in DispatchLog but may be fused into downstream
  kernels → overcounting.
- **`aten::copy_` (M2D memcpy)**: On XPU, may show large device time in trace but overlaps with
  compute kernels — does not contribute to wall time. Not in calc_flops output.
- **Profiler overhead**: Always verify `sum(kernel durations) ≈ E2E GPU time`. If kernel sum
  exceeds E2E by >10%, trace has profiler overhead inflation and must be manually inspected.
  See https://github.com/intel/torch-xpu-ops/issues/3048 for known XPU profiler issues.
- **Remote SSH profiling**: Traces generated via remote SSH tend to show higher kernel times than
  those run directly on the machine. Prefer running `--profile_test` locally and copying trace back.

---

## Step 7: Running on CUDA

Commands are the same as XPU but with `-d cuda`:

```bash
# Baseline (T2)
CUDA_VISIBLE_DEVICES=<value> python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test eval

# Profiler (trace.json)
CUDA_VISIBLE_DEVICES=<value> python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test eval --profile_test

# Calculate_Flops (T1)
CUDA_VISIBLE_DEVICES=<value> Calculate_Flops=1 python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test eval
```

### CUDA-specific differences from XPU

1. **SDPA op name**: CUDA uses `aten::_scaled_dot_product_flash_attention` which decomposes to
   `aten::_flash_attention_forward`. XPU uses
   `aten::_scaled_dot_product_fused_attention_overrideable`. The comparison script normalizes both
   to `aten::sdpa_forward`.

2. **No M2D memcpy overhead**: CUDA trace shows negligible `aten::copy_` time vs XPU which may
   show large overlapped memcpy.

3. **CUPTI overhead**: CUDA profiler traces may have inflated kernel times due to CUPTI
   instrumentation. Always verify `sum(kernel durations) ≈ E2E`. If inflated (>10% difference),
   regenerate trace.

4. **No unitrace**: Unitrace is XPU-only. For CUDA per-op timing, use torch profiler trace.

5. **HuggingFace cache**: On CUDA machines, set `HF_HOME` to a local cache directory to avoid
   re-downloading models. See `config/machines.yaml` for machine-specific paths.

---

## Step 8: T1/T2/R Software Efficiency Analysis

### 8.1 Framework

- **T1** = Roofline projection time = **Σ max(op_compute_time, op_memory_time)** for each op.
  Each op is independently classified as compute-bound or memory-bound based on its arithmetic
  intensity vs the platform's ridge point. T1 is the sum of per-op projections, NOT the
  aggregate `max(total_compute, total_memory)`.
- **T2** = End-to-end CPU wall clock time (from benchmark). This is the user experience metric.
- **T2_device** = Sum of kernel durations from trace.json or unitrace. Used for per-op breakdown.
- **R = T1 / T2** = Software efficiency (closer to 1 = better). Always use T2, not T2_device.
- **R_xpu / R_cuda** = Relative software efficiency ratio across platforms

### 8.2 Why R < 1 (common causes)

1. **Cache effect (biggest factor at small batch sizes)** — Working set partially fits in GPU cache.
   Roofline assumes DRAM bandwidth, but cache hits make actual faster than projected for
   memory-bound ops. R improves at larger batch sizes when data exceeds cache.

2. **Kernel inefficiency** — Specific ops achieve much less than peak bandwidth or TFLOPS. If one
   op has consistently low projection/actual ratio across batch sizes and platforms, it's a real
   kernel performance problem.

3. **Projection undercounting** — FLOPs or memory bytes estimate is wrong. Examples:
   - `native_layer_norm`: FLOPs calculation may undercount actual kernel work
   - SDPA: complex memory access patterns not captured by simple roofline
   - `topk`: memory estimate too small

### 8.3 Why R > 1 (common causes)

1. **Peak calibration** — If hardware spec peak is conservative (e.g., B70 calibrated peak of
   154 TFLOPS exceeds the 120 TFLOPS used in older versions of context_func.py), T1 is
   underestimated → R > 1. Ensure `context_func.py` uses values from `config/hardware_specs.yaml`.

2. **FLOPs overcounting** — DispatchLog may count ops that get fused away at runtime (e.g.,
   unfused SDPA components when runtime uses fused kernel).

### 8.4 Sanity checks

- Always verify: `sum(kernel durations from trace) ≈ T2_device`
- If `sum(kernels) > T2_device` → trace has profiler overhead inflation. Regenerate trace or
  identify and exclude inflated ops. Do NOT use inflated traces for per-op analysis.
- If `T2 ≈ T2_device` → minimal host overhead, GPU is the bottleneck (good).
- If `T2 >> T2_device` → significant host overhead. Check for excessive CPU work, synchronization
  points, or kernel launch overhead.

### 8.5 Report generation

See `skills/oob_report_eager.md` for report generation methodology, template, and script usage.

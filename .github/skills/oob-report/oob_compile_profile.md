<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# OOB 300 Models — Compile Mode Profiling

Step-by-step instructions for torch.compile mode performance profiling of the OOB 300 model
subset. Collects T2 (compile wall clock), T1 inputs (Triton kernel bytes + eager calcflops
for extern ops), profiler trace, and unitrace.

**Workload**: OOB 300 models via `run.py` in the `RUIJIEZHONG66166/benchmark` repo (branch `develop`).

**Prerequisites**: Complete `skills/env_setup.md` before starting.

---

## Key Concepts

### T1 Compile Model

```
T1_compile = T1_triton + T1_extern
```

- **T1_triton** = Σ(kernel_bytes / BW) — all fused Triton kernels are memory-bound by definition
- **T1_extern** = Σ max(FLOPs/peak, bytes/BW) — unfused library calls (GEMM, SDPA, conv)
  sourced from eager calcflops output

### SDPA Cases

| Case | Eager | Compile | Detection | Action |
|------|-------|---------|-----------|--------|
| 1 (decomposition) | has SDPA | decomposes to BMM+softmax+BMM | `_safe_softmax` in Triton kernel names | Replace flash proj with decomposed BMM roofline |
| 2 (fused) | has SDPA | keeps fused SDPA | SDPA in eager, no `_safe_softmax` in compile | No correction needed |
| 3 (graph fusion) | manual bmm+softmax | fuses into SDPA | No SDPA in eager, `_scaled_dot_product` in compile | Replace BMM projs with SDPA proj |
| 4 (no SDPA) | no SDPA | no SDPA | — | No action |

---

## Step 0: Confirm Parameters

Before running, confirm with the user:

| Parameter | Options |
|-----------|---------|
| Platform | B580 (xpu), B70/G31 (xpu), 4080S (cuda) |
| Workload | inference (eval, fp16) or training (train, bf16) |
| Scope | all models, specific suite, or specific model names |

---

## Step 1: Batch Run (5-Pass Pipeline)

**Script**: `scripts/oob300/run_all_models_compile.sh`

### 1.1 Usage

```bash
# Run all inference models (default: reads all *_inference.yaml):
bash scripts/oob300/run_all_models_compile.sh <device> <output_dir> <PLATFORM>

# Run specific models only:
bash scripts/oob300/run_all_models_compile.sh <device> <output_dir> <PLATFORM> model1,model2

# Run training models:
bash scripts/oob300/run_all_models_compile.sh <device> <output_dir> <PLATFORM> "" \
  benchmark/oob300/torchbench_training.yaml \
  benchmark/oob300/timm_training.yaml \
  benchmark/oob300/huggingface_training.yaml

# Skip install step:
SKIP_INSTALL=1 bash scripts/oob300/run_all_models_compile.sh <device> <output_dir> <PLATFORM>
```

Arguments:

- `device`: xpu or cuda
- `output_dir`: directory to store results
- `PLATFORM`: G31, B580, or 4080 (for calc_t1_compile.py)
- `model_filter` (4th arg, optional): comma-separated model names to run
- `yaml_files` (5th+ args, optional): YAML files from `benchmark/oob300/`
  (default: all `*_inference.yaml`). Precision and test mode are read from the YAML header.

### 1.2 5 Passes Per Model

| Pass | Purpose | Command | Output |
|------|---------|---------|--------|
| 1 | T2 compile | `python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval\|train> --compile` | `*_t2_compile.txt` |
| 2 | T1_extern (eager calcflops) | `Calculate_Flops=1 python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval\|train>` | `*_calcflops.txt` |
| 3 | T1_triton (kernel bytes) | `TORCHINDUCTOR_BENCHMARK_KERNEL=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python run.py ... --compile` | `*_t1_triton.txt` |
| 4 | Profiler (per-kernel actual) | `python run.py ... --compile --profile_test` | `*_trace_compile.json` |
| 5 | Unitrace (XPU L0 times) | `unitrace --chrome-kernel-logging --start-paused python run.py ... --compile` | `*_unitrace_compile.json` |

### 1.3 Triton Patch for Pass 3

The Triton heuristics patch must be applied **only for Pass 3** to instrument kernel byte
counting. Apply before Pass 3, revert immediately after:

```bash
# Before Pass 3:
python scripts/oob300/patch_triton_heuristics.py apply

# Run Pass 3 ...

# After Pass 3 (before Pass 4/5):
python scripts/oob300/patch_triton_heuristics.py revert
```

**Important**: The patch modifies Triton's heuristics.py in-place. If left applied during
Pass 4/5, it may interfere with normal compilation. Always revert after Pass 3 completes.

### 1.4 Behavior

- Precision and test mode are read from the YAML header (same as eager `run_all_models.sh`).
- Skips models that already have a `*_compile_result.txt` file (idempotent).
- If T2 compile fails (OOM, dynamo error), the model is skipped entirely.
- Timeout per pass: 900s (15 min). Compile mode is significantly slower due to graph capture.
- **`--channels-last` is always used.**
- **Pass 5 (unitrace) is skipped on CUDA** — unitrace is XPU-only.
- **Pass 2 (calcflops) runs in eager mode** (no `--compile`) — it captures the op-level
  FLOPs/bytes that remain as extern calls after compilation.

### 1.5 Output files per model

```
<output_dir>/<model>_bs<BS>_t2_compile.txt        # Pass 1: T2
<output_dir>/<model>_bs<BS>_calcflops.txt         # Pass 2: eager FLOPs/bytes
<output_dir>/<model>_bs<BS>_t1_triton.txt         # Pass 3: TRITON_KERNEL_BYTES lines
<output_dir>/<model>_bs<BS>_trace_compile.json    # Pass 4: profiler trace
<output_dir>/<model>_bs<BS>_unitrace_compile.json # Pass 5: unitrace (XPU only)
<output_dir>/<model>_bs<BS>_compile_result.txt    # Final: T1/R calculation output
```

### 1.6 Post-processing: Generate reports

See `skills/oob_compile_report.md` for per-model report generation.

---

## Step 2: Compute T1 and R

**Script**: `scripts/oob300/calc_t1_compile.py`

```bash
python scripts/oob300/calc_t1_compile.py \
    --calcflops <output_dir>/<model>_bs<BS>_calcflops.txt \
    --triton-bytes <output_dir>/<model>_bs<BS>_t1_triton.txt \
    --t2 <T2_ms> \
    --platform <PLATFORM> \
    --verbose
```

Machine-readable output:
```
R_COMPILE=0.9583
T1_TRITON_MS=50.6849
T1_EXTERN_MS=131.9599
T1_TOTAL_MS=182.6448
T2_COMPILE_MS=190.5920
SDPA_DECOMPOSED=0
EAGER_NEEDS_SDPA=0
```

---

## Manual Single-Model Steps (Reference)

The following steps document what `run_all_models_compile.sh` does internally. Use these when
you need to manually run or debug a specific model.

---

## Step 2: T2 Compile Benchmark

### 2.1 Purpose

Measure end-to-end wall clock time (T2) for the compiled model. This is the denominator in
R = T1/T2.

### 2.2 Run

```bash
python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile
```

### 2.3 Output

- **GPU Time per batch** = **T2** (end-to-end wall clock for the compiled graph execution).
- First invocation triggers `torch.compile` graph capture and compilation (slow).
  Subsequent iterations run the compiled graph at full speed. The benchmark reports the
  steady-state time after warmup.

---

## Step 3: CalcFlops Eager (T1_extern)

### 3.1 Purpose

Capture per-op FLOPs and memory bytes for ops that remain as **extern calls** after compilation
(GEMM, SDPA, conv). These are not fused by Triton and must be projected individually.

### 3.2 Why eager mode (no --compile)

`torch.compile` replaces most ops with fused Triton kernels. `Calculate_Flops` (DispatchLog)
needs to intercept individual aten ops to count their FLOPs/bytes. Running in eager mode
captures the full op graph — the extern ops (GEMM, SDPA, conv) appear identically in both
eager and compile, so their FLOPs/bytes are the same.

### 3.3 Run

```bash
Calculate_Flops=1 python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train>
```

**Important**: Only use the **last iteration** output for T1 calculation. See
`skills/oob_profile_eager.md` Step 5 for details on DispatchLog output format.

### 3.4 Output

Same format as eager calcflops — per-op cumulative lines. `calc_t1_compile.py` filters to
only extern ops (GEMM, conv, SDPA) based on `EXTERN_OPS` list.

---

## Step 4: Triton Kernel Bytes (T1_triton)

### 4.1 Purpose

Measure the total memory bytes read+written by each fused Triton kernel. Since Triton kernels
fuse elementwise/reduction/normalization ops that are inherently memory-bound, their roofline
projection is simply `T1_triton = bytes / BW`.

### 4.2 How it works

The `patch_triton_heuristics.py` script patches Triton's `heuristics.py` to print
`TRITON_KERNEL_BYTES <kernel_name> <bytes>` for each kernel invocation during compilation.
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` forces fresh compilation (ignoring cached kernels)
so the patched code executes.

### 4.3 Run

```bash
# Apply patch BEFORE this pass:
python scripts/oob300/patch_triton_heuristics.py apply

# Run:
TORCHINDUCTOR_BENCHMARK_KERNEL=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile

# Revert patch AFTER this pass:
python scripts/oob300/patch_triton_heuristics.py revert
```

### 4.4 Output

Lines in stdout:
```
TRITON_KERNEL_BYTES triton_poi_fused_add_mul_pow_tanh_view_2 805306368
TRITON_KERNEL_BYTES triton_per_fused_add_native_layer_norm_view_7 703643648
...
```

Each line = one kernel invocation. Multiple invocations of the same kernel (e.g., across
attention layers) appear as separate lines. `calc_t1_compile.py` sums them.

### 4.5 Validation

- Total triton bytes should be plausible: typically 5-50 GB for batch sizes used.
- If output is empty, the patch was not applied or caches were not cleared.

---

## Step 5: Profile with PyTorch Profiler (Compile)

### 5.1 Purpose

Collect per-kernel actual GPU durations in compile mode. Used for projection quality analysis
(comparing projected time vs actual time per kernel/op).

### 5.2 Run

```bash
python run.py <model> -d <device> --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile --profile_test
```

### 5.3 Output

Saved to `timeline/`:
- `trace.json` — Chrome trace format with both Triton kernel and extern kernel durations.

### 5.4 Understanding the compile trace

Unlike eager traces, compile traces show:
- **Triton kernels**: `triton_poi_fused_*`, `triton_per_fused_*`, `triton_red_fused_*`
  (pointwise, persistent, reduction fusion patterns)
- **Extern kernels**: `gemm_kernel`, `gen_conv_kernel`, `micro_sdpa*` — library calls that
  were not fused into Triton
- Triton kernel names encode which ops were fused (e.g., `triton_poi_fused_add_mul_pow_tanh_view_2`
  = fused add + mul + pow + tanh + view)

### 5.5 Kernel sum validation

Same as eager: verify `sum(all kernel durations) ≈ T2`. If significantly less, there is
inter-kernel gap (host overhead, launch latency). If significantly more, profiler has
overhead inflation.

---

## Step 6: Unitrace (XPU Only)

### 6.1 Purpose

Same as eager Step 4: collect per-kernel GPU times without profiler overhead. More accurate
than torch profiler for XPU.

### 6.2 Run

```bash
UNITRACE_LAST_ITER=1 \
  <unitrace_bin> --chrome-kernel-logging --start-paused \
  python run.py <model> -d xpu --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile
```

### 6.3 Output

Produces `python.<pid>.json` with per-kernel durations. Same format as eager unitrace output.

### 6.4 Note

Unitrace is **not available on CUDA**. Pass 5 is skipped on 4080S.

---

## Step 7: Compute T1 and R

### 7.1 Purpose

Combine T1_triton (from Pass 3) and T1_extern (from Pass 2) to compute the compile-mode
roofline projection and software efficiency R.

### 7.2 Script

```bash
python scripts/oob300/calc_t1_compile.py \
    --calcflops <output_dir>/<model>_bs<BS>_calcflops.txt \
    --triton-bytes <output_dir>/<model>_bs<BS>_t1_triton.txt \
    --t2 <T2_ms> \
    --platform <PLATFORM> \
    --verbose
```

### 7.3 How T1_compile is computed

```
T1_compile = T1_triton + T1_extern

T1_triton = Σ(kernel_bytes / DRAM_BW)        # all Triton kernels, memory-bound
T1_extern = Σ max(FLOPs/peak, bytes/BW)      # per extern op, roofline-classified
```

### 7.4 SDPA correction

If compile decomposes SDPA (Case 1: `_safe_softmax` in Triton kernel names), the script:
1. Removes the fused SDPA flash projection from T1_extern
2. Adds the decomposed BMM + softmax + BMM projection (which includes FP32 attention matrix
   memory traffic)

If compile keeps fused SDPA (Case 2), no correction needed.

### 7.5 Machine-readable output

```
R_COMPILE=0.9583
T1_TRITON_MS=50.6849
T1_EXTERN_MS=131.9599
T1_TOTAL_MS=182.6448
T2_COMPILE_MS=190.5920
SDPA_DECOMPOSED=0
EAGER_NEEDS_SDPA=0
```

---

## Step 8: Running on CUDA

Commands are the same as XPU but with `-d cuda`:

```bash
# T2:
CUDA_VISIBLE_DEVICES=<value> python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile

# CalcFlops:
CUDA_VISIBLE_DEVICES=<value> Calculate_Flops=1 python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test <eval|train>

# Triton bytes:
CUDA_VISIBLE_DEVICES=<value> TORCHINDUCTOR_BENCHMARK_KERNEL=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile

# Profiler:
CUDA_VISIBLE_DEVICES=<value> python run.py <model> -d cuda --precision <dtype> --channels-last --bs <batch_size> --test <eval|train> --compile --profile_test
```

### CUDA-specific differences

1. **No unitrace** — Pass 5 is skipped. Use profiler trace for actual kernel times.
2. **Triton kernels are the same** — `torch.compile` generates the same Triton kernel names
   on both XPU and CUDA (same inductor graph).
3. **Extern kernels differ** — CUDA uses cuBLAS GEMM / cuDNN conv; XPU uses oneMKL GEMM /
   oneDNN conv. Performance differs but projection methodology is the same.

---

## Step 9: T1/T2/R Software Efficiency Analysis

### 9.1 Framework

- **T1_compile** = T1_triton + T1_extern
  - **T1_triton** = Σ(kernel_bytes / DRAM_BW) — all fused Triton kernels are memory-bound
  - **T1_extern** = Σ max(FLOPs/peak, bytes/BW) — per extern op, roofline-classified
- **T2** = End-to-end CPU wall clock time for the compiled model (steady-state after warmup).
  This is the user-experience metric (same as eager T2).
- **T2_device** = Sum of kernel durations from profiler trace (compile). Used for per-kernel
  breakdown and projection quality analysis.
- **R = T1_compile / T2** = Software efficiency. Closer to 1 = compile-mode execution is
  well-predicted by the roofline model. Always use T2, not T2_device.
- **R_kernel** = For individual Triton kernels: `(kernel_bytes / BW) / actual_kernel_duration`.
  Shows whether a specific Triton kernel achieves near-peak bandwidth.

### 9.2 Why R < 1 (common causes in compile mode)

1. **Kernel launch overhead** — Compiled graphs still dispatch individual Triton kernels
   sequentially. Between-kernel gaps (host dispatch, L0/CUDA launch latency) are not captured
   by T1 but contribute to T2. More pronounced when the model has many small kernels.

2. **Triton autotuning suboptimality** — Triton selects tile sizes and configs via autotuning.
   If autotuning is incomplete (e.g., `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` was used at
   runtime), kernels may run slower than optimal. This affects T2 but not T1.

3. **Extern kernel inefficiency** — GEMM/conv library calls (oneMKL, cuBLAS) may not reach
   peak TFLOPS at the specific shapes used. T1_extern assumes peak, actual may be lower.

4. **Cache effect** — Same as eager: at small batch sizes, working set fits in GPU cache,
   making actual Triton kernel execution faster than the DRAM-bandwidth-based projection.
   This makes T2 < T1, so R > 1 (see 9.3). At larger batch sizes where data exceeds cache,
   R may drop below 1 if other overheads dominate.

5. **Graph breaks** — If `torch.compile` encounters unsupported patterns, it inserts graph
   breaks that add host-device synchronization. These gaps inflate T2 but are invisible to T1.

### 9.3 Why R > 1 (common causes in compile mode)

1. **Triton bytes undercounting** — The `TRITON_KERNEL_BYTES` instrumentation reports the
   *logical* tensor bytes (numel × element_size for each input/output). Actual memory traffic
   may be less due to:
   - GPU cache hits (L2 reuse between adjacent kernels)
   - Triton tiling efficiency (partial loads not reflected in logical size)
   This means T1_triton is overestimated → R > 1.

2. **Peak calibration** — If hardware spec peak is conservative, T1_extern is overestimated.
   Ensure `config/hardware_specs.yaml` values match `context_func.py`.

3. **Extern FLOPs overcounting** — CalcFlops runs in eager mode. If compile fuses some ops
   that are counted as extern in eager (e.g., bias add fused into GEMM at runtime), T1_extern
   includes ops that no longer execute as extern calls → T1 > actual.

### 9.4 Sanity checks

- **Kernel sum vs T2**: `sum(all kernel durations from compile trace) ≈ T2_device`. If
  significantly less, there are inter-kernel gaps (launch overhead, graph breaks).
- **T1_triton reasonableness**: `T1_triton / T2` should be between 0.1 and 0.9 for most models.
  If T1_triton ≈ 0, likely no Triton kernels were generated (model is all extern ops like GEMM).
  If T1_triton ≈ T1_total, model is mostly memory-bound fused ops with few extern calls.
- **R range**: Expect 0.7 – 1.1 for well-behaved models. R < 0.5 suggests significant overhead
  or measurement issues. R > 1.2 suggests bytes overcounting or cache effects.
- **Cross-platform consistency**: R should be comparable across B580/B70/4080S for the same model.
  Large discrepancies suggest platform-specific kernel performance issues.

### 9.5 Report generation

See `skills/oob_compile_report.md` for per-model compile report generation methodology, template,
and cross-platform comparison.

---

## Known Issues

- **Dynamo errors**: Some models (hf_Bert, BERT_pytorch) fail during torch.compile graph capture
  on certain platforms.
- **Timeout**: Large models (densenet121) may exceed the 900s timeout for compilation.
- **4080S has no unitrace**: Pass 5 is skipped on CUDA. Use profiler trace for actual kernel times.

---

## Key Rules

- **Precision**: fp16 for inference, bf16 for training
- **`--channels-last` is always used**
- **Pass 2 (calcflops) runs in eager mode** (no `--compile`) — it captures the op-level FLOPs/bytes
  that remain as extern calls after compilation
- **Pass 3 requires `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`** to force recompilation so the patched
  Triton outputs fresh byte counts (not cached)
- **Triton kernels are assumed memory-bound** — T1_triton = bytes/BW. This is valid because Triton
  fuses elementwise, reductions, and normalization ops that are inherently bandwidth-limited
- **Vector engine ops** (softmax, layer_norm, batch_norm, max_pool) have FLOPs zeroed in T1_extern
  since they use the vector engine, not the matrix engine
- **Do not mix data from old/new scripts** — if the pipeline changes, re-run all passes

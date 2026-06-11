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
      rcpi1-ins0.log
    xpu_profiler/
      timeline/
        trace.json
    cuda_profiler/
      timeline/
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
| `t1/rcpi1-ins0.log` | T1 projection: cumulative per-op FLOPs and bytes. Use the last benchmark iteration only. |
| `xpu_profiler/timeline/trace.json` | XPU per-op actual timing; bridge for unitrace-to-op mapping; graph consistency trace analysis |
| `cuda_profiler/timeline/trace.json` | CUDA per-op actual timing; graph consistency trace analysis |
| `unitrace/python.<pid>.json` | XPU kernel-level timing without profiler overhead. Preferred source for XPU per-op time. Single-iteration window. |
| `xpu_t2/rcpi1-ins0.log` | XPU wall-clock T2. Extract from line: `GPU Time per batch:  209.353 milliseconds` |
| `cuda_t2/rcpi1-ins0.log` | CUDA wall-clock T2. Same format. |

## Calcflops File Column Format

The calcflops file (`t1/rcpi1-ins0.log`) uses `|`-separated columns. The column layout is fixed:

```
col  0: op_name            (e.g. aten::convolution)
col  1: cum_flops          (cumulative FLOPs, raw total)
col  2: cum_memory_raw     (cumulative memory bytes, raw — do NOT use for T1)
col  3: cum_gemm_conv_flops
col  4: cum_gemm_conv_memory
col  5: cum_mem_B580       (cache-adjusted bytes for B580 — USE for B580 T1)
col  6: cum_mem_4080       (cache-adjusted bytes for 4080S — USE for 4080S/CUDA T1)
col  7: cum_mem_G31        (cache-adjusted bytes for B70/G31 — USE for XPU T1)
...
col 23: args:(...)
col last: zero:True/False
```

**Critical rules:**
- Always use the platform-specific cache-adjusted memory column (col 5/6/7), not raw memory (col 2). Using raw memory will severely overcount T1 for memory-bound ops with high cache reuse (e.g. Longformer, BERT attention ops).
- Diff consecutive rows to get per-op delta FLOPs and delta bytes (values are cumulative).
- Benchmark iterations are delimited by a reset in cumulative values. Detect them by finding rows where `cum_mem_platform[i] < cum_mem_platform[i-1] * 0.5`. Use the last iteration only.
- Column count may vary; always split on `|` and index from 0. Stop at the `args:` field.

## Op Name Normalization (for trace ↔ calcflops matching)

Apply these normalizations before matching op names between calcflops and profiler traces:

```
# Backend variants → canonical dispatch name
aten::convolution_overrideable  → aten::convolution
aten::cudnn_convolution         → aten::convolution
aten::mkldnn_convolution        → aten::convolution
aten::_convolution              → aten::convolution
aten::conv2d                    → aten::convolution

# Wrapper chains → leaf dispatch name
aten::linear                    → aten::addmm
aten::layer_norm                → aten::native_layer_norm
aten::batch_norm                → aten::native_batch_norm
aten::_batch_norm_impl_index    → aten::native_batch_norm
aten::softmax                   → aten::_softmax
aten::log_softmax               → aten::_log_softmax
aten::matmul                    → aten::bmm
aten::adaptive_avg_pool2d       → aten::mean
aten::max_pool2d                → aten::max_pool2d_with_indices

# relu kernel implementations (clamp_min is the GPU kernel for relu)
aten::clamp_min                 → aten::relu
aten::clamp_min_                → aten::relu_

# In-place / out-of-place merging (calcflops may functionalize in-place ops)
aten::add_                      → aten::add
aten::mul_                      → aten::mul
aten::sub_                      → aten::sub
aten::div_                      → aten::div
aten::masked_fill_              → aten::masked_fill

# SDPA variants → common name
aten::scaled_dot_product_attention               → aten::sdpa_forward
aten::_scaled_dot_product_flash_attention        → aten::sdpa_forward
aten::_scaled_dot_product_fused_attention_overrideable → aten::sdpa_forward
aten::_flash_attention_forward                   → aten::sdpa_forward

# View/metadata ops (no GPU kernel) → drop from comparison
aten::unbind, aten::contiguous, aten::reshape, aten::t → __view_noop__

# Strip overload suffixes
aten::add.Tensor → aten::add
aten::mul.Scalar → aten::mul
(general rule: strip everything after the last "." in the op suffix)
```

## Vector Engine Ops: Zero FLOPs for T1

The following ops run on the vector engine, not the matrix engine. Their FLOPs from calcflops are unreliable and will misclassify them as compute-bound. **Force their delta_flops to zero** so T1 uses memory bandwidth only:

```
aten::_softmax
aten::native_layer_norm
aten::native_batch_norm
aten::max_pool2d_with_indices
```

## Model Completeness

A model is complete for cross-platform comparison when it has all six files above.

A model is minimally usable for single-platform analysis when it has:
- `t1/rcpi1-ins0.log`
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

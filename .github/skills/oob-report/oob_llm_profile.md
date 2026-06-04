<!-- Copyright 2024-2026 Intel Corporation -->
<!-- Co-authored with GitHub Copilot -->
<!-- Licensed under the Apache License, Version 2.0 -->

# HuggingFace OOB LLM — Profiling

Step-by-step instructions for profiling HuggingFace OOB LLM/VLM models on Intel XPU (B70)
and NVIDIA CUDA (4080S). Covers T1 (roofline projection), T2 (e2e latency), and per-op
analysis via torch profiler traces.

**Workload**: HuggingFace OOB LLM benchmark via `run_benchmarks.py` in the
`HuggingFace_OOB_transformers` repo. Models defined in `benchmark/OOB_llm/models.yaml`.

**Prerequisites**: Complete `skills/env_setup.md` before starting.

---

## Step 0: Confirm parameters

Before running, confirm with the user:

| Parameter | Options |
|-----------|---------|
| Platform | B70 (xpu), 4080S (cuda) |
| Scope | all models, or specific model names |

Fixed parameters (not user-configurable):
- dtype = bfloat16
- sequence_length = 1024
- num_tokens_to_generate = 32
- mode = generate (autoregressive text generation)

---

## Step 1: Model list

Models are defined in `benchmark/OOB_llm/models.yaml` — 41 LLM/VLM models across 7 task types.
Each entry specifies `model_id`, `task`, and `batch_size`.

The batch_size is tuned per-model to avoid OOM. Larger models get smaller batch sizes.

---

## Step 2: Batch run all models

**Script**: `scripts/oob_llm/run_all_llm.sh`

### 2.1 Usage

```bash
# B70 (XPU):
bash scripts/oob_llm/run_all_llm.sh xpu G31 http://proxy.ims.intel.com:911

# 4080S (CUDA):
bash scripts/oob_llm/run_all_llm.sh cuda 4080 http://proxy.ims.intel.com:911
```

Arguments:
- `device`: xpu or cuda
- `platform`: G31 or 4080 (used for calcflops platform-specific memory columns)
- `proxy` (optional): HTTP proxy for model downloads

### 2.2 Three-pass pipeline

The script runs each model in 3 passes:

| Pass | Purpose | Output |
|------|---------|--------|
| 1. Baseline | T2 measurement (e2e_latency) | `baseline/{model_safe}/*.json` |
| 2. CalcFlops | T1 projection (roofline) | `calcflops/{model_safe}_bs{N}_calcflops.txt` |
| 3. Profiler | Per-op GPU times (trace.json) | `profiler/{model_safe}/trace.json` |

### 2.3 Output directory structure

```
{output_dir}/
  baseline/{model_safe}/
    {model_safe}_benchmark_{timestamp}.json     # HF OOB benchmark JSON with e2e_latency
  calcflops/
    {model_safe}_bs{N}_calcflops.txt            # stdout from calcflops pass
  profiler/{model_safe}/
    trace.json                                  # torch profiler trace (moved from profiler_profiles/)
    {model_safe}_benchmark_{timestamp}.json     # benchmark JSON from profiler pass
  profiler_profiles/
    w3_i10-monitored-b{BS}_s1024_n32-sdpa-...json  # raw profiler traces (pre-move)
```

**Note on profiler traces**: The `run_benchmarks.py` profiler saves traces to
`profiler_profiles/` named by config (batch_size). The script attempts to move them
to per-model directories. If the move fails (script crash, naming conflict), traces
remain in `profiler_profiles/` and must be manually matched by batch_size.

### 2.4 Behavior

- Skips models where all 3 passes are done (`.done` marker file).
- Skips individual passes if output already exists.
- OOM models fail on baseline pass and are skipped entirely.
- CalcFlops failures also skip the model (no projection = no report).
- Timeout per pass: 1800s (30 min).
- B70 (XPU): Sets `REQUESTS_CA_BUNDLE='' CURL_CA_BUNDLE=''` for SSL cert workaround.

### 2.5 Platform-specific notes

**B70**:
- GPU memory: 30.3 GB (not 48GB)
- Proxy: `http://proxy.ims.intel.com:911`
- Source conda: `source /root/miniforge3/etc/profile.d/conda.sh && conda activate jianyi`
- Source oneAPI: `source /opt/intel/oneapi/setvars.sh`
- HF cache via NFS: `export HF_HOME=/mnt/nfs_data/huggingface HF_HUB_OFFLINE=1`
  - NFS mount: `10.112.228.229:/mnt/data` → `/mnt/nfs_data` (persistent in fstab)
  - All 41 models pre-cached — no network download needed
- HF token: set `HF_TOKEN` environment variable (only needed if HF_HUB_OFFLINE=0)
- SSL workaround: `REQUESTS_CA_BUNDLE='' CURL_CA_BUNDLE=''`

**4080S**:
- HF cache: `/mnt/ssd1/huggingface/hub` (2.2TB, pre-cached)
- Source conda: `source /root/miniforge3/etc/profile.d/conda.sh && conda activate jianyi`
- HF token: set `HF_TOKEN` environment variable
- `model.generation_config.disable_compile = True` — prevents StaticCache triggering
  auto-compilation on models that use it

---

## Step 3: Extracting T2 from baseline JSON

The baseline JSON has two possible formats:

**Format 1** (4080S CUDA):
```json
{"hash_key": {"metadata": {...}, "measurements": {"e2e_latency": [2.076, ...]}, "config": {...}}}
```

**Format 2** (B70 XPU):
```json
{"metadata": {...}, "measurements": {"e2e_latency": [1.854, ...]}, "config": {...}}
```

T2 = median of `measurements.e2e_latency` array. Values are in **seconds** — convert to ms
for report generation (`T2_ms = median(e2e_latency) * 1000`).

The `config` section contains: `batch_size`, `sequence_length`, `num_tokens_to_generate`,
`dtype`, `device`, `attn_implementation` (sdpa).

---

## Step 4: CalcFlops output format

Same 24-column pipe-delimited format as OOB 300 models. See `skills/oob_profile_eager.md`
Step 5.6 for column definitions.

Key columns:
- [0]: op_name (e.g., `aten::mm`)
- [1]: cumulative FLOPs
- [2]: cumulative memory (bytes)
- [5]: cache-adjusted memory for B580
- [6]: cache-adjusted memory for 4080S
- [7]: cache-adjusted memory for B70 (G31)
- [23]: args with tensor shapes

The script `compare_projection_vs_actual.py` from `scripts/oob300/` parses this format
and is reused by the LLM report generator.

---

## Step 5: Profiler traces

Both B70 and 4080S use torch profiler (no unitrace for LLM).

Trace format is standard Chrome trace JSON with:
- `cpu_op` events: aten:: ops with `External id` linking to device kernels
- `kernel` events: GPU compute kernels with duration
- `gpu_memcpy` events: memory transfers
- XPU traces also have `xpu_runtime` and `xpu_driver` events (filtered by parser)

The `parse_trace_ops()` function from `scripts/oob300/compare_projection_vs_actual.py`
handles both CUDA and XPU traces identically.

---

## Step 6: Differences from OOB 300

| Aspect | OOB 300 | HF OOB LLM |
|--------|---------|-------------|
| Benchmark | `run.py` (TorchBench) | `run_benchmarks.py` (HF OOB) |
| T2 source | baseline.txt ("CPU Wall Time per batch:") | baseline JSON (median e2e_latency, seconds) |
| T2 scope | forward (eval) or forward+backward+optimizer (train) | full generate loop (prefill + decode) |
| Precision | fp16 (eval) / bf16 (train) | bfloat16 |
| Mode | eval / train | generate |
| Unitrace | Available on XPU | Not available (profiler only) |
| Actual source | unitrace (XPU) / profiler (CUDA) | profiler (both platforms) |
| Batch size | From YAML per suite | From models.yaml per model |
| Models | 170 (3 suites × ~55-60) | 41 LLM/VLM models |

---

## Step 7: Report generation

See `skills/oob_llm_report.md` for per-model and fleet-level report generation.

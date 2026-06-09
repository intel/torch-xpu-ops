# Data Contracts

This file defines the canonical data layout and artifact contracts for the OOB eager workflow.

The purpose is to remove layout ambiguity and eliminate the need for intermediate translation layouts.

## Core Rule

Use exactly one canonical local session layout.

Do not preserve a separate flat per-platform symlink layout as a required intermediate format.

Downstream workflows should read the canonical session layout directly.

## Canonical Session Layout

```text
raw_logs/<session_name>/
  metadata.json
  <model>/
    t1/
      calcflops.txt
    xpu_profiler/
      trace.json
      profile_parser.log
    cuda_profiler/
      trace.json
      profile_parser.log
    unitrace/
      python.<pid>.json
    xpu_t2/
      rcpi1-ins0.log
    cuda_t2/
      rcpi1-ins0.log
```

## Session-Level Contract

### `raw_logs/<session_name>/`

Represents one Jenkins-backed OOB eager analysis session.

It must contain:

1. one `metadata.json`
2. zero or more per-model directories

Session name rule:

1. default to the YAML file name without extension when starting from session YAML
2. preserve an explicit user-provided session name if one exists

## Metadata Contract

`metadata.json` should capture enough information to resume or analyze the session without re-querying Jenkins.

At minimum it should include:

1. session output directory
2. trigger-job ids per pass
3. pass result per pass
4. discovered models
5. batch size per model
6. sub-job ids per model per pass when available

Recommended fields:

1. model name
2. short model name
3. batch size
4. precision
5. mode
6. compile mode
7. pass completeness information

### Example Schema

```json
{
  "session_dir": "raw_logs/my_session",
  "passes": {
    "xpu_t2": { "trigger_job_id": 12345, "result": "SUCCESS" },
    "cuda_t2": { "trigger_job_id": 12346, "result": "SUCCESS" },
    "xpu_profiler": { "trigger_job_id": 12347, "result": "SUCCESS" },
    "cuda_profiler": { "trigger_job_id": 12348, "result": "SUCCESS" },
    "unitrace": { "trigger_job_id": 12349, "result": "SUCCESS" },
    "t1": { "trigger_job_id": 12350, "result": "SUCCESS" }
  },
  "models": [
    {
      "model_name": "timm_vision_transformer",
      "short_name": "vit",
      "batch_size": 64,
      "precision": "amp",
      "mode": "inference",
      "compile_mode": "eager",
      "complete": true,
      "sub_jobs": {
        "xpu_t2": 22001,
        "cuda_t2": 22002,
        "xpu_profiler": 22003,
        "cuda_profiler": 22004,
        "unitrace": 22005,
        "t1": 22006
      }
    }
  ]
}
```

Fields may vary per session; the schema above is representative, not exhaustive.

## Model Directory Contract

Each model directory represents one model at one batch size inside the session.

Expected shape:

```text
<model>/
  t1/
    calcflops.txt
  xpu_profiler/
    trace.json
    profile_parser.log
  cuda_profiler/
    trace.json
    profile_parser.log
  unitrace/
    python.<pid>.json
  xpu_t2/
    rcpi1-ins0.log
  cuda_t2/
    rcpi1-ins0.log
```

## File-Level Contracts

### `t1/calcflops.txt`

Meaning:

1. the T1 input for roofline projection

Required semantics:

1. must contain cumulative per-op calcflops output
2. downstream analysis must use the last benchmark iteration

Migration note:

1. standardize the file name as `calcflops.txt`
2. do not keep using `t1/rcpi1-ins0.log` as the canonical calcflops name

### `xpu_profiler/trace.json`

Meaning:

1. PyTorch profiler trace for XPU

Used for:

1. trace-based actual timing
2. kernel-to-op mapping bridge for unitrace
3. graph-consistency trace analysis

### `cuda_profiler/trace.json`

Meaning:

1. PyTorch profiler trace for CUDA

Used for:

1. actual timing on CUDA
2. graph-consistency trace analysis

### `unitrace/python.<pid>.json`

Meaning:

1. XPU kernel-level timing without profiler overhead

Rules:

1. XPU-only
2. preferred source for XPU per-op actual timing
3. should represent a clean single-iteration collection window when possible

### `xpu_t2/rcpi1-ins0.log` and `cuda_t2/rcpi1-ins0.log`

Meaning:

1. wall-clock latency logs used to extract T2

Expected line:

```text
GPU Time per batch:  209.353 milliseconds
```

Rule:

1. if this line is missing, T2 is unavailable for that platform on that model

## Completeness Contract

### Minimal Completeness For A Single-Platform Model

A model is minimally usable for one platform when it has:

1. `t1/calcflops.txt`
2. platform-specific `trace.json`
3. platform-specific T2 log

### Completeness For Cross-Platform B70 vs 4080S Comparison

A model is complete for B70 vs 4080S eager comparison when it has:

1. `t1/calcflops.txt`
2. `xpu_profiler/trace.json`
3. `cuda_profiler/trace.json`
4. `xpu_t2/rcpi1-ins0.log`
5. `cuda_t2/rcpi1-ins0.log`
6. `unitrace/python.<pid>.json` for the XPU side when unitrace is expected

The same pattern applies to B580 vs 4080S comparisons. The platform pair is determined by the session configuration; the file layout is identical regardless of which XPU platform is used.

### Incomplete Model Handling

If a model is incomplete:

1. keep its artifacts
2. do not delete partial state
3. mark it incomplete in metadata or workflow notes
4. allow downstream workflows to skip it explicitly

## Report Output Contract

### Per-Model Reports

Recommended output location:

```text
reports/<session_name>/models/
```

Each report should be uniquely attributable to:

1. model
2. batch size
3. precision
4. mode

### Fleet Summary

Recommended output location:

```text
reports/<session_name>/summary_eager_inference.md
```

### Graph Consistency Report

Recommended output location:

```text
reports/<session_name>/graph_consistency_eager_inference.md
```

### Insights Summary

Recommended output location:

```text
reports/<session_name>/insights_summary.md
```

## Pairing Contract

Cross-platform comparisons must pair models using a stable key composed from:

1. model name
2. batch size
3. precision
4. test mode

Do not pair models only by file name prefix if those fields are ambiguous.

## Validation Contract

Before generating any report, validate:

1. canonical session directory exists
2. metadata exists or the session can be reconstructed from layout alone
3. required files exist for each included model
4. T2 is extractable from the T2 logs
5. calcflops file is parseable
6. trace file is parseable

For XPU unitrace workflows, additionally validate:

1. unitrace file exists when expected
2. unitrace kernel sum is not obviously inconsistent with T2

## Migration Rules

1. Replace script-specific layout assumptions with canonical session layout assumptions.
2. Remove the requirement for `flat_views/` as an intermediate artifact.
3. Keep file naming and section structure stable enough that downstream insights workflows can rely on them.

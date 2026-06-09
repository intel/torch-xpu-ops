# Eager Session Workflow

Use this workflow when the user wants to start from Jenkins session inputs, existing Jenkins pass URLs, or a fresh model list that must be launched on Jenkins.

This document is the canonical source of truth for session launch, artifact download, and canonical session layout preparation.

## Goal

Convert one OOB eager Jenkins session into a complete canonical local session layout that is ready for per-model report generation and fleet summary analysis.

## Supported Scenarios

### Scenario 1: Download + Report

The user already has a session YAML containing the six Jenkins trigger-job URLs.

This is the primary supported flow.

### Scenario 2: Launch + Download + Report

The user provides a model list and wants a fresh Jenkins session launched first, then downloaded and analyzed.

This should follow the same final artifact contract as Scenario 1.

## Required Inputs

One of the following:

1. A session YAML containing the six Jenkins trigger-job URLs
2. A model list for launching a fresh Jenkins session
3. Existing trigger-job build numbers or URLs for one or more passes

Additional required context:

1. Desired session name if it cannot be derived from the YAML file name
2. Jenkins server or trigger-job URL base if not explicit in the YAML
3. Optional suite metadata location if downstream fleet summary needs suite grouping

## Six Jenkins Passes

Use the following six-pass contract as the source of truth:

| Pass | Device | Purpose |
|------|--------|---------|
| `t1` | CUDA | FLOPs and bytes projection input for T1 |
| `unitrace` | XPU | Kernel-level GPU timing on XPU |
| `xpu_profiler` | XPU | Per-op GPU timing trace on XPU |
| `cuda_profiler` | CUDA | Per-op GPU timing trace on CUDA |
| `xpu_t2` | XPU | XPU wall-clock batch latency |
| `cuda_t2` | CUDA | CUDA wall-clock batch latency |

See also `jenkins-pass-reference.md`.

## Session YAML Contract

Expected shape:

```yaml
t1: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
unitrace: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
xpu_profiler: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
cuda_profiler: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
xpu_t2: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
cuda_t2: https://<server>/job/newOOB_launch_benchmark_trigger/<N>/
```

Session name:

1. Use the YAML filename without extension as the default session name.
2. If the user provides an explicit name, prefer the explicit name.

Models:

1. Do not require the user to provide the full discovered model list in the YAML.
2. Parse the model list from Jenkins `summary.log` during artifact collection.

## Workflow

### Step 1: Confirm Inputs

Before collecting data, confirm:

1. Whether the user is in Download + Report mode or Launch + Download + Report mode
2. The session name
3. Whether all six passes are expected
4. Whether partial success is acceptable if one or more passes failed

### Step 2: Resolve Jenkins Jobs

For each pass:

1. Resolve the trigger-job URL or build number
2. Query trigger job status via Jenkins API
3. Record build result, device, and any pass metadata needed downstream

If launching a fresh session:

1. build all six pass definitions from the model list
2. trigger them consistently
3. save a resumable session manifest immediately after launch

### Step 3: Parse `summary.log`

Use `summary.log` from each trigger job to recover model-level information.

Expected columns:

```text
framework,model_name,mode_name,compile_mode,precision,batch_size,cores_per_instance,instance,valid_ins,throughput,link,device
```

Important rule:

1. The `link` field may contain trailing whitespace and must be trimmed before parsing the sub-job artifact path.

From each row, recover:

1. model name
2. batch size
3. device
4. sub-job build number
5. artifact directory path

### Step 4: Download Sub-Job Artifacts

For each discovered model and pass:

1. query the sub-job artifact list
2. download the files needed for that pass
3. place them under the canonical session layout

Expected downloaded artifact classes:

1. T1 log or calcflops output
2. profiler trace artifacts
3. unitrace JSON
4. T2 log
5. optional parser logs and metadata

### Step 5: Build Canonical Session Layout

Write artifacts directly into the canonical session layout defined in `data-contracts.md`.

Important migration rule:

1. Do not preserve a separate flat-view or symlink-view stage as a required workflow phase.
2. Downstream report and fleet workflows should read the canonical session layout directly.

### Step 6: Generate Session Metadata

At minimum, session metadata should record:

1. trigger-job ids and results per pass
2. session output directory
3. discovered models
4. batch size per model
5. discovered sub-job ids per pass

This metadata becomes the local session index for downstream report workflows.

### Step 7: Validate Completeness

Per model, verify whether the expected eager report inputs are present:

1. T1 data exists
2. XPU T2 exists
3. CUDA T2 exists
4. XPU profiler trace exists
5. CUDA profiler trace exists
6. XPU unitrace exists when expected

If a model is incomplete:

1. keep it in the session layout
2. mark it incomplete in metadata or workflow notes
3. allow downstream workflows to skip it explicitly

## Canonical Output Layout

The canonical local session layout is:

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

## Completion Criteria

The session workflow is complete when:

1. the session name is fixed
2. the six passes are resolved or launched
3. artifacts are downloaded into canonical layout
4. metadata is written
5. complete and incomplete models are distinguishable
6. the session is ready for `eager-report-workflow.md`

## Common Failure Modes

1. trigger job still running
2. trigger job failed but has partial artifacts
3. `summary.log` missing or malformed
4. sub-job artifact path cannot be parsed from `link`
5. expected trace or T2 files missing
6. one pass succeeded while another failed for the same model

In all of these cases, prefer preserving partial artifacts and recording incompleteness rather than deleting the model.

## References

- [data-contracts.md](data-contracts.md)
- [jenkins-pass-reference.md](jenkins-pass-reference.md)
- [troubleshooting.md](troubleshooting.md)

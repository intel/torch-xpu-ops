# triage.json

One per triaged issue, at `e2e_output/<issue>-<model>/triage.json` (`draft-<model>`
in a dry run, where `item.issue` is null), written in
three steps:

| Step | Writes |
|---|---|
| `e2e_accuracy_triage.py deep` | `item`, `experiments`, and the logs next to the file |
| e2e-accuracy-triage skill | `result`, and the reproducer file |
| `e2e_accuracy_triage.py verify` | `verify` |

`e2e_accuracy_create_issues.py post` reads all of it.

```jsonc
{
  "item": {                    // one deep_queue entry of created.json
    "issue": 6100, "category": "accuracy", "priority": 0, "sku": "pvc-1100",
    "key": "timm_models,bfloat16,training,convnextv2_nano.fcmae_ft_in22k_in1k",
    "value": "fail_accuracy", "batch_size": "8",
    "message": "Accuracy failed for key name ...", "log_excerpt": ["..."]
  },
  "experiments": [{
    "name": "rerun",
    "cmd": "python benchmarks/dynamo/...", "env": {"TORCHDYNAMO_REPRO_AFTER": "aot"},
    "rc": 1,                   // or "timeout"
    "result": "fail_accuracy", // last token the benchmark printed, or "not run: item out of time"
    "elapsed_s": 412,
    "log": "rerun.log",        // full log, next to triage.json
    "notes": ["RMSE ...", "memory: eager: 30.1 GB, dynamo: 41.0 GB, ratio: 0.73"],
    "tail": ["..."]
  }],
  "result": {"category": "...", "owner": "...", "confidence": "...", "evidence": ["..."],
             "ops": "...", "dtypes": "...", "shapes": "...", "layouts": "...", "slowest_stage": "...",
             "memory": "...", "repro_script": "repro.py", "signature": "...", "conclusion": "..."},
  "verify": {"passed": false, "reason": "reference run fails"}
}
```

Experiments, each read against `rerun` (the CI command unchanged):

| category | experiments |
|---|---|
| accuracy | `fp32`, `aot_eager`, `output_code` (TORCH_LOGS=output_code), `minify_dynamo` (`--minify`), `minify_aot` (only if no repro yet) |
| crash | `aot_eager`, `eager`, `minify_dynamo`, `minify_aot` (REPRO_LEVEL=3) |
| eager_nondeterministic | `deterministic` (`--deterministic`), `fp32` |
| eager_fail | `cpu_eager`, `xpu_eager` |
| oom | `memory` (performance mode, `--print-memory`), `half_batch` |
| timeout | `trace` (TORCH_TRACE for tlparse), `compile_time` |

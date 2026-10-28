# run_benchmarks

Multi-worker E2E benchmark runner for PyTorch Dynamo benchmarks (huggingface, timm_models, torchbench) and the pt2e quantization suite, targeting XPU (default), CUDA, or CPU.

It is copied into the PyTorch checkout as `run_e2e` and executed from there (`python -m run_e2e`). The CI entry point is [linux-e2etest/action.yml](../../actions/linux-e2etest/action.yml).

## Design goals

- **One runner, many tiers.** Test scope lives in data (tier YAML files), not in code. Adding or removing models means editing `.ci/benchmarks/<tier>/config.yaml`.
- **Saturate devices.** Each GPU (or NUMA-bound CPU slice) gets one worker thread that pulls tasks from a shared queue, so a slow task does not idle the other devices.
- **Never hang the pipeline.** Each task is a subprocess with a wall-clock timeout, a GPU/host memory watchdog, and output-pattern kill rules. A failed task is recorded as a row and the run continues.
- **Uniform results.** Every task, regardless of suite, produces a CSV row with the same prefix columns, so results can be merged into per-scenario summaries.

## Components

| Module | Responsibility |
| --- | --- |
| [__main__.py](__main__.py) | CLI parsing, task-file / filter handling, worker setup, run loop, exit code |
| [config.py](config.py) | `TestTask` dataclass, valid value sets, backend flags/env, timeout, kill patterns |
| [tasks.py](tasks.py) | Model list loading, skip lists, task file parsing, tier scope expansion, default task generation |
| [suites.py](suites.py) | Per-suite command construction and result collection (`InductorSuite`, `PT2ESuite`) |
| [workers.py](workers.py) | CPU topology detection, GPU/CPU worker generation, NUMA/`numactl` prefixes |
| [runner.py](runner.py) | Single-task execution (subprocess, output reader, watchdog), worker dispatch, summaries |
| [monitor.py](monitor.py) | `xpu-smi` / host memory polling used by the watchdog |
| [log.py](log.py) | Timestamped, worker-tagged logging and banners |

## Execution flow

1. Parse CLI arguments and apply overrides (timeout, memory threshold, monitor on/off).
2. Build the task list, either from `--task-file` or from the `--suite/--dt/--mode/--scenario` filters.
3. Detect workers: `--numactl-args` if given, otherwise one per GPU (`ZE_AFFINITY_MASK` / `NUM_GPUS` aware), or CPU slices on `--device cpu`.
4. For each task file (or the single default run), dispatch tasks to workers:
   - a worker takes a task, runs `run_single`, and appends a result row;
   - `run_single` starts the command under `numactl`/affinity, streams output, and kills the process on timeout, memory threshold, or matched error pattern;
   - the suite parses the outcome into a CSV row (`failed` plus a reason if killed or unparsable).
5. After all tasks, per-task CSVs under the log root are merged into `summary_accuracy.csv` and `summary_performance.csv`.
6. Print a summary. The process exits with code 1 if any task failed, after all task files have run.

### Task sources

- **Tier YAML** (`.yaml`/`.yml`): a file such as [p0/config.yaml](../../../.ci/benchmarks/p0/config.yaml) declares its own `backends`, `precision`, `suite`, `scenario`, and `models` (per suite and mode). Each precision without a `mode` runs in both modes. CLI `--suite/--dt/--mode/--scenario` act as filters on top.
- **CSV/TXT** rows: either the legacy 5-column format (`suite, dtype, mode, model, result`) or an output CSV with 8+ columns (`dev, backend, suite, dtype, mode, name, scenario, batch_size, elapsed, result[, quant]`). The backend comes from `--backend`.
- **No task file**: the cross product of the filters is generated from the model list files (`benchmarks/dynamo/<suite>_models_list.txt`), or from `--model-only`.

Models listed in `benchmarks/dynamo/<suite>.yaml` under `skip` (including `device.xpu`) are dropped with a warning.

### Task counting

- Inductor task = suite x dtype x mode x scenario x backend x model.
- pt2e task = dtype x scenario x model, inference only. int8 performance expands to two tasks (`symm` and `asymm`). pt2e is not multiplied by backend.
- Tiers: p0 = 2328, p1 = 4656, p2 = 2230 tasks. `full` is the sum of p0, p1, and p2.

## Suites

- **InductorSuite** (huggingface, timm_models, torchbench): runs the upstream `benchmarks/dynamo/<suite>.py` with `--backend`/flags from `BACKEND_FLAGS` and env from `BACKEND_ENV`. Timeout is `--timeout=<task_timeout_seconds>`. Results are read from the suite's output CSV and a row is appended to `inductor-<suite>-<dt>-<mode>-<device>-<scenario>.csv`.
- **PT2ESuite**: accuracy runs `pt2e-accuracy/.../inductor_quant_acc.py` (needs `--dataset-dir`) and parses Acc@1/Acc@5 from the log. Performance runs `pt2e-performance/run_benchmark.py` and reads the `eval_throughput` value from a per-run scratch directory, which is removed afterward. Quant variant is passed via `XPU_QUANT_CONFIG` and `--quantization pt2e`.

## Backends

`--backend` (or per-tier `backends`) selects an inductor variant: `default`, `dynamic`, `cpp_wrapper`, `freezing`, `aot_inductor`, `inductor_deterministic_perf`. Each maps to CLI flags in `BACKEND_FLAGS` and env vars in `BACKEND_ENV` (for example `cpp_wrapper` sets `TORCHINDUCTOR_CPP_WRAPPER=1`).

## Safety rules

- **Timeout**: `--task-timeout` (default 10800 s) kills a task that runs too long.
- **Memory watchdog**: GPU memory is polled with `xpu-smi` (or the host memory on CPU). Crossing the threshold kills the task. The default in [config.py](config.py) is 1.0 on Linux and Windows. Disable with `--no-gpu-memory-monitor`.
- **Error patterns**: output containing `out of memory`, `OutOfMemory`, or `UR_RESULT_ERROR` (case-insensitive) kills the task.
- Failures are recorded as rows with `failed`, so a single bad model never aborts the run.

## Output layout

```
inductor_log/<label>/<suite>/<dtype>/<mode>/<scenario>[/<backend>][/<quant>]/
    inductor-<suite>-<dtype>-<mode>-<device>-<scenario>.csv
    logs-<model>-worker<N>-card<M>.log
inductor_log/<label>/summary_accuracy.csv
inductor_log/<label>/summary_performance.csv
```

- `<label>` is the task file name (the parent folder for `config.yaml`) when more than one task file is given, and empty otherwise.
- `<backend>` is added only for non-default backends, and `<quant>` only for pt2e int8 performance. The default layout is unchanged.
- Summary files merge rows by their `scenario` column. Their header is the union of columns seen.

## Usage

Run from the PyTorch checkout, with `run_benchmarks/` copied to `./run_e2e` (CI does this with rsync):

```bash
# Tier scope (one or more files, run sequentially)
python -m run_e2e --task-file ./benchmarks/dynamo/p0/config.yaml
python -m run_e2e --task-file ./benchmarks/dynamo/p0/config.yaml,./benchmarks/dynamo/p1/config.yaml

# Tier scope narrowed by filters
python -m run_e2e --task-file ./benchmarks/dynamo/p0/config.yaml --suite huggingface --dtype float32 --mode inference

# Explicit filters without a task file
python -m run_e2e --suite huggingface --dt bfloat16 --mode inference --scenario performance

# Only some models
python -m run_e2e --suite torchbench --mode inference --model-only alexnet,dcgan

# CSV task rows with a specific backend
python -m run_e2e --task-file tasks.csv --backend cpp_wrapper

# pt2e accuracy (needs ImageNet)
python -m run_e2e --suite pt2e --scenario accuracy --dataset-dir /path/to/imagenet

# CPU, 8 cores per worker instance
python -m run_e2e --device cpu --cores-per-instance 8 --task-file ./benchmarks/dynamo/p0/config.yaml
```

### Options

| Option | Meaning |
| --- | --- |
| `--suite`, `--dt`/`--dtype`, `--mode`, `--scenario` | Comma/space-separated filters. Empty means all. With `--task-file` they filter the file. |
| `--task-file` | Comma-separated tier YAML or CSV/TXT sources, run one after another. Cannot be combined with `--model-only`. |
| `--model-only` | Model names, a CSV with a `<suite> <mode>` column, or tier YAML files. |
| `--backend` | Inductor backend for CSV/TXT sources and filter runs. |
| `--device` | `xpu` (default), `cuda`, or `cpu`. |
| `--numactl-args` | Override per-worker `numactl` arguments, separated by `;`. Needs `NUM_GPUS`. |
| `--cores-per-instance` | CPU only. Default is all cores of a NUMA node. |
| `--dataset-dir` | ImageNet directory for pt2e accuracy. |
| `--task-timeout` | Per-task wall-clock limit in seconds. |
| `--gpu-memory-threshold` | Memory kill threshold between 0 and 1. |
| `--no-gpu-memory-monitor` | Disable memory polling. |

### Environment

- `ZE_AFFINITY_MASK`: selects GPUs (ranges such as `0-3,6` are supported). Unset means all `NUM_GPUS` GPUs.
- `NUM_GPUS`: required for GPU devices unless `ZE_AFFINITY_MASK` is set.
- `NUMACTL_ARGS` / `--numactl-args`: per-GPU or per-worker binding.

## CI integration

[linux-e2etest/action.yml](../../actions/linux-e2etest/action.yml) maps the `model_list` input to task files:

- `p0`, `p1`, `p2` map to `./benchmarks/dynamo/<tier>/config.yaml`.
- `full` maps to p0, p1, and p2 in that order.
- Its `suite`, `dt`, `mode`, and `scenario` inputs are passed as filters.
- `MODEL_ONLY_NAME`, when set, overrides the task file with `--model-only`.

The step pipes output through `tee` with `|| true`, so the job does not fail on benchmark failures. Consumers should check the summary CSVs and the log instead.

## Troubleshooting

- `benchmarks/dynamo directory not found`: run from the PyTorch checkout, or pass `--task-file`.
- `NUM_GPUS must be set`: export `NUM_GPUS` or set `ZE_AFFINITY_MASK`.
- `Model list file not found`: the suite's `benchmarks/dynamo/<suite>_models_list.txt` is missing. Use a tier YAML or `--model-only`.
- `Column '<suite> <mode>' not found`: a CSV `--model-only` file lacks the expected column.
- Many `failed` rows with `Memory>...`: lower the workload or raise `--gpu-memory-threshold`.
- Tasks killed by `timeout`: raise `--task-timeout`.

# torch-xpu-ops microbench

Operator microbenchmarks with an A/B performance gate for PR CI and the same
framework for nightly runs. Run every command from `test/`:

```bash
cd test
python -m microbench <command> --help
```

## Layout

| Path | Purpose |
|---|---|
| `case.py` | Case model (`OpBench`, `Case`, stable `case_id`), input helpers, case discovery |
| `measure.py` | Measurement protocol (cache flush, input rotation, E2E and device time), worker, case validator, canary |
| `run.py` | A/B scheduler over cards (`A B / B A` rounds); card layout, GPU frequency lock (xe and i915), pre-run checks |
| `judge.py` | Hierarchical bootstrap, Holm, two-stage verdicts, English report, check job |
| `selection.py` | Case selection from a diff, new XPU op detection |
| `github.py` | Labels, maintainers, override record, PR comment (privileged workflows) |
| `tools/` | `maintenance.py` (coverage report, unmapped-file report for `op_source_map.yaml`, noise calibration) and the config consistency check `test_registry.py` |
| `cases/` | One module per op family (`elementwise.py`, `pooling.py`, ...); one `OpBench` class per op and pass set |
| `config/` | `op_source_map.yaml`, `core_set.yaml`, `coverage_exclude.yaml`, `gating.yaml` (thresholds, budget, quarantine), `machines.yaml`, `history.json` (A/A noise statistics, case costs, canary history) |

Listing and selecting cases never imports torch: cases reach torch through
lazy proxies in `case.py`, so the ubuntu CI jobs only need `pyyaml` and `numpy`.

## Writing a case

Add the op to the module of its family in `cases/`:

```python
from ..case import FLOAT_DTYPES, grid, OpBench, randn, State, torch


class Softmax(OpBench):
    op = "softmax"                      # unique op name (selection, reports)
    family = "softmax"                  # core-set group / reporting
    passes = ("fwd", "bwd")
    expected = {"fwd": ("_softmax",), "bwd": ("_softmax_backward_data",)}
    params = grid(shape=[(8192, 8192), (64, 8192)], dtype=FLOAT_DTYPES, dim=[0, 1])
    pr = [dict(shape=(8192, 8192), dtype="bf16", dim=1)]   # PR tier subset

    def setup(self, p, device):         # untimed, once per input set
        return State(x=randn(p["shape"], p["dtype"], device), dim=p["dim"])

    def fwd(self, s):                   # the only code in the timed region
        return torch.softmax(s.x, s.dim)
```

Rules:

* `fwd()` calls the op and nothing else. No input construction, no
  host-to-device copies, no cache flush; the validator rejects random factories
  and host/device copies inside the timed region.
* `bwd` cases build the graph and `grad_out` in setup; the timed region is
  `torch.autograd.grad(out, inputs, grad_out, retain_graph=True)`. Override
  `grad_inputs()` (default `[s.x]` or `s.inputs`) to differentiate several
  inputs, or override `bwd()` to call an aten `*_backward` op directly.
* In-place ops that accumulate state set `inplace = True` and implement
  `reset(s)`; it runs before the flush of every iteration, outside the timed
  region.
* `expected` lists the aten ops (no `aten::` prefix) the pass must execute on
  the device; `a|b` accepts either. Ops without an XPU kernel of their own set
  `composite = True` and list the XPU ops they decompose into.
* `noise_class`: `normal`, `atomic` (atomics/contention), `random` (RNG ops) or
  `host_bound` (host syncs). Only `normal` cases can be in the core set.
* `gating = False` keeps an op out of PR gating (e.g. `matmul`, oneDNN).
* Values that drive performance (indices, masks, repeats, targets) come from
  the CPU generator (`randint`/`mask` in `case.py`), so A and B see identical
  data even if a PR changes an XPU RNG kernel.
* Keep single tensors under ~512M elements (BMG memory with backward and
  input rotation); scale the batch down and say so in a comment.
* `case_id` is a hash of `(op, pass, params)`. Changing params creates new
  cases (new history, new calibration).
* A PR that changes a case module validates (and A/A runs) the cases of the
  ops whose definitions it touches; a change to a shared helper, table or base
  class covers the whole module. Validation takes one dtype per (op, pass,
  shape), PR-tier cases first, within `--validate-budget-min` (20 min);
  cases left over are reported as not validated.

### Table-driven cases

Ops with real-model shapes use `table_bench()`: it builds a `TableBench`
class from a list of cases (each in every float dtype unless it fixes
`dtype`) and the op call; `setup()` builds the inputs from the keys of each
case. `pr` indexes the list (bf16, or the fixed dtype).

```python
Mul = table_bench(
    "mul",
    [
        dict(shape=(16, 12, 512, 64), in_stride=(393216, 64, 768, 1), other=0.354),
        dict(shape=(1000, 81), other_shape=(1000, 81), dtype="bool"),
    ],
    pr=[0],
    family="elementwise",
    fwd=lambda self, s: torch.mul(s.x, s.other),
)
```

Keys:

* `shape`, `in_stride`, `in_dtype`: the main input `s.x` (stride only when
  non-contiguous, including 0 strides of broadcast inputs; dtype only when it
  differs from the case dtype);
* `<arg>_shape`, `<arg>_stride`, `<arg>_dtype`: tensor `s.<arg>`;
* `shapes`, `strides`: tensor list `s.xs` (`cat`, `_foreach_*`); `lists`
  names more list arguments with the same shapes;
* any other key is a scalar `s.<key>` (`dim`, `other`, `alpha`, ...);
  `defaults` fills keys that only some cases set.

`values` picks value ranges where performance or validity depends on them
(`("uniform", lo, hi)`, `("int", lo, hi)`, `("mask", p)`, `("nonfinite",
frac)`). `inplace=True` (class or case) restores the input before every
iteration; `idempotent=True` skips that for ops such as `relu_`. Backward
classes pass `passes=("bwd",)` and `bwd=...` calling the aten backward op
(`expected` defaults to `<op>_backward`).

## Measurement protocol

Per iteration: `reset()` (in-place only), sync, `torch.sum(flush_buffer)`
(512 MB, allocated once per process), sync, then the timed `run()` + sync.

* **E2E**: `time.perf_counter()` around `run()`; no profiler.
* **Device time**: a second pass under the profiler with
  `ZE_ENABLE_TRACING_LAYER=1`. Kernels launched inside the `__cache_flush__`
  annotation are dropped; `kernel`/`gpu_memcpy`/`gpu_memset` events launched
  inside `__mb_iter__` form the sample. An iteration whose flush kernel cannot
  be identified is invalid; more than 10% invalid marks the case invalid.
* **Input rotation**: `L2 * 4 / set_bytes` input sets (at most 20, at most 25%
  of free memory), cycled every iteration.
* **Warmup** by time (>= 200 ms) until three consecutive windows agree within
  5%; at least 50 iterations and 40 ms of op time; one sample per iteration.

## Validator

```bash
python -m microbench validate --op softmax -v             # on XPU
python -m microbench validate --op softmax --device cpu   # dispatch-level checks only
```

Checks: every expected op is dispatched and (on GPU) launches device work,
the first expected op receives the declared shape/dtype/stride, no input
construction in the timed region, flush kernels separate cleanly. PR CI runs
it on the cases a PR changes; a failure fails `microbench-run`.

## Running locally

```bash
python -m microbench list --op softmax                   # cases and ids
python -m microbench run --op softmax --pass bwd --case '{"dim":1,"dtype":"bf16","shape":[8192,8192]}'
python -m microbench env status                          # GPU freq / temperature per card
sudo python -m microbench env lock --ratio 0.95          # pin min=max at 95% of RP0

# A/B: two venvs with the base and the PR wheel (same PyTorch commit)
python -m microbench select --base origin/main --head HEAD --out sel.json
python -m microbench schedule --python-a venv_base/bin/python --python-b venv_pr/bin/python \
    --cases sel.json --validate sel.json --out-dir out --env-check --canary
python -m microbench check --data-dir out --selection sel.json --out-dir report
```

`schedule` without `--python-a` runs A/A with one wheel. `--device cpu --quick
--binding none` gives a functional smoke test without a GPU.

## Statistics

`r = median(B) / median(A)` of device time. The 95% CI and one-sided p-values
come from a hierarchical bootstrap (processes first, then iterations). A case
is a regression only if `r > 1.10`, the CI lower bound exceeds `1 + delta`
(per-case noise), the Holm-adjusted p-value is below 0.05, and both the
screening stage (3 processes per side) and the confirmation stage (6 per
side, B first, on another card) agree. E2E is report-only. A canary that
drifts makes the run inconclusive. Parameters live in `config/gating.yaml`.

Only cases with A/A calibration data (`noise_stats` in `config/history.json`,
`calibrated: true`) can block a PR:

```bash
python -m microbench noise calibrate out_aa_* --out microbench/config/history.json --min-runs 20
```

## CI

`pull.yml`:

1. `microbench-select` (ubuntu): config consistency check
   (`tools/test_registry.py`); resolves the PyTorch SHA once
   (both builds use it) and the merge-base; selects cases from the diff
   (`config/op_source_map.yaml`, `config/core_set.yaml`); detects new XPU
   ops; honours a maintainer-added `disable_microbench`.
2. `linux-build` (PR wheel) and `microbench-build-base` (merge-base wheel,
   cached by `(pytorch_sha, merge_base)`), in parallel.
3. `microbench / microbench-run` (a `bmg-test` runner): both wheels in
   separate venvs on the same runner, measured on the cards of its
   `ZE_AFFINITY_MASK` only; SHA check, environment check (idle cards,
   temperature; the hosts manage GPU frequency themselves and its range is
   recorded in the report), validator, A/B
   scheduler. Fails only for infrastructure errors or validator failures.
4. `microbench / microbench-check` (ubuntu): verdicts, report, comment
   payload. Fails only for a blocking regression without a valid override.
5. `microbench_comment.yml` (`workflow_run`) posts/updates the single
   `<!-- microbench-report -->` comment, also for fork PRs.
6. `microbench_override.yml` (`pull_request_target`): a maintainer's
   `perf_regression_accepted` records the approved head SHA and reruns only
   the check job; new pushes remove the label.

Required checks: `linux-build`, `microbench-select`,
`microbench-build-base / build-status-check`, `microbench / microbench-run`,
`microbench / microbench-check`. A skipped job counts as passed, so each job
that can fail must be required.

Labels: `disable_microbench` (maintainers only; skips build-base, run and
check), `perf_regression_accepted` (maintainers only; accepts the regressions
of the current head commit). Neither can be applied by auto-label. On fork
PRs the select job's token cannot look up maintainers, so `disable_microbench`
is ignored there and the benchmark runs.

`nightly_ondemand.yml` (`ut` containing `microbench`):

* nightly: all cases, previous nightly wheel (A) vs this nightly (B),
  coverage report, noise summary and case costs as artifacts; the job may
  take up to 24 h (scheduler budget 20 h);
* on demand: `microbench_base` (commit/branch, `merge-base`, or empty for
  A/A) and `microbench_cases` (`selected`, `core`, `all`, `pr`, or op names).

## Configuration files

* `op_source_map.yaml`: changed file -> ops, maintained by hand;
  `python -m microbench srcmap` lists unmapped kernel files (report only).
* `core_set.yaml`: 1-2 cases per shared code path (plan 6.1); maintainer review.
* `coverage_exclude.yaml`: `excluded` / `deferred` XPU ops with reasons;
  `python -m microbench coverage --native-functions <yaml>` reports coverage.
* `gating.yaml`: verdict thresholds, PR time budget, and the `quarantine` list
  (cases/ops reported but never blocking).
* `history.json`: A/A noise statistics (which cases may block), measured case
  costs (LPT sharding, budget), canary history per host/card.
* `machines.yaml`: per-host cards, NUMA/CPU binding, L2 size, limits.

# torch-xpu-ops microbench

Operator microbenchmarks with a performance gate for PR CI and a nightly
regression detector. PRs are screened against reference data of the last
nightlies and A/B-tested against their merge-base only when suspicious;
nightlies detect steps against earlier nightlies, recheck them with A/B,
attribute them to torch-xpu-ops or PyTorch and file issues. Run every
command from `test/`:

```bash
cd test
python -m microbench <command> --help
```

## Layout

| Path | Purpose |
|---|---|
| `case.py` | Case model (`OpBench`, `Case`, stable `case_id`), input helpers, case discovery |
| `measure.py` | Measurement protocol (cache flush, input rotation, E2E and device time), worker, case validator, canary |
| `run.py` | Scheduler over cards: A/B (`A B / B A` rounds), single-sided and multi-wheel rounds; records (`records.jsonl`) and per-case logs; card layout and identity, GPU frequency, pre-run checks; `ci-run`, the microbench-run CI job of each scope |
| `judge.py` | Hierarchical bootstrap, Holm, two-stage verdicts, PR conclusion matrix, English report, check job (PR / on-demand, and the nightly check: attribution, state update, nightly report, issue evidence) |
| `selection.py` | Case selection from a diff, new XPU op detection, reference matching (code equivalence), direct A/B decision, candidate commits |
| `reference.py` | Reference state (nightly measurements and confirmed jumps), noise statistics, screening / detection against it |
| `github.py` | Labels, maintainers, override record, PR comment (privileged workflows) |
| `tools/` | `maintenance.py` (coverage report, unmapped-file report for `op_source_map.yaml`, noise calibration) and the config consistency check `test_registry.py` |
| `cases/` | One module per op family (`elementwise.py`, `pooling.py`, ...); one `OpBench` class per op and pass set |
| `config/` | `op_source_map.yaml` (changed file -> ops, core set), `coverage_exclude.yaml`, `gating.yaml` (A/B thresholds, quarantine, screening / detection thresholds, `always_ab`, direct A/B switches, PR budgets, machines); optional `history.json` written by `noise calibrate` (A/A calibration, case costs, canary history) |

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
  cases (new history, new calibration). `case_code_hash` hashes the case
  module: editing a module keeps the ids but its cases lose their reference
  records until the next nightlies measure them again.
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
* **`PROTOCOL_VERSION`** (`__init__.py`): bump it with any change that moves
  measured values; reference records of another version are never compared.
  Changing `measure.py` or `case.py` without a bump gives a
  report-only warning.

Every run writes the logs of each case to `logs/<op>/<pass>_<case_id>/`
(`case_id` is a hash of the op, pass and parameters; `logs/index.csv` lists
the parameters of every case and its directory). Every file starts with
`<op>_<pass>_`:

* `<op>_<pass>_summary.log`: configuration, commits and card of each side,
  per-process device/E2E statistics, errors with the worker's stderr, A/B
  ratio and CI, screening/detection result;
* `<op>_<pass>_samples.jsonl`: every iteration;
* `<op>_<pass>_<stage>[_<side>]_proc<k>_e2e.log` / `_device.log`: warmup,
  every timed iteration with its input set, every device event counted or
  dropped as flush, why a sample is invalid.

`--keep-traces` also keeps the chrome traces (large).

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
python -m microbench select --base origin/main --head HEAD --flow ab --out sel.json
python -m microbench schedule --python-a venv_base/bin/python --python-b venv_pr/bin/python \
    --cases sel.json --validate sel.json --out-dir out --env-check --canary
python -m microbench check --ab-dir out --selection sel.json --out-dir report

# screening as in PR CI: reference state of the last nightly, then the PR wheel only
python -m microbench reference fetch --repo intel/torch-xpu-ops --out-dir ref
python -m microbench select --repo .. --base $(git merge-base origin/main HEAD) --head HEAD \
    --reference-state ref/<run>/reference.jsonl --out sel.json
python -m microbench ci-run --scope pr-vs-reference --selection sel.json \
    --python venv_pr/bin/python --out-dir screen
python -m microbench ci-run --scope pr-vs-base --selection sel.json --screen screen/screen.json \
    --python venv_pr/bin/python --python-base venv_base/bin/python --out-dir ab
python -m microbench check --screen-dir screen --ab-dir ab --out-dir report
```

`schedule` without `--python-a` runs A/A with one wheel, `--single` measures
one wheel, `--wheel NAME=PYTHON` (repeated) alternates several wheels.
`--device cpu --quick --binding none` gives a functional smoke test without a
GPU.

## Statistics

**A/B (confirmation).** `r = median(B) / median(A)` of device time. The 95%
CI and one-sided p-values come from a hierarchical bootstrap (processes first,
then iterations). A case is a regression only if `r > 1.10`, the CI lower
bound exceeds `1 + delta` (per-case noise) and the Holm-adjusted p-value is
below 0.05. Cases without a screening direction need two A/B stages that
agree (3 processes per side, then 6, B first, on another card); cases that
screening flagged need one 6-process A/B in the same direction. E2E is
report-only. A drifting canary makes the run inconclusive. Parameters:
`config/gating.yaml`.

**Screening (PR) and detection (nightly)** compare the median of the
process medians with the reference value: the median over the most recent
(2-5) nightly records of the same environment key, preferably the same card,
after the case's last confirmed jump. For PRs the records must also come from
commits that equal the merge-base in every file related to the case.
Threshold `t = min(0.05, 0.10 - 2.33 * sigma_tot)` (a real 10% step is caught
with 99% probability); `sigma_tot` combines the cross-run, cross-card and
process noise of the case, computed from the reference state. Screening only
decides whether to build the base and A/B; only A/B results block a PR or
file an issue. Parameters: the screening sections of `config/gating.yaml`.

Only cases with A/A calibration data (`noise_stats` in `config/history.json`,
`calibrated: true`) can block a PR:

```bash
python -m microbench noise calibrate out_aa_* --out microbench/config/history.json --min-runs 20
```

## CI

`pull.yml`:

1. `microbench-select` (ubuntu, parallel with lint): config consistency check;
   resolves the PyTorch SHA once (both builds use it) and the merge-base;
   selects cases (`op_source_map.yaml`); detects new XPU
   ops; honours a maintainer-added `disable_microbench`; reads the latest
   reference state and matches code-equivalent records; picks the path:
   `vs-reference`, or `vs-base` (direct A/B: a case that can block has no
   reference or is `always_ab`).
2. `microbench / microbench-run (pr-vs-reference)` (`bmg-test`): the PR wheel
   only, screened against the reference data; output `suspect`.
3. `microbench-build-base`: the merge-base, torch only, same PyTorch commit;
   after linux-build (direct A/B) or after screening found suspects.
4. `microbench / microbench-run (pr-vs-base)`: A/B of all cases (direct A/B) or
   the suspects.
5. `microbench / microbench-check` (ubuntu): final verdict, report,
   comment payload (`microbench-comment-<pr>` with `all_cases.md/csv` and
   `verdicts.json`). Fails only for a blocking regression without a valid
   override.
6. `microbench_comment.yml` (`workflow_run`) posts/updates the single
   `<!-- microbench-report -->` comment, also for fork PRs.
7. `microbench_override.yml` (`pull_request_target`): a maintainer's
   `perf_regression_accepted` records the approved head SHA and reruns only
   the check job; new pushes remove the label.

Run jobs upload one artifact, `microbench-data-<id>-<scope>-<attempt>`
(results, raw samples, `records.jsonl` and the per-case logs).

Required checks: `linux-build`, `microbench-select`,
`microbench / microbench-run (pr-vs-reference)`,
`microbench-build-base / build-status-check`,
`microbench / microbench-run (pr-vs-base)`, `microbench / microbench-check`.
A skipped job counts as passed, so each job that can fail must be required.

Labels: `disable_microbench` (maintainers only; skips everything),
`perf_regression_accepted` (maintainers only; accepts the regressions of the
current head commit). Neither can be applied by auto-label. On fork PRs the
select job's token cannot look up maintainers, so `disable_microbench` is
ignored there and the benchmark runs.

`nightly_ondemand.yml`, build-nightly and build-weekly:

1. `microbench / microbench-run (nightly-vs-reference)`, one job per shard
   (`MICROBENCH_NIGHTLY_SHARDS`, default 4): every case of this nightly wheel
   N (3 processes), A/A of a rotating seventh of the PR-tier cases, the
   case validator (coverage), detection against the reference state, recheck
   1 of the suspects against the previous nightly wheel P on the same card,
   candidate commits (P..N commits touching files related to the case).
2. `microbench-build-base` builds P' (P's xpu-ops, N's PyTorch, torch only)
   when a confirmed regression has candidates, and
   `microbench / microbench-run (nightly-xpu-ops-vs-pytorch)` alternates P,
   P' and N.
3. `microbench / microbench-check`: attribution (xpu-ops / pytorch / mixed),
   new reference state (`microbench-reference-state`: this night's
   measurements and confirmed jumps, 30 days), nightly report, coverage
   report, `always_ab` / quarantine suggestions, issue evidence
   (`microbench-nightly-<run>`).
4. `microbench-auto-issue` (`_microbench_auto_issue.yml`): evidence, LLM
   drafts (`microbench-issue-authoring` skill), checked filing. Issues are
   filed only when `MICROBENCH_AUTO_ISSUE_ENABLED` is `true`; otherwise the
   drafts are in the report artifact.

On demand (`ut` containing `microbench`): `microbench_base` empty runs the PR
flow against `main`'s merge-base; a commit/branch or `merge-base` runs a
direct A/B; `none` an A/A; `nightly` simulates a nightly (its state is
uploaded as `microbench-reference-state-ondemand`, which only
`microbench_reference_run` reads; no issues). `microbench_cases`: `selected`,
`core`, `all`, `pr`, or op names. With `MICROBENCH_REFERENCE_WRITE_DISPATCH`
set, a `nightly` dispatch on `main` publishes the real state (rebuild after a
driver upgrade).

## Configuration files

* `op_source_map.yaml`: changed file -> ops, maintained by hand, and the
  core set (1-2 cases per shared code path; maintainer review);
  `python -m microbench srcmap` lists unmapped kernel files (report only).
* `coverage_exclude.yaml`: `excluded` / `deferred` XPU ops with reasons;
  `python -m microbench coverage --native-functions <yaml>` reports coverage.
* `gating.yaml`: A/B verdict thresholds, the `quarantine` list (cases/ops
  reported but never blocking), screening / detection thresholds,
  `always_ab` cases (suggested by the nightly report, confirmed by a
  maintainer), direct A/B switches, the improvements policy, the PR budgets,
  and `machines` (per-host cards, NUMA/CPU binding, protocol overrides,
  pre-run limits).
* `history.json` (optional, written by `noise calibrate`): A/A noise
  statistics (which cases may block), measured case costs (LPT sharding,
  budget), canary history per host/card.

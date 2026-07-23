# Classification, Adaptation, and Ledger

## Provisional buckets

Assign exactly one scan bucket from observed execution:

| Bucket | Use when |
|---|---|
| `confirmed` | XPU reached the intended stage and showed the same behavior/signature as upstream. |
| `related-failure` | XPU reached the intended path but failed differently from upstream. |
| `not-reproduced` | The intended XPU stage ran and the upstream failure was absent. |
| `blocked-env` | A dependency, topology, loader, or shared environment failure prevented execution. |
| `blocked-platform` | The required path has no XPU equivalent. |
| `blocked-fetch` | Required source details could not be retrieved. |
| `blocked-script-error` | The harness failed, ran on CPU, or did not prove the target stage. |
| `needs-performance-harness` | A performance-only claim needs an unavailable benchmark. |

These are provisional runtime results. `confirmed` does not prove that the inputs
are valid, the behavior is a bug, or XPU needs an independent fix.
`related-failure` does not claim reproduction of the upstream bug. Independent
review owns those decisions.

Title and deep rejects do not receive a local bucket. Only executed or terminally
blocked repro rows set `local_status: done`.

## Repro fidelity

Before assigning a runtime bucket:

- preserve upstream inputs, shapes, dtypes, mode, and oracle
- seed random inputs and reuse the same input for compared executions
- replace uninitialized or contract-invalid inputs only as an explicit diagnostic;
  do not call the changed scenario an upstream reproduction
- prove a key input, output, or target execution is on XPU
- record CPU fallback and shared-frontend failures
- for compiler cases, record an eager baseline and prove the target compiler stage
  was reached

Tiny expected floating-point noise, documented unsupported behavior, and a broken
harness are not confirmed behavior.

## CUDA to XPU adaptation

Change only device mechanics:

- `"cuda"` to `"xpu"` for device placement
- `torch.cuda.*` to `torch.xpu.*`
- `torch.cuda.synchronize()` to `torch.xpu.synchronize()`
- CUDA-only test markers and autocast device strings to their XPU equivalents

Do not change the numerical scenario or oracle to make the repro pass or fail.

## Abnormal termination

Normal execution ends with `RESULT: <bucket>`. A child process cannot emit this
after a segfault, abort, or forced timeout, so append a parent-observed record to
the same log:

```text
PARENT_RESULT: command=<...>; exit=<code>; signal=<signal-or-none>;
timeout=<true|false>; target_stage_reached=<true|false>
```

An abnormal termination may receive `confirmed` or `related-failure` only when the
log proves the intended stage was reached. Otherwise use `blocked-script-error`.
Independent review requires a second fresh-process/cache attempt with the same
decisive abnormal signature before treating it as a real bug.

Use `PARENT_RESULT` only for an attempted child process. A
`needs-performance-harness` placeholder records the missing workload, baseline,
threshold, or environment and must not claim that a target stage ran.

## Provisional routing

Routing suggests implementation ownership; it does not authorize filing:

| Suggested repository | Code ownership |
|---|---|
| `pytorch/pytorch` | Shared Inductor, Dynamo, autograd, dispatcher, ATen, Triton, runtime, or upstream XPU code. |
| `intel/torch-xpu-ops` | XPU kernel or backend code maintained only in torch-xpu-ops. |
| Review required | Ownership, fallback, or shared-fix effect is unclear. |

A shared or CUDA/reference fix that naturally covers XPU should be tracked
upstream, not implemented again for XPU.

## Ledger contract

Keep `artifacts/candidate_ledger.jsonl` backward-compatible with historical runs.
Each raw candidate has exactly one row with at least:

| Field | Values |
|---|---|
| `id` | Stable candidate id |
| `kind` | `issue`, `pr`, or `commit` |
| `title` | Source title or commit message |
| `url` | Source URL |
| `title_status` | `pass` or `reject` |
| `deep_status` | `pending`, `pass`, or `reject` |
| `local_status` | `pending` or `done` |
| `local_bucket` | One provisional bucket when done |

Keep a concise rejection/blocker reason when applicable. An actionable-pending row
has `title_status == pass`, `deep_status != reject`, and
`local_status == pending`.

---
name: xpu-alignment-buckets-and-routing
description: How to label and route candidates. Lists the result buckets (confirmed, not-reproduced, blocked, etc.), what counts as a confirmed bug, how to rewrite a CUDA reproducer for XPU, which repo implements a fix, and the columns of the candidate ledger file. Read this for Steps 1.1 through 3.
---

# Classification, Adaptation & Ledger

## Bucket vocabulary

Assign exactly one provisional bucket as the `RESULT:` of each repro (Step 2c).
Pick by what actually happened when you ran it:

| Bucket | Apply when... |
|--------|---------------|
| `confirmed` | the repro ran on XPU and showed the **same** bug as upstream |
| `related-failure` | the repro ran on XPU but failed in a **different** way than upstream |
| `not-reproduced` | the repro ran on XPU and the upstream failure did **not** happen |
| `blocked-env` | the repro could not start: a dependency was missing or it needed a distributed/multi-GPU setup |
| `blocked-platform` | the repro needed a code path XPU does not have at all |
| `blocked-fetch` | you could not retrieve the issue/PR/commit details to build a repro |
| `blocked-script-error` | the repro failed before producing a verdict: it crashed before the check, or it silently ran on CPU instead of XPU |
| `needs-performance-harness` | the bug is performance-only and needs a benchmark you do not have |

Candidates rejected before a repro runs do not get a bucket: title rejects are
recorded with `title_status: reject` (Step 1.1) and deep-filter rejects with
`deep_status: reject` (Step 2a), including commits with insufficient context.
Only repros that actually run receive a `local_bucket`.

## Confirmation criteria

Use this when deciding between `confirmed` and `not-reproduced` (Step 2c) -- i.e.
whether what you saw on XPU matches the upstream behavior. These scan buckets do
not decide whether the behavior is a real bug, requires an XPU-specific fix, or
should be filed. The independent review makes those decisions.

**Counts as a bug**: crash, segfault, assertion failure, hang, wrong numerical
result, wrong shape/stride/dtype, off-by-one beyond atol=1e-4.

**Does NOT count as a bug**: tiny float noise within tolerance, documented
unsupported behavior, an invalid repro setup.

## CUDA -> XPU adaptation

Use this when writing a repro from upstream CUDA code (Step 2b). Map the device
APIs and keep everything else identical:

- `cuda` -> `xpu` for `.to("cuda")`, `device="cuda"`, `torch.cuda.*` -> `torch.xpu.*`.
- `torch.cuda.synchronize()` -> `torch.xpu.synchronize()`.
- `@onlyCUDA` / `requires_cuda` test markers -> run directly on XPU.
- Drop CUDA-only kwargs (e.g., `device_type="cuda"` in autocast) and substitute `"xpu"`.
- Keep the numerical scenario, shapes, dtypes, and oracle identical; only the
  device changes.

## Provisional routing (confirmed / related-failure)

Use this when suggesting where a `confirmed` / `related-failure` bug would be
implemented (Step 2d). This is the implementation repository, not the tracking
repository. Do not file or hand off based on this suggestion. The independent
review must first establish a `needs-xpu-fix` verdict, canonical tracker, and
current fix state.

| Implement in... | When the bug is in... |
|--------------|-----------------------|
| `pytorch/pytorch` | shared code: Inductor, autograd, dispatcher, ATen, Triton, runtime |
| `pytorch/pytorch` | an XPU kernel that lives upstream in `aten/src/ATen/native/xpu/` |
| `pytorch/pytorch` | a CPU-only crash that affects all backends |
| `intel/torch-xpu-ops` | an XPU kernel that is **not** upstream |
| `intel/torch-xpu-ops` | an XPU backend gap (different error or missing feature vs CUDA) |
| `pytorch/pytorch` | anything you are unsure about (default) |

After review, every `needs-xpu-fix` case is tracked in `intel/torch-xpu-ops`
regardless of its implementation repository.

## Ledger schema (`artifacts/candidate_ledger.jsonl`)

The ledger is the resume point for the batched pipeline. It is **agent-maintained**:
there is no script that updates it. The agent reads and rewrites rows directly as
each stage completes. The JSONL format is kept for backward compatibility with the
legacy scripted workflow.

Each row tracks at least:

| Field | Values | Set during |
|-------|--------|-----------|
| `id` | candidate id | Step 1.1 |
| `kind` | `issue` / `pr` / `commit` | Step 1.1 |
| `title` | candidate title or commit message | Step 1.1 |
| `url` | evidence URL | Step 1.1 |
| `title_status` | `pass` / `reject` | Step 1.1 |
| `deep_status` | `pending` / `pass` / `reject` | Step 2a |
| `local_status` | `pending` / `done` | Step 2c |
| `local_bucket` | a bucket vocabulary value | Step 2c |

An actionable-pending row is one with `title_status == pass` AND
`deep_status != reject` AND `local_status == pending`.

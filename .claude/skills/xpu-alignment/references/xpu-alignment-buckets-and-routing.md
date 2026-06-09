---
name: xpu-alignment-buckets-and-routing
description: How to label and route candidates. Lists the result buckets (confirmed, not-reproduced, blocked, etc.), what counts as a confirmed bug, how to rewrite a CUDA reproducer for XPU, which repo to file a bug in, and the columns of the candidate ledger file. Read this for Steps 1.1 through 3.
---

# Classification, Adaptation & Ledger

## Bucket vocabulary

Assign exactly one bucket as the `RESULT:` of each repro (Step 2c). Pick by what
actually happened when you ran it:

| Bucket | Apply when... |
|--------|---------------|
| `confirmed` | the repro ran on XPU and showed the **same** bug as upstream |
| `related-failure` | the repro ran on XPU but failed in a **different** way than upstream |
| `not-reproduced` | the repro ran on XPU and the upstream failure did **not** happen |
| `blocked-env` | the repro could not start: a dependency was missing or it needed a distributed/multi-GPU setup |
| `blocked-platform` | the repro needed a code path XPU does not have at all |
| `blocked-fetch` | you could not retrieve the issue/PR/commit details to build a repro |
| `blocked-commit-context` | a commit had too little context to construct a meaningful repro |
| `blocked-script-error` | the repro crashed before reaching the check (e.g. it ran on CPU instead of XPU) |
| `needs-performance-harness` | the bug is performance-only and needs a benchmark you do not have |
| `not-applicable` | the candidate was rejected before running anything (title or deep-filter) |

## Confirmation criteria

Use this when deciding between `confirmed` and `not-reproduced` (Step 2c) -- i.e.
whether what you saw on XPU actually counts as a bug.

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

## Routing rules (confirmed / related-failure)

Use this when deciding which repo to file a `confirmed` / `related-failure` bug
into (Step 2d). Pick by where the buggy code lives:

| File into... | When the bug is in... |
|--------------|-----------------------|
| `pytorch/pytorch` | shared code: Inductor, autograd, dispatcher, ATen, Triton, runtime |
| `pytorch/pytorch` | an XPU kernel that lives upstream in `aten/src/ATen/native/xpu/` |
| `pytorch/pytorch` | a CPU-only crash that affects all backends |
| `intel/torch-xpu-ops` | an XPU kernel that is **not** upstream |
| `intel/torch-xpu-ops` | an XPU backend gap (different error or missing feature vs CUDA) |
| `pytorch/pytorch` | anything you are unsure about (default) |

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
| `local_bucket` | a Bucket Vocabulary value | Step 2c |

An actionable-pending row is one with `title_status == pass` AND
`deep_status != reject` AND `local_status == pending`.

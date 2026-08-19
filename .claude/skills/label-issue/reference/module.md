# Canonical Category Taxonomy (13 buckets)

**Authoritative enum** for issue "Category" — no other values
permitted:

`distributed`, `sdpa`, `inductor`, `dynamo`, `torchAO`, `sparse`, `profiler`,
`torch-ops-gemm`, `torch-ops-eltwise`, `torch-ops-reduction`,
`torch-ops-others`, `torch-runtime`, `others`

`extract.json`'s `module` field carries exactly one of these, or `""`, and its
`module_label` field carries the corresponding GitHub label from the table
below. Both are read off the issue's EXISTING `module:` label, so treat them as
a **prior, not an answer**: keep the bucket unless the traced root cause
contradicts it, and derive the bucket yourself from the trace when the field is
`""`. When you override the bucket, take its label from the table below.

## Bucket to label mapping

Two bucket names differ from their label, because the label predates the
taxonomy. Emit the **label** column in `labels.md`, never the bucket name.

| Bucket | GitHub label |
|---|---|
| `distributed` | `module: distributed` |
| `sdpa` | `module: sdpa` |
| `inductor` | `module: inductor` |
| `dynamo` | `module: dynamo` |
| `torchAO` | `module: ao` |
| `sparse` | `module: sparse` |
| `profiler` | `module: profiler` |
| `torch-ops-gemm` | `module: torch-ops-gemm` |
| `torch-ops-eltwise` | `module: torch-ops-eltwise` |
| `torch-ops-reduction` | `module: torch-ops-reduction` |
| `torch-ops-others` | `module: torch-ops-others` |
| `torch-runtime` | `module: core` |
| `others` | `module: others` |

`module: ut` is NOT a category. It is a test-module signal already carried by the
`test_module` axis, so it never appears as a `module:` value here.

### Decision Priority Order

When an issue matches multiple categories, apply in this order (first match
wins):

1. `distributed` — anything tagged `[distributed]` or involving
   XCCL/ProcessGroup/DDP/FSDP/DTensor/symm_mem/collective ops
2. `sdpa` — SDPA / flash / efficient attention kernels (unless
   already claimed by Distributed)
3. `sparse` — sparse tensor formats/ops, nested tensors
4. `profiler` — `torch.profiler`, Kineto, `record_function`, chrome traces, ITT
5. `inductor` — the torch.compile **backend**: Triton codegen, lowering,
   FxGraphCache/codecache, `output_code`, AOTAutograd
6. `dynamo` — the torch.compile **frontend**: graph breaks, guards,
   `symbolic_convert`, `fullgraph`, tracing errors
7. `torchAO` — quantization (int4/int8/fp8/PT2E quant/torchao)
8. `torch-ops-gemm` — matrix multiplication family (see subcategories)
9. `torch-ops-eltwise` — elementwise/pointwise operations
10. `torch-ops-reduction` — reduction operations
11. `torch-ops-others` — other ATen/native ops not fitting gemm/eltwise/reduction,
    including autograd, optimizer, `torch.fx`, and `torch.export` failures that
    resolve to an op
12. `torch-runtime` — torch.xpu.* runtime, memory/OOM, RNG, streams, IPC,
    device management
13. `others` — CI/infra/tracking/build/doc/test-harness/meta — the catch-all

`inductor` outranks `dynamo` deliberately. Every torch.compile failure walks
through `torch/_dynamo/` frames, so a traceback mentioning Dynamo proves nothing
on its own; choose `dynamo` only when the defect is in tracing/guards and there
is no backend-codegen signal.

When judging an issue by keyword, ignore the `## Versions` / `Collecting
environment` dump at the end of the body. It lists every installed package
(`onemkl-sycl-sparse`, `torchao`, ...) and matching against it misclassifies
nearly every issue, so exclude that section from consideration.

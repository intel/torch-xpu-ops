# Canonical Category Taxonomy (17 buckets)

**Authoritative enum** for issue "Category" — no other values
permitted:

`distributed`, `sdpa`, `inductor`, `dynamo`, `torchAO`, `sparse`, `profiler`,
`torch-ops-gemm`, `torch-ops-eltwise`, `torch-ops-reduction`,
`torch-ops-others`, `torch-runtime`, `build`, `infra`, `rfc`, `dpclang`,
`utils`

`extract.json`'s `module` field carries exactly one of these, or `""`, read off
the issue's EXISTING `module:` label. When it is non-blank, preserve it: return
that bucket directly and take its label from the table below — the issue is
already labeled, so do not re-decide this axis. When it is `""`, derive the
bucket yourself from the traced root cause. Take its label from the table below.

## Bucket to label mapping

Some bucket names differ from their label. Emit the **label** column in
`labels.md`, never the bucket name. The authoritative label list and per-label
`keywords` live in `categories.module` of `proposed_labels.json`; read the
keywords from there rather than hard-coding them here. The keywords supplement the
Decision Priority Order below — the traced root cause stays primary.

| Bucket | GitHub label |
|---|---|
| `distributed` | `module: distributed` |
| `sdpa` | `module: sdpa` |
| `inductor` | `module: inductor` |
| `dynamo` | `module: dynamo` |
| `torchAO` | `module: ao` |
| `sparse` | `module: sparse` |
| `profiler` | `module: profiler` |
| `torch-ops-gemm` | `module: gemm` |
| `torch-ops-eltwise` | `module: eltwise` |
| `torch-ops-reduction` | `module: reduction` |
| `torch-ops-others` | `module: ops` |
| `torch-runtime` | `module: core` |
| `build` | `module: build` |
| `infra` | `module: infra` |
| `rfc` | `module: rfc` |
| `dpclang` | `module: dpclang` |
| `utils` | `module: utils` |

There is no generic `module: others` catch-all. A failure that would previously
fall to "others" resolves to one of the refined buckets: `build` (build/compile
failure), `infra` (CI/CD, tracking, meta, test-harness), `rfc` (design proposal),
`dpclang` (DPC++ compiler support), or `utils` (torch utils). Documentation
issues are carried by the native Type / `documentation` label, not a `module:`
value.

`module: ut` is NOT a category. It is a test-surface signal carried by the `test`
axis (`test: ut`), so it never appears as a `module:` value here. The `test` axis
covers only `ut`/`e2e`/`oob`.

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
13. `build` — build/compile failure (CMake, linker, missing symbol, codegen build)
14. `dpclang` — DPC++ compiler (icpx/dpcpp) support issues surfacing during compile
15. `rfc` — RFC / design proposal / tracking-only feature request
16. `utils` — torch utils not resolving to any op or runtime surface
17. `infra` — CI/CD, tracking, meta, test-harness — the catch-all

`inductor` outranks `dynamo` deliberately. Every torch.compile failure walks
through `torch/_dynamo/` frames, so a traceback mentioning Dynamo proves nothing
on its own; choose `dynamo` only when the defect is in tracing/guards and there
is no backend-codegen signal.

When judging an issue by keyword, ignore the `## Versions` / `Collecting
environment` dump at the end of the body. It lists every installed package
(`onemkl-sycl-sparse`, `torchao`, ...) and matching against it misclassifies
nearly every issue, so exclude that section from consideration.

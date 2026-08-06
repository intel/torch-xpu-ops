# Canonical Category Taxonomy (11 buckets)

**Authoritative enum** for issue "Category" — no other values
permitted:

`distributed`, `sdpa`, `inductor`, `torchAO`, `sparse`,
`torch-ops-gemm`, `torch-ops-eltwise`, `torch-ops-reduction`.
`torch-ops-others`, `torch-runtime`, `others`

### Decision Priority Order

When an issue matches multiple categories, apply in this order (first match
wins):

1. `distributed` — anything tagged `[distributed]` or involving
   XCCL/ProcessGroup/DDP/FSDP/DTensor/symm_mem/collective ops
2. `sdpa` — SDPA / flash / efficient attention kernels (unless
   already claimed by Distributed)
3. `inductor` — torch.compile / Dynamo / AOTAutograd / Triton codegen /
   benchmark failures via the inductor path
4. `torchAO` — quantization (int4/int8/fp8/PT2E quant/torchao)
5. `sparse` — sparse tensor formats/ops
6. `torch-ops-gemm` — matrix multiplication family (see subcategories)
7. `torch-ops-eltwise` — elementwise/pointwise operations
8. `torch-ops-reduction` — reduction operations
9. `torch-ops-others` — other ATen/native ops not fitting gemm/eltwise/reduction
10. `torch-runtime` — torch.xpu.* runtime, memory/OOM, profiler, RNG,
    streams, IPC, device management
11. `others` — CI/infra/tracking/build/doc/test-harness/meta — the catch-all

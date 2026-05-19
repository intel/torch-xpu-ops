# DeePEP Dispatch vs Allgather Local Permute Fusion — Performance Results

## Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| topk | 8 |
| num_experts | 128 |
| world_size | 4 GPUs |
| dtype | bfloat16 |
| warmup | 10 iterations |
| timed loops | 100 iterations |

**Environment variable**: `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=0`
(disables D2D copy engine overlap, ensures apple-to-apple single-stream comparison)

## Results

### tokens_per_rank = 2048

| Method | Avg (ms) | Min (ms) | Notes |
|--------|----------|----------|-------|
| **EP Dispatch (ring-ordered)** | **1.110** | **1.108** | Single kernel, no overlap |
| Allgather single-stream | 1.846 | 1.842 | backend_stream = current_stream |
| Allgather two-stream | 1.339 | 1.295 | Default (uses separate backend stream) |

### tokens_per_rank = 4096

| Method | Avg (ms) | Min (ms) | Notes |
|--------|----------|----------|-------|
| **EP Dispatch (ring-ordered)** | **2.161** | **2.160** | Single kernel, no overlap |
| Allgather single-stream | 3.764 | 3.652 | backend_stream = current_stream |
| Allgather two-stream | 2.725 | 2.575 | Default (uses separate backend stream) |

## Analysis

- **EP Dispatch is 40% faster** than allgather single-stream (fair single-kernel comparison)
- **EP Dispatch is 18% faster** than allgather two-stream (even though allgather uses stream overlap)
- Performance scales linearly with token count (2x tokens → ~2x latency)

## Key Optimizations

1. **Ring-ordered work decomposition**: `src_rank = (rank + step + 1) % world_size` ensures no two ranks read from the same source simultaneously, avoiding interconnect congestion
2. **Interleaved work groups**: Adjacent work groups target different source ranks, enabling GPU to overlap cross-device memory access latency
3. **Vectorized reads (VEC_SIZE=8)**: Each work item handles 8 bf16 values (16 bytes), achieving coalesced memory access
4. **Precomputed rank_buffers_ptr**: Device tensor with symmetric memory pointers built once, eliminating per-call host→device copy overhead (~0.5ms savings)

## How to Reproduce

```bash
# Build the kernel
cd test/xpu/csrc && python build.py

# Run the benchmark (all 3 methods, 2048 and 4096 tokens)
cd test/xpu/distributed
SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=0 mpirun -np 4 python bench_compare.py

# Run the standalone correctness + performance test
SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=0 mpirun -np 4 python test_deepep_dispatch.py
```

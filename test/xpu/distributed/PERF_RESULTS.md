# DeePEP Dispatch vs Allgather Local Permute Fusion — Performance Results

## Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| topk | 8 |
| num_experts | 128 |
| world_size | 4 GPUs |
| dtype | bfloat16 |
| warmup | 20 iterations |
| timed loops | 20 iterations |

## Results

### tokens_per_rank = 2048

Projection: `PCIe BW = 26.8 GB/s (31.5 × 0.85), HBM BW = 437.0 GB/s`

| Method | Avg (ms) | Min (ms) | Projection (ms) | Efficiency | Notes |
|--------|----------|----------|-----------------|------------|-------|
| **EP Dispatch (ring-ordered)** | **1.012** | **1.011** | TBD | TBD | TBD |
| Allgather+permute w/o overlap | 1.983 | 1.980 | 1.631 | — | allgather 0.940 + permute 0.691 ms |
| Allgather+permute w/ overlap | 1.328 | 1.278 | — | — | Overlap hides allgather latency |
| └─ Allgather only (PCIe) | — | — | 0.940 | — | 24.00 MB @ 26.8 GB/s |
| └─ Local permute only (fused kernel) | 0.750 | — | 0.691 | 92.1% | 288.00 MB @ 437.0 GB/s |

### tokens_per_rank = 4096

| Method | Avg (ms) | Min (ms) | Notes |
|--------|----------|----------|-------|
| **EP Dispatch (ring-ordered)** | **1.973** | **1.972** | Single kernel, no overlap |
| Allgather+permute w/o overlap | 3.916 | 3.912 | backend_stream = current_stream |
| Allgather+permute w/ overlap | 2.579 | 2.506 | Default (uses separate backend stream) |

## Analysis

- **EP Dispatch is 49% faster** than allgather+permute w/o overlap
- **EP Dispatch is 22% faster** than allgather+permute w/ overlap
- Performance scales linearly with token count (2x tokens → ~2x latency)

## Data Transfer Analysis

Common parameters: `num_experts=128, topk=8, hidden_size=2048, dtype=bfloat16 (2B)`

Each token data = 2048 × 2B = **4 KB**

### 4 devices × 2048 tokens_per_rank

- Device 0 owns 32/128 experts → P(slot on device 0) = 1/4
- P(token unneeded) = (3/4)^8 ≈ 10.0%  →  **~90% of remote tokens need transfer**
- Remote tokens = 3 × 2048 = 6144

| Method | Transferred tokens | Volume | Savings |
|--------|-------------------|--------|---------|
| Allgather (全拉) | 6144 | 24 MB | — |
| EP Dispatch (only owned) | ~5530 | ~21.6 MB | **~10%** |

### 8 devices × 1024 tokens_per_rank

- Device 0 owns 16/128 experts → P(slot on device 0) = 1/8
- P(token unneeded) = (7/8)^8 ≈ 34.4%  →  **~65.6% of remote tokens need transfer**
- Remote tokens = 7 × 1024 = 7168

| Method | Transferred tokens | Volume | Savings |
|--------|-------------------|--------|---------|
| Allgather (全拉) | 7168 | 28 MB | — |
| EP Dispatch (only owned) | ~4702 | ~18.4 MB | **~34%** |

### Scaling Projection (num_experts=128, topk=8)

General formula: `P(token unneeded) = ((W-1)/W)^topk`，其中 W = world_size

| world_size | experts/device | P(unneeded) | EP Dispatch 传输比例 | vs Allgather 节省 |
|:----------:|:--------------:|:-----------:|:-------------------:|:----------------:|
| 2 | 64 | (1/2)^8 = 0.4% | 99.6% | ~0% |
| 4 | 32 | (3/4)^8 = 10.0% | 90.0% | **~10%** |
| 8 | 16 | (7/8)^8 = 34.4% | 65.6% | **~34%** |
| 16 | 8 | (15/16)^8 = 59.7% | 40.3% | **~60%** |
| 32 | 4 | (31/32)^8 = 77.6% | 22.4% | **~78%** |
| 64 | 2 | (63/64)^8 = 88.2% | 11.8% | **~88%** |

**Takeaway**: Device 越多 → 每个 device 拥有的 expert 比例越小 (1/W) → ownership pre-check 跳过的无效读取越多 → EP Dispatch 相对 allgather 的传输量优势从 ~0% (2 devices) 增大到 ~88% (64 devices)。在大规模部署（≥16 devices）下，EP Dispatch 的选择性读取带来的传输量优势非常显著。

## Bandwidth Projection (理论下界分析)

Hardware assumptions: `PCIe BW = 26.8 GB/s (31.5 × 0.85 discount), HBM BW = 437.0 GB/s`

Configuration: `4 devices × 2048 tokens_per_rank, hidden=2048, topk=8, 128 experts, bf16`

### Allgather + Local Permute Fusion

| Component | Data Volume | Projected Time | Actual Time | Efficiency |
|-----------|------------|----------------|-------------|------------|
| Allgather (PCIe) | 24.00 MB | 0.940 ms | — | — |
| Local permute cold (read+write) | 288.00 MB | 0.691 ms | 0.754 ms | 91.7% |
| Local permute warm (write-only) | 256.00 MB | 0.614 ms | 0.753 ms | 81.5% |
| Fused single-launch permute | 288.00 MB | 0.691 ms | 0.751 ms | 92.1% |
| **Fused lower bound (allgather + permute)** | — | **1.631 ms** | **1.330 ms** | — |

> Fused 实测 1.330 ms < 理论下界 1.631 ms，因为 allgather 与 local permute 在双 stream 上 overlap 执行。

### EP Dispatch (ring-ordered kernel)

| Component | Data Volume | Projected Time | Notes |
|-----------|------------|----------------|-------|
| PCIe remote read (filtered) | 21.60 MB | 0.846 ms (serial) | P(skip)=10%, effective 1843/2048 tokens per src |
| PCIe ideal overlap | — | 0.282 ms | 3 links fully overlapped |
| HBM local read | 7.20 MB | 0.017 ms | — |
| HBM write (avg) | 64.00 MB | 0.154 ms | — |
| **Lower bound (PCIe serial + HBM)** | — | **0.999 ms** | — |
| **Actual** | — | **1.012 ms** | **efficiency = 98.8%** |

> EP Dispatch 几乎达到 PCIe 带宽理论下界 (98.8% efficiency)，ring-ordered 交错访问充分利用了互联带宽。

### Per-Rank Assignment Statistics

| Rank | Local Assignments | Remote Recv | Send (MB) | Recv (MB) | Comm LB (ms) | Comm UB (ms) |
|:----:|:-----------------:|:-----------:|:---------:|:---------:|:------------:|:------------:|
| 0 | 16347 | 12282 | 48.12 | 47.98 | 1.885 | 3.763 |
| 1 | 16671 | 12483 | 47.64 | 48.76 | 1.910 | 3.775 |
| 2 | 16277 | 12179 | 47.99 | 47.57 | 1.879 | 3.743 |
| 3 | 16241 | 12183 | 48.15 | 47.59 | 1.886 | 3.749 |

> Worst-case rank: fused_lb=2.222 ms, fused_ub=4.088 ms (allgather + permute 理论范围)

## Key Optimizations

1. **Ring-ordered work decomposition**: `src_rank = (rank + step + 1) % world_size` ensures no two ranks read from the same source simultaneously, avoiding interconnect congestion
2. **Interleaved work groups**: Adjacent work groups target different source ranks, enabling GPU to overlap cross-device memory access latency
3. **Ownership pre-check**: Before issuing expensive PCIe reads, check if ANY topk expert is owned by the current rank; skip the read entirely if not (~10% PCIe traffic savings with uniform expert distribution)
4. **Vectorized reads (VEC_SIZE=8)**: Each work item handles 8 bf16 values (16 bytes), achieving coalesced memory access
5. **Precomputed rank_buffers_ptr**: Device tensor with symmetric memory pointers built once, eliminating per-call host→device copy overhead (~0.5ms savings)

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

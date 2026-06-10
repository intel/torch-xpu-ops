# Allgather Latency Benchmark Results

- Date: 2026-06-11
- Platform: 4x BMG GPUs, mpirun -np 4
- Dtype: bfloat16, warmup=20, loop=100, measure(last)=50
- Prefill: 64M bf16 × 100 reps/elem
- Sizes (total output): 8 KB .. 512 MB (per-rank send chunk = total / 4)
- Benchmark: `bench_ccl_allgather_latency.cpp`
- `busBW(GB/s)` uses allgather correction factor `(n-1)/n`, with n=4 → factor 0.75.

---

## Test 1: 2025.3 setvars

Environment: `source /root/hanchao/intel/oneapi/setvars.sh`

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     4.29 |     3.98 |     4.56 |   0.58 |        1.431 |
| 16.00 KB   |     5.77 |     5.19 |     6.19 |   0.99 |        2.128 |
| 32.00 KB   |     6.09 |     5.69 |     6.43 |   0.74 |        4.037 |
| 64.00 KB   |     6.07 |     5.73 |     6.45 |   0.71 |        8.102 |
| 128.00 KB  |     7.30 |     7.15 |     7.47 |   0.32 |       13.474 |
| 256.00 KB  |    12.04 |    11.51 |    12.53 |   1.02 |       16.335 |
| 512.00 KB  |    21.77 |    20.87 |    22.81 |   1.94 |       18.063 |
| 1.00 MB    |    40.76 |    39.87 |    41.68 |   1.81 |       19.293 |
| 2.00 MB    |    78.39 |    77.46 |    79.28 |   1.83 |       20.064 |
| 4.00 MB    |   153.70 |   152.70 |   154.60 |   1.90 |       20.467 |
| 8.00 MB    |   304.29 |   303.54 |   305.00 |   1.46 |       20.676 |
| 16.00 MB   |   605.39 |   604.80 |   605.93 |   1.13 |       20.785 |
| 32.00 MB   |  1206.86 |  1206.30 |  1207.46 |   1.16 |       20.852 |
| 64.00 MB   |  2410.92 |  2410.34 |  2411.46 |   1.12 |       20.877 |
| 128.00 MB  |  4816.53 |  4815.93 |  4817.10 |   1.18 |       20.900 |
| 256.00 MB  |  9628.80 |  9628.25 |  9629.38 |   1.13 |       20.909 |
| 512.00 MB  | 19245.54 | 19245.04 | 19246.06 |   1.02 |       20.922 |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     5.19 |     4.86 |     5.45 |   0.58 |        1.184 |
| 16.00 KB   |     6.62 |     6.06 |     7.02 |   0.96 |        1.856 |
| 32.00 KB   |     6.96 |     6.60 |     7.28 |   0.67 |        3.532 |
| 64.00 KB   |     6.93 |     6.62 |     7.30 |   0.68 |        7.095 |
| 128.00 KB  |     8.19 |     8.04 |     8.34 |   0.30 |       12.001 |
| 256.00 KB  |    12.89 |    12.36 |    13.38 |   1.02 |       15.258 |
| 512.00 KB  |    22.62 |    21.72 |    23.66 |   1.94 |       17.385 |
| 1.00 MB    |    41.61 |    40.71 |    42.54 |   1.82 |       18.898 |
| 2.00 MB    |    79.33 |    78.43 |    80.19 |   1.76 |       19.828 |
| 4.00 MB    |   154.58 |   153.60 |   155.50 |   1.90 |       20.350 |
| 8.00 MB    |   305.33 |   304.59 |   306.03 |   1.44 |       20.605 |
| 16.00 MB   |   606.56 |   605.96 |   607.08 |   1.12 |       20.745 |
| 32.00 MB   |  1208.02 |  1207.48 |  1208.60 |   1.13 |       20.832 |
| 64.00 MB   |  2412.04 |  2411.47 |  2412.58 |   1.10 |       20.867 |
| 128.00 MB  |  4817.72 |  4817.10 |  4818.30 |   1.19 |       20.894 |
| 256.00 MB  |  9629.94 |  9629.40 |  9630.54 |   1.14 |       20.906 |
| 512.00 MB  | 19246.71 | 19246.20 | 19247.23 |   1.03 |       20.921 |

---

## Test 2: 2026.0 setvars

Environment: `source /root/hanchao/2026.0/intel/oneapi/setvars.sh`

> **NOTE**: Method A (ccl_event) returns near-zero values for sizes ≥ 8 MB — same profiling bug observed in allreduce 2026 build. Use Method B (barrier) for reliable data at large sizes.

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     5.76 |     5.48 |     5.99 |   0.51 |        1.066 |
| 16.00 KB   |     8.96 |     8.48 |     9.34 |   0.86 |        1.372 |
| 32.00 KB   |     8.14 |     7.80 |     8.40 |   0.60 |        3.021 |
| 64.00 KB   |     8.56 |     8.30 |     8.78 |   0.48 |        5.744 |
| 128.00 KB  |     9.26 |     9.06 |     9.45 |   0.39 |       10.613 |
| 256.00 KB  |    13.98 |    13.74 |    14.23 |   0.49 |       14.068 |
| 512.00 KB  |    23.76 |    22.97 |    24.66 |   1.69 |       16.549 |
| 1.00 MB    |    42.79 |    42.04 |    43.58 |   1.53 |       18.379 |
| 2.00 MB    |    81.16 |    80.42 |    81.75 |   1.33 |       19.379 |
| 4.00 MB    |   158.67 |   158.16 |   159.13 |   0.97 |       19.825 |
| 8.00 MB    |     0.11 |     0.10 |     0.13 |   0.03 | ⚠️ INVALID |
| 16.00 MB   |     0.11 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |
| 32.00 MB   |     0.12 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |
| 64.00 MB   |     0.11 |     0.10 |     0.13 |   0.03 | ⚠️ INVALID |
| 128.00 MB  |     0.11 |     0.10 |     0.13 |   0.03 | ⚠️ INVALID |
| 256.00 MB  |     0.11 |     0.10 |     0.14 |   0.03 | ⚠️ INVALID |
| 512.00 MB  |     0.12 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     6.70 |     6.43 |     6.91 |   0.48 |        0.917 |
| 16.00 KB   |     9.87 |     9.40 |    10.23 |   0.83 |        1.245 |
| 32.00 KB   |     9.07 |     8.75 |     9.32 |   0.58 |        2.711 |
| 64.00 KB   |     9.44 |     9.18 |     9.67 |   0.49 |        5.208 |
| 128.00 KB  |    10.20 |    10.01 |    10.35 |   0.34 |        9.637 |
| 256.00 KB  |    14.88 |    14.65 |    15.12 |   0.47 |       13.217 |
| 512.00 KB  |    24.67 |    23.90 |    25.57 |   1.67 |       15.937 |
| 1.00 MB    |    43.70 |    42.95 |    44.47 |   1.52 |       17.998 |
| 2.00 MB    |    82.04 |    81.28 |    82.62 |   1.35 |       19.173 |
| 4.00 MB    |   159.64 |   159.14 |   160.12 |   0.98 |       19.705 |
| 8.00 MB    |   247.84 |   247.44 |   248.28 |   0.84 |       25.385 |
| 16.00 MB   |   471.95 |   471.50 |   472.40 |   0.90 |       26.661 |
| 32.00 MB   |   930.26 |   929.72 |   930.78 |   1.06 |       27.053 |
| 64.00 MB   |  1842.04 |  1841.38 |  1842.68 |   1.30 |       27.324 |
| 128.00 MB  |  3644.27 |  3643.54 |  3644.99 |   1.45 |       27.622 |
| 256.00 MB  |  7246.69 |  7246.05 |  7247.29 |   1.25 |       27.782 |
| 512.00 MB  | 14453.79 | 14452.63 | 14454.87 |   2.24 |       27.858 |

---

## Comparison: min_us (Method B: barrier)

2026.0 vs 2025.3 — large-message regime (≥8 MB) shows ~30% latency reduction due to higher bus bandwidth (~27.8 vs ~20.9 GB/s). Small-message (≤512 KB) is slightly slower in 2026.0.

| Size | Test1 min_us (25.3) | Test2 min_us (26.0) | Diff (us) | Speedup |
|------|-------------:|-------------:|-----------:|--------:|
| 8.00 KB    |     4.86 |     6.43 |  +1.57 | 0.76x |
| 16.00 KB   |     6.06 |     9.40 |  +3.34 | 0.64x |
| 32.00 KB   |     6.60 |     8.75 |  +2.15 | 0.75x |
| 64.00 KB   |     6.62 |     9.18 |  +2.56 | 0.72x |
| 128.00 KB  |     8.04 |    10.01 |  +1.97 | 0.80x |
| 256.00 KB  |    12.36 |    14.65 |  +2.29 | 0.84x |
| 512.00 KB  |    21.72 |    23.90 |  +2.18 | 0.91x |
| 1.00 MB    |    40.71 |    42.95 |  +2.24 | 0.95x |
| 2.00 MB    |    78.43 |    81.28 |  +2.85 | 0.96x |
| 4.00 MB    |   153.60 |   159.14 |  +5.54 | 0.97x |
| 8.00 MB    |   304.59 |   247.44 | −57.15 | **1.23x** |
| 16.00 MB   |   605.96 |   471.50 |−134.46 | **1.29x** |
| 32.00 MB   |  1207.48 |   929.72 |−277.76 | **1.30x** |
| 64.00 MB   |  2411.47 |  1841.38 |−570.09 | **1.31x** |
| 128.00 MB  |  4817.10 |  3643.54 |−1173.56| **1.32x** |
| 256.00 MB  |  9629.40 |  7246.05 |−2383.35| **1.33x** |
| 512.00 MB  | 19246.20 | 14452.63 |−4793.57| **1.33x** |

**Takeaway**: 2026.0 oneCCL achieves a clear ~1.3× speedup at large message sizes (≥8 MB), saturating at ~27.8 GB/s busBW vs ~20.9 GB/s in 2025.3. The small-message regime sees a modest regression of 1–3 µs, likely from increased dispatch overhead. Method A (ccl_event) is broken for ≥8 MB on 2026.0 and should be avoided.

---

## Reproduction

```bash
# 2025.3
source /root/hanchao/intel/oneapi/setvars.sh --force
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib    -L${I_MPI_ROOT}/lib    -lccl -lmpi \
     bench_ccl_allgather_latency.cpp -o bench_ccl_allgather_latency_25
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_allgather_latency_25 \
  --min 12 --max 28 --warmup 20 --loop 100 \
  --prefill-n 67108864 --prefill-reps 100

# 2026.0
source /root/hanchao/2026.0/intel/oneapi/setvars.sh --force
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib    -L${I_MPI_ROOT}/lib    -lccl -lmpi \
     bench_ccl_allgather_latency.cpp -o bench_ccl_allgather_latency_26
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_allgather_latency_26 \
  --min 12 --max 28 --warmup 20 --loop 100 \
  --prefill-n 67108864 --prefill-reps 100
```

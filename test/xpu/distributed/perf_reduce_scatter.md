# Reduce-Scatter Latency Benchmark Results

- Date: 2026-06-11
- Platform: 4x BMG GPUs, mpirun -np 4
- Dtype: bfloat16, warmup=20, loop=100, measure(last)=50
- Prefill: 64M bf16 × 100 reps/elem
- Sizes (total input): 8 KB .. 512 MB (per-rank recv chunk = total / 4)
- Benchmark: `bench_ccl_reduce_scatter_latency.cpp`
- `busBW(GB/s)` uses reduce_scatter correction factor `(n-1)/n`, with n=4 → factor 0.75.

---

## Test 1: 2025.3 setvars

Environment: `source /root/hanchao/intel/oneapi/setvars.sh`

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     7.17 |     6.71 |     7.53 |   0.81 |        0.857 |
| 16.00 KB   |     7.33 |     6.78 |     7.76 |   0.97 |        1.677 |
| 32.00 KB   |     6.93 |     6.53 |     7.21 |   0.68 |        3.547 |
| 64.00 KB   |     7.44 |     7.15 |     7.69 |   0.54 |        6.605 |
| 128.00 KB  |     8.57 |     7.86 |     9.02 |   1.16 |       11.471 |
| 256.00 KB  |    12.63 |    12.21 |    13.04 |   0.83 |       15.567 |
| 512.00 KB  |    22.20 |    21.52 |    23.08 |   1.56 |       17.712 |
| 1.00 MB    |    41.11 |    40.38 |    41.96 |   1.57 |       19.128 |
| 2.00 MB    |    79.57 |    78.56 |    80.42 |   1.86 |       19.768 |
| 4.00 MB    |   154.11 |   153.16 |   155.12 |   1.97 |       20.412 |
| 8.00 MB    |   305.12 |   304.25 |   305.84 |   1.59 |       20.619 |
| 16.00 MB   |   606.27 |   605.75 |   606.74 |   0.99 |       20.755 |
| 32.00 MB   |  1207.73 |  1207.35 |  1208.15 |   0.79 |       20.837 |
| 64.00 MB   |  2412.39 |  2412.01 |  2412.78 |   0.77 |       20.864 |
| 128.00 MB  |  4819.87 |  4819.46 |  4820.29 |   0.83 |       20.885 |
| 256.00 MB  |  9639.90 |  9639.43 |  9640.34 |   0.91 |       20.885 |
| 512.00 MB  | 19270.07 | 19269.52 | 19270.59 |   1.07 |       20.895 |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |     8.02 |     7.57 |     8.38 |   0.81 |        0.766 |
| 16.00 KB   |     8.22 |     7.68 |     8.65 |   0.97 |        1.495 |
| 32.00 KB   |     7.80 |     7.40 |     8.10 |   0.70 |        3.152 |
| 64.00 KB   |     8.32 |     8.04 |     8.59 |   0.54 |        5.905 |
| 128.00 KB  |     9.48 |     8.78 |     9.92 |   1.14 |       10.371 |
| 256.00 KB  |    13.48 |    13.05 |    13.90 |   0.85 |       14.582 |
| 512.00 KB  |    23.06 |    22.37 |    23.95 |   1.58 |       17.053 |
| 1.00 MB    |    41.97 |    41.25 |    42.80 |   1.55 |       18.739 |
| 2.00 MB    |    80.48 |    79.50 |    81.34 |   1.84 |       19.543 |
| 4.00 MB    |   155.02 |   154.06 |   156.03 |   1.98 |       20.293 |
| 8.00 MB    |   306.20 |   305.31 |   306.92 |   1.61 |       20.547 |
| 16.00 MB   |   607.42 |   606.88 |   607.90 |   1.02 |       20.715 |
| 32.00 MB   |  1208.91 |  1208.52 |  1209.33 |   0.81 |       20.817 |
| 64.00 MB   |  2413.51 |  2413.11 |  2413.92 |   0.81 |       20.854 |
| 128.00 MB  |  4821.03 |  4820.62 |  4821.46 |   0.84 |       20.880 |
| 256.00 MB  |  9641.04 |  9640.55 |  9641.49 |   0.94 |       20.882 |
| 512.00 MB  | 19271.25 | 19270.74 | 19271.75 |   1.01 |       20.894 |

---

## Test 2: 2026.0 setvars

Environment: `source /root/hanchao/2026.0/intel/oneapi/setvars.sh`

> **NOTE**: Method A (ccl_event) returns near-zero values for sizes ≥ 16 MB — same profiling bug observed in allreduce / allgather 2026 builds. Use Method B (barrier) for reliable data at large sizes.

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |    12.01 |    11.69 |    12.33 |   0.63 |        0.512 |
| 16.00 KB   |    10.02 |     9.44 |    10.49 |   1.04 |        1.226 |
| 32.00 KB   |    10.15 |     9.78 |    10.45 |   0.67 |        2.421 |
| 64.00 KB   |    11.82 |    11.57 |    12.07 |   0.49 |        4.158 |
| 128.00 KB  |    11.46 |    11.24 |    11.66 |   0.41 |        8.578 |
| 256.00 KB  |    14.51 |    14.09 |    15.01 |   0.92 |       13.547 |
| 512.00 KB  |    24.27 |    23.53 |    25.14 |   1.61 |       16.203 |
| 1.00 MB    |    43.42 |    42.39 |    44.55 |   2.16 |       18.113 |
| 2.00 MB    |    81.78 |    80.81 |    82.68 |   1.88 |       19.234 |
| 4.00 MB    |   158.47 |   157.87 |   159.02 |   1.14 |       19.851 |
| 8.00 MB    |   309.55 |   309.01 |   310.09 |   1.08 |       20.325 |
| 16.00 MB   |     0.11 |     0.10 |     0.13 |   0.02 | ⚠️ INVALID |
| 32.00 MB   |     0.12 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |
| 64.00 MB   |     0.11 |     0.10 |     0.14 |   0.03 | ⚠️ INVALID |
| 128.00 MB  |     0.11 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |
| 256.00 MB  |     0.11 |     0.10 |     0.13 |   0.03 | ⚠️ INVALID |
| 512.00 MB  |     0.12 |     0.10 |     0.14 |   0.04 | ⚠️ INVALID |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB    |    12.95 |    12.67 |    13.24 |   0.57 |        0.474 |
| 16.00 KB   |    10.89 |    10.32 |    11.36 |   1.04 |        1.128 |
| 32.00 KB   |    11.08 |    10.73 |    11.37 |   0.64 |        2.218 |
| 64.00 KB   |    12.67 |    12.44 |    12.94 |   0.50 |        3.879 |
| 128.00 KB  |    12.39 |    12.16 |    12.60 |   0.44 |        7.936 |
| 256.00 KB  |    15.42 |    14.99 |    15.90 |   0.91 |       12.753 |
| 512.00 KB  |    25.18 |    24.45 |    26.04 |   1.59 |       15.616 |
| 1.00 MB    |    44.32 |    43.31 |    45.43 |   2.12 |       17.745 |
| 2.00 MB    |    82.69 |    81.72 |    83.61 |   1.90 |       19.021 |
| 4.00 MB    |   159.43 |   158.79 |   160.01 |   1.23 |       19.731 |
| 8.00 MB    |   310.65 |   310.08 |   311.23 |   1.15 |       20.253 |
| 16.00 MB   |   524.48 |   523.88 |   525.09 |   1.21 |       23.991 |
| 32.00 MB   |  1024.70 |  1023.96 |  1025.47 |   1.50 |       24.559 |
| 64.00 MB   |  1992.26 |  1991.51 |  1993.03 |   1.52 |       25.264 |
| 128.00 MB  |  3916.32 |  3915.44 |  3917.25 |   1.81 |       25.704 |
| 256.00 MB  |  7765.05 |  7764.02 |  7766.08 |   2.05 |       25.927 |
| 512.00 MB  | 15520.88 | 15520.00 | 15521.81 |   1.81 |       25.943 |

---

## Comparison: min_us (Method B: barrier)

2026.0 vs 2025.3 — large-message regime (≥16 MB) shows ~25% latency reduction, saturating at ~25.9 GB/s busBW vs ~20.9 GB/s. Small-message (≤2 MB) is slightly slower in 2026.0.

| Size | Test1 min_us (25.3) | Test2 min_us (26.0) | Diff (us) | Speedup |
|------|-------------:|-------------:|-----------:|--------:|
| 8.00 KB    |     7.57 |    12.67 |  +5.10 | 0.60x |
| 16.00 KB   |     7.68 |    10.32 |  +2.64 | 0.74x |
| 32.00 KB   |     7.40 |    10.73 |  +3.33 | 0.69x |
| 64.00 KB   |     8.04 |    12.44 |  +4.40 | 0.65x |
| 128.00 KB  |     8.78 |    12.16 |  +3.38 | 0.72x |
| 256.00 KB  |    13.05 |    14.99 |  +1.94 | 0.87x |
| 512.00 KB  |    22.37 |    24.45 |  +2.08 | 0.91x |
| 1.00 MB    |    41.25 |    43.31 |  +2.06 | 0.95x |
| 2.00 MB    |    79.50 |    81.72 |  +2.22 | 0.97x |
| 4.00 MB    |   154.06 |   158.79 |  +4.73 | 0.97x |
| 8.00 MB    |   305.31 |   310.08 |  +4.77 | 0.98x |
| 16.00 MB   |   606.88 |   523.88 |  −83.00 | **1.16x** |
| 32.00 MB   |  1208.52 |  1023.96 | −184.56 | **1.18x** |
| 64.00 MB   |  2413.11 |  1991.51 | −421.60 | **1.21x** |
| 128.00 MB  |  4820.62 |  3915.44 | −905.18 | **1.23x** |
| 256.00 MB  |  9640.55 |  7764.02 |−1876.53 | **1.24x** |
| 512.00 MB  | 19270.74 | 15520.00 |−3750.74 | **1.24x** |

**Takeaway**: 2026.0 oneCCL achieves a ~1.2× speedup for reduce_scatter at large message sizes (≥16 MB), saturating at ~25.9 GB/s busBW vs ~20.9 GB/s in 2025.3. The transition point happens at 16 MB (vs 8 MB for allgather), suggesting the SUM-reduction kernel keeps the algorithm bandwidth-limited longer. The small-message regime sees a 2–5 µs regression. Method A (ccl_event) is broken for ≥16 MB on 2026.0 and should be avoided.

### Cross-collective comparison @ 512 MB, Method B

| Collective | 2025.3 busBW | 2026.0 busBW | 2026.0 / 2025.3 |
|---|---:|---:|---:|
| allreduce       | ~20.9 GB/s | ~27.x GB/s | ~1.30x |
| allgather       | 20.92 GB/s | 27.86 GB/s | 1.33x |
| reduce_scatter  | 20.89 GB/s | 25.94 GB/s | 1.24x |

`reduce_scatter` lags `allgather` slightly on 2026.0 — consistent with the reduction kernel adding a small per-element compute cost on top of the same wire-data movement.

---

## Reproduction

```bash
# 2025.3
source /root/hanchao/intel/oneapi/setvars.sh --force
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib    -L${I_MPI_ROOT}/lib    -lccl -lmpi \
     bench_ccl_reduce_scatter_latency.cpp -o bench_ccl_reduce_scatter_latency_25
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_reduce_scatter_latency_25 \
  --min 12 --max 28 --warmup 20 --loop 100 \
  --prefill-n 67108864 --prefill-reps 100 \
  > rs_25.3.log 2>&1

# 2026.0
source /root/hanchao/2026.0/intel/oneapi/setvars.sh --force
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib    -L${I_MPI_ROOT}/lib    -lccl -lmpi \
     bench_ccl_reduce_scatter_latency.cpp -o bench_ccl_reduce_scatter_latency_26
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_reduce_scatter_latency_26 \
  --min 12 --max 28 --warmup 20 --loop 100 \
  --prefill-n 67108864 --prefill-reps 100 \
  > rs_26.0.log 2>&1
```

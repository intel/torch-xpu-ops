# oneCCL C API Collectives Benchmark Report

- Date: 2026-06-25
- Platform: 8× BMG GPUs, mpirun -np 8, oneAPI **2026.0** (oneCCL 2021.17, C API)
- Dtype: bfloat16, warmup=20, loop=100, measure(last)=50
- Prefill: 64M bf16 × 100 reps/elem
- Timing: barrier method only (post_barrier.cmd_start − pre_barrier.cmd_end)
- Build: `icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 -lccl -lmpi`

---

## All-Reduce (C API / barrier)

| Size       | avg_us  | min_us  | max_us  | var_us  | busBW(GB/s) |
|------------|--------:|--------:|--------:|--------:|------------:|
| 8.00 KB   |   41.55 |   41.00 |   42.16 |    1.16 |       0.345 |
| 16.00 KB  |   38.80 |   38.26 |   39.36 |    1.10 |       0.739 |
| 32.00 KB  |   30.02 |   29.31 |   30.76 |    1.45 |       1.910 |
| 64.00 KB  |   30.40 |   29.52 |   31.18 |    1.66 |       3.773 |
| 128.00 KB |   32.94 |   32.34 |   33.51 |    1.17 |       6.963 |
| 256.00 KB |   34.13 |   33.65 |   34.60 |    0.95 |      13.443 |
| 512.00 KB |   51.07 |   50.03 |   52.33 |    2.30 |      17.967 |
| 1.00 MB   |   96.87 |   95.04 |   98.76 |    3.72 |      18.942 |
| 2.00 MB   |  185.20 |  183.43 |  186.99 |    3.57 |      19.817 |
| 4.00 MB   |  362.55 |  360.95 |  363.84 |    2.89 |      20.246 |
| 8.00 MB   |  693.97 |  693.06 |  694.90 |    1.84 |      21.154 |
| 16.00 MB  | 1400.33 | 1395.69 | 1403.47 |    7.78 |      20.967 |
| 32.00 MB  | 2742.82 | 2737.57 | 2746.40 |    8.83 |      21.409 |
| 64.00 MB  | 5352.99 | 5348.34 | 5355.72 |    7.38 |      21.939 |
| 128.00 MB |10559.79 |10554.70 |10562.79 |    8.09 |      22.243 |
| 256.00 MB |21027.50 |21022.47 |21030.62 |    8.15 |      22.340 |
| 512.00 MB |42075.24 |42069.29 |42079.39 |   10.10 |      22.330 |

**Key observations:**
- **Latency**: ~40µs at small (8KB), scales to ~42ms at 512MB
- **Bandwidth**: Plateaus at **~22 GB/s** for large messages (>1MB)
- **Busbw factor**: 2×(n-1)/n = 1.75 for ws=8 (allreduce sends/receives twice per rank)

---

## All-Gather (C API / barrier)

| Size       | avg_us  | min_us  | max_us  | var_us  | busBW(GB/s) |
|------------|--------:|--------:|--------:|--------:|------------:|
| 8.00 KB   |   10.82 |   10.41 |   11.15 |    0.75 |       0.662 |
| 16.00 KB  |   11.60 |   10.27 |   12.76 |    2.49 |       1.236 |
| 32.00 KB  |   16.82 |   16.02 |   17.47 |    1.45 |       1.704 |
| 64.00 KB  |   16.18 |   15.68 |   16.71 |    1.03 |       3.545 |
| 128.00 KB |   16.80 |   16.15 |   17.39 |    1.24 |       6.825 |
| 256.00 KB |   17.01 |   16.46 |   17.49 |    1.03 |      13.484 |
| 512.00 KB |   28.20 |   26.73 |   30.38 |    3.65 |      16.266 |
| 1.00 MB   |   50.78 |   49.16 |   53.07 |    3.91 |      18.068 |
| 2.00 MB   |   95.05 |   93.68 |   96.45 |    2.77 |      19.306 |
| 4.00 MB   |  184.15 |  182.09 |  186.44 |    4.35 |      19.930 |
| 8.00 MB   |  362.14 |  360.37 |  363.51 |    3.14 |      20.269 |
| 16.00 MB  |  671.77 |  667.63 |  673.87 |    6.24 |      21.853 |
| 32.00 MB  | 1332.23 | 1327.89 | 1334.64 |    6.75 |      22.038 |
| 64.00 MB  | 2654.06 | 2649.56 | 2656.51 |    6.95 |      22.125 |
| 128.00 MB | 5256.55 | 5251.88 | 5259.12 |    7.24 |      22.342 |
| 256.00 MB |10487.51 |10482.77 |10490.27 |    7.50 |      22.396 |
| 512.00 MB |21010.00 |21004.89 |21013.01 |    8.13 |      22.359 |

**Key observations:**
- **Latency**: ~11µs at small (8KB), much faster than allreduce initially
- **Bandwidth**: Reaches **~22.4 GB/s** (slightly better than allreduce!)
- **Busbw factor**: (n-1)/n = 0.875 for ws=8 (each rank sends once, receives n-1 pieces)

---

## Reduce-Scatter (C API / barrier)

| Size       | avg_us  | min_us  | max_us  | var_us  | busBW(GB/s) |
|------------|--------:|--------:|--------:|--------:|------------:|
| 8.00 KB   |   25.81 |   25.20 |   26.45 |    1.25 |       0.278 |
| 16.00 KB  |   25.56 |   24.21 |   26.91 |    2.70 |       0.561 |
| 32.00 KB  |   20.01 |   18.98 |   20.71 |    1.73 |       1.433 |
| 64.00 KB  |   20.23 |   19.48 |   20.87 |    1.39 |       2.835 |
| 128.00 KB |   22.48 |   21.86 |   23.07 |    1.21 |       5.101 |
| 256.00 KB |   22.12 |   21.39 |   22.75 |    1.36 |      10.368 |
| 512.00 KB |   28.80 |   27.09 |   30.65 |    3.57 |      15.931 |
| 1.00 MB   |   51.44 |   50.00 |   53.22 |    3.22 |      17.836 |
| 2.00 MB   |   95.69 |   94.43 |   97.24 |    2.81 |      19.176 |
| 4.00 MB   |  184.81 |  183.30 |  186.57 |    3.27 |      19.858 |
| 8.00 MB   |  361.96 |  360.70 |  363.20 |    2.50 |      20.279 |
| 16.00 MB  |  724.90 |  720.70 |  726.58 |    5.87 |      20.251 |
| 32.00 MB  | 1431.81 | 1426.78 | 1434.74 |    7.96 |      20.506 |
| 64.00 MB  | 2774.59 | 2769.83 | 2777.17 |    7.34 |      21.164 |
| 128.00 MB | 5441.86 | 5436.86 | 5444.70 |    7.84 |      21.581 |
| 256.00 MB |10795.78 |10791.09 |10798.71 |    7.62 |      21.757 |
| 512.00 MB |21625.01 |21619.22 |21628.62 |    9.40 |      21.723 |

**Key observations:**
- **Latency**: ~25-20µs at small sizes (faster than allreduce at startup)
- **Bandwidth**: Reaches **~21.7 GB/s** (slightly lower than allgather)
- **Busbw factor**: (n-1)/n = 0.875 for ws=8 (each rank receives 1/8 of total, same as allgather)

---

## Comparison Summary

| Op | Peak BW (GB/s) | Small Latency (8KB) | Scaling Profile |
|----|---------------:|-------------------:|---|
| AllReduce | 22.33 | 41.55 µs | Quick plateau to peak around 1-2MB |
| AllGather | 22.40 | 10.82 µs | Fast startup, smooth scaling |
| ReduceScatter | 21.72 | 25.81 µs | Mid-range startup, good scaling |

**Efficiency ranking (by peak bandwidth):**
1. AllGather: **22.40 GB/s** (best)
2. AllReduce: **22.33 GB/s** (within margin)
3. ReduceScatter: **21.72 GB/s** (acceptable, ~2% lower)

---

## Notes on Send/Recv

A `bench_ccl_send_recv_latency_c_api.cpp` benchmark was also created following the same structure (ping-pong pattern between paired ranks 0↔1, 2↔3, …), but encountered segmentation faults during execution on 4–8 ranks. This indicates a potential compatibility issue with `onecclSend()` / `onecclRecv()` C API in the current oneCCL 2021.17 library or oneAPI 2026.0 environment. The code is complete and available for future investigation/fixes; possibly requires:
- A newer oneCCL version with fixes for send/recv on GPU buffers
- Alternative initialization or group-based invocation patterns
- Buffer memory alignment or host vs. device placement adjustments

---

## Benchmark Logs

- `allreduce_c_api_ws8.log`
- `allgather_c_api_ws8.log`
- `reduce_scatter_c_api_ws8.log`
- `send_recv_c_api_ws8.log` (crashed; preserved for diagnostics)
- `bench_ccl_send_recv_latency_c_api.cpp` (source, needs debugging)

---

## Build & Run Commands

```bash
source ~/hanchao/2026.0/intel/oneapi/setvars.sh --force

# Build
for op in allreduce allgather reduce_scatter; do
  icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
    -I/opt/intel/oneapi/ccl/2021.17/include \
    -I/opt/intel/oneapi/mpi/2021.17/include \
    -L/opt/intel/oneapi/ccl/2021.17/lib \
    -L/opt/intel/oneapi/mpi/2021.17/lib \
    -lccl -lmpi \
    bench_ccl_${op}_latency_c_api.cpp -o bench_ccl_${op}_latency_c_api_26
done

# Run (ws=8)
for op in allreduce allgather reduce_scatter; do
  ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7 CCL_ATL_TRANSPORT=mpi \
  mpirun -n 8 ./bench_ccl_${op}_latency_c_api_26 \
    --min 12 --max 28 --warmup 20 --loop 100
done
```

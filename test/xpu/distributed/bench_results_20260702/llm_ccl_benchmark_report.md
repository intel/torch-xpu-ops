# oneCCL Collective Communication Benchmark Report

**Machine**: B60-5468-1 (10.239.167.23)  
**Date**: 2026-07-02  
**GPU**: 8x Intel B60 (BMG) [0xe211]  
**Toolchain**: oneAPI 2026.0  
**Docker**: intelgpu/ubuntu-26.04-rolling:26.18

---

## Summary

### Peak Bandwidth (GB/s)

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllReduce | 27.88 | 22.33 |
| AllGather | 27.88 | 22.40 |
| ReduceScatter | 26.01 | 21.77 |

### Small Message Latency (8 KB, µs)

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllReduce | 20.01 | 35.15 |
| AllGather | 7.60 | 10.92 |
| ReduceScatter | 13.64 | 22.41 |

---

## AllReduce 4-GPU (GPU 4-7)

| Size | Latency (µs) | Bandwidth (GB/s) |
|------|-------------:|-----------------:|
| 8 KB | 20.01 | 0.61 |
| 16 KB | 17.99 | 1.37 |
| 32 KB | 18.63 | 2.64 |
| 64 KB | 17.75 | 5.54 |
| 128 KB | 18.19 | 10.81 |
| 256 KB | 25.33 | 15.52 |
| 512 KB | 44.94 | 17.50 |
| 1 MB | 82.50 | 19.06 |
| 2 MB | 159.43 | 19.73 |
| 4 MB | 313.77 | 20.05 |
| 8 MB | 494.74 | 25.43 |
| 16 MB | 976.67 | 25.77 |
| 32 MB | 1882.83 | 26.73 |
| 64 MB | 3680.21 | 27.35 |
| 128 MB | 7274.41 | 27.68 |
| 256 MB | 14456.69 | 27.85 |
| 512 MB | 28885.92 | **27.88** |

---

## AllReduce 8-GPU

| Size | Latency (µs) | Bandwidth (GB/s) |
|------|-------------:|-----------------:|
| 8 KB | 35.15 | 0.41 |
| 16 KB | 34.09 | 0.84 |
| 32 KB | 32.05 | 1.79 |
| 64 KB | 30.44 | 3.77 |
| 128 KB | 33.06 | 6.94 |
| 256 KB | 34.23 | 13.40 |
| 512 KB | 50.90 | 18.03 |
| 1 MB | 96.66 | 18.99 |
| 2 MB | 185.36 | 19.80 |
| 4 MB | 362.11 | 20.27 |
| 8 MB | 691.16 | 21.24 |
| 16 MB | 1405.48 | 20.89 |
| 32 MB | 2742.68 | 21.41 |
| 64 MB | 5347.55 | 21.96 |
| 128 MB | 10554.91 | 22.25 |
| 256 MB | 21024.52 | 22.34 |
| 512 MB | 42073.81 | **22.33** |

---

## AllGather 4-GPU (GPU 4-7)

| Size (OUT) | Latency (µs) | Bandwidth (GB/s) |
|------------|-------------:|-----------------:|
| 8 KB | 7.60 | 0.81 |
| 16 KB | 12.30 | 1.00 |
| 32 KB | 11.41 | 2.16 |
| 64 KB | 12.10 | 4.06 |
| 112 KB | 9.84 | 8.74 |
| 128 KB | 10.14 | 9.70 |
| 224 KB | 13.86 | 12.41 |
| 256 KB | 14.87 | 13.22 |
| 512 KB | 24.67 | 15.94 |
| 32 MB | 942.68 | 26.70 |
| 64 MB | 1840.12 | 27.35 |
| 112 MB | 3191.89 | 27.60 |
| 224 MB | 6345.52 | 27.76 |
| 392 MB | 11076.48 | 27.83 |
| 448 MB | 12653.69 | 27.84 |
| 896 MB | 25270.87 | **27.88** |

---

## AllGather 8-GPU

| Size (OUT) | Latency (µs) | Bandwidth (GB/s) |
|------------|-------------:|-----------------:|
| 8 KB | 10.92 | 0.66 |
| 16 KB | 11.80 | 1.22 |
| 32 KB | 17.33 | 1.65 |
| 64 KB | 16.74 | 3.43 |
| 128 KB | 17.42 | 6.59 |
| 256 KB | 17.00 | 13.49 |
| 512 KB | 27.92 | 16.43 |
| 1 MB | 50.72 | 18.09 |
| 2 MB | 94.98 | 19.32 |
| 4 MB | 184.27 | 19.92 |
| 8 MB | 361.66 | 20.30 |
| 16 MB | 669.28 | 21.93 |
| 32 MB | 1329.82 | 22.08 |
| 64 MB | 2652.83 | 22.14 |
| 128 MB | 5254.58 | 22.35 |
| 256 MB | 10486.13 | **22.40** |
| 512 MB | 21008.54 | 22.36 |

---

## ReduceScatter 4-GPU (GPU 4-7)

| Size (IN) | Latency (µs) | Bandwidth (GB/s) |
|-----------|-------------:|-----------------:|
| 8 KB | 13.64 | 0.45 |
| 16 KB | 12.28 | 1.00 |
| 32 KB | 12.83 | 1.92 |
| 64 KB | 12.82 | 3.83 |
| 112 KB | 12.56 | 6.85 |
| 128 KB | 12.30 | 7.99 |
| 224 KB | 14.30 | 12.03 |
| 256 KB | 15.40 | 12.77 |
| 512 KB | 25.09 | 15.67 |
| 32 MB | 1041.22 | 24.17 |
| 64 MB | 1990.70 | 25.28 |
| 112 MB | 3432.48 | 25.66 |
| 128 MB | 3912.65 | 25.73 |
| 224 MB | 6803.89 | 25.89 |
| 448 MB | 13599.43 | 25.91 |
| 784 MB | 23741.26 | 25.97 |
| 896 MB | 27125.10 | 25.98 |
| 1792 MB | 54177.74 | **26.01** |

---

## ReduceScatter 8-GPU

| Size (IN) | Latency (µs) | Bandwidth (GB/s) |
|-----------|-------------:|-----------------:|
| 8 KB | 22.41 | 0.32 |
| 16 KB | 21.66 | 0.66 |
| 32 KB | 21.64 | 1.33 |
| 64 KB | 20.27 | 2.83 |
| 128 KB | 22.57 | 5.08 |
| 256 KB | 22.10 | 10.38 |
| 512 KB | 28.70 | 15.99 |
| 1 MB | 51.53 | 17.80 |
| 2 MB | 95.64 | 19.19 |
| 4 MB | 184.59 | 19.88 |
| 8 MB | 362.49 | 20.25 |
| 16 MB | 725.66 | 20.23 |
| 32 MB | 1432.68 | 20.49 |
| 64 MB | 2769.85 | 21.20 |
| 128 MB | 5434.75 | 21.61 |
| 256 MB | 10790.45 | **21.77** |
| 512 MB | 21621.55 | 21.73 |

---

## Key Observations

1. **Peak Bandwidth**: 
   - 4-GPU achieves ~27.88 GB/s for AllReduce/AllGather, ~26 GB/s for ReduceScatter
   - 8-GPU achieves ~22 GB/s across all collectives

2. **4-GPU vs 8-GPU**:
   - 4-GPU shows ~25% higher peak bandwidth than 8-GPU
   - Likely due to PCIe topology (4 GPUs may share closer switch)

3. **Small Message Latency**:
   - AllGather has lowest latency (~7-11 µs)
   - AllReduce has highest latency (~20-35 µs) due to reduction computation
   - ReduceScatter is in between (~13-22 µs)

4. **Bandwidth Saturation**:
   - Bandwidth saturates around 8-16 MB message size
   - Smaller messages are latency-bound

---

## Log Files

- `allreduce_4gpu.log`
- `allreduce_8gpu.log`
- `allgather_4gpu.log`
- `allgather_8gpu.log`
- `allgather_exact_prefill.log`
- `allgather_exact_decode.log`
- `reduce_scatter_4gpu.log`
- `reduce_scatter_8gpu.log`
- `reduce_scatter_exact_prefill.log`
- `reduce_scatter_exact_decode.log`

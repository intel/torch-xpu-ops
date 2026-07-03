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
| AllGather | 27.86 | 22.40 |
| ReduceScatter | 25.95 | 21.77 |

### Small Message Latency (8 KB, µs)

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllReduce | 20.01 | 35.15 |
| AllGather | 7.24 | 10.92 |
| ReduceScatter | 13.32 | 22.41 |

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
| 8 KB | 7.24 | 0.85 |
| 16 KB | 10.84 | 1.13 |
| 32 KB | 9.97 | 2.46 |
| 64 KB | 10.54 | 4.66 |
| 128 KB | 10.19 | 9.65 |
| 256 KB | 14.84 | 13.25 |
| 512 KB | 24.66 | 15.94 |
| 1 MB | 43.82 | 17.95 |
| 2 MB | 81.94 | 19.20 |
| 4 MB | 158.56 | 19.84 |
| 8 MB | 245.79 | 25.60 |
| 16 MB | 470.49 | 26.74 |
| 32 MB | 928.80 | 27.10 |
| 64 MB | 1839.91 | 27.36 |
| 128 MB | 3641.89 | 27.64 |
| 256 MB | 7245.04 | 27.79 |
| 512 MB | 14452.72 | **27.86** |

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
| 8 KB | 13.32 | 0.46 |
| 16 KB | 12.53 | 0.98 |
| 32 KB | 12.92 | 1.90 |
| 64 KB | 12.81 | 3.84 |
| 128 KB | 12.38 | 7.94 |
| 256 KB | 15.37 | 12.79 |
| 512 KB | 25.09 | 15.67 |
| 1 MB | 44.21 | 17.79 |
| 2 MB | 82.72 | 19.01 |
| 4 MB | 159.86 | 19.68 |
| 8 MB | 309.91 | 20.30 |
| 16 MB | 525.04 | 23.97 |
| 32 MB | 1023.96 | 24.58 |
| 64 MB | 1988.88 | 25.31 |
| 128 MB | 3913.90 | 25.72 |
| 256 MB | 7759.98 | 25.94 |
| 512 MB | 15516.66 | **25.95** |

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
   - 4-GPU achieves ~27.88 GB/s for AllReduce/AllGather, ~25.95 GB/s for ReduceScatter
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

- `allreduce_4gpu.log` - AllReduce 4-GPU (2^12 - 2^28)
- `allreduce_8gpu.log` - AllReduce 8-GPU (2^12 - 2^28)
- `allgather_4gpu.log` - AllGather 4-GPU (2^12 - 2^28)
- `allgather_8gpu.log` - AllGather 8-GPU (2^12 - 2^28)
- `reduce_scatter_4gpu.log` - ReduceScatter 4-GPU (2^12 - 2^28)
- `reduce_scatter_8gpu.log` - ReduceScatter 8-GPU (2^12 - 2^28)
- `*_exact_*.log` - 精确尺寸测试 (用于 LLM 模型)

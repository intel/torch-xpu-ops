# LLM Collective Communication Benchmark Report

**Machine**: B60-5468-1 (10.239.167.23)  
**Date**: 2026-07-02  
**GPU**: 8x Intel B60 (BMG) [0xe211]  
**Toolchain**: oneAPI 2026.0  
**Docker**: intelgpu/ubuntu-26.04-rolling:26.18

---

## 1. Benchmark Results Summary

### Peak Bandwidth (GB/s)

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllGather | 27.86 | 22.40 |
| ReduceScatter | 25.95 | 21.77 |

### Small Message Latency (8 KB)

| Collective | 4-GPU (µs) | 8-GPU (µs) |
|------------|----------:|----------:|
| AllGather | 7.42 | 10.92 |
| ReduceScatter | 13.69 | 22.41 |

---

## 2. LLM Model Performance

### Model Configurations

| Model | hidden_size | intermediate | num_experts | topk | layers | Source |
|-------|-------------|--------------|-------------|------|--------|--------|
| DeepSeek 4 Pro | 7168 | 3072 (MoE) | 384 | 6 | 61 | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| DeepSeek 4 Flash | 4096 | 2048 (MoE) | 256 | 6 | 43 | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| Qwen 3 235B | 4096 | 1536 (MoE) | 128 | 8 | 94 | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Hunyuan 3 | 4096 | 1536 (MoE) | 192 | 8 | 80 | [HF](https://huggingface.co/tencent/Hy3-preview) |

### Message Size Calculation

- **AllGather OUTPUT**: `seq × bs × hidden × dtype_size`
- **ReduceScatter INPUT**: `seq × bs × hidden × dtype_size`

Note: 
- **Exact values** are from benchmark_exact_v2 with precise sizes
- **~values** are interpolated from power-of-2 benchmark
- **N/A** means size exceeds benchmark range

### DECODE Phase (seq = 1)

| Model | TP | BS | AG Size (OUT) | AG Latency (µs) | RS Size (IN) | RS Latency (µs) |
|-------|---:|---:|--------------:|----------------:|-------------:|----------------:|
| DeepSeek 4 Flash | 4 | 64 | 256 KB | 14.82 | 512 KB | 25.06 |
| DeepSeek 4 Pro | 8 | 16 | 112 KB | ~17.25 | 224 KB | ~22.22 |
| Qwen 3 235B (MXFP8) | 4 | 16 | 64 KB | 10.33 | 128 KB | 12.43 |
| Qwen 3 235B (BF16) | 4 | 4 | 32 KB | 10.14 | 32 KB | 14.20 |
| Qwen 3 235B (BF16) | 8 | 8 | 64 KB | 16.74 | 64 KB | 20.27 |
| Hunyuan 3 (MXFP8) | 4 | 4 | 16 KB | 10.23 | 32 KB | 14.20 |
| Hunyuan 3 (BF16) | 4 | 2 | 16 KB | 10.23 | 16 KB | 12.74 |

### PREFILL Phase (seq = 3584, input_len for LLM)

**Updated with exact-size benchmarks (TP4 GPU 4-7)**:

| Model | TP | BS | AG Size (OUT) | AG Time (ms) | RS Size (IN) | RS Time (ms) |
|-------|---:|---:|--------------:|-------------:|-------------:|-------------:|
| DeepSeek 4 Flash | 4 | 64 | **896 MB** | **25.27** | **1792 MB** | **54.18** |
| DeepSeek 4 Pro | 8 | 16 | 392 MB | ~16.08 | 784 MB | ~23.21 |
| Qwen 3 235B (MXFP8) | 4 | 16 | 224 MB | ~6.34 | 448 MB | ~13.58 |
| Qwen 3 235B (BF16) | 4 | 4 | 112 MB | ~3.19 | 112 MB | ~3.43 |
| Hunyuan 3 (MXFP8) | 4 | 4 | 64 MB | 1.84 | 128 MB | 3.91 |
| Hunyuan 3 (BF16) | 4 | 2 | 64 MB | 1.84 | 64 MB | 1.99 |

**Exact Prefill Benchmark Results (TP4)**:

| Size (MB) | AllGather (ms) | AllGather BW | ReduceScatter (ms) | RS BW |
|----------:|---------------:|-------------:|-------------------:|------:|
| 64 | 1.84 | 27.35 GB/s | 1.99 | 25.31 GB/s |
| 128 | 3.64 | 27.63 GB/s | 3.91 | 25.72 GB/s |
| 256 | 7.25 | 27.78 GB/s | 7.76 | 25.94 GB/s |
| 384 | 10.85 | 27.83 GB/s | — | — |
| 448 | 12.65 | 27.84 GB/s | — | — |
| 768 | — | — | 23.21 | 26.02 GB/s |
| 896 | **25.27** | **27.89 GB/s** | **27.12** | **25.98 GB/s** |
| 1792 | — | — | **54.18** | **26.01 GB/s** |

---

## 3. Detailed Benchmark Results

### AllGather 4-GPU (GPU 4-7)

| Size (OUT) | Latency (µs) | Bandwidth (GB/s) |
|------------|-------------:|-----------------:|
| 8 KB | 7.42 | 0.83 |
| 16 KB | 10.23 | 1.20 |
| 32 KB | 10.14 | 2.42 |
| 64 KB | 10.33 | 4.76 |
| 128 KB | 10.19 | 9.65 |
| 256 KB | 14.82 | 13.27 |
| 512 KB | 24.66 | 15.95 |
| 1 MB | 43.76 | 17.97 |
| 2 MB | 82.24 | 19.13 |
| 4 MB | 158.40 | 19.86 |
| 8 MB | 245.76 | 25.60 |
| 16 MB | 470.44 | 26.75 |
| 32 MB | 928.90 | 27.09 |
| 64 MB | 1839.75 | 27.36 |
| 128 MB | 3642.08 | 27.64 |
| 256 MB | 7244.96 | 27.79 |
| 512 MB | 14452.89 | **27.86** |

### AllGather 8-GPU

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

### ReduceScatter 4-GPU (GPU 4-7)

| Size (IN) | Latency (µs) | Bandwidth (GB/s) |
|-----------|-------------:|-----------------:|
| 8 KB | 13.69 | 0.45 |
| 16 KB | 12.74 | 0.97 |
| 32 KB | 14.20 | 1.73 |
| 64 KB | 12.80 | 3.84 |
| 128 KB | 12.43 | 7.91 |
| 256 KB | 15.34 | 12.82 |
| 512 KB | 25.06 | 15.69 |
| 1 MB | 44.14 | 17.82 |
| 2 MB | 82.62 | 19.04 |
| 4 MB | 159.22 | 19.76 |
| 8 MB | 310.41 | 20.27 |
| 16 MB | 524.94 | 23.97 |
| 32 MB | 1024.04 | 24.58 |
| 64 MB | 1989.19 | 25.30 |
| 128 MB | 3914.22 | 25.72 |
| 256 MB | 7760.06 | 25.94 |
| 512 MB | 15516.88 | **25.95** |

### ReduceScatter 8-GPU

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

## 4. Key Observations

1. **Peak Bandwidth**: 
   - 4-GPU: AllGather 27.86 GB/s, ReduceScatter 25.95 GB/s
   - 8-GPU: AllGather 22.40 GB/s, ReduceScatter 21.77 GB/s

2. **Small Message Latency**: 
   - AllGather: ~7-11 µs base latency
   - ReduceScatter: ~13-22 µs base latency

3. **4-GPU vs 8-GPU**:
   - 4-GPU shows ~25% higher peak bandwidth
   - Likely due to PCIe topology (4 GPUs on same switch)

4. **Decode vs Prefill**:
   - Decode phase is latency-bound (small messages ~10-25 µs)
   - Prefill phase is bandwidth-bound (large messages approach peak BW)

5. **LLM Prefill Timing** (TP4, exact sizes):
   - DeepSeek 4 Flash: AG 896MB = 25.27ms, RS 1792MB = 54.18ms
   - Per-layer overhead ~80ms for DeepSeek 4 Flash prefill

---

## 5. Benchmark Tools

### Exact Size Benchmark (New)
```bash
# AllGather with exact MB sizes
mpirun -n 4 ./bench_ccl_allgather_exact_v2 --sizes-mb 384,448,896 --warmup 10 --loop 50

# ReduceScatter with exact MB sizes  
mpirun -n 4 ./bench_ccl_reduce_scatter_exact_v2 --sizes-mb 768,896,1792 --warmup 10 --loop 50
```

### Log Files

Saved to `/data/hanchao/bench_logs/`:
- `allgather_4gpu.log`
- `allgather_8gpu.log`
- `reduce_scatter_4gpu.log`
- `reduce_scatter_8gpu.log`

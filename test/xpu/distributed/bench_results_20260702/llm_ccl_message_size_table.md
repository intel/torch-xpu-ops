# LLM Collective Communication Message Size 表

**Machine**: B60-5468-1 (10.239.167.23)  
**Date**: 2026-07-02  
**GPU**: 8x Intel B60 (BMG)  
**Toolchain**: oneAPI 2026.0

## 模型配置

| Model             | hidden_size | 来源 |
|-------------------|-------------|------|
| DeepSeek 4 Pro    | 7168        | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| DeepSeek 4 Flash  | 4096        | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| Qwen 3 235B       | 4096        | [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B) |
| Hunyuan 3         | 4096        | [HF](https://huggingface.co/tencent/Hy3-preview) |

---

## Benchmark Message Size 定义

- **AllGather**: `Size(out)` = OUTPUT size = `seq × bs × hidden × dtype_size` (完整 tensor)
- **ReduceScatter**: `Size(in)` = INPUT size = `seq × bs × hidden × dtype_size` (完整 tensor)

**注意**: 这是 benchmark 报告的 size，实际每个 rank 发送的数据量 = size / TP

---

## PREFILL 阶段 (seq = input_len)

### TP4 配置 (GPU 4-7)

| Model | Assumption | TP | BS | seq | hidden | AG Size (OUT) | AG Time (ms) | RS Size (IN) | RS Time (ms) |
|-------|------------|----|----|-----|--------|---------------|--------------|--------------|--------------|
| DeepSeek 4 Flash | 4bit MoE, 8bit dense | 4 | 64 | 3584 | 4096 | **896 MB** | **25.27** | **1792 MB** | **54.18** |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 3584 | 4096 | **224 MB** | **6.35** | **448 MB** | **13.60** |
| Qwen 3 235B | BF16 | 4 | 4 | 3584 | 4096 | **112 MB** | **3.19** | **112 MB** | **3.43** |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 3584 | 4096 | **448 MB** | **12.65** | **896 MB** | **27.13** |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | 4096 | **64 MB** | **1.84** | **128 MB** | **3.91** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | 4096 | **64 MB** | **1.84** | **64 MB** | **1.99** |

### TP8 配置 (GPU 0-7)

| Model | Assumption | TP | BS | seq | hidden | AG Size (OUT) | AG Time (ms) | RS Size (IN) | RS Time (ms) |
|-------|------------|----|----|-----|--------|---------------|--------------|--------------|--------------|
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 3584 | 7168 | **392 MB** | **16.07** | **784 MB** | **33.11** |
| Qwen 3 235B | BF16 | 8 | 8 | 3584 | 4096 | **224 MB** | **9.18** | **224 MB** | **9.45** |

---

## DECODE 阶段 (seq = 1)

### TP4 配置 (GPU 4-7)

| Model | Assumption | TP | BS | hidden | AG Size (OUT) | AG Latency (µs) | RS Size (IN) | RS Latency (µs) |
|-------|------------|----|----|--------|---------------|-----------------|--------------|-----------------|
| DeepSeek 4 Flash | 4bit MoE, 8bit dense | 4 | 64 | 4096 | **256 KB** | **14.87** | **512 KB** | **25.09** |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 4096 | **64 KB** | **12.10** | **128 KB** | **12.30** |
| Qwen 3 235B | BF16 | 4 | 4 | 4096 | **32 KB** | **11.41** | **32 KB** | **12.83** |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 4096 | **128 KB** | **10.14** | **256 KB** | **15.40** |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | **16 KB** | **12.30** | **32 KB** | **12.83** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | **16 KB** | **12.30** | **16 KB** | **12.28** |

### TP8 配置 (GPU 0-7)

| Model | Assumption | TP | BS | hidden | AG Size (OUT) | AG Latency (µs) | RS Size (IN) | RS Latency (µs) |
|-------|------------|----|----|--------|---------------|-----------------|--------------|-----------------|
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 7168 | **112 KB** | **15.68** | **224 KB** | **26.09** |
| Qwen 3 235B | BF16 | 8 | 8 | 4096 | **64 KB** | **15.66** | **64 KB** | **24.15** |

**所有数值均为精确尺寸实测结果**

---

## 精确尺寸 Benchmark 结果

### TP4 AllGather Prefill (GPU 4-7)

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 32 MB | 0.94 | 26.70 |
| 64 MB | 1.84 | 27.35 |
| 112 MB | 3.19 | 27.60 |
| 224 MB | 6.35 | 27.76 |
| 448 MB | 12.65 | 27.84 |
| 896 MB | **25.27** | **27.88** |

### TP4 ReduceScatter Prefill (GPU 4-7)

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 32 MB | 1.04 | 24.17 |
| 64 MB | 1.99 | 25.28 |
| 112 MB | 3.43 | 25.66 |
| 128 MB | 3.91 | 25.73 |
| 224 MB | 6.80 | 25.89 |
| 448 MB | 13.60 | 25.91 |
| 896 MB | 27.13 | 25.98 |
| 1792 MB | **54.18** | **26.01** |

### TP8 AllGather Prefill (GPU 0-7)

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 224 MB | 9.18 | 22.39 |
| 392 MB | **16.07** | **22.38** |

### TP8 ReduceScatter Prefill (GPU 0-7)

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 224 MB | 9.45 | 21.76 |
| 784 MB | **33.11** | **21.73** |

### TP4 Decode (GPU 4-7)

| Size | AG (µs) | RS (µs) |
|------|---------|---------|
| 8 KB | 7.60 | 13.64 |
| 16 KB | 12.30 | 12.28 |
| 32 KB | 11.41 | 12.83 |
| 64 KB | 12.10 | 12.82 |
| 112 KB | 9.84 | 12.56 |
| 128 KB | 10.14 | 12.30 |
| 224 KB | 13.86 | 14.30 |
| 256 KB | 14.87 | 15.40 |
| 512 KB | 24.67 | 25.09 |

### TP8 Decode (GPU 0-7)

| Size | AG (µs) | RS (µs) |
|------|---------|---------|
| 64 KB | 15.66 | 24.15 |
| 112 KB | 15.68 | — |
| 224 KB | — | 26.09 |

---

## Benchmark 命令

```bash
# 环境设置
source /opt/intel/oneapi/setvars.sh --force
export CCL_ATL_TRANSPORT=mpi

# TP4 (GPU 4-7)
export ZE_AFFINITY_MASK=4,5,6,7
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --sizes-mb 32,64,112,224,448,896 --warmup 10 --loop 50
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --sizes-mb 32,64,112,128,224,448,896,1792 --warmup 10 --loop 50

# TP8 (GPU 0-7)
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
mpirun -n 8 ./bench_ccl_allgather_latency_c_api --sizes-mb 224,392 --warmup 10 --loop 50
mpirun -n 8 ./bench_ccl_reduce_scatter_latency_c_api --sizes-mb 224,784 --warmup 10 --loop 50
```

### 精确尺寸参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--sizes-mb` | 以 MB 为单位的精确尺寸列表 | `--sizes-mb 64,128,256,896` |
| `--sizes-kb` | 以 KB 为单位的精确尺寸列表 | `--sizes-kb 112,224,256,512` |
| `--sizes` | 以 bf16 元素数量为单位 | `--sizes 33554432,67108864` |
| `--min/--max` | 2^n 尺寸范围 (log2) | `--min 24 --max 28` |

---

## Peak Bandwidth Summary

| Collective | TP4 (4-GPU) | TP8 (8-GPU) |
|------------|------------:|------------:|
| AllReduce | 27.88 GB/s | 22.33 GB/s |
| AllGather | 27.88 GB/s | 22.40 GB/s |
| ReduceScatter | 26.01 GB/s | 21.77 GB/s |

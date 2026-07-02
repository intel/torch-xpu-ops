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

| Model | Assumption | TP | BS | seq | hidden | AG Size (OUT) | AG Time (ms) | RS Size (IN) | RS Time (ms) |
|-------|------------|----|----|-----|--------|---------------|--------------|--------------|--------------|
| DeepSeek 4 Flash | 4bit MoE, 8bit dense | 4 | 64 | 3584 | 4096 | **896 MB** | **25.27** | **1792 MB** | **54.18** |
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 3584 | 7168 | **392 MB** | **11.08** | **784 MB** | **23.75** |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 3584 | 4096 | **224 MB** | **6.35** | **448 MB** | **13.60** |
| Qwen 3 235B | BF16 | 4 | 4 | 3584 | 4096 | **112 MB** | **3.19** | **112 MB** | **3.43** |
| Qwen 3 235B | BF16 | 8 | 8 | 3584 | 4096 | **224 MB** | **6.35** | **224 MB** | **6.81** |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 3584 | 4096 | **448 MB** | **12.65** | **896 MB** | **27.13** |
| Hunyuan 3 | MXFP8, 8bit GQA | 2 | 2 | 4096 | 4096 | **32 MB** | **0.94** | **64 MB** | **1.99** |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | 4096 | **64 MB** | **1.84** | **128 MB** | **3.91** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | 4096 | **64 MB** | **1.84** | **64 MB** | **1.99** |

---

## DECODE 阶段 (seq = 1)

| Model | Assumption | TP | BS | hidden | AG Size (OUT) | AG Latency (µs) | RS Size (IN) | RS Latency (µs) |
|-------|------------|----|----|--------|---------------|-----------------|--------------|-----------------|
| DeepSeek 4 Flash | 4bit MoE, 8bit dense | 4 | 64 | 4096 | **256 KB** | **14.84** | **512 KB** | **25.09** |
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 7168 | **112 KB** | **11.52** | **224 KB** | **16.23** |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 4096 | **64 KB** | **10.54** | **128 KB** | **12.38** |
| Qwen 3 235B | BF16 | 4 | 4 | 4096 | **32 KB** | **9.97** | **32 KB** | **12.92** |
| Qwen 3 235B | BF16 | 8 | 8 | 4096 | **64 KB** | **10.54** | **64 KB** | **12.81** |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 4096 | **128 KB** | **10.19** | **256 KB** | **15.37** |
| Hunyuan 3 | MXFP8, 8bit GQA | 2 | 2 | 4096 | **8 KB** | **7.24** | **16 KB** | **12.53** |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | **16 KB** | **10.84** | **32 KB** | **12.92** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | **16 KB** | **10.84** | **16 KB** | **12.53** |

**所有数值均为精确尺寸实测结果**

---

## 精确尺寸 Benchmark 结果 (TP4, GPU 4-7)

### AllGather Prefill

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 32 MB | 0.94 | 26.73 |
| 64 MB | 1.84 | 27.35 |
| 112 MB | 3.19 | 27.60 |
| 224 MB | 6.35 | 27.76 |
| 392 MB | 11.08 | 27.83 |
| 448 MB | 12.65 | 27.84 |
| 896 MB | **25.27** | **27.88** |

### ReduceScatter Prefill

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 32 MB | 1.03 | 24.37 |
| 64 MB | 1.99 | 25.30 |
| 112 MB | 3.43 | 25.66 |
| 128 MB | 3.91 | 25.72 |
| 224 MB | 6.81 | 25.88 |
| 448 MB | 13.60 | 25.91 |
| 784 MB | 23.75 | 25.96 |
| 896 MB | 27.13 | 25.97 |
| 1792 MB | **54.18** | **26.01** |

### Decode (精确尺寸)

| Size | AG (µs) | RS (µs) |
|------|---------|---------|
| 8 KB | 7.24 | 13.32 |
| 16 KB | 10.84 | 12.53 |
| 32 KB | 9.97 | 12.92 |
| 64 KB | 10.54 | 12.81 |
| 112 KB | **11.52** | **17.15** |
| 128 KB | 10.19 | 12.38 |
| 224 KB | **14.75** | **16.23** |
| 256 KB | 14.84 | 15.37 |
| 512 KB | 24.66 | 25.09 |

---

## Benchmark 命令

```bash
# 环境设置
source /opt/intel/oneapi/setvars.sh --force
export ZE_AFFINITY_MASK=4,5,6,7  # 使用 GPU 4-7 (避免与其他用户冲突)
export CCL_ATL_TRANSPORT=mpi

# Decode 阶段 - 精确尺寸测试
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --sizes-kb 8,16,32,64,112,128,224,256,512 --warmup 20 --loop 100
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --sizes-kb 8,16,32,64,112,128,224,256,512 --warmup 20 --loop 100

# Prefill 阶段 - 精确尺寸测试
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --sizes-mb 32,64,112,224,392,448,896 --warmup 10 --loop 50
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --sizes-mb 32,64,112,128,224,448,784,896,1792 --warmup 10 --loop 50
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

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllGather | 27.88 GB/s | 22.40 GB/s |
| ReduceScatter | 26.01 GB/s | 21.77 GB/s |

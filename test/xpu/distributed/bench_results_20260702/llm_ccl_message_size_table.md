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
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 3584 | 7168 | **392 MB** | ~14.40 | **784 MB** | ~23.21 |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 3584 | 4096 | **224 MB** | ~8.06 | **448 MB** | ~17.23 |
| Qwen 3 235B | BF16 | 4 | 4 | 3584 | 4096 | **112 MB** | ~4.04 | **112 MB** | ~4.35 |
| Qwen 3 235B | BF16 | 8 | 8 | 3584 | 4096 | **224 MB** | ~9.15 | **224 MB** | ~10.16 |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 3584 | 4096 | **448 MB** | **12.65** | **896 MB** | **27.12** |
| Hunyuan 3 | MXFP8, 8bit GQA | 2 | 2 | 4096 | 4096 | **32 MB** | ~0.93 | **64 MB** | ~1.99 |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | 4096 | **64 MB** | **1.84** | **128 MB** | **3.91** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | 4096 | **64 MB** | **1.84** | **64 MB** | **1.99** |

**说明**: 
- **粗体数值** = 精确尺寸测试结果
- ~数值 = 从 2^n 尺寸插值估算

---

## DECODE 阶段 (seq = 1)

| Model | Assumption | TP | BS | hidden | AG Size (OUT) | AG Latency (µs) | RS Size (IN) | RS Latency (µs) |
|-------|------------|----|----|--------|---------------|-----------------|--------------|-----------------|
| DeepSeek 4 Flash | 4bit MoE, 8bit dense | 4 | 64 | 4096 | **256 KB** | **14.82** | **512 KB** | **25.06** |
| DeepSeek 4 Pro | 4bit MoE, 8bit dense | 8 | 16 | 7168 | **112 KB** | ~17.42 | **224 KB** | ~22.10 |
| Qwen 3 235B | MXFP8, BF16 GQA | 4 | 16 | 4096 | **64 KB** | **10.33** | **128 KB** | **12.43** |
| Qwen 3 235B | BF16 | 4 | 4 | 4096 | **32 KB** | **10.14** | **32 KB** | **14.20** |
| Qwen 3 235B | BF16 | 8 | 8 | 4096 | **64 KB** | **16.74** | **64 KB** | **20.27** |
| Qwen 3 235B | MXFP4 MoE | 4 | 32 | 4096 | **128 KB** | **10.19** | **256 KB** | **15.34** |
| Hunyuan 3 | MXFP8, 8bit GQA | 2 | 2 | 4096 | **8 KB** | ~7.42 | **16 KB** | ~12.74 |
| Hunyuan 3 | MXFP8, 8bit GQA | 4 | 4 | 4096 | **16 KB** | **10.23** | **32 KB** | **14.20** |
| Hunyuan 3 | BF16 | 4 | 2 | 4096 | **16 KB** | **10.23** | **16 KB** | **12.74** |

---

## 精确尺寸 Benchmark 结果 (TP4, GPU 4-7)

### AllGather

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 64 MB | 1.84 | 27.35 |
| 128 MB | 3.64 | 27.63 |
| 256 MB | 7.25 | 27.78 |
| 384 MB | 10.85 | 27.83 |
| 448 MB | 12.65 | 27.84 |
| 896 MB | **25.27** | **27.89** |

### ReduceScatter

| Size | Time (ms) | Bandwidth (GB/s) |
|------|-----------|------------------|
| 64 MB | 1.99 | 25.31 |
| 128 MB | 3.91 | 25.72 |
| 256 MB | 7.76 | 25.94 |
| 768 MB | 23.21 | 26.02 |
| 896 MB | 27.12 | 25.98 |
| 1792 MB | **54.18** | **26.01** |

---

## Benchmark 命令

```bash
# 环境设置
source /opt/intel/oneapi/setvars.sh --force
export ZE_AFFINITY_MASK=4,5,6,7  # 使用 GPU 4-7 (避免与其他用户冲突)
export CCL_ATL_TRANSPORT=mpi

# Decode 阶段 - 小消息延迟测试
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --min 12 --max 18 --warmup 20 --loop 100
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --min 12 --max 18 --warmup 20 --loop 100

# Prefill 阶段 - 精确尺寸测试 (推荐)
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --sizes-mb 64,128,256,384,448,896 --warmup 10 --loop 50
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --sizes-mb 64,128,256,768,896,1792 --warmup 10 --loop 50

# 或者使用 2^n 尺寸范围
mpirun -n 4 ./bench_ccl_allgather_latency_c_api --min 24 --max 29 --warmup 10 --loop 50
mpirun -n 4 ./bench_ccl_reduce_scatter_latency_c_api --min 24 --max 29 --warmup 10 --loop 50
```

### 精确尺寸参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--sizes-mb` | 以 MB 为单位的精确尺寸列表 | `--sizes-mb 64,128,256,896` |
| `--sizes-kb` | 以 KB 为单位的精确尺寸列表 | `--sizes-kb 256,512,1024` |
| `--sizes` | 以 bf16 元素数量为单位 | `--sizes 33554432,67108864` |
| `--min/--max` | 2^n 尺寸范围 (log2) | `--min 24 --max 28` |

---

## Peak Bandwidth Summary

| Collective | 4-GPU | 8-GPU |
|------------|------:|------:|
| AllGather | 27.86 GB/s | 22.40 GB/s |
| ReduceScatter | 25.95 GB/s | 21.77 GB/s |

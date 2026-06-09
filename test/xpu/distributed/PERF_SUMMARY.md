# oneCCL Allreduce Performance Benchmark

测试平台：BMG ×4 (`ZE_AFFINITY_MASK=0,1,2,3`)，oneCCL native (oneAPI 2026.0)
数据类型：bfloat16，world_size = 4
Benchmark：`symm/bench_native/bench_ccl_prefill.cpp`（fire-and-forget，SYCL event profiling）
环境变量：`NEOReadDebugKeys=1 OverrideL1CachePolicyInSurfaceStateAndStateless=2 NEO_CACHE_PERSISTENT=0`

---

## 1. Allreduce 性能 Summary

### 组 1：≤ 2MB（小消息，延迟受限区）

| Size | avg_us | var_us | algBW (GB/s) | busBW (GB/s) |
|---------:|--------:|--------:|-------------:|-------------:|
| 256 B    |   20.10 |  149.19 |        0.013 |        0.019 |
| 512 B    |   19.48 |   32.37 |        0.026 |        0.039 |
| 1.00 KB  |   19.32 |   38.00 |        0.053 |        0.079 |
| 2.00 KB  |   19.56 |   33.98 |        0.105 |        0.157 |
| 4.00 KB  |   19.71 |   34.92 |        0.208 |        0.312 |
| 8.00 KB  |   23.80 |   44.38 |        0.344 |        0.516 |
| 16.00 KB |   25.16 |   25.00 |        0.651 |        0.977 |
| 32.00 KB |   27.50 |   29.01 |        1.192 |        1.787 |
| 64.00 KB |   32.11 |   28.70 |        2.041 |        3.062 |
| 128.00 KB|   32.24 |   28.88 |        4.066 |        6.098 |
| 256.00 KB|   32.42 |   47.06 |        8.085 |       12.128 |
| 512.00 KB|   46.87 |   55.05 |       11.185 |       16.777 |
| 1.00 MB  |   83.91 |   29.69 |       12.497 |       18.745 |
| 2.00 MB  |  157.66 |   49.43 |       13.302 |       19.953 |

**观察**：≤4KB 几乎是固定延迟 ~19–20 µs（launch/同步开销主导）。从 512KB 开始带宽接近饱和（~11–13 GB/s algBW）。var_us 反映 iteration 间抖动，大多在 25–55 µs 范围。

### 组 2：> 2MB ～ 256MB（大消息，带宽受限区）

| Size | avg_us | var_us | algBW (GB/s) | busBW (GB/s) |
|---------:|---------:|---------:|-------------:|-------------:|
| 4.00 MB   |   312.06 |    48.12 |       13.441 |       20.161 |
| 8.00 MB   |   505.65 |   108.52 |       16.590 |       24.885 |
| 16.00 MB  |   968.96 |   102.27 |       17.315 |       25.972 |
| 32.00 MB  |  1875.21 |     6.04 |       17.894 |       26.841 |
| 64.00 MB  |  3675.42 |    11.09 |       18.259 |       27.388 |
| 128.00 MB |  7265.78 |    16.51 |       18.473 |       27.709 |
| 256.00 MB | 14449.51 |    73.52 |       18.577 |       27.866 |

**观察**：8MB 以上进入高带宽区，algBW 从 ~16.6 GB/s 逐步爬升到 ~18.58 GB/s，busBW 稳定在 ~24.9–27.9 GB/s。≥32MB 时抖动极小（var_us < 100 µs），性能非常稳定。

> 列含义：
> - `avg_us` = average(per-rank min iteration time)，即每个 rank 取其所有 iteration 中的最小值，再跨 rank 取平均（µs）
> - `var_us` = average(per-rank (max−min) iteration time)，即每个 rank 的 iteration 中最大值减最小值，再跨 rank 取平均（µs），反映抖动
> - `algBW`  = bytes / avg_us
> - `busBW`  = algBW × 2(n−1)/n（allreduce bus 因子，n=4）

---

## 2. unitrace `--device-timing` Kernel 级 GPU 执行时间

由于 oneCCL `allreduce_ll_ring` 单次操作内部会提交数百个 L0 GPU kernel，对完整 benchmark 直接 `unitrace --device-timing` 会因逐 kernel timestamp fence 而极慢。因此采用：**独立 prefill kernel 用 unitrace 测量** + **程序内 SYCL event 测量 CCL**。

### Prefill (GEMM-like) kernel 独立测量

unitrace 报告的 kernel 名：
`main::{lambda(handler&)#N}::operator()...{lambda(id<1>)#1}[SIMD32 {...} {1024;1;1}]`

| 配置 (prefill_n × reps) | unitrace device time | SYCL event span |
|------------------------:|---------------------:|----------------:|
| 64M × 200 | ~12.6–19.1 ms | 13.4 ms |
| 1M  × 200 | ~234–237 µs   | 0.97 ms |
| 64M × 10  | ~2.0–2.4 ms   | 3.25 ms |

unitrace 设备时间与程序内 SYCL event 测量一致 —— 二者同源于 GPU timestamp counter。

---

## 复现命令

```bash
source /root/hanchao/2026.0/intel/oneapi/setvars.sh --force
cd /root/hanchao/symm/bench_native
export NEOReadDebugKeys=1
export OverrideL1CachePolicyInSurfaceStateAndStateless=2
export NEO_CACHE_PERSISTENT=0

# 组 1：<= 2MB
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_prefill --op ar --min 7 --max 20 --warmup 20 --loop 50 \
  --prefill-n 1048576 --prefill-reps 50

# 组 2：> 2MB .. 256MB
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_prefill --op ar --min 21 --max 27 --warmup 15 --loop 30 \
  --prefill-n 1048576 --prefill-reps 50

# Prefill kernel 的 unitrace device-timing
UNITRACE=/root/hanchao/applications.analyzers.profilingtoolsinterfaces.sdk/tools/unitrace/build/unitrace
ZE_AFFINITY_MASK=0 $UNITRACE --device-timing -v ./prefill_only_trace 67108864 200

```

# oneCCL Allreduce 性能与 GEMM-Prefill 重叠分析

测试平台：BMG ×4 (`ZE_AFFINITY_MASK=0,1,2,3`)，oneCCL native (oneAPI 2025.3)
数据类型：bfloat16，world_size = 4，transport = MPI
Benchmark：`symm/bench_native/bench_ccl_prefill.cpp`（fire-and-forget，SYCL event profiling）

---

## 1. Allreduce 性能 Summary

### 组 1：≤ 2MB（小消息，延迟受限区）

| Size | min_us | span_us | algBW (GB/s) | busBW (GB/s) |
|---------:|--------:|--------:|-------------:|-------------:|
| 256 B    |    8.63 |   30.74 |        0.030 |        0.044 |
| 512 B    |    8.63 |   22.11 |        0.059 |        0.089 |
| 1.00 KB  |    8.84 |   15.52 |        0.116 |        0.174 |
| 2.00 KB  |    8.53 |   17.71 |        0.240 |        0.360 |
| 4.00 KB  |    8.11 |   15.65 |        0.505 |        0.757 |
| 8.00 KB  |   13.62 |   18.86 |        0.601 |        0.902 |
| 16.00 KB |   12.79 |   16.26 |        1.281 |        1.921 |
| 32.00 KB |   13.52 |   17.11 |        2.424 |        3.636 |
| 64.00 KB |   14.46 |   17.43 |        4.533 |        6.800 |
| 128.00 KB|   15.29 |   18.06 |        8.574 |       12.860 |
| 256.00 KB|   21.63 |   27.26 |       12.118 |       18.178 |
| 512.00 KB|   40.35 |   46.31 |       12.993 |       19.489 |
| 1.00 MB  |   78.21 |   83.15 |       13.408 |       20.111 |
| 2.00 MB  |  153.82 |  161.70 |       13.634 |       20.451 |

**观察**：≤16KB 几乎是固定延迟 ~8–14 µs（launch/同步开销主导）。从 256KB 开始带宽接近饱和 (~12–13 GB/s algBW)。

### 组 2：> 2MB ～ 256MB（大消息，带宽受限区）

| Size | min_us | span_us | algBW (GB/s) | busBW (GB/s) |
|---------:|---------:|---------:|-------------:|-------------:|
| 4.00 MB   |   304.72 |   312.53 |       13.764 |       20.647 |
| 8.00 MB   |   605.28 |   618.23 |       13.859 |       20.789 |
| 16.00 MB  |  1206.30 |  1211.60 |       13.908 |       20.862 |
| 32.00 MB  |  2407.29 |  2415.09 |       13.939 |       20.908 |
| 64.00 MB  |  4809.79 |  4819.69 |       13.953 |       20.929 |
| 128.00 MB |  9613.86 |  9627.22 |       13.961 |       20.941 |
| 256.00 MB | 19219.82 | 19236.66 |       13.967 |       20.950 |

**观察**：带宽完全饱和，algBW 稳定在 ~13.96 GB/s，busBW ~20.95 GB/s。延迟随 size 线性增长，对每翻倍 size 时间近似翻倍。

> 列含义：
> - `min_us` = 各 rank 各 iteration 中最小的 SYCL event 时间（µs）
> - `span_us` = (GPU total span)/loop，pipeline 吞吐延迟（µs）
> - `algBW`  = bytes / min_us
> - `busBW`  = algBW × 2(n−1)/n（allreduce bus 因子，n=4）

---

## 2. unitrace `--device-timing` Kernel 级 GPU 执行时间

由于 oneCCL `allreduce_ll_ring` 单次操作内部会提交数百个 L0 GPU kernel，对完整 benchmark 直接 `unitrace --device-timing` 会因逐 kernel timestamp fence 而极慢。因此采用：**独立 prefill kernel 用 unitrace 测量** + **程序内 SYCL event 测量 CCL**。

### Prefill (GEMM-like) kernel 独立测量

unitrace 报告的 kernel 名：
`main::{lambda(handler&)#N}::operator()...{lambda(id<1>)#1}[SIMD32 {...} {1024;1;1}]`

| 配置 (prefill_n × reps) | unitrace device time | SYCL event span |
|------------------------:|---------------------:|----------------:|
| 64M × 200 | ~13.0–14.0 ms | 13.8 ms |
| 1M  × 200 | ~216–218 µs   | 1.13 ms |
| 64M × 10  | ~2.1–2.5 ms   | 3.4 ms |

unitrace 设备时间与程序内 SYCL event 测量一致 —— 二者同源于 GPU timestamp counter。

---

## 3. GEMM / CCL 重叠对比（loop = 1，allreduce 2MB）

prefill = 64M × 200 reps，warmup = 5：

| 指标 | 值 |
|------|----:|
| prefill_ms (GEMM-like kernel GPU 时间) | 18.0 ms |
| ccl_tot_ms (CCL allreduce GPU span) | 2.4 ms |
| **GEMM / CCL** | **7.63×** |

**结论**：在小迭代数（loop=1）下，prefill GEMM 的 GPU 执行时间约为 CCL allreduce 的 **7.6 倍**，GEMM 完全主导 GPU timeline，CCL 通信可被 GEMM 计算完全掩盖。

---

## 复现命令

```bash
source ~/hanchao/intel/oneapi/setvars.sh --force
cd /root/hanchao/symm/bench_native

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

# GEMM/CCL 重叠 (loop=1)
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi mpirun -n 4 \
  ./bench_ccl_prefill --op ar --min 20 --max 20 --warmup 5 --loop 1 \
  --prefill-n 67108864 --prefill-reps 200
```

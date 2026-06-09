# Allreduce Latency Benchmark Results

- Date: 2026-06-10 (re-tested with prefill_reps=100, prefill ~5-11 ms)
- Platform: 4x BMG GPUs, mpirun -np 4
- Dtype: bfloat16, warmup=20, loop=100, measure(last)=50
- Prefill: 64M bf16 × 100 reps/elem
- Sizes: 8 KB .. 512 MB

---

## Test 1: 2025.3 setvars

Environment: `source /root/hanchao/intel/oneapi/setvars.sh`

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB | 13.58 | 13.17 | 13.99 | 0.82 | 0.905 |
| 16.00 KB | 13.12 | 12.65 | 13.61 | 0.95 | 1.873 |
| 32.00 KB | 10.63 | 10.27 | 10.98 | 0.71 | 4.624 |
| 64.00 KB | 11.28 | 11.05 | 11.51 | 0.47 | 8.716 |
| 128.00 KB | 12.84 | 12.59 | 13.10 | 0.51 | 15.317 |
| 256.00 KB | 22.01 | 21.32 | 22.72 | 1.40 | 17.865 |
| 512.00 KB | 41.45 | 40.40 | 42.48 | 2.08 | 18.975 |
| 1.00 MB | 79.09 | 78.23 | 79.83 | 1.60 | 19.888 |
| 2.00 MB | 154.63 | 153.92 | 155.26 | 1.34 | 20.343 |
| 4.00 MB | 304.37 | 303.46 | 305.16 | 1.70 | 20.670 |
| 8.00 MB | 606.27 | 605.75 | 606.77 | 1.01 | 20.755 |
| 16.00 MB | 1207.65 | 1207.17 | 1208.10 | 0.93 | 20.839 |
| 32.00 MB | 2410.70 | 2410.28 | 2411.10 | 0.82 | 20.878 |
| 64.00 MB | 4816.54 | 4816.17 | 4816.96 | 0.79 | 20.899 |
| 128.00 MB | 9619.31 | 9618.89 | 9619.72 | 0.82 | 20.929 |
| 256.00 MB | 19232.52 | 19232.08 | 19232.96 | 0.89 | 20.936 |
| 512.00 MB | 38457.71 | 38457.13 | 38458.33 | 1.20 | 20.940 |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB | 14.54 | 14.15 | 14.94 | 0.79 | 0.845 |
| 16.00 KB | 14.16 | 13.68 | 14.65 | 0.96 | 1.735 |
| 32.00 KB | 11.68 | 11.34 | 12.01 | 0.67 | 4.209 |
| 64.00 KB | 12.34 | 12.10 | 12.57 | 0.47 | 7.969 |
| 128.00 KB | 13.89 | 13.64 | 14.15 | 0.51 | 14.150 |
| 256.00 KB | 23.02 | 22.33 | 23.74 | 1.41 | 17.085 |
| 512.00 KB | 42.43 | 41.40 | 43.44 | 2.05 | 18.534 |
| 1.00 MB | 80.11 | 79.25 | 80.84 | 1.60 | 19.634 |
| 2.00 MB | 155.81 | 155.08 | 156.45 | 1.37 | 20.190 |
| 4.00 MB | 305.55 | 304.64 | 306.39 | 1.75 | 20.590 |
| 8.00 MB | 607.55 | 607.03 | 608.09 | 1.06 | 20.711 |
| 16.00 MB | 1208.93 | 1208.46 | 1209.38 | 0.92 | 20.817 |
| 32.00 MB | 2411.89 | 2411.45 | 2412.30 | 0.85 | 20.868 |
| 64.00 MB | 4817.82 | 4817.43 | 4818.24 | 0.80 | 20.894 |
| 128.00 MB | 9620.48 | 9620.05 | 9620.89 | 0.84 | 20.927 |
| 256.00 MB | 19233.78 | 19233.34 | 19234.21 | 0.88 | 20.935 |
| 512.00 MB | 38458.84 | 38458.28 | 38459.44 | 1.16 | 20.939 |

---

## Test 2: 2026.0 setvars

Environment: `source /root/hanchao/2026.0/intel/oneapi/setvars.sh`

> **NOTE**: Method A (ccl_event) returns near-zero values for sizes ≥ 8 MB — likely a profiling bug in this oneAPI version. Use Method B (barrier) for reliable data at large sizes.

### Method A: ccl_event

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB | 14.97 | 14.18 | 15.76 | 1.58 | 0.821 |
| 16.00 KB | 14.63 | 14.16 | 15.11 | 0.95 | 1.679 |
| 32.00 KB | 15.13 | 14.68 | 15.59 | 0.91 | 3.248 |
| 64.00 KB | 16.80 | 16.56 | 17.07 | 0.50 | 5.850 |
| 128.00 KB | 17.11 | 16.92 | 17.30 | 0.38 | 11.490 |
| 256.00 KB | 24.40 | 23.66 | 25.15 | 1.49 | 16.118 |
| 512.00 KB | 43.91 | 43.08 | 44.67 | 1.58 | 17.911 |
| 1.00 MB | 81.65 | 80.90 | 82.24 | 1.34 | 19.264 |
| 2.00 MB | 158.32 | 157.86 | 158.78 | 0.91 | 19.869 |
| 4.00 MB | 311.65 | 311.17 | 312.19 | 1.02 | 20.188 |
| 8.00 MB | 0.12 | 0.10 | 0.15 | 0.04 | ⚠️ INVALID |
| 16.00 MB | 0.11 | 0.10 | 0.13 | 0.03 | ⚠️ INVALID |
| 32.00 MB | 0.12 | 0.10 | 0.16 | 0.05 | ⚠️ INVALID |
| 64.00 MB | 0.12 | 0.10 | 0.14 | 0.04 | ⚠️ INVALID |
| 128.00 MB | 0.11 | 0.10 | 0.13 | 0.03 | ⚠️ INVALID |
| 256.00 MB | 0.11 | 0.10 | 0.14 | 0.03 | ⚠️ INVALID |
| 512.00 MB | 0.11 | 0.10 | 0.14 | 0.04 | ⚠️ INVALID |

### Method B: barrier

| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) |
|------|--------|--------|--------|--------|-------------|
| 8.00 KB | 15.87 | 15.11 | 16.65 | 1.54 | 0.774 |
| 16.00 KB | 15.55 | 15.10 | 16.03 | 0.93 | 1.580 |
| 32.00 KB | 16.05 | 15.60 | 16.50 | 0.90 | 3.063 |
| 64.00 KB | 17.70 | 17.47 | 17.97 | 0.50 | 5.553 |
| 128.00 KB | 18.00 | 17.80 | 18.20 | 0.40 | 10.924 |
| 256.00 KB | 25.31 | 24.58 | 26.05 | 1.47 | 15.538 |
| 512.00 KB | 44.82 | 44.00 | 45.56 | 1.56 | 17.546 |
| 1.00 MB | 82.54 | 81.79 | 83.13 | 1.34 | 19.055 |
| 2.00 MB | 159.30 | 158.85 | 159.75 | 0.90 | 19.747 |
| 4.00 MB | 312.73 | 312.22 | 313.28 | 1.06 | 20.118 |
| 8.00 MB | 496.02 | 495.53 | 496.52 | 0.99 | 25.368 |
| 16.00 MB | 976.76 | 976.17 | 977.38 | 1.21 | 25.765 |
| 32.00 MB | 1881.55 | 1880.85 | 1882.22 | 1.37 | 26.750 |
| 64.00 MB | 3684.45 | 3683.64 | 3685.26 | 1.62 | 27.321 |
| 128.00 MB | 7276.60 | 7275.38 | 7277.78 | 2.40 | 27.668 |
| 256.00 MB | 14461.28 | 14459.70 | 14462.91 | 3.21 | 27.844 |
| 512.00 MB | 28889.31 | 28887.06 | 28891.51 | 4.46 | 27.876 |

---

## Comparison: min_us (Method B: barrier)

| Size | Test1 min_us | Test2 min_us | Diff (us) | Speedup |
|------|-------------|-------------|-----------|---------|
| 8.00 KB | 14.15 | 15.11 | +0.96 | 0.94x |
| 16.00 KB | 13.68 | 15.10 | +1.42 | 0.91x |
| 32.00 KB | 11.34 | 15.60 | +4.26 | 0.73x |
| 64.00 KB | 12.10 | 17.47 | +5.37 | 0.69x |
| 128.00 KB | 13.64 | 17.80 | +4.16 | 0.77x |
| 256.00 KB | 22.33 | 24.58 | +2.25 | 0.91x |
| 512.00 KB | 41.40 | 44.00 | +2.60 | 0.94x |
| 1.00 MB | 79.25 | 81.79 | +2.54 | 0.97x |
| 2.00 MB | 155.08 | 158.85 | +3.77 | 0.98x |
| 4.00 MB | 304.64 | 312.22 | +7.58 | 0.98x |
| 8.00 MB | 607.03 | 495.53 | -111.50 | 1.23x |
| 16.00 MB | 1208.46 | 976.17 | -232.29 | 1.24x |
| 32.00 MB | 2411.45 | 1880.85 | -530.60 | 1.28x |
| 64.00 MB | 4817.43 | 3683.64 | -1133.79 | 1.31x |
| 128.00 MB | 9620.05 | 7275.38 | -2344.67 | 1.32x |
| 256.00 MB | 19233.34 | 14459.70 | -4773.64 | 1.33x |
| 512.00 MB | 38458.28 | 28887.06 | -9571.22 | 1.33x |

> Speedup = Test1_min / Test2_min. Values > 1.0x mean 2026.0 is faster.
> 
> **Summary**: 2026.0 oneapi is ~30% faster for large messages (≥8 MB) but ~7-31% slower for small messages (≤128 KB).

---

## Benchmark 构建逻辑 (`bench_ccl_allreduce_latency.cpp`)

### 整体架构

纯 C++/SYCL/oneCCL 实现，不依赖 PyTorch。每个 MPI rank 绑定一个 GPU，使用 in-order + profiling 队列。

### 测量策略（每个 message size）

```
1. Warmup: 同步执行 N 次 allreduce（每次 ccl_ev.wait()）
2. Prefill kernel: 提交一个大 parallel_for（64M bf16 × 100 reps ≈ 5-10ms）
   → 目的: 在 host 派发 timing loop 期间保持 GPU 繁忙，避免空闲 drain
3. Fire-and-forget loop（无 per-iter 同步）:
   for i in [0, loop):
       pre_evs[i]  = q.single_task(nop)     ← 前哨 barrier kernel
       ccl_evs[i]  = ccl::allreduce(...)     ← 存 event，不 wait
       post_evs[i] = q.single_task(nop)      ← 后哨 barrier kernel
4. q.wait() — 单次 GPU 全局同步
5. 丢弃前 (loop - 50) 次迭代，仅统计最后 50 次
6. 双路计时:
   Method A (ccl_event):  ccl_ev.command_end − ccl_ev.command_start
   Method B (barrier):    post_barrier.command_start − pre_barrier.command_end
7. MPI_Gather 各 rank 数据到 rank 0
8. 输出: per-rank per-iter 明细 + 汇总表 (avg/min/max/var_us, busBW)
```

### 关键设计点

| 要素 | 说明 |
|------|------|
| Prefill kernel | 占住 GPU 执行单元，让后续 allreduce 在 queue 中排队而非立即执行完毕 |
| 双路计时 | Method A 依赖 CCL event profiling（2026.0 大 size 有 bug）；Method B 用夹逼法兜底 |
| 丢弃前段 | loop=100 时丢前 50 次，排除 prefill overlap 和 CCL 内部 lazy-init 干扰 |
| In-order queue | 保证 pre → allreduce → post 严格顺序，barrier 计时有效 |
| 带宽计算 | algBW = bytes/time；busBW = algBW × 2(n-1)/n（allreduce 修正因子） |

### CLI 参数

```
--min  N          log2 最小 numel (default 12 → 4K elems = 8 KB)
--max  N          log2 最大 numel (default 28 → 256M elems = 512 MB)
--step N          log2 步长 (default 1)
--warmup N        warmup 迭代数 (default 20)
--loop   N        计时迭代数 (default 100)
--prefill-n N     prefill 元素数 (default 64M)
--prefill-reps N  每元素计算轮数 (default 100)
```

### 编译与运行

```bash
# Build
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -O2 -std=c++17 \
     -I${CCL_ROOT}/include -I${I_MPI_ROOT}/include \
     -L${CCL_ROOT}/lib -L${I_MPI_ROOT}/lib \
     -lccl -lmpi \
     bench_ccl_allreduce_latency.cpp -o bench_ccl_allreduce_latency

# Run (4 GPUs)
ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi \
mpirun -n 4 ./bench_ccl_allreduce_latency

# 自定义 size range
mpirun -n 4 ./bench_ccl_allreduce_latency --min 4 --max 14 --loop 200
```

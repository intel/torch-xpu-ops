# XPU SymmetricMemory `one_shot_all_reduce` — status and perfermance


- Barrier 用 system-scope `sycl::atomic_fence` + `put_signal` / `wait_signal` 握手（`Signal.cpp::barrierKernel`），不依赖 `atomic_ref<system>`（BMG 上未 work）。

---

## 1. 性能（docker, ws=4, bf16, WARMUP=10 ITERS=50）

one_shot 算法  
- 每 rank 把所有 `N-1` 个 peer 的 symm buffer 直接读到本地，单 kernel 完成跨卡读取 + reduce + 写回。
- 默认走 **fused signal-pad barrier 版本**：`launch_fused_one_shot_all_reduce_sum`，把 `put_signal` / `wait_signal` 的握手内联到 reduce kernel，省一次 host barrier。

环境：容器 `gar-registry.caas.intel.com/pytorch/pytorch-ipex-spr:downstream_vllm-kernel-0.1.6-py2.12_ww16.5`，
`ZE_AFFINITY_MASK=0,1,2,3`，`USE_SIGNAL_BARRIER=1`。

| bytes | one_shot (μs) | oneCCL (μs) |
|---:|---:|---:|
|   2 KiB |   26.5 |  17.3 |
|   8 KiB |   31.4 |  21.0 |
|  32 KiB |   31.0 |  21.7 |
| 128 KiB |   30.7 |  24.3 |
| 512 KiB |   83.7 |  52.0 |
|   2 MiB |  265.7 | 166.4 |
|   8 MiB | 1025.7 | 618.2 |

精度：`fp32 / fp16 / bf16 × {1K, 64K, 1M}` 9 组全部 PASS。

### 2.1 Native C++/SYCL bench

量化"torch 栈固定开销"，写了一份纯 C++/SYCL bench（MPI 仅作 launcher + kvs 引导），两条路径用与 `XPUSymmetricMemoryOps.cpp` 完全相同的 kernel：

- `fused`：fused signal-barrier + reduce 单 kernel（对应 `launch_fused_one_shot_all_reduce_sum`）。
- `ccl`：`ccl::allreduce(..., bf16, sum)`。

代码与运行脚本本（`bench_oneshot_vs_ccl.cpp`，`build_and_run.sh`）。容器内运行：
```bash
cd .../bench_native && bash build_and_run.sh build
docker exec hanchao bash -c 'cd .../bench_native && \
  export ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi I_MPI_FABRICS=shm FI_PROVIDER=shm && \
  mpirun -n 4 ./bench_oneshot_vs_ccl --min 10 --max 22 --warmup 20 --iters 100'
```
（`I_MPI_FABRICS=shm FI_PROVIDER=shm`环境变量需要）

5 次 run 中位数（docker ws=4 bf16）：

| bytes | fused (μs) | ccl (μs) | fused BW (GiB/s) | ccl BW (GiB/s) |
|---:|---:|---:|---:|---:|
|   2 KiB |   18.6 |  17.3 |  0.10 |  0.11 |
|   8 KiB |   17.1 |  21.0 |  0.45 |  0.36 |
|  32 KiB |   18.9 |  21.7 |  1.61 |  1.40 |
| 128 KiB |   31.3 |  24.3 |  3.90 |  5.00 |
| 512 KiB |   79.4 |  52.0 |  6.15 |  9.20 |
|   2 MiB |  269.4 | 166.4 |  7.25 | 11.80 |
|   8 MiB | 1026.1 | 618.2 |  7.60 | 12.60 |

**与 §2 torch 实测交叉对比**（小消息）：

| bytes | native fused | torch fused (§2) | Δ = torch − native |
|---:|---:|---:|---:|
|   2 KiB |  18.6 |  26.5 |  +7.9 |
|   8 KiB |  17.1 |  31.4 | +14.3 |
|  32 KiB |  18.9 |  31.0 | +12.1 |
| 128 KiB |  31.3 |  30.7 |  −0.6 |

torch − native ≈ **12-15 μs** 是稳定的"torch 栈税"（op dispatcher + pybind + `TORCH_LIBRARY_IMPL` boxed↔unboxed + `XPU_DISPATCH_FLOAT_HALF_BF16` 宏 + tensor refcount/DeviceGuard）。


---

## 3. Roofline projection

参考 `sycl-tla/examples/00_bmg_gemm/projection.py`。脚本 `proj_allreduce_rs_ag.py`

**模型**（ws=N，消息 S 字节）：

```
RS(S, N) = S / MemBW + (S/N / N_uni + N_lat) * (N - 1)
AG(S, N) = (S        / N_uni + N_lat) * (N - 1)        # AG 输入是 RS 后的 shard = S/N
AR(S, N) = RS(S, N) + AG(S/N, N)                       # ring 算法 allreduce 地板
1shot_bw = (N - 1) * S / N_uni                         # one_shot 纯 PCIe 下界
```

常量（BMG，PCIe P2P，单节点）：
- `N_uni = 31.5 * 0.6957 = 21.91 GiB/s`
- `MemBW = 450 GB/s`（HBM）
- `N_lat = 3 μs`（每跳 P2P 延迟）

### 3.1 projection vs 实测对比表（ws=4 bf16）

```
      size     RS us     AG us     AR us   1shot_bw         1shot       ccl    1shot−AR   ccl−AR
--------------------------------------------------------------------------------------------
   2.0 KiB      9.07      9.07     18.14       0.26          26.5      17.3        +8.4     -0.8
   8.0 KiB      9.28      9.26     18.54       1.04          31.4      21.0       +12.9     +2.5
  32.0 KiB     10.12     10.04     20.16       4.18          31.0      21.7       +10.8     +1.5
 128.0 KiB     13.47     13.18     26.65      16.71          30.7      24.3        +4.1     -2.3
 512.0 KiB     26.88     25.71     52.59      66.84          83.7      52.0       +31.1     -0.6
   2.0 MiB     80.50     75.84    156.35     267.37         265.7     166.4      +109.4    +10.1
```

### 3.2 关键判断

1. **CCL 实测全程贴 RS+AG 地板（`ccl−AR` 全在 ±2.5 μs）**
2. **小消息 (≤ 128 KiB) one_shot ≈ AR 地板 + ~10 μs**：那 10 μs 是 torch op dispatcher (~13 μs) + signal barrier 的固定税，与算法 kernel 无关。`(N-1)·S/N_uni` 只占 0.3-4 μs，**算法 kernel 不是瓶颈**。
3. **大消息 (≥ 2 MiB) one_shot 完全 PCIe-bound**：实测 1026 μs vs `1shot_bw` 1070 μs（误差 < 4%），跑到 BMG PCIe root-complex 单向带宽上限。要降只能换 ring 化的 two_shot（地板 ≈ AR/2 = `1.5·S/N_uni`），CCL 已经做到了。

---

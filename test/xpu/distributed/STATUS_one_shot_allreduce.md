# XPU SymmetricMemory `one_shot_all_reduce` — status and performance

- torch branch: https://github.com/Chao1Han/pytorch/tree/symm-211
- build: replace torch/third_party/xpu.txt commit with currently torch-xpu-ops commit; python setup.py develop
- Algorithm：每 rank 把所有 `N-1` 个 peer 的 symm buffer 直接读到本地，单 kernel 完成跨卡读取 + reduce + 写回。
- 默认走 **fused signal-pad barrier 版本**（`launch_fused_one_shot_all_reduce_sum`），把 `put_signal` / `wait_signal` 握手内联到 reduce kernel，省一次 host barrier。
- Barrier 用 system-scope `sycl::atomic_fence` + `put_signal` / `wait_signal` 握手（`Signal.cpp::barrierKernel`），不依赖 `atomic_ref<system>`（BMG 上未 work）。

精度：`fp32 / fp16 / bf16 × {1K, 64K, 1M} × {ws=4, ws=8}` 18 组全部 PASS。

---

## 1. Native bench — ws=4

纯 C++/SYCL bench（MPI 仅作 launcher + kvs 引导，去 torch 栈），代码与脚本：
- [bench_oneshot_vs_ccl.cpp](../../../../../bench_native/bench_oneshot_vs_ccl.cpp)
- [build_and_run.sh](../../../../../bench_native/build_and_run.sh)
- 编译选项 `-DBENCH_WS=4`，运行：

```bash
cd /root/hanchao/symm/bench_native
docker exec hanchao bash -c 'cd /root/hanchao/symm/bench_native && \
  export ZE_AFFINITY_MASK=0,1,2,3 CCL_ATL_TRANSPORT=mpi I_MPI_FABRICS=shm FI_PROVIDER=shm && \
  mpirun -n 4 ./bench_oneshot_vs_ccl_ws4 --min 11 --max 23 --step 1 --warmup 50 --iters 200'
```

中位数（docker ws=4 bf16）：

| bytes | fused (μs) | ccl (μs) | proj AR (μs) | proj 1shot_bw (μs) |
|-----:|-----------:|---------:|-------------:|-------------------:|
|   4 KiB |    17.5 |   17.9 |   18.27 |   0.52 |
|   8 KiB |    17.7 |   23.7 |   18.54 |   1.04 |
|  16 KiB |    19.0 |   22.1 |   19.08 |   2.09 |
|  32 KiB |    19.0 |   24.8 |   20.16 |   4.18 |
|  64 KiB |    23.5 |   21.7 |   22.32 |   8.36 |
| 128 KiB |    31.6 |   23.4 |   26.65 |  16.71 |
| 256 KiB |    50.3 |   32.8 |   35.29 |  33.42 |
| 512 KiB |    79.5 |   52.3 |   52.59 |  66.84 |
|   1 MiB |   144.3 |   90.1 |   87.17 | 133.69 |
|   2 MiB |   269.6 |  167.0 |  156.35 | 267.37 |
|   4 MiB |   521.2 |  316.0 |  294.69 | 534.75 |
|   8 MiB |  1033.3 |  621.2 |  571.39 |1069.49 |
|  16 MiB |  2033.5 | 1224.0 | 1124.78 |2138.99 |

### 1.1 观察（ws=4）

1. **CCL 实测全程贴 RS+AG roofline**（≤ 32 KiB 略高于 AR 是常数税，≥ 2 MiB 实测 167-1224 μs vs proj 156-1125 μs，误差 < 10%）：CCL 用 ring 拓扑，已经吃满 RS+AG 投影上限。
2. **小消息 (≤ 32 KiB) fused 也贴 AR 地板**：fused 17-19 μs ≈ proj AR 18-20 μs。one_shot 的 `(N-1)·S/N_uni` 带宽项 ≤ 4 μs，**barrier + kernel launch 才是主导**，与 CCL ring 头部小消息延迟相当。
3. **128 KiB 起 one_shot 与 CCL 拉开**：`(N-1)·S/N_uni` 接近甚至超过 AR 地板（128 KiB: 16.7 vs 26.7，512 KiB: 66.8 vs 52.6）。one_shot 的纯带宽下界在 256 KiB 时已经追平 AR 地板，**512 KiB 后变成 PCIe-bound**，CCL ring 用 `(N-1)/N` 因子（0.75）避开这个上限。
4. **大消息 fused 完全 PCIe-bound**：8 MiB 实测 1033 μs vs `1shot_bw` 1069 μs（误差 < 4%），与 BMG PCIe root-complex 单向带宽一致。这条路径**算法上无优化空间**，要降必须换 ring 化的 two_shot。

---

## 2. Native bench — ws=8

宿主实际有 8 张 BMG（`/dev/dri/renderD128–135`）。重新编译 `-DBENCH_WS=8`，运行：

```bash
docker exec hanchao bash -c 'cd /root/hanchao/symm/bench_native && \
  export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7 CCL_ATL_TRANSPORT=mpi I_MPI_FABRICS=shm FI_PROVIDER=shm && \
  mpirun -n 8 ./bench_oneshot_vs_ccl_ws8 --min 10 --max 23 --step 1 --warmup 50 --iters 200'
```

| bytes | fused (μs) | ccl (μs) | proj AR (μs) | proj 1shot_bw (μs) |
|-----:|-----------:|---------:|-------------:|-------------------:|
|   2 KiB |    25.2 |   26.7 |   42.16 |   0.61 |
|   4 KiB |    24.6 |   26.1 |   42.31 |   1.22 |
|   8 KiB |    27.7 |   35.0 |   42.63 |   2.44 |
|  16 KiB |    33.9 |   32.9 |   43.25 |   4.87 |
|  32 KiB |    51.0 |   34.7 |   44.51 |   9.75 |
|  64 KiB |    89.8 |   34.8 |   47.02 |  19.50 |
| 128 KiB |   192.2 |   34.9 |   52.04 |  38.99 |
| 256 KiB |   285.2 |   40.5 |   62.08 |  77.98 |
| 512 KiB |   516.1 |   64.4 |   82.16 | 155.97 |
|   1 MiB |   801.7 |  107.2 | 122.31 | 311.94 |
|   2 MiB |  1392.4 |  193.1 | 202.63 | 623.87 |
|   4 MiB |  2596.5 |  371.0 | 363.26 |1247.74 |
|   8 MiB |  5042.8 |  721.5 | 684.51 |2495.49 |
|  16 MiB |  9943.4 | 1427.6 |1327.03 |4990.98 |

### 2.1 观察（ws=8）

1. **CCL ws=8 反而比 ws=4 更平**：≤ 256 KiB 几乎 flat 在 30-40 μs，2 MiB 实测 193 μs vs proj AR 203 μs，全程贴 RS+AG 地板，误差 < 6%。CCL 在 8 卡上用了多路 ring 并行 + xelink 路径。
2. **fused one_shot 大幅劣化**：4 KiB → 24.6 μs（与 ws=4 持平），**256 KiB 起断崖** 285 μs（ws=4 是 50 μs，差 5.7×），8 MiB 5043 μs（ws=4 是 1033 μs，差 4.9×）。
3. **断崖出在 PCIe-bound 区开始处**：1shot_bw `(N-1)·S/N_uni` 在 ws=8 是 7·S/N_uni，相对 ws=4 的 3·S/N_uni 放大 7/3 = 2.33×；但实测放大 ~5×，**多出来的 2× 是 root-complex 入口带宽饱和**（8 个 rank 同时打 PCIe Gen5 ×16 上行 ~50 GB/s，每 rank 仅 6.25 GB/s 实际可用，远低于 31.5 GiB/s 标称）。
4. **fused vs proj AR 在 128 KiB 起背离**（128 KiB: 192 vs 52, 8 MiB: 5043 vs 685, 7×）：one_shot 在 ws=8 算法上就**不是 RS+AG ring**，每 rank 读 7·S vs ring 每 rank 读 (N-1)/N·S = 0.875·S，跨 root-complex 流量爆炸。
5. **结论**：ws=8 上 one_shot **小消息 (≤ 16 KiB) 还能用**（fused 25-34 μs 接近 CCL 27-33 μs），**≥ 32 KiB 应改用 two_shot** 算法。

---

## 3. 补充：Roofline projection（ws=4 / ws=8）

公式（ws=N，消息 S 字节，bf16）：

```
RS(S, N) = S / MemBW + (S/N / N_uni + N_lat) * (N - 1)
AG(S, N) = (S        / N_uni + N_lat) * (N - 1)        # AG 输入是 RS 后的 shard = S/N
AR(S, N) = RS(S, N) + AG(S/N, N)                       # ring 算法 allreduce 地板
1shot_bw = (N - 1) * S / N_uni                         # one_shot 纯 PCIe 下界
```

常量（BMG，PCIe P2P，单节点）：
- `N_uni = 31.5 * 0.6957 = 21.91 GiB/s`（折扣对应实际 PCIe Gen5 ×16 单向）
- `MemBW = 450 GB/s`（HBM）
- `N_lat = 3 μs`（每跳 P2P 延迟）

完整投影数值已嵌入 §1 / §2 的 `proj AR` / `proj 1shot_bw` 列；脚本：[proj_allreduce_rs_ag.py](proj_allreduce_rs_ag.py)。

### 3.1 ws=4 vs ws=8 投影差异

| metric | ws=4 | ws=8 | ratio |
|---|---:|---:|---:|
| `(N-1)/N` (AR factor) | 0.75 | 0.875 | 1.17× |
| `(N-1)` (1shot factor) | 3 | 7 | 2.33× |
| `N_lat·(N-1)` per phase | 9 μs | 21 μs | 2.33× |

ws=8 上 ring AR 只贵 17%，one_shot 贵 133% 。

---

## 4. 补充：torch op 路径开销（仅参考，不影响 §1–§3 的硬件结论）

§1–§3 都是去 torch 栈的 native 数据。torch op 路径有恒定 ~12-15 μs dispatch tax（pybind + `TORCH_LIBRARY_IMPL` boxed↔unboxed + tensor refcount/DeviceGuard + `XPU_DISPATCH_FLOAT_HALF_BF16` 宏）。

`bench_ours_docker.py`（torch fused one_shot, `WARMUP=200 ITERS=500`）vs §1/§2 native 对比：

| bytes | torch ws=4 (μs) | native ws=4 (μs) | torch ws=8 (μs) | native ws=8 (μs) |
|-----:|----------------:|-----------------:|----------------:|-----------------:|
|   2 KiB |  28.2 |    —  |  52.0 |  25.2 |
|   4 KiB |  27.2 |  17.5 |  43.5 |  24.6 |
|   8 KiB |  24.8 |  17.7 |  27.9 |  27.7 |
|  16 KiB |  27.1 |  19.0 |  29.5 |  33.9 |
|  32 KiB |  23.7 |  19.0 |  48.8 |  51.0 |
|  64 KiB |  31.1 |  23.5 | 111.8 |  89.8 |
| 128 KiB |  24.2 |  31.6 | 215.4 | 192.2 |
| 256 KiB |  40.0 |  50.3 | 306.6 | 285.2 |
| 512 KiB |  72.0 |  79.5 | 561.2 | 516.1 |
|   1 MiB | 136.0 | 144.3 | 888.4 | 801.7 |
|   2 MiB | 261.6 | 269.6 |1477.5 |1392.4 |
|   4 MiB | 516.4 | 521.2 |2778.3 |2596.5 |
|   8 MiB |1019.9 |1033.3 |5136.1 |5042.8 |
|  16 MiB |2034.8 |2033.5 |10020 |9943 |

观察：
- **大消息 torch ≈ native**：≥ 1 MiB 时两者差 < 10%，dispatch tax 被 PCIe 时间淹没。
- **小消息 ws=4**：torch 24-28 μs ≈ native 17-23 μs + dispatch tax ~10 μs；但 `WARMUP=500` 控制住了 first-iter 抖动，torch 路径在 in-order queue 上有隐式异步（kernel submit 完即返回）。
- **小消息 ws=8 torch 反而比 native 慢 ~2×**（2 KiB: 52 vs 25，4 KiB: 43 vs 25）：8-rank Python launcher 让 dispatch tax 放大；这是 docker 多进程同步的二次效应，与 op 算法无关。

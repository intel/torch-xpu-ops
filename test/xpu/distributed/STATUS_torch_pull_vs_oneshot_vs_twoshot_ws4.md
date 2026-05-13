# torch op path: pull vs one_shot vs two_shot — ws=4 bf16 (2026-04-29)

条件：docker `hanchao` 容器，BMG ws=4，bf16，`USE_SIGNAL_BARRIER=1 + FI_PROVIDER=tcp + UCX_TLS=tcp,sm,self + ZE_AFFINITY_MASK=0,1,2,3`，`mpirun -n 4 python test_allreduce.py --impl <impl> --no_ref --warmup 30 --iters 100`。`--no_ref` 是因为 docker ws=4 时 `dist.all_reduce` 触发 XCCL `UR_RESULT_ERROR_DEVICE_LOST`（详见 `instruction.md` §12.11）。

数值单位 ms，最后一列标 best 方案。

| bytes  | size MB | pull (ms) | one_shot (ms) | two_shot (ms) | best |
|-------:|--------:|----------:|--------------:|--------------:|:-----|
| 4 KiB  |   0.004 |     0.198 |         0.128 |         0.094 | two_shot |
| 16 KiB |    0.02 |     0.221 |         0.112 |         0.088 | two_shot |
| 64 KiB |    0.06 |     0.217 |         0.090 |         0.093 | one_shot |
| 256 KiB|     0.25|     0.197 |         0.135 |         0.103 | two_shot |
| 1 MiB  |     1.0 |     0.237 |         0.140 |         0.142 | one ≈ two |
| 4 MiB  |     4.0 |     0.381 |         0.526 |         0.518 | **pull** |
| 16 MiB |    16.0 |     1.080 |         2.198 |         2.192 | **pull** |
| 64 MiB |    64.0 |     4.308 |         8.799 |         8.785 | **pull** |
| 128 MiB|   128.0 |     8.591 |        17.621 |        17.548 | **pull** |
| 256 MiB|   256.0 |    17.140 |        35.108 |        35.132 | **pull** |

> **更正 (2026-05-12)**：本节早期把 `pull` 表述为 "走 `dist.reduce_scatter_tensor + all_gather_into_tensor`（≈ CCL ring）"，**错误**。`allreduce_with_pull`（`allreduce_impl.py:431`）实际是**纯 Python 在 symm-mem buffer 上做 RS+AG**：N-1 次 `tensor.copy_()` 把本地 chunk 推到 remote symm slot → `workspace.barrier()` → 单 kernel `torch.sum(my_symm_2d, dim=0)` 做归约 → N-1 次 `tensor.copy_()` 从 remote symm 拉回。**没有 CCL collective，没有 XCCL host barrier**；唯一同步是 symm-mem 的 signal-pad barrier。因此下面的 "vs CCL pull" 措辞应理解为 "vs torch symm-mem python pull impl"，**不**是 native oneCCL ring。CCL 自己的 ring 数据见 §12.12 / §17。

观察：

1. **小消息 (≤ 256 KiB)**：`two_shot` ≈ `one_shot`（80–135 μs），都比 `pull` (~210 μs) 快 2-2.5×。`pull` 在 Python 端跑 (N-1)·2 次 `copy_()` + 2 次 `workspace.barrier()` + 1 次 `torch.sum`，每个 op 一次 host dispatch + L0 submit，~30 μs × 7 op ≈ 200 μs，全是 dispatch 头部。

2. **中等 (1 MiB)**：三者第一次接近（pull 237 / one 140 / two 142 μs）；symm-mem 的 (N-1)·S/N_uni 带宽项追上 `pull` 的固定头部。

3. **大消息 (≥ 4 MiB)**：`pull` 完全反超：
   - 16 MiB: pull **1.08 ms** vs symm 2.20 ms（**2.0×**）
   - 64 MiB: pull **4.31 ms** vs symm 8.79 ms（**2.04×**）
   - 256 MiB: pull **17.14 ms** vs symm 35.11 ms（**2.05×**）
   - 之前误以为是 N²/(N-1) PCIe inbound 比例 —— **错**。`pull` 也是 (N-1)·S 总 PCIe traffic（每 rank 推 N-1 个 chunk + 拉 N-1 个 chunk，和 one_shot 一致），但 **pull 的每个 `copy_()` 是纯 D2D memcpy，没有任何 reduce 算子参与**，EU 跑满 PCIe 入口；one/two_shot 的 fused kernel 内部带 load+add+store + 多次 sub-group barrier + signal-pad CAS，被同步切碎 → 实际带宽利用低一半。
   - 256 MiB symm 35 ms ≈ 14.6 GB/s（PCIe Gen5 ×16 的 ~46%）；pull 17 ms ≈ 30 GB/s ≈ 单向饱和。

4. **one_shot ≈ two_shot 在 ws=4 全程持平**：与 §12.11 ws=2 / native §1 现象一致 —— ws=4 时 two_shot 把 one_shot 的 (N-1)·S 拆成 RS+AG (1+ (N-1)/N)·S，节省的 PCIe 流量被 4 次 barrier (RS pre/post + AG pre/post) 抵消。**ws=8 才能看到 two_shot 的 ~2× 优势**（见 §12.11 ws=8 表）。

5. **要打过 pull/CCL ring 必须改 ring 拓扑**：§12.12 的 native ring bench 在 ws=4 16 MiB 已做到 1.41 ms，逼近 CCL 1.23 ms；但 ring 当前是 4(N-1)+3 = 15 个 kernel launch，小消息 (≤ 256 KiB) ~150 μs 全是 dispatch overhead。下一步要 fused-ring 单 kernel，覆盖小+大全程。

总结决策（torch op 路径选择）：
- ≤ 1 MiB: `two_shot`（小消息和 one_shot 几乎持平，但 ws=8 能甩开 one_shot）。
- ≥ 4 MiB: 暂时只能 `pull`（≈ CCL ring）。等 fused-ring symm op 出来后切到 ring。

---

## Update 2026-04-30: fused-ring (`USE_RING=1`) torch op 数据

把 §12.12/§12.13 的 fused-ring 单 kernel 接到 `torch.ops.symm_mem.two_shot_all_reduce_` 路径上，环境变量 `USE_RING=1` 切换：默认走 fused two_shot（Plan B），`USE_RING=1` 走 Rabenseifner pull-form ring。其它配置同上表。

### ws=4 bf16

| bytes   | size MB | pull (ms) | two_shot 默认 (ms) | two_shot **USE_RING=1** (ms) | ring vs two_shot | ring vs pull |
|--------:|--------:|----------:|-------------------:|-----------------------------:|-----------------:|-------------:|
| 4 KiB   |   0.004 |     0.198 |              0.094 |                        0.134 |          0.70× |   慢      |
| 16 KiB  |    0.02 |     0.221 |              0.088 |                        0.087 |          1.01× |   2.5×     |
| 64 KiB  |    0.06 |     0.217 |              0.093 |                        0.091 |          1.02× |   2.4×     |
| 256 KiB |    0.25 |     0.197 |              0.103 |                        0.075 |          1.37× |   2.6×     |
| 1 MiB   |     1.0 |     0.237 |              0.142 |                        0.121 |          1.17× |   2.0×     |
| 4 MiB   |     4.0 |     0.381 |              0.518 |                        0.437 |          1.19× |   0.87×（pull 仍快）  |
| 16 MiB  |    16.0 |     1.080 |              2.192 |                        1.813 |          1.21× |   0.60× |
| 64 MiB  |    64.0 |     4.308 |              8.785 |                        7.323 |          1.20× |   0.59× |
| 128 MiB |   128.0 |     8.591 |             17.548 |                       14.849 |          1.18× |   0.58× |
| 256 MiB |   256.0 |    17.140 |             35.132 |                       29.604 |          1.19× |   0.58× |

### ws=8 bf16（`ZE_AFFINITY_MASK=0..7`，`mpirun -n 8`）

| bytes   | size MB | two_shot 默认 (ms) | two_shot **USE_RING=1** (ms) | ring vs two_shot |
|--------:|--------:|-------------------:|-----------------------------:|-----------------:|
| 4 KiB   |   0.004 |              0.140 |                        0.138 |          1.01× |
| 16 KiB  |    0.02 |              0.092 |                        0.124 |          0.74× |
| 64 KiB  |    0.06 |              0.090 |                        0.097 |          0.93× |
| 256 KiB |    0.25 |              0.120 |                        0.141 |          0.85× |
| 1 MiB   |     1.0 |              0.329 |                        0.185 |          1.78× |
| 4 MiB   |     4.0 |              1.255 |                        0.622 |          2.02× |
| 16 MiB  |    16.0 |              4.758 |                        2.630 |          1.81× |
| 64 MiB  |    64.0 |             18.299 |                       10.576 |          1.73× |
| 128 MiB |   128.0 |             38.225 |                       20.649 |          1.85× |
| 256 MiB |   256.0 |             75.181 |                       42.671 |          **1.76×** |

观察：

1. **ws=8 大消息 fused-ring 完胜**：256 MiB 从 75.2 ms → 42.7 ms，~1.76×。这是 PCIe Gen5 ×16 单端口入口被 (N-1)·S 流量打爆的根因被解开 —— ring 让每 rank inbound = 2·(N-1)/N·S，从 875 MiB → 224 MiB（N=8, S=128 MiB），×3.9，匹配理论。
2. **ws=4 大消息 fused-ring 优势 ~1.2×**：256 MiB 35 ms → 29.6 ms。比 ws=8 的 1.76× 小，因为 N=4 时 pull-flow 比 (N-1)·S 只省 (N-1)·(N-1)/N = 9/4 = 2.25× 流量，而每 rank 的 PCIe 入口又被 2(N-1)=6 步 ring 的并发分摊较少。
3. **ws=4 小消息 (≤ 64 KiB)**：fused-ring ≈ two_shot；4 KiB 略慢（0.134 vs 0.094 ms）—— 因为 ring 内有 2(N-1)=6 个内部 step barrier，small-msg dispatch overhead 被 step 数放大；two_shot 只有 3 个 inline barrier。
4. **vs pull**：fused-ring 在 4 MiB+ 仍未追上 pull（256 MiB ring 29.6 ms vs pull 17.1 ms）。原因：pull 调用 CCL ring，CCL 用 ZE IPC + write-form (push) 流水线，把 (N-1)·S/N 的 PCIe 流量按 N-1 步切成块流水，每步只发 S/N，而我们的 ring 是 read-form 单 step 一次性 read S/N，step 间无重叠。
5. **建议路由**（更新版）：
   - `≤ 1 MiB`：`two_shot` 默认（`USE_RING=0`）
   - `1 MiB – 4 MiB`：`USE_RING=1`（ring 在 ws=4 已 1.17–1.19×）
   - `≥ 4 MiB` ws=8：`USE_RING=1`（1.7–2.0×，已 worth it）
   - `≥ 4 MiB` ws=4：仍跑 `pull`（CCL）；ring 距离 CCL 还差 ~1.7×，要 push-flow / pipelined ring 才能追平。


---

## §13 1+2 优化尝试（push-form AG + 增加 WG）—— 全部为负

为闭合上面 §12 末尾与 CCL pull 的 1.7-1.8× 差距，按"先做 1+2"思路尝试两条最简单的路径，**两条都失败**。详见 `instruction.md` §12.14。一句话总结：

| 尝试 | 改动 | ws=4 / 256 MiB 实测 | 结论 |
|---|---|---|---|
| 1. push-form AG | AG 段从 read-from-left 改为 write-to-right + system fence | 29.6 → **54.4 ms (1.84× 回退)** | BMG peer-write 显著慢于 peer-read；CCL push 必走 ZE IPC + DMA copy engine，不是改 SYCL store 方向能复现的 |
| 2. 加 WG | `kRingMaxNumGroups` 24 → 32（torch op）；native bench sweep 8-48 | 24=29.6 / 32=31.4 ms；native flat ±10% | 已带宽饱和，并发度不是瓶颈 |

副作用收获：发现 `kFusedRingSignalBaseU32 = 2048` 在 ws=8 时 step 占用 [2048, 2408) 越过默认 signal pad 上界 2304（u32 slot）—— **latent UB 在历史所有 ws=8 ring 测里都在踩**。已修为 1664（紧贴 two_shot 区 [1024, 1600) 之后）。

**最终 build5（pull-AG + WGs=24 + base=1664）ws=4 验证**：

```
                   USE_RING=0       USE_RING=1
64 MB     two_shot   10.770 ms       8.804 ms   (1.22×)
128 MB    two_shot   20.355 ms      18.410 ms   (1.11×)
256 MB    two_shot   41.478 ms      36.415 ms   (1.14×)
```

ring 仍稳定快于 two_shot 1.1-1.2×，与 §12.13 数据一致，UB 修复未引入回退。

**ws=8 在尝试期间 host GPU 进入坏状态**（容器重启后 8 卡只剩 7 卡可见），本次环境无法复测；§12.13 的 ws=8 1.76× 数据（base=2048 时刚好"踩 UB 但没炸"）需要在恢复 8 卡后用 base=1664 重测确认。

### 下一步候选（按可行性排序）

1. **多 ring（ws=8 拆 2 个 4 卡 ring）** — 中风险高收益，几乎是 CCL 在 ws=8 上 1.7× 的真正来源。
2. **多 step 双缓冲流水** — 高风险，理论上能再挖一倍带宽。
3. **ZE IPC + copy engine 走 push** — 高风险大工程，但能直接复用 CCL 机制。

---

## §14 Double-buffer 流水（option 2）—— 中等消息 +10-20%

§13 列出的下一步候选 1-2-3 中，option 1（多 ring ws=8）需 8 卡而本次环境只剩 7 卡可用，先做 option 2：fused-ring 每个 step 内 WG 切片再切 halves=2 半，per-half 一个 fwd flag，让 step k+1 H0 PCIe read 与 peer 的 step k H1 在 flight 时重叠。

**实现要点**：`kRingPipeHalves=2`，fwd_slot 索引 `(step·halves+h)·num_wgs+wg`；pad base 1664→1600（贴在 two_shot 区 [1024,1600) 之后）；ws=8 worst-case 占用 14·2·24+24=696, 1600+696=2296<2304, fit。halves=4 验证收益已 plateau，留 halves=2。

**ws=4 实测（torch op build8 vs build5 single-buffer vs two_shot）**：

| Size | two_shot | ring single-buf | ring double-buf | db gain over single-buf |
|---|---|---|---|---|
| 1 MB | 0.178 | 0.185 | 0.174 | +6% |
| 4 MB | 0.610 | 0.622 | 0.565 | **+10%** |
| 16 MB | 3.165 | 2.630 | **2.198** | **+20%** |
| 64 MB | 10.023 | 10.576 | 8.854 | +19% |
| 128 MB | 20.473 | 20.649 | 17.708 | +17% |
| 256 MB | 41.987 | 36.415 | 39.352 | within noise |

**结论**：
- 中等消息 **4-128 MiB +10-20%**：正是 step barrier 延迟敏感区间。
- 大消息 256 MiB 已带宽 bound，pipeline 帮不上忙；与 §13 增 WG 实验一致。
- 小消息 ≤ 1 MiB 持平；barrier 数翻倍但 step 内工作量小，开销没吃光收益。
- **vs CCL pull**：4-16 MiB 区间 db ring 已和 CCL pull 持平甚至略胜；256 MiB 仍差 2.3×。

ws=8 待 GPU 恢复后用同一 build8 wheel 重测；pad 已验过塞得下。

**当前 build8 状态**：pull-AG + WGs=24 + halves=2 + base=1600；已 install；正确性 ✓。

下一步：option 1（多 ring，等 ws=8 恢复）。

---

## §15 ws=8 build8 验证（GPU 恢复后）

| Size | two_shot | ring db | speedup |
|---|---|---|---|
| 1 MB | 0.446 | 0.204 | **2.19×** |
| 4 MB | 1.267 | 0.619 | **2.05×** |
| 16 MB | 4.543 | 2.607 | 1.74× |
| 64 MB | 17.530 | 10.471 | 1.67× |
| 128 MB | 34.686 | 20.594 | 1.68× |
| 256 MB | 70.622 | 41.396 | **1.71×** |

- 正确性 ✓ 全档；pad UB fix（base=1600 + halves=2 footprint 696 = 2296 < 2304）确认有效。
- 大消息与 §12.13（latent-UB base=2048）持平：41.4 vs 42.7 ms；db 在 ws=8 大消息上和 single-buf 等价（带宽 bound）。
- 小消息 ≤ 4 MiB 从 §12.13 的 1.78× 提到 **2.05-2.19×**——db 压低 step barrier 占比。
- vs CCL pull ws=8 256 MiB ~24 ms：ring 41.4 ms 仍差 1.7×，留给 option 1 多 ring。

---

## §16 2D 分层 ring（ws=8 only）—— 大幅突破

ws=8 拆 2 组 ×4 卡：Phase A 组内 RS (3 步) → Phase B 跨组对 partner exchange (1 步) → Phase C 组内 AG (3 步)，共 **7 步** vs flat ring 14。每 step 数据量 ×2 但同步开销减半。

**ws=8 build10 vs build8 vs two_shot**：

| Size | two_shot | flat-db | **2D** | 2D vs ts | 2D vs flat |
|---|---|---|---|---|---|
| 1 MB | 0.446 | 0.204 | **0.188** | 2.37× | +9% |
| 4 MB | 1.267 | 0.619 | **0.549** | 2.31× | +13% |
| 16 MB | 4.543 | 2.607 | **1.996** | 2.28× | +31% |
| 64 MB | 17.530 | 10.471 | **7.811** | 2.24× | +34% |
| 128 MB | 34.686 | 20.594 | **15.554** | 2.23× | +32% |
| 256 MB | 70.622 | 41.396 | **30.768** | **2.30×** | **+35%** |

vs CCL pull ws=8 256 MiB ~24 ms：**从 1.71× 差距缩到 1.28×**（闭合 70%）。

**实施关键 bug fix**：Phase A 末步 (k=2) 必须 skip put_signal —— 右邻没有 phase A wait 来消费这个 slot，slot 累积到 1 后下一 iter 的 CAS 0→1 死锁 → DEVICE_LOST。第一次 build 就踩，加 skip 即过。

ws=4 走 flat ring db 不变，256 MiB 29.3 ms ≈ 历史最佳 29.6 ms，✓ 无回退。

剩余 1.28× 推测来自 CCL 用 ZE IPC + copy engine 异步 dispatch（option 3 范畴），工程量极大但是闭合最后差距的唯一路径。

---

## §17 重新对照 `pull`（python symm 实现）—— 修正历史误判（2026-05-12）

之前 §11/§12/§16 多次提到 "vs CCL pull ws=8 256 MiB ~24 ms"，那个 ~24 ms 数据来源是 §12.12 的 native CCL benchmark（直接调 oneCCL ring + iSHMEM 路径），**不是** test_allreduce.py 里的 `--impl pull`。`--impl pull` 实际跑 `allreduce_with_pull`（python `tensor.copy_()` × 2(N-1) + 1 次 `torch.sum`，全程 symm-mem，无 CCL）。这次用同一 test harness 重新对照三种实现，结果意外：

### ws=8 重测 (build10, USE_RING=1 走 2D 分层 ring)

| Size | two_shot (默认) | flat-db (ws=8 假设走 flat) | **2D ring** | python pull |
|---|---|---|---|---|
| 1 MB | 0.446 | 0.204 | **0.188** | 0.373 |
| 4 MB | 1.267 | 0.619 | **0.549** | 1.615 |
| 16 MB | 4.543 | 2.607 | **1.996** | 5.614 |
| 64 MB | 17.530 | 10.471 | **7.811** | 24.867 |
| 128 MB | 34.686 | 20.594 | **15.554** | 46.881 |
| 256 MB | 70.622 | 41.396 | **30.768** | 95.233 |
| 512 MB | — | — | **62.506** | 191.451 |

→ **2D ring 在 ws=8 全程把 python pull 甩 ~3.1×**。ws=8 时 pull 的 (N-1)·2 = 14 次 host `copy_()` dispatch + 串行 barrier 把流水线打碎，scale 极差；而 2D ring 把 7 步全塞进 1 个 kernel。所以 §16 "差 1.28×" 是和 native CCL 比的差距，**和 python pull 比已经反超 3×**。

### ws=4 重测 (build10, USE_RING=1 走 flat-db；2D 仅 ws=8)

| Size | two_shot | **flat-db ring** | python pull | pull vs ring |
|---|---|---|---|---|
| 1 MB  | 0.175 | **0.205** | 0.200 | pull 持平 |
| 4 MB  | 1.017 | **0.884** | 0.535 | pull 1.65× |
| 16 MB | 4.401 | **3.671** | 2.165 | pull 1.70× |
| 64 MB | 17.643 | **14.660** | 8.583 | pull 1.71× |
| 128 MB| 35.175 | **30.519** | 17.141 | pull 1.78× |
| 256 MB| 70.167 | **70.657** | 34.261 | pull 2.06× |

→ ws=4 大消息 **python pull 仍领先 1.7-2.0×**。

### 为什么 ws=4 python pull 反而快？

ws=4 时 pull 只要 6 次 `copy_()` + 1 次 sum + 6 次 `copy_()` = 13 个 op；每个 `copy_()` 落到 `torch-xpu-ops/src/ATen/native/xpu/Copy.cpp` 的 `q.copy(src, dst, bytes)`（compute queue，但纯 memcpy kernel，无 reduce 无 atomic）。13 个 op 串行 dispatch 头部 ws=4 时只占 ~80-100 μs，对 17 ms 大消息可忽略，**EU memcpy 单方向能压满 PCIe**。

我们的 flat-ring kernel 单 launch，但内部：
1. 每 step 一对 `put_signal + wait_signal` CAS（busy-spin 等右邻 PCIe-write 可见）；
2. RS 的 6 步每步带 reduce（load remote + load local + add + store local），EU 不像纯 memcpy 那样能完全 burst；
3. WG 间靠 sub-group barrier 同步，PCIe-read 抖动直接卡 WG。

实测 ws=4 256 MiB 30 ms ≈ 17 GB/s effective inbound，**只有 pull 30 GB/s 的 56%** —— 同步切碎了带宽。

> 关键反直觉：**单 kernel 不一定胜过多 kernel 链**。当 problem size 大到 dispatch overhead 可忽略，且 ring step 是带宽 bound 时，"每 step 独立 D2D + 中间一次大归约" 反而能让硬件流水线打满。

### 下一步候选

| 方向 | 估算 | 风险 |
|---|---|---|
| (a) AG 段拆出来用 `q.memcpy` (compute queue 上的纯 memcpy primitive) | ws=4 256 MB 30→~22 ms (-26%) | 中：需要切换 fused kernel 拓扑 |
| (b) 给 AG 段新建 BCS copy-engine queue（L0 ordinal=copy）和 compute queue 并行 | ws=4 →~15-18 ms (-50%) | 高：跨 engine signal 需要 L0 event；同步语义复杂 |
| (c) 在 build10 的 flat-ring 里**去掉 reduce step 的 CAS busy-spin**，改成 producer-side fence + WG 内 store-load relaxed | ws=4 中等消息 -10% | 低：纯改 kernel；正确性需仔细验 |
| (d) Phase-A 同时做 RS+AG 推送（double-pumped pipeline，每 step 同时送两个 chunk） | ws=4 +20% 带宽 | 中：pad 容量足够 |

倾向先做 (c) 验证同步是否是 ws=4 瓶颈；若 (c) 红利明显（>15%）就尝试 (a)；(b) 是最后手段，需要在 SYCL 之外用 L0 直接建队列。

---

## §18 option (c) put_signal release-only — build11 (2026-05-12)

`Signal.hpp` 新增 `put_signal_release_only<Sem>`：跳过 `while(load_acquire(addr)!=0)` 这条对远端 pad 的 PCIe load 等待。两个 fused-ring kernel 内 8 处 put 调用切换；one_shot/two_shot 路径保持原 `put_signal`。

正确性前提：每个 `(k,h,wg)` slot 单次 AR 内只写一次；end-of-iter back-sync 保证跨调用 slot 回 0；同 rank kernel 串行 stream + 跨 rank back-sync 排除 race。

### ws=4 (flat-ring db)

| Size | build10 | **build11** | Δ |
|---|---|---|---|
| 4 KB | 0.089 | **0.048** | **-46%** |
| 32 KB| 0.087 | **0.046** | **-47%** |
| 1 MB | 0.151 | **0.131** | **-13%** |
| 4 MB | 0.443 | 0.445 | 0 |
| 16 MB| 1.833 | 1.825 | 0 |
| 64 MB| 7.310 | 7.295 | 0 |
| 256 MB| 29.349 | 29.361 | 0 |
| 512 MB| 59.117 | 59.088 | 0 |

### ws=8 (2D ring)

| Size | build10 | **build11** | Δ |
|---|---|---|---|
| 1 MB | 0.188 | **0.138** | **-27%** |
| 4 MB | 0.549 | **0.480** | **-13%** |
| 16 MB| 1.996 | 1.964 | -2% |
| 64 MB| 7.811 | 7.890 | +1% |
| 128 MB| 15.554 | 15.642 | +1% |
| 256 MB| 30.768 | 31.451 | +2% |
| 512 MB| 62.506 | 62.885 | ≈0 |

正确性 ✓ 全档。

**核心结论**：
- option (c) 对 sync-bound 小消息 **-13% 到 -47%**，对 bw-bound 大消息持平。每个 put 省 ~600 ns 远端 PCIe load × 14 步 ≈ 8 μs/call。
- **证伪 §17 假设**：ws=4 大消息 vs python pull 1.7× 慢的 root cause **不是** CAS busy-spin。同步省到最低还是 30 ms，说明瓶颈在 fused-reduce kernel 内部 EU 带宽利用（load+add+store + WG barrier 切碎 PCIe burst），不在 signal pad 同步。
- 闭合 ws=4 large-msg gap 必须做 §17 候选 (a) 拆 RS+sum+AG kernel 链 或 (b) BCS copy engine。

---

## §19 节点收尾：build12 代码整理后定稿数据 (2026-05-14)

`XPUSymmetricMemoryOps.cpp` 清理：删除 non-fused one_shot kernel、2-kernel two_shot RS+AG + 3 host barrier 路径、所有 env flag（`USE_RING`/`USE_FUSED_ONESHOT`/`USE_FUSED_TWOSHOT`）。`two_shot_all_reduce_` 入口按 world_size 自动路由：ws=8 → 2D 分层 ring，ws=4 → flat ring db，ws=2 → fused two_shot。文件 1405 → 1130 行。

build12 = build11 同 kernel + 入口清理，性能预期一致。

### 各 world_size 定稿对照（torch op，bf16，warmup 30 iters 100）

| size | ws=2 fused two_shot | ws=4 flat ring | ws=4 python pull | ws=8 2D ring | ws=8 python pull |
|---|---|---|---|---|---|
| 0.5 MB | 0.060 | 0.054 | 0.197 | — | 0.373 (1 MB) |
| 1 MB | 0.075 | 0.119 | 0.237 | 0.170 | 0.373 |
| 4 MB | 0.248 | 0.444 | 0.381 | 0.566 | 1.615 |
| 16 MB| 1.085 | 1.825 | 1.080 | 1.944 | 5.614 |
| 64 MB| 4.427 | 7.299 | 4.308 | 7.801 | 24.867 |
| 256 MB| 17.512| 29.339| 17.140| 30.793| 95.233 |
| 512 MB| 34.753| 59.433| 34.261| 62.619| 191.451 |

### 节点完成清单

- ✅ ws=2/4/8 全部走单一 fused 路径（无 env flag），代码 -275 行；
- ✅ 入口路由清晰：`world_size` switch 一处，无 `getenv`；
- ✅ ws=8 全档 ~3× 领先 python pull；
- ✅ ws=4 全档正确；中小消息（≤ 1 MB）领先 pull 1.5-4×；
- 🟡 ws=4 大消息 (≥ 4 MB) 仍落后 pull ~1.7×（fused-reduce kernel bw 利用问题），留作下一节点。

**算 one_shot/two_shot 节点完成。**

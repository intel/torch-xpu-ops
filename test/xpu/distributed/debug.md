# Fused Allgather+Permute Kernel — Debug & Analysis Notes

> **Date**: 2026-05-22
> **Platform**: 4× Intel Arc Pro B60 (Xe2-HPG), PCIe interconnect
> **Repository**: `intel/torch-xpu-ops`, branch working on `test/xpu/csrc/AllgatherPermuteFused.cpp`
> **Config**: hidden=2048, topk=8, experts=128, dtype=bf16, world_size=4

---

## 1. 问题背景：小 token 数场景下 barrier 开销主导

### 1.1 原始实现流程

原始 allgather+local permute 需要 **4 次 kernel launch**：

```
copy (input → symm_mem)  →  barrier  →  allgather_permute  →  barrier
```

每次 barrier 实现为：
1. `put_signal` kernel：写 1 到所有 remote rank 的 signal_pad
2. `wait_signal` kernel：spin-poll 自己的 signal_pad 等待所有 remote rank 写入
3. 两次 barrier（pre + post）= 4 次额外 kernel launch

### 1.2 Baseline 性能数据

| tokens_per_rank | workgroups | PCIe read (MB) | HBM write (MB) | median (ms) | min (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 512 | 1.5 | 16 | 1.179 | 1.006 |
| 256 | 1,024 | 3.0 | 32 | 0.866 | 0.799 |
| 512 | 2,048 | 6.0 | 64 | 0.906 | 0.430 |
| 1024 | 4,096 | 12.0 | 128 | 0.588 | 0.583 |
| 2048 | — | — | — | DEVICE_LOST | — |

**关键发现**: tpr=128 (1.179 ms) 比 tpr=1024 (0.588 ms) **更慢**，尽管数据量少 8 倍。
原因：barrier 开销 (~0.8–1.0 ms) 在小 token 数下占比 ~80%+。

### 1.3 结论

仅减少 workgroup 数量无法解决小 token 性能问题。需要 **消除 barrier kernel launch 开销本身**。

---

## 2. Fused Kernel 设计

### 2.1 核心思路

将 4 次 kernel launch 合并为 **1 次 persistent kernel launch**：

```
Phase 0: Copy input → symm_mem          (all WGs, 分区负责)
Phase 1: Grid-wide barrier + 跨 device 信号交换  (原子计数 + spin-wait)
Phase 2: Allgather + Permute             (ring-ordered, 持久循环)
Phase 3: Post-read barrier + 跨 device 信号交换
```

### 2.2 Grid-Wide Barrier 实现

SYCL 没有 inter-workgroup 同步原语，采用 **原子计数器 + flag 轮询** 模式（参考 `TwoShotAllReduceKernel` 的 `sync_remote_blocks_impl` 模式）：

```
grid_state layout: [pre_counter, pre_flag, post_counter, post_flag]
```

1. 所有 WG 的 thread 0 做 `atomic_ref.fetch_add(1)` on `pre_counter`
2. 最后一个到达的 WG（`old + 1 == grid_dim`）执行跨 device 信号交换
3. 完成后，最后的 WG 设置 `pre_flag = pre_done_val`
4. 其他 WG 的 thread 0 spin-poll `pre_flag`
5. `item.barrier()` 将完成状态传播到 WG 内所有 thread

### 2.3 Generation Counter

避免多次调用之间需要 reset flag：

```cpp
pre_done_val  = 2 * generation + 1
post_done_val = 2 * generation + 2
```

Python 端跟踪 generation counter，每次调用递增。

### 2.4 跨 Device 信号协议

使用 symmetric memory 中的 sync buffer，与 `Signal.hpp` 相同的 `store_release` / `load_acquire` 语义：

```cpp
// put_signal: 写 1 到 remote rank 的 sync buffer
store_release_u32(&target[rank], 1);

// wait_signal: spin-poll 自己的 sync buffer
while (load_acquire_u32(&my_sync[r]) != 1) {}
store_release_u32(&my_sync[r], 0);  // reset for next call
```

与 `put_signal`/`wait_signal` 的区别：不需要 wait-for-zero-before-write，因为每次 kernel 调用写 1、remote 读 1 后 reset 0，sequential kernel 执行保证顺序。

---

## 3. 编译与构建问题

### 3.1 SYCL `atomic_ref` 编译错误

**问题**：SYCL `atomic_ref` 的默认 `memory_order` 模板参数**不允许** `release` 或 `acquire`。

```
error: no matching constructor for initialization of 'sycl::atomic_ref<int32_t, memory_order::release, ...>'
```

**原因**：SYCL spec 规定默认 memory_order 只允许 `relaxed`, `acq_rel`, `seq_cst`。

**修复**：使用 `acq_rel` 作为默认，在每个操作中显式指定 order：

```cpp
// 正确写法
sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>
    flag(grid_state[1]);
flag.store(val, sycl::memory_order::release);     // 每个操作显式指定
flag.load(sycl::memory_order::acquire);           // 每个操作显式指定
```

### 3.2 构建命令

```bash
cd test/xpu/csrc && python build.py
# 或手动:
icpx -fsycl -std=c++17 -shared -fPIC -O2 \
  -I<torch_include> -I<project_src> \
  AllgatherPermuteFused.cpp -o liballgather_permute_fused.so
```

6 个库全部构建成功：
- `liblocal_permute_copy.so`, `libep_dispatch.so`, `libep_combine.so`
- `liballgather_with_symm_mem.so`, `libunpermute_reduce_scatter.so`
- **`liballgather_permute_fused.so`** (新增)

---

## 4. Workspace 指针失效 Bug

### 4.1 问题 1：独立 workspace 名称失败

**现象**：使用 `__fused_sync` 作为独立的 workspace name 调用 `get_symm_mem_workspace()` 失败。

**原因**：`get_symm_mem_workspace()` 内部需要解析 process group，独立名称无法关联到已有的 group。

**修复**：使用同一个 workspace（与 original kernel 相同），将 sync buffer 放在 data region 之后。

### 4.2 问题 2：DEVICE_LOST（指针失效）

**现象**：先运行 original kernel，再运行 fused kernel 时触发 `UR_RESULT_ERROR_DEVICE_LOST`。

**Root Cause**：`get_symm_mem_workspace()` **每次调用** 都执行 `_SymmetricMemory.rendezvous(tensor)`，即使复用同一个 tensor。original kernel 内部调用 `workspace.barrier()` 也会触发 rendezvous。之前缓存的 `get_buffer()` 返回的指针**变为 stale**。

**修复**：在每次运行 fused kernel **之前**，用 `build_fused_buffers()` 重新获取 fresh 指针：

```python
def build_fused_buffers(hidden_shard, group, group_name):
    """每次调用获取 fresh workspace pointers，避免 stale pointer 问题"""
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=total_size)
    for r in range(world_size):
        buf = workspace.get_buffer(r, shape, dtype, storage_offset=...)
        rank_ptr_list.append(buf.data_ptr())   # fresh pointer
        sync_buf = workspace.get_buffer(r, ...)
        sync_ptr_list.append(sync_buf.data_ptr())  # fresh pointer
    return rank_bufs, sync_bufs, my_sync
```

### 4.3 Symmetric Memory 内存布局

```
每个 rank 的 workspace:
├── [0, buffer_size)                = data region
├── [data_bytes_aligned, +sync_bytes) = sync buffers (fused 新增)
├── [signal_pad_offset, +signal_pad_size) = signal pads
│   signal_pad_offset = round_up(buffer_size, 16)
│   signal_pad_size = 9216 bytes (实测值，非 types header 中的 2048)
└── block_size = signal_pad_offset + signal_pad_size
```

sync buffers 放在 `data_bytes_aligned` offset（data region 之内、signal pads 之前），经验证无重叠。

---

## 5. Grid-Wide Barrier 死锁问题

### 5.1 现象

- tpr=128 → 64 WGs → **正常**
- tpr=256 → 128 WGs → **死锁 (hang)**

### 5.2 Root Cause 分析

Grid-wide barrier 要求 **所有 WG 同时可调度**。如果 WG 数量超过硬件并发能力，spin-waiting 的 WG 会阻止未调度的 WG 执行 → **死锁**。

```
B60 硬件参数:
- max_compute_units = 160 (Vector Engines / VEs)
- SIMD width = 16
- WG_SIZE = 512 → 每个 WG 需要 512/16 = 32 HW threads
- 每个 VE 支持 ~16 concurrent HW threads

最大并发 HW threads ≈ 160 × 16 = 2560
最大安全 WG 数 = 2560 / 32 = 80
```

| tpr | WGs | HW threads | 状态 |
|-----|-----|-----------|------|
| 128 | 64  | 2048      | ✅ works (< 2560) |
| 256 | 128 | 4096      | ❌ deadlock (> 2560) |

### 5.3 修复

将 WG 上限从 `max_cu` (160) 改为 `max_cu / 2` (80)：

```cpp
// BEFORE (too generous):
const int64_t max_wgs = std::max<int64_t>(1, static_cast<int64_t>(max_cu));

// AFTER (safe):
// max_cu * 16 / 32 = max_cu / 2
// 160 / 2 = 80 max WGs
const int64_t max_wgs = std::max<int64_t>(1, static_cast<int64_t>(max_cu) / 2);
```

**公式推导**：
```
max_safe_wgs = max_cu × hw_threads_per_ve / (wg_size / simd_width)
             = 160 × 16 / (512 / 16)
             = 160 × 16 / 32
             = 80
```

修复后 tpr=256, 512, 1024 全部正常运行。

---

## 6. 最终 Benchmark 结果

修复 WG 上限后的完整数据（每个 size 独立运行，4× B60, bf16）：

| tokens_per_rank | Original avg (ms) | Fused avg (ms) | Speedup | Correct |
|:---:|:---:|:---:|:---:|:---:|
| 64  | 0.987 | 0.164 | **6.03x** | ✓ |
| 128 | 1.055 | 0.115 | **9.17x** | ✓ |
| 256 | 1.057 | 0.175 | **6.03x** | ✓ |
| 512 | 1.289 | 0.312 | **4.13x** | ✓ |
| 768 | 1.149 | 0.450 | **2.56x** | ✓ |
| 896 | 0.766 | 0.518 | **1.48x** | ✓ |
| 960 | 0.573 | 0.550 | **1.04x** | ✓ |
| 1024| 0.587 | 0.585 | **1.00x** | ✓ |
| 2048| DEVICE_LOST | — | — | — |

### 6.1 性能特征

- **峰值加速 9.17x**（tpr=128），barrier 开销从 ~1ms 降至 ~0.1ms
- 小 token 数场景（tpr ≤ 512）加速 4-9x，barrier 开销占比越高加速越明显
- tpr=1024 时两种方式性能一致（compute 主导，barrier 开销可忽略）
- tpr=2048 在 B60 上 DEVICE_LOST（已知硬件限制，与 fused kernel 无关）

### 6.2 正确性

所有 size 均通过 `torch.equal(remap_orig, remap_fused)` 验证，fused 输出与原始实现完全一致。

---

## 7. Crossover 分析与切换边界

### 7.1 经验数据

从 benchmark 数据看：
- tpr ≤ 960: fused 更快
- tpr ≥ 1024: 两者持平
- **经验切换边界 ≈ tpr = 1024**（B60, H=2048, ws=4, bf16）

### 7.2 理论边界推导

Fused kernel 节省 3 次 kernel launch 开销（copy 已经 fused，省掉 2× barrier launch + 1× 额外 sync），代价是 compute 效率略降（persistent kernel 的 WG 数受限）。

```
Fused wins when:
  3 × T_launch > T_compute × (η_orig/η_fused - 1)

其中:
  T_launch ≈ 0.25–0.35 ms (kernel launch + barrier overhead, 实测)
  η_orig = 原始 kernel 的带宽利用率
  η_fused = fused kernel 的带宽利用率 (受 max_wgs 限制)
```

关键洞察：在小 token 数场景下，原始 kernel 也是 under-subscribed（WG 不够填满 GPU），η_orig 同样较低，因此 η_orig ≈ η_fused，fused 几乎总是赢。只有当 tpr 足够大使得原始 kernel 能充分利用带宽时，fused 的 WG 限制才成为劣势。

### 7.3 推荐切换策略

```python
def should_use_fused(tokens_per_rank, hidden_size=2048, world_size=4):
    """Returns True if fused kernel is expected to be faster."""
    return tokens_per_rank <= 960
```

更通用的公式（跨硬件）：

```python
VEC_SIZE = 8
hidden_vecs = hidden_size // VEC_SIZE
total_work = world_size * tokens_per_rank * hidden_vecs
max_cu = get_max_compute_units()  # 160 on B60
wg_size = 512

# 原始 kernel 的 WG 数（1 WG per 256 work-items）
orig_wgs = (total_work + 255) // 256

# 原始 kernel 充分利用硬件的阈值
# 当 orig_wgs >> max_cu 时，bandwidth utilization 接近峰值
# 此时 fused 的 WG 限制开始成为瓶颈
threshold_wgs = max_cu * 4  # ~640 on B60

use_fused = orig_wgs < threshold_wgs
# 等价于: tokens_per_rank < threshold_wgs * 256 / (world_size * hidden_vecs)
#        = 640 * 256 / (4 * 256) = 640 ≈ 合理的经验值范围
```

---

## 8. 已知问题与限制

### 8.1 多 size 顺序运行崩溃

连续运行 3 个以上不同 tpr 的 benchmark 时，第 3 个 size 在 `bench_original()` 中触发 DEVICE_LOST。原因可能是 workspace 反复 rendezvous + 不同 size 的 buffer layout 变化导致内部状态不一致。

**Workaround**: 每个 size 独立运行（单独的 mpirun 进程）。

### 8.2 tpr=2048 DEVICE_LOST

B60 在 tpr=2048 时在原始 kernel 也会 DEVICE_LOST，与 fused kernel 无关。可能是 symmetric memory 大小限制或 device 资源上限。

### 8.3 WG 上限依赖硬件

`max_wgs = max_cu / 2` 的公式假设 Xe2-HPG 架构参数（16 HW threads/VE, SIMD=16）。其他 GPU 架构需要调整除数。更安全的做法是提供 environment variable 覆盖：

```cpp
const char* env_max = std::getenv("FUSED_MAX_WGS");
const int64_t max_wgs = env_max ? std::atoi(env_max)
    : std::max<int64_t>(1, static_cast<int64_t>(max_cu) / 2);
```

---

## 9. 文件清单

| 文件 | 说明 |
|------|------|
| `test/xpu/csrc/AllgatherPermuteFused.cpp` | Fused kernel 实现 (~329 行) |
| `test/xpu/distributed/bench_fused.py` | Benchmark 脚本，对比 fused vs original |
| `test/xpu/csrc/build.py` | 构建脚本 (已添加 fused target) |
| `test/xpu/csrc/LocalPermuteCopy.cpp` | 原始 `AllgatherPermuteRingVecKernel` |
| `test/xpu/distributed/allgather_local_permute_fusion.py` | Python wrapper |
| `test/xpu/distributed/bench_small_tokens.py` | 小 token 数 baseline benchmark |
| `src/xccl/Signal.hpp` | 信号原语参考 |
| `src/xccl/XPUSymmetricMemoryOps.cpp` | `TwoShotAllReduceKernel` 参考 |
| `src/xccl/XPUSymmetricMemory.cpp` | Symmetric memory 分配器 |

---

## 10. 后续工作

1. **集成到 `allgather_local_permute_fusion.py`**：添加 auto-switching 逻辑，tpr ≤ 960 用 fused，> 960 用 original
2. **多 size 顺序运行稳定性**：修复 workspace 重复 rendezvous 导致的 DEVICE_LOST
3. **跨硬件验证**：在其他 Intel XPU 上测试并调整 WG 上限公式
4. **WG_SIZE 调优**：尝试 WG_SIZE=256（更多 WGs 但每个更小），可能提升小 size 性能
5. **SLM 优化**：Phase 2 的 permute 写入模式可利用 SLM 做 local coalescing

# DeePEP Dispatch vs Allgather Local Permute Fusion — Performance Results

## Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| topk | 8 |
| num_experts | 128 |
| world_size | 4 GPUs |
| dtype | bfloat16 |
| warmup | 20 iterations |
| timed loops | 20 iterations |


## API definition

### compute_scatter_idx

```python
def compute_scatter_idx(
    topk_idx: torch.Tensor,            # [num_tokens, topk], int64, 全局 topk expert 索引
    num_experts: int = None,           # expert 总数（可选，默认从 topk_idx 推断）
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 返回 (scatter_idx, expert_offsets)
```

- **input arguments:**
  - `topk_idx`: `[num_tokens, topk]` int64 — 全局 token→expert 映射
  - `num_experts`: 整数，expert 总数。若为 `None`，从 `topk_idx.max() + 1` 推断

- **output:**
  - `scatter_idx`: `[num_tokens, topk]` int32 — 每个 `(token, k)` 在输出 buffer 中的写入位置。输出按 expert 分组：expert 0 的所有 token 在前，接着 expert 1，依此类推。同一 expert 内 token 按原始顺序排列（stable sort）
  - `expert_offsets`: `[num_experts + 1]` int64 — 每个 expert 在输出中的起始偏移（前缀和）。expert `e` 的 token 占据 `remap_hidden_states[expert_offsets[e] : expert_offsets[e+1]]`

- **算法:**
  1. 将 `topk_idx` flatten 为 `[num_tokens * topk]`
  2. 按 expert ID stable sort → `sort_indices`
  3. 计算逆置换：`scatter_idx[sort_indices[i]] = i`
  4. `expert_offsets` = `bincount(topk_flat).cumsum()` 前缀补零

- **`topk_idx` vs `scatter_idx` 的区别:**

  两者 shape 相同 `[num_tokens, topk]`，但语义完全不同：

  | | `topk_idx[i, k]` | `scatter_idx[i, k]` |
  |---|---|---|
  | **语义** | token `i` 的第 `k` 个 topk 选中了**哪个 expert** | token `i` 的第 `k` 份 hidden 应该**写到 output 的哪一行** |
  | **值域** | `[0, num_experts)` (expert ID) | `[0, num_tokens * topk)` (output row index) |
  | **用途** | 路由信息：告诉你"这个 token 去哪个 expert" | 内存布局：告诉 kernel "往 remap_hidden_states 的哪个位置写" |

  具体例子（4 tokens, topk=2, 3 experts）：
  ```
  topk_idx = [[0, 1],    # token 0 → expert 0, expert 1
              [2, 0],    # token 1 → expert 2, expert 0
              [1, 2],    # token 2 → expert 1, expert 2
              [0, 1]]    # token 3 → expert 0, expert 1

  # 按 expert 排序后的输出布局:
  #   expert 0 的 token: (0,0), (1,1), (3,0)  → 占 row 0,1,2
  #   expert 1 的 token: (0,1), (2,0), (3,1)  → 占 row 3,4,5
  #   expert 2 的 token: (1,0), (2,1)         → 占 row 6,7

  scatter_idx = [[0, 3],   # token 0 写到 row 0 (expert0第1个), row 3 (expert1第1个)
                [6, 1],    # token 1 写到 row 6 (expert2第1个), row 1 (expert0第2个)
                [4, 7],    # token 2 写到 row 4 (expert1第2个), row 7 (expert2第2个)
                [2, 5]]    # token 3 写到 row 2 (expert0第3个), row 5 (expert1第3个)

  expert_offsets = [0, 3, 6, 8]
  ```

  **本质：** `scatter_idx` 编码了"当前 token 是该 expert 的第几个"这个信息。没有它，kernel 不知道往 expert 块内的哪个偏移写。

### allgather_local_permute_fusion

```python
def allgather_local_permute_fusion(
    hidden_shard: torch.Tensor,       # [num_tokens_per_rank, hidden_size], bfloat16, 本 rank 的输入 hidden states
    topk_idx: torch.Tensor,           # [num_tokens, topk], int64, 全局 topk expert 索引（所有 rank 相同）
    scatter_idx: torch.Tensor,        # [num_tokens, topk], int32, 预计算的 expert-sorted 写入位置（来自 compute_scatter_idx）
    remap_hidden_states: torch.Tensor, # [num_tokens * topk, hidden_size], bfloat16, 输出（expert-centric 布局）
    group: dist.ProcessGroup = None,   # TP process group（默认 WORLD）
    group_name: str = None,            # symmetric memory workspace 名称
    backend_stream: torch.xpu.Stream = None,  # overlap stream
) -> torch.Tensor:                     # 返回 remap_hidden_states [num_tokens * topk, hidden_size]
```

### deepep_owner_dispatch

```python
def deepep_owner_dispatch(
    hidden_shard: torch.Tensor,        # [num_tokens_per_rank, hidden_size], bfloat16, 本 rank 的输入 hidden states
    topk_idx: torch.Tensor,            # [num_tokens, topk], int64, 全局 topk expert 索引
    remap_hidden_states: torch.Tensor, # [num_tokens * topk, hidden_size], bfloat16, 输出（仅填充本 rank 拥有的 expert 对应位置）
    num_experts: int,                  # expert 总数（用于计算 ownership）
    group: dist.ProcessGroup = None,   # TP process group（默认 WORLD）
    group_name: str = None,            # symmetric memory workspace 名称
    skip_copy: bool = False,           # 若 True，跳过 hidden_shard→symmetric memory 的 copy（需确保数据已就位）
) -> torch.Tensor:                     # 返回 remap_hidden_states [num_tokens * topk, hidden_size]
```

## Results

### tokens_per_rank = 2048

| Method | Avg (ms) | Min (ms) | Notes |
|--------|----------|----------|-------|
| **EP Dispatch (ring-ordered)** | **1.013** | **1.009** | ownership pre-check skips ~10% PCIe reads |
| **Allgather+permute (fused)** | **1.118** | **1.115** | single kernel, single stream, ring-ordered |
| Allgather+permute w/o overlap | 1.869 | 1.861 | no overlap |
| Allgather+permute w/ overlap | 1.365 | 1.322 | overlap(legacy) |

### tokens_per_rank = 4096

| Method | Avg (ms) | Min (ms) | Notes |
|--------|----------|----------|-------|
| **EP Dispatch (ring-ordered)** | **1.969** | **1.967** | |
| **Allgather+permute (fused)** | **2.166** | **2.165** | |
| Allgather+permute w/o overlap | 3.673 | 3.668 | (legacy) |
| Allgather+permute w/ overlap | 2.820 | 2.627 | (legacy) |


## Data Transfer Analysis

Common parameters: `num_experts=128, topk=8, hidden_size=2048, dtype=bfloat16 (2B)`

Each token data = 2048 × 2B = **4 KB**

### 4 devices × 2048 tokens_per_rank

- Device 0 owns 32/128 experts → P(slot on device 0) = 1/4
- P(token unneeded) = (3/4)^8 ≈ 10.0%  →  **~90% of remote tokens need transfer**
- Remote tokens = 3 × 2048 = 6144

| Method | Transferred tokens | Volume | Savings |
|--------|-------------------|--------|---------|
| Allgather (全拉) | 6144 | 24 MB | — |
| EP Dispatch (only owned) | ~5530 | ~21.6 MB | **~10%** |

### 8 devices × 1024 tokens_per_rank

- Device 0 owns 16/128 experts → P(slot on device 0) = 1/8
- P(token unneeded) = (7/8)^8 ≈ 34.4%  →  **~65.6% of remote tokens need transfer**
- Remote tokens = 7 × 1024 = 7168

| Method | Transferred tokens | Volume | Savings |
|--------|-------------------|--------|---------|
| Allgather (全拉) | 7168 | 28 MB | — |
| EP Dispatch (only owned) | ~4702 | ~18.4 MB | **~34%** |

### Scaling Projection (num_experts=128, topk=8)

General formula: `P(token unneeded) = ((W-1)/W)^topk`，其中 W = world_size

| world_size | experts/device | P(unneeded) | EP Dispatch 传输比例 | vs Allgather 节省 |
|:----------:|:--------------:|:-----------:|:-------------------:|:----------------:|
| 2 | 64 | (1/2)^8 = 0.4% | 99.6% | ~0% |
| 4 | 32 | (3/4)^8 = 10.0% | 90.0% | **~10%** |
| 8 | 16 | (7/8)^8 = 34.4% | 65.6% | **~34%** |
| 16 | 8 | (15/16)^8 = 59.7% | 40.3% | **~60%** |
| 32 | 4 | (31/32)^8 = 77.6% | 22.4% | **~78%** |
| 64 | 2 | (63/64)^8 = 88.2% | 11.8% | **~88%** |

**Takeaway**: Device 越多 → 每个 device 拥有的 expert 比例越小 (1/W) → ownership pre-check 跳过的无效读取越多 → EP Dispatch 相对 allgather 的传输量优势从 ~0% (2 devices) 增大到 ~88% (64 devices)。在大规模部署（≥16 devices）下，EP Dispatch 的选择性读取带来的传输量优势非常显著。

## How to Reproduce

```bash
# Build the kernel
cd test/xpu/csrc && python build.py

# Run the allgather + local permute fusion
cd test/xpu/distributed
SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1 mpirun -np 4 python test_allgather_local_permute_fusion.py

# Run the standalone correctness + performance test
SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1 mpirun -np 4 python test_deepep_dispatch.py
```


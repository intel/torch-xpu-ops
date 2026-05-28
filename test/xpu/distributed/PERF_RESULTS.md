# SymmBuffer Fusion API — Performance Results

## Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| topk | 8 |
| num_experts | 128 |
| world_size | 4 GPUs |
| dtype | bfloat16 |
| PCIe BW (discounted) | 22.0 GB/s |
| HBM BW | 437.0 GB/s |
| warmup | 20 iterations |
| timed loops | 20 iterations |


## SymmBuffer Fusion API

### SymmBuffer.allgather_local_permute_fusion

```python
def allgather_local_permute_fusion(
    self,
    hidden_shard: torch.Tensor,       # [num_tokens_per_rank, hidden] bfloat16, 本 rank 的输入 hidden states
    topk_idx: torch.Tensor,           # [num_tokens_per_rank, topk] int32, 本 rank 的 expert 分配索引
    topk_weights: torch.Tensor,       # [num_tokens_per_rank, topk] float32, 本 rank 的路由权重
    num_experts: int,                 # expert 总数
    remap_hidden_states: torch.Tensor, # [num_tokens * topk, hidden] bfloat16, 预分配的输出 buffer
) -> Tuple[torch.Tensor, SymmHandle]:
```

- **功能：** 融合 allgather + local permute。通过 symmetric memory 将各 rank 的 hidden states、topk_idx、topk_weights 广播到所有 rank，然后使用 notify_dispatch_v2 内核直接计算 scatter_idx（无需外部预计算），最后调用 allgather_permute 内核将 hidden states 按 expert-centric 布局排列。
- **输入要求：**
  - `hidden_shard`: 形状 `[num_tokens_per_rank, hidden]`，dtype 为 `bfloat16`，本 rank 拥有的 token hidden states
  - `topk_idx`: 形状 `[num_tokens_per_rank, topk]`，dtype 为 `int32`，每个 token 选中的 expert ID（仅本 rank 的 token）
  - `topk_weights`: 形状 `[num_tokens_per_rank, topk]`，dtype 为 `float32`，路由权重（仅本 rank 的 token）
  - `num_experts`: 整数，expert 总数（例如 128）
  - `remap_hidden_states`: 预分配的输出 tensor，形状 `[num_tokens * topk, hidden]`，dtype 为 `bfloat16`，其中 `num_tokens = num_tokens_per_rank * world_size`
- **输出：**
  - `remap_hidden_states`: expert-centric 布局的 hidden states
  - `SymmHandle`: 包含 scatter_idx、global_topk_weights、global_topk_idx 等信息，供 `unpermute_reducescatter_fusion` 使用
- **注意：** 需要 `notify_dispatch_v2` 和 `allgather_permute` 内核已加载

### SymmBuffer.unpermute_reducescatter_fusion

```python
def unpermute_reducescatter_fusion(
    self,
    expert_output: torch.Tensor,   # [num_tokens * topk, hidden] bfloat16, expert 输出（expert-centric 布局）
    handle: SymmHandle,            # 由 allgather_local_permute_fusion 返回的句柄
    output: torch.Tensor,          # [num_tokens_per_rank, hidden] bfloat16, 预分配的输出 buffer
) -> torch.Tensor:
```

- **功能：** 融合 unpermute + reduce-scatter。流水线实现：先对远端 chunk 计算 local unpermute 并通过 symmetric memory 推送到目标 rank，最后计算本 rank 的 chunk（与最后一次推送重叠）。所有 rank 的部分结果通过 sum_reduction 内核聚合。
- **输入要求：**
  - `expert_output`: 形状 `[num_tokens * topk, hidden]`，dtype 为 `bfloat16`，expert 处理后的输出（与 allgather_local_permute_fusion 的 remap_hidden_states 形状相同）
  - `handle`: `SymmHandle` 对象，必须由同一轮 `allgather_local_permute_fusion` 返回，包含 `abs_scatter_idx` 和 `global_topk_weights`
  - `output`: 预分配的输出 tensor，形状 `[num_tokens_per_rank, hidden]`，dtype 为 `bfloat16`
- **输出：** `output` tensor，包含 reduce-scatter 聚合后的结果
- **注意：** 需要 `local_unpermute_copy_` 内核已加载；当 `num_ranks > 2` 时使用 `sum_reduction` 内核加速聚合

### SymmHandle 数据结构

```python
@dataclass
class SymmHandle:
    scatter_idx: torch.Tensor         # [num_tokens, topk] int32，expert-relative 位置
    global_topk_weights: torch.Tensor # [num_tokens, topk] float32，全局路由权重
    num_tokens_per_rank: int          # 每个 rank 的 token 数
    global_topk_idx: torch.Tensor     # [num_tokens, topk] int32，全局 expert 索引
    rows_per_expert: torch.Tensor     # [num_experts] int32，每个 expert 的 token 行数
    abs_scatter_idx: torch.Tensor     # [num_tokens, topk] int32，绝对写入位置（用于 unpermute）
```


## Results — allgather_local_permute_fusion

### tokens_per_rank = 1024

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **0.639** | **0.632** | **0.649** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| allgather (PCIe) | 12.0 MB | 22.0 GB/s | 0.571 |
| permute R+W (HBM) | 144.0 MB | 437.0 GB/s | 0.346 |
| **lower_bound** | — | — | **0.916** |
| **efficiency** | — | — | **143.4%** |

### tokens_per_rank = 2048

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **1.196** | **1.189** | **1.200** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| allgather (PCIe) | 24.0 MB | 22.0 GB/s | 1.141 |
| permute R+W (HBM) | 288.0 MB | 437.0 GB/s | 0.691 |
| **lower_bound** | — | — | **1.832** |
| **efficiency** | — | — | **153.2%** |

### tokens_per_rank = 4096

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **2.441** | **2.311** | **4.840** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| allgather (PCIe) | 48.0 MB | 22.0 GB/s | 2.283 |
| permute R+W (HBM) | 576.0 MB | 437.0 GB/s | 1.382 |
| **lower_bound** | — | — | **3.665** |
| **efficiency** | — | — | **150.1%** |

### tokens_per_rank = 8192

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **4.540** | **4.533** | **4.543** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| allgather (PCIe) | 96.0 MB | 22.0 GB/s | 4.565 |
| permute R+W (HBM) | 1152.0 MB | 437.0 GB/s | 2.764 |
| **lower_bound** | — | — | **7.329** |
| **efficiency** | — | — | **161.4%** |


## Results — unpermute_reducescatter_fusion

### tokens_per_rank = 1024

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **0.708** | **0.701** | **0.738** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| reduce_scatter (PCIe) | 12.0 MB | 22.0 GB/s | 0.571 |
| unpermute R+W (HBM) | 132.0 MB | 437.0 GB/s | 0.317 |
| **lower_bound** | — | — | **0.887** |
| **efficiency** | — | — | **125.4%** |

### tokens_per_rank = 2048

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **1.348** | **1.344** | **1.355** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| reduce_scatter (PCIe) | 24.0 MB | 22.0 GB/s | 1.141 |
| unpermute R+W (HBM) | 264.0 MB | 437.0 GB/s | 0.633 |
| **lower_bound** | — | — | **1.775** |
| **efficiency** | — | — | **131.7%** |

### tokens_per_rank = 4096

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **2.612** | **2.609** | **2.616** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| reduce_scatter (PCIe) | 48.0 MB | 22.0 GB/s | 2.283 |
| unpermute R+W (HBM) | 528.0 MB | 437.0 GB/s | 1.267 |
| **lower_bound** | — | — | **3.550** |
| **efficiency** | — | — | **135.9%** |

### tokens_per_rank = 8192

| | Avg (ms) | Min (ms) | Max (ms) |
|---|:---:|:---:|:---:|
| **Measured** | **5.153** | **5.148** | **5.159** |

| Projection | Data Volume | BW | Time (ms) |
|---|:---:|:---:|:---:|
| reduce_scatter (PCIe) | 96.0 MB | 22.0 GB/s | 4.565 |
| unpermute R+W (HBM) | 1056.0 MB | 437.0 GB/s | 2.534 |
| **lower_bound** | — | — | **7.099** |
| **efficiency** | — | — | **137.8%** |


## Summary

| tokens_per_rank | AG+Permute Avg (ms) | AG+Permute Efficiency | Unperm+RS Avg (ms) | Unperm+RS Efficiency |
|:---:|:---:|:---:|:---:|:---:|
| 1024 | 0.639 | 143.4% | 0.708 | 125.4% |
| 2048 | 1.196 | 153.2% | 1.348 | 131.7% |
| 4096 | 2.441 | 150.1% | 2.612 | 135.9% |
| 8192 | 4.540 | 161.4% | 5.153 | 137.8% |

> Efficiency = lower_bound / measured × 100%。Efficiency > 100% 表示实测延迟低于 non-overlapped projection（PCIe + HBM 串行），说明 allgather/reduce-scatter 与 permute/unpermute 有效重叠。

## How to Reproduce

```bash
# Build the kernel
cd test/xpu/csrc && python build.py

# Run the SymmBuffer fusion performance benchmark
cd test/xpu/distributed
SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1 TOKENS_PER_RANK=2048 mpirun -np 4 python test_symm_buffer_fusion_perf.py
```


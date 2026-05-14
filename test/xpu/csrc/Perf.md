# LocalPermuteCopy XPU Kernel

Fused local permute copy SYCL kernel：将 `[tokens_per_rank, hidden]` 按 `topk_idx` 映射到 `remap_hidden_states[token * topk + k]`。

注册为 `torch.ops.symm_mem.local_permute_copy_`。

## 前置条件

- Intel oneAPI 工具链（`icpx` 编译器）
- PyTorch（已安装且支持 XPU）
- XPU 设备可用

## 构建

```bash
cd test/xpu/csrc
python build.py
```

构建过程会：
1. 自动检测 PyTorch 的 include/lib 路径
2. 自动检测 `_GLIBCXX_USE_CXX11_ABI` 设置
3. 使用 `icpx -fsycl` 编译 `LocalPermuteCopy.cpp`
4. 生成 `liblocal_permute_copy.so` 到当前目录

清理构建产物：

```bash
python build.py clean
```

## 运行测试

```bash
# 使用 unittest
python test_local_permute_copy.py

# 或使用 pytest（更详细的输出）
pytest test_local_permute_copy.py -v
```

测试用例包括：
- `test_basic_float32` — float32 基本功能
- `test_basic_bfloat16` — bfloat16 基本功能
- `test_offset_zero` — offset 为 0 的情况
- `test_large_topk` — 较大 topk 值
- `test_single_token` — 单 token
- `test_zero_tokens` — 空输入（no-op）
- `test_inplace_semantics` — 验证 in-place 修改语义

## 在 Python 中使用

```python
from local_permute_copy import local_permute_copy

# src_hidden:          [num_tokens_per_rank, hidden_size], XPU tensor
# topk_idx:            [num_tokens, topk],                 XPU tensor
# remote_token_offset: int
# remap_hidden_states: [num_tokens * topk, hidden_size],   XPU tensor (输出，in-place 修改)
result = local_permute_copy(src_hidden, topk_idx, remote_token_offset, remap_hidden_states)
```

或者直接加载库调用底层 op：

```python
import torch
torch.ops.load_library("liblocal_permute_copy.so")

result = torch.ops.symm_mem.local_permute_copy_(
    src_hidden, topk_idx, remote_token_offset, remap_hidden_states
)
```

## 分布式测试（allgather + permute fusion）

```bash
cd test/xpu/distributed
source env.sh
mpirun -n 2 python test_allgather_local_permute_fusion_dist.py
mpirun -n 4 python test_allgather_local_permute_fusion_dist.py
```

测试包含正确性校验（对比 all_gather + Python reference）和性能计时。

## 性能数据

测试配置：`TOKENS_PER_RANK=2048, HIDDEN_SIZE=2048, TOPK=8, dtype=bfloat16`

| world_size | avg latency (ms) | 说明 |
|:----------:|:-----------------:|:-----|
| 2          | 2.86              | native kernel（PCIe） |
| 4          | 5.87              | native kernel（PCIe） |

> 对比 Python fallback 实现（2 ranks）：~555 ms → **~194× 加速**

## 文件说明

| 文件 | 说明 |
|------|------|
| `LocalPermuteCopy.cpp` | SYCL kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx） |
| `CMakeLists.txt` | CMake 集成（项目内构建） |
| `local_permute_copy.py` | Python 封装接口 |
| `test_local_permute_copy.py` | 单元测试 |

# EpDispatch XPU Kernel

TP+EP owner-based dispatch SYCL kernel：通过 symmetric memory 实现跨 rank 的 token dispatch。

单次 kernel launch 处理所有 `(token, k, hidden)` 工作项：
1. 每个 thread 计算 `(global_token_idx, k, h)` → 查 `topk_idx` 得到 expert
2. 计算 expert owner rank（支持余数分配）
3. 若 expert **不属于**当前 rank → 直接 skip（不读 remote memory）
4. 若属于当前 rank → 通过 rank 指针表从 source rank 的 symmetric memory **选择性读取**对应 token → 写入 `remap_hidden_states`

注册为 `torch.ops.symm_mem.ep_dispatch`。

## 前置条件

- Intel oneAPI 工具链（`icpx` 编译器）
- PyTorch（已安装且支持 XPU）
- XPU 设备可用

## 构建

```bash
cd test/xpu/csrc
python build.py
```

清理构建产物：

```bash
python build.py clean
```

## 运行分布式测试

```bash
cd test/xpu/distributed
source env.sh
mpirun -n 2 python test_deepep_dispatch.py
mpirun -n 4 python test_deepep_dispatch.py
```

测试包含正确性校验（对比 all_gather + Python reference）和性能计时。

## 性能数据

测试配置：`TOKENS_PER_RANK=256, HIDDEN_SIZE=512, TOPK=4, NUM_EXPERTS=8, dtype=bfloat16`

| world_size | avg latency (ms) | per-rank latencies (ms, last 5 iters) |
|:----------:|:-----------------:|:--------------------------------------|
| 2          | 0.968             | 0.75 ~ 1.41                           |
| 4          | 1.181             | 0.85 ~ 1.26                           |

> 对比 Python fallback 实现（2 ranks）：~67 ms → **~50× 加速**

## 在 Python 中使用

```python
import torch
import torch.distributed as dist
from deepep_dispatch import deepep_owner_dispatch

# hidden_shard:        [num_tokens_per_rank, hidden_size], XPU tensor
# topk_idx:            [num_tokens, topk],                 XPU tensor
# remap_hidden_states: [num_tokens * topk, hidden_size],   XPU tensor (输出)
result = deepep_owner_dispatch(
    hidden_shard, topk_idx, remap_hidden_states,
    num_experts=8, group=dist.group.WORLD,
)
```

自动检测 `libep_dispatch.so`，若不存在则使用 Python fallback。

## 文件说明

| 文件 | 说明 |
|------|------|
| `EpDispatch.cpp` | SYCL kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx） |
| `CMakeLists.txt` | CMake 集成（项目内构建） |
| `test_deepep_dispatch.py` | 分布式正确性 + 性能测试 |
| `deepep_dispatch.py` | Python 封装（kernel + fallback） |

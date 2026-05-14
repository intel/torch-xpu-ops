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

## 文件说明

| 文件 | 说明 |
|------|------|
| `LocalPermuteCopy.cpp` | SYCL kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx） |
| `CMakeLists.txt` | CMake 集成（项目内构建） |
| `local_permute_copy.py` | Python 封装接口 |
| `test_local_permute_copy.py` | 单元测试 |

# EpDispatch XPU Kernel

Fused TP+EP owner-based dispatch SYCL kernel: 将 `hidden_shard` 按 `topk_idx` 和 expert ownership 映射到 `remap_hidden_states`。

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

构建过程会：
1. 自动检测 PyTorch 的 include/lib 路径
2. 自动检测 `_GLIBCXX_USE_CXX11_ABI` 设置
3. 使用 `icpx -fsycl` 编译 `EpDispatch.cpp`
4. 生成 `libep_dispatch.so` 到当前目录

清理构建产物：

```bash
python build.py clean
```

## 运行测试

```bash
# 使用 unittest
python test_ep_dispatch.py

# 或使用 pytest（更详细的输出）
pytest test_ep_dispatch.py -v
```

测试用例包括：
- `test_basic_float32` — float32 基本功能
- `test_basic_bfloat16` — bfloat16 基本功能
- `test_large_topk` — 较大 topk 值
- `test_zero_tokens` — 空输入（no-op）
- `test_inplace_semantics` — 验证 in-place 修改语义

## 在 Python 中使用

```python
from ep_dispatch import ep_dispatch

# hidden_shard:       [num_tokens_per_rank, hidden_size], XPU tensor
# topk_idx:           [num_tokens, topk],                 XPU tensor
# remap_hidden_states: [num_tokens * topk, hidden_size],   XPU tensor (输出，in-place 修改)
result = ep_dispatch(hidden_shard, topk_idx, remap_hidden_states, num_experts, rank, world_size)
```

或者直接加载库调用底层 op：

```python
import torch
torch.ops.load_library("libep_dispatch.so")

result = torch.ops.symm_mem.ep_dispatch(
    hidden_shard, topk_idx, remap_hidden_states, num_experts, rank, world_size
)
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `EpDispatch.cpp` | SYCL kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx） |
| `CMakeLists.txt` | CMake 集成（项目内构建） |
| `ep_dispatch.py` | Python 封装接口 |
| `test_ep_dispatch.py` | 单元测试 |

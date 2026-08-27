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

# AllgatherWithSymmMem XPU Kernel

Pure allgather SYCL kernel via symmetric memory: ring-ordered pull from all ranks' symmetric memory buffers, no permutation.

注册为 `torch.ops.symm_mem.allgather_with_symm_mem`。

## 前置条件

- Intel oneAPI 工具链（`icpx` 编译器）
- PyTorch（已安装且支持 XPU）
- XPU 设备可用

## 构建

```bash
cd test/xpu/csrc
python build.py
```

构建过程会生成 `liballgather_with_symm_mem.so` 到当前目录。

## 运行分布式测试

```bash
cd test/xpu/distributed
source env.sh
mpirun -n 2 python test_allgather_local_permute_fusion_dist.py
```

在 `test_allgather_local_permute_fusion_dist.py` 的 `main` 中启用 `check_allgather_with_symm_mem()` 即可测试。

## 在 Python 中使用

```python
from allgather_local_permute_fusion import allgather_with_symm_mem

# input_shard: [numel_per_rank] XPU tensor
# output:      [numel_per_rank * world_size] XPU tensor
allgather_with_symm_mem(input_shard, output_tensor=output, group=group)
```

自动检测 `liballgather_with_symm_mem.so`，若不存在则使用 Python fallback。

## 文件说明

| 文件 | 说明 |
|------|------|
| `AllgatherWithSymmMem.cpp` | SYCL kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx） |
| `allgather_local_permute_fusion.py` | Python 封装（kernel + fallback） |

# LowLatencyDispatchRoleSplitIshmem XPU Kernel

DeepSymm `LowLatencyDispatchRoleSplitKernelBK`（`moe_ep/internode_ll.cpp`）低延迟
MoE dispatch kernel的简化 reproducer：单个 kernel launch 内把 work-group 分成两种
角色 —— "expert" WG（每个 WG 独占一个全局 expert id，扫描本 rank 的本地 token，把命中
的 token 通过 ISHMEM RDMA（`ishmemx_putmem_nbi_work_group_qp`）推送到 expert 所在
rank 的接收 buffer，再发布完成 flag/count）和 "receiver" WG（每个本 rank 拥有的
local expert 一个，等待所有 source rank 的完成 flag 后，把到达的 token gather 进
`packed_recv_x` / `packed_recv_src_info`，并记录 DeepEP 风格的 `(count, begin)`
layout range）。省略了生产 kernel 的 mask buffer / fp8 cast / cumulative stats /
NVLink+RDMA 分层等附加特性，只保留 role-split + RDMA + on-device 完成信号这一核心
模式，便于单独做功能验证和性能测试。

注册为 `torch.ops.symm_mem.low_latency_dispatch_role_split_ishmem`。

## 前置条件

- Intel oneAPI 工具链（`icpx` 编译器）+ ISHMEM（设置 `ISHMEM_HOME`/`ISHMEM_ROOT`）
- PyTorch（已安装且支持 XPU），多张 XPU 设备 + RDMA 网卡（NIC 间可互通）
- MPI（用于 ISHMEM 的 host bootstrap）

## 构建

```bash
cd test/xpu/csrc
python build.py
```

生成 `liblow_latency_dispatch_role_split_ishmem.so`。

## 运行测试（功能正确性 + 性能）

```bash
cd test/xpu/distributed
./test_low_latency_dispatch_role_split_ishmem.sh
# 或者手动：
mpirun -np 4 --prepend-rank python test_low_latency_dispatch_role_split_ishmem.py
```

测试用例校验 `packed_recv_x` / `packed_recv_src_info` / `packed_recv_count` /
`packed_recv_layout_range` 与 python 端按相同 role-split 语义重放的参考实现完全一致，
随后在 timed loop 中测量端到端延迟，并（默认开启 `ENABLE_PROFILE=1`）用
`torch.profiler` 抓取 chrome trace，从 trace 中提取纯 kernel 耗时计算
GB/s/PE 带宽。

主要环境变量：`TOKENS_PER_RANK` (128)、`HIDDEN_SIZE` (2048)、`TOPK` (8)、
`NUM_EXPERTS` (32)、`CAPACITY_MULT` (4)、`LOOP` (40)、`WARMUP` (20)。

## 文件说明

| 文件 | 说明 |
|------|------|
| `LowLatencyDispatchRoleSplitIshmem.cpp` | SYCL + ISHMEM kernel 实现 + op 注册 |
| `build.py` | 编译脚本（icpx + ISHMEM 静态库） |
| `../distributed/test_low_latency_dispatch_role_split_ishmem.py` | 正确性 + 性能 UT |
| `../distributed/test_low_latency_dispatch_role_split_ishmem.sh` | mpirun 启动脚本 |

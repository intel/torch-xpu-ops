## 0. 核心工作流

1. `symm_mem.empty(...)` 分配对称张量；
2. `symm_mem.rendezvous(tensor, group=...)` 在进程组内交换句柄，得到可远程寻址的
   指针；
3. 通过 `torch.ops.symm_mem.*` 主机端算子，或在 Triton kernel 内通过设备端 SHMEM
   原语进行通信。

三个**分配后端**（host 侧）：`set_backend("CUDA" | "NVSHMEM" | "NCCL")`。

---

## 1. 后端实现（C++ / CUDA）

源码目录：[torch/csrc/distributed/c10d/symm_mem/](torch/csrc/distributed/c10d/symm_mem/)

### 1.1 通用框架层

| 文件 | 作用 |
|---|---|
| [SymmetricMemory.hpp](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp) / [SymmetricMemory.cpp](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp) | 定义 `SymmetricMemory` / `SymmetricMemoryAllocator` 抽象接口；通过 `TORCH_LIBRARY_FRAGMENT(symm_mem, m)` 注册所有 `symm_mem::*` 算子 schema；实现 `_rendezvous`、`_barrier`、后端注册与选择 |
| [DMAConnectivity.hpp](torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp) / [DMAConnectivity.cpp](torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.cpp) | 抽象“设备间可直接 DMA 访问”的连通性检测 |
| [CudaDMAConnectivity.cpp](torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp) | CUDA NVLink 拓扑连通性探测 |
| [cuda_mem_pool.cpp](torch/csrc/distributed/c10d/symm_mem/cuda_mem_pool.cpp) | 对称内存的 CUDA MemPool 分配器，支撑 `get_mem_pool` |
| [intra_node_comm.*](torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cpp) | 节点内 NVLink 直连通信原语（`IntraNodeComm`） |

### 1.2 后端具体实现

| 后端 | 文件 | 说明 |
|---|---|---|
| CUDA（NVLink/P2P/multicast） | [CUDASymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu)、[CUDASymmetricMemory-inl.cuh](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh)、[CUDASymmetricMemoryOps.cu](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu)、[CUDASymmetricMemoryUtils.cpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp) | 基于 CUDA VMM / IPC 句柄与 NVLink SHARP（multicast）的实现；`one_shot`/`two_shot`/`multimem` all-reduce、`barrier`、`put_signal`/`wait_signal`、`stream_write_value32` 等设备算子 |
| NVSHMEM | [NVSHMEMSymmetricMemory.cpp](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cpp)、[nvshmem_extension.cu](torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu)、[nvshmem_team_manager.hpp](torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp) | 基于 NVSHMEM 的对称堆分配与 put/get/broadcast/all-to-all/signal 等算子，支持跨节点（RDMA） |
| NCCL | [NCCLSymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu)、[nccl_extension.cu](torch/csrc/distributed/c10d/symm_mem/nccl_extension.cu)、[nccl_devcomm_manager.hpp](torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp)、[nccl_ep.cu](torch/csrc/distributed/c10d/symm_mem/nccl_ep.cu) | 基于 NCCL 窗口/设备通信器的实现；`nccl_put`/`nccl_get`/`put_signal`/`wait_signal`/`reduce_scatter_offset`/`all_to_all_nd`；支持注册外部 NCCL comm |
| 设备端算子 | [ops/](torch/csrc/distributed/c10d/symm_mem/ops/) | 各后端共享 / 分派的 device-side 算子实现 |

### 1.3 host 侧注册的算子（节选）

`one_shot_all_reduce(_out/_copy)`、`two_shot_all_reduce_`、`multimem_all_reduce_`、
`multimem_one_shot_all_reduce(_out)`、`multimem_all_gather_out`、`reduce_scatter_out`、
`_async_input_mm`、`stream_write_value32_`、`memset32_`、`memcpy_to_multicast_`、
`nvshmem_put/get/broadcast/put_with_signal/wait_for_signal`、
`nccl_put/get/put_signal/wait_signal/reduce_scatter_offset/all_to_all_nd`、
`all_to_all_vdev(_2d/_2d_offset)`、`tile_reduce`、`multi_root_tile_reduce`、
`_rendezvous`、`_barrier`。

### 1.4 对应测试

- 后端 / 连通性 / 分配 / rendezvous：[test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py) 中的
  `SymmetricMemoryTest`：`test_has_multicast_support`、`test_cuda_nvlink_connectivity_detection`、
  `test_large_alloc`、`test_get_signal_pad`、`test_rendezvous_via_pg_allgather`、
  `test_rendezvous_custom_backend`、`test_pg_rendezvous_abort_after`、`test_subgroup`、
  `test_dispatcher_torchbind_symmetric_memory`。
- multimem / signal 设备算子：同文件 `SymmetricMemoryTest.test_get`，以及
  `test_multimem_all_reduce`、`test_multimem_one_shot_all_reduce`、
  `test_multimem_one_shot_reduce_out`、`test_multimem_all_gather`。
- NVSHMEM 后端 host 侧算子：[test/distributed/test_nvshmem.py](test/distributed/test_nvshmem.py)
  中的 `NVSHMEMSymmetricMemoryTest`（`test_nvshmem_put`/`test_nvshmem_get`/
  `test_get_remote_tensor(s)`/`test_multicast_ptr` 等）、`NVSHMEMAll2AllTest`
  （`test_all_to_all_vdev(_2d/_2d_offset)`）、`NVSHMEMTileCommTest`、`DispatchCombineTest`。

---

## 2. 前端 API（Python）

包目录：[torch/distributed/_symmetric_memory/](torch/distributed/_symmetric_memory/)

### 2.1 核心门面

[torch/distributed/_symmetric_memory/__init__.py](torch/distributed/_symmetric_memory/__init__.py)
封装底层 `torch._C._distributed_c10d._SymmetricMemory`，暴露面向用户的 API：

| API | 作用 |
|---|---|
| `empty(*size, dtype, device)` | 分配对称张量 |
| `rendezvous(tensor, group)` | 交换句柄，返回可远程访问的 `_SymmetricMemory` |
| `is_symm_mem_tensor(t)` | 判断张量是否来自对称内存 |
| `set_backend(name)` / `get_backend(device)` | 选择/查询分配后端（CUDA/NVSHMEM/NCCL） |
| `is_nvshmem_available()` | 探测 NVSHMEM 是否可用 |
| `get()` / `put_signal()` / `wait_signal()` | 单边通信 + 信号同步 |
| `reduce_scatter_offset()` / `all_to_all_nd()` | 高层集合通信 |
| `get_mem_pool(device)` | 返回对称内存 MemPool（配合 `torch.cuda.use_mem_pool`） |
| `get_signal_pad_size()` / `set_signal_pad_size()` | 信号 pad 配置 |
| `get_remote_tensors()` | 获取各 peer 的远程张量视图 |

### 2.2 backend 辅助模块

| 模块 | 作用 |
|---|---|
| [_nccl.py](torch/distributed/_symmetric_memory/_nccl.py) | `register_external_nccl_comm(...)`：把外部持有的 NCCL comm（如 torchcomms）注册进 symm mem 注册表，返回 RAII 句柄 `NcclCommRegistration` |
| [_shmem_triton.py](torch/distributed/_symmetric_memory/_shmem_triton.py) | 后端无关派发：`get_shmem_backend_module()`（CUDA→`_nvshmem_triton`）、`requires_shmem` 装饰器 |
| [_nvshmem_triton.py](torch/distributed/_symmetric_memory/_nvshmem_triton.py) | NVSHMEM 的 Triton 设备端支持（详见第 4 节） |
| [_shmem_triton_utils.py](torch/distributed/_symmetric_memory/_shmem_triton_utils.py) | 共享的 `ShmemKernelRegistry`、`run_shmem_init_hook` |

### 2.3 对应测试

- 用户 API：[test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py)
  `SymmetricMemoryTest.test_is_symm_mem_tensor`、`test_get_backend`、
  `test_get_signal_pad_size`、`test_set_signal_pad_size(_with_allocation)`、
  `test_allow_overlapping_devices`。
- MemPool：[test/distributed/test_nvshmem.py](test/distributed/test_nvshmem.py)
  `test_mempool_tensor_factory`、`test_mempool_tensor_w_collective`、
  `test_mempool_compute_ops`、`test_handle_offset`。

---

## 3. 典型用法

### 3.1 基础：分配 + rendezvous + 单边 put

```python
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

symm_mem.set_backend("NVSHMEM")
group_name = dist.group.WORLD.group_name

src = symm_mem.empty(nelems, dtype=torch.int64, device=device).fill_(rank)
dst = symm_mem.empty(nelems, dtype=torch.int64, device=device).fill_(-1)
symm_mem.rendezvous(src, group=group_name)
symm_mem.rendezvous(dst, group=group_name)

dist.barrier()
peer = 1 - rank
if rank == 0:
    symm_mem.put(dst, src, peer)   # 把本地 src 写入 peer 的 dst
dist.barrier()
```

### 3.2 集合通信（one-shot / multimem all-reduce）

```python
x = symm_mem.empty(N, device=device)
symm_mem.rendezvous(x, group=group_name)
torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

### 3.3 通过 MemPool 隐式分配

```python
mempool = symm_mem.get_mem_pool(device)
with torch.cuda.use_mem_pool(mempool):
    x = torch.arange(128, device=device)   # x 来自对称内存
torch.ops.symm_mem.one_shot_all_reduce(x, "sum", group_name)
```

对应测试：`SymmetricMemoryTest.test_get`、`test_multimem_*`（见第 1.4 节），以及
`test_nvshmem.py` 的 MemPool 系列。文档另见
[docs/source/symmetric_memory.md](docs/source/symmetric_memory.md)。

---

## 4. 设备端 SHMEM + Triton

让 Triton JIT kernel **内部**直接调用 SHMEM 设备端原语（put/get/signal/collective），
实现 kernel 级细粒度、可与计算重叠的单边通信。CUDA 平台后端固定为 **NVSHMEM**
（`libnvshmem_device.bc`）。

### 4.1 实现

[_nvshmem_triton.py](torch/distributed/_symmetric_memory/_nvshmem_triton.py)：

- `NvshmemLibFinder`：定位 `libnvshmem_device.bc`；
- `@requires_nvshmem`（CUDA 专用）/ `@requires_shmem`（跨平台，见
  [_shmem_triton.py](torch/distributed/_symmetric_memory/_shmem_triton.py)）：
  在 Triton 编译期链接 SHMEM 设备库并注册 init hook；
- 一批 `@core.extern` 包装的设备函数：`putmem_block`、`getmem_block(_nbi)`、
  `putmem_signal_block`、`signal_wait_until`、`signal_op`、`fence`、`quiet`、
  `my_pe`、`n_pes`、`barrier_all`、team 级 `alltoall`/`broadcast`/`reduce`。

### 4.2 用法

```python
import torch.distributed._symmetric_memory._shmem_triton as shmem_triton
from torch.distributed._symmetric_memory._shmem_triton import requires_shmem
from torch._inductor.runtime.triton_compat import triton

shmem = shmem_triton.get_shmem_backend_module()  # CUDA -> nvshmem

@requires_shmem
@triton.jit
def my_put_kernel(dest, src, nelems, pe):
    shmem.put(dest, src, nelems, pe)

# host 侧：empty + rendezvous 后即可在 kernel 内发起通信
my_put_kernel[(1,)](dst, src, nelems, peer)
```

### 4.3 对应测试

专门测试文件：[test/distributed/test_shmem_triton.py](test/distributed/test_shmem_triton.py)
（`SHMEMTritonTest`，需 H100 + NVSHMEM + Triton）：

| 测试 | 覆盖原语 |
|---|---|
| `test_triton_put` | `put` |
| `test_triton_get`（参数化 `nbi`） | `get` / `get_nbi`+`quiet` |
| `test_triton_get_ring` | 环形 `get` |
| `test_triton_put_signal_set` / `test_triton_put_signal_add` | `putmem_signal_block`（SET/ADD） |
| `test_triton_wait_until` / `test_triton_signal_wait_until` | `wait_until` / `signal_wait_until` |
| `test_triton_fence` / `test_triton_quiet` | `fence` / `quiet` |
| `test_triton_barrier` / `test_triton_sync` | `barrier_all` / `my_pe`/`n_pes` |
| `test_triton_alltoall` / `test_triton_broadcast` | team `alltoall` / `broadcast` |
| `test_triton_sum_reduce` / `test_triton_minmax_reduce` / `test_triton_prod_reduce`（参数化 `dtype`） | team `reduce`（sum/min/max/prod） |

```bash
python test/distributed/test_shmem_triton.py
```

---

## 5. 上层融合算子：Async-TP / fused all-gather-matmul

这些高层算子把张量并行中的通信（all-gather / reduce-scatter）与相邻的 matmul
融合并流水化（pipeline），用对称内存实现计算-通信重叠。定义在
[torch/distributed/_symmetric_memory/__init__.py](torch/distributed/_symmetric_memory/__init__.py)
（通过 `lib = torch.library.Library("symm_mem", "DEF")` 注册 `fused_*` 算子）。

### 5.1 主要算子与实现变体

| 算子 | 说明 |
|---|---|
| `_fused_all_gather_matmul`（`fused_all_gather_matmul`） | all-gather + matmul 融合，含 `_fallback` / `_native`（sm>=90）/ `_multimem` 变体 |
| `_fused_all_gather_scaled_matmul` | FP8 scaled 版本；`_ScaleMode`（UNSCALED/TENSOR_WISE/ROW_WISE_SHARDED/ROW_WISE_REPLICATED） |
| `_fused_matmul_reduce_scatter` | matmul + reduce-scatter 融合 |
| `_fused_scaled_matmul_reduce_scatter` | FP8 scaled reduce-scatter 版本 |
| `_low_contention_all_gather(_ce_multicast)` / `_low_contention_reduce_scatter` | 低竞争集合通信 |
| pipeline 原语 | `_pipelined_all_gather_and_consume`、`_pipelined_produce_and_all2all`、`_pipelined_multi_all_gather_and_consume` |

相关环境变量：`TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP`、`TORCH_SYMMMEM_IMPLICIT_POOL`、
`TORCH_SYMM_MEM_DISABLE_MULTICAST`。

### 5.2 对应测试

[test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py)
中的 `AsyncTPTest`（`@skip_if_rocm_multiprocess`）：

- `test_fused_all_gather_matmul`（参数化 `gather_dim`）
- `test_fused_all_gather_matmul_native`（`TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP=1`）
- `test_multimem_all_gather_matmul`
- `test_fused_all_gather_scaled_matmul`
- `test_fused_matmul_reduce_scatter`（参数化 `scatter_dim`）
- `test_fused_scaled_matmul_reduce_scatter`

低竞争集合通信：同文件 `SymmetricMemoryTest.test_low_contention_all_gather`、
`test_low_contention_all_gather_ce_multicast(_out)`、`test_low_contention_reduce_scatter`。

```bash
python test/distributed/test_symmetric_memory.py AsyncTPTest
```

---

## 6. 与 TorchInductor 的结合

有两条独立的集成路径。

### 6.1 micro-pipeline TP 编译器 pass

TorchInductor 的 FX pass
[torch/_inductor/fx_passes/micro_pipeline_tp.py](torch/_inductor/fx_passes/micro_pipeline_tp.py)
（由 `torch._inductor.config._micro_pipeline_tp` 开关控制）自动识别图中的
all-gather→matmul / matmul→reduce-scatter 模式，并替换为第 5 节的 `symm_mem`
融合算子，从而在 `torch.compile` 下自动开启 Async-TP。

对应测试：[test/distributed/tensor/parallel/test_micro_pipeline_tp.py](test/distributed/tensor/parallel/test_micro_pipeline_tp.py)

- `MicroPipelineTPTest`：`test_find_all_gather_patterns`、`test_find_reduce_scatter_patterns`、
  `test_get_unexposed_collectives`、`test_fuse_all_gather_matmul(_view_optimization/_slice_cat/_slice_cat_trim)`、
  `test_fuse_all_gather_scaled_matmul`、`test_fuse_matmul_reduce_scatter(_slice_cat)`、
  `test_fuse_scaled_matmul_reduce_scatter(_rowwise_scales_reshape_mm_reshape)`、
  `test_dtensor_seq_par`。
- `MicroPipelineTP4GPUTest`：`test_extra_collectives`。

```bash
python test/distributed/tensor/parallel/test_micro_pipeline_tp.py
```

### 6.2 low-contention collectives 降级 pass

[torch/_inductor/fx_passes/low_contention_collectives.py](torch/_inductor/fx_passes/low_contention_collectives.py)
在 Inductor 中把标准集合通信替换为 `symm_mem._low_contention_*` 实现。

对应测试：[test/inductor/test_low_contention_collectives.py](test/inductor/test_low_contention_collectives.py)
（断言图中出现 `symm_mem._low_contention_all_gather.default` 等目标）。

### 6.3 symm_mem 参数注册系统（custom op 实体化）

为了让 Inductor 在调用自定义算子前，把标记为 symm_mem 的输入正确“实体化”
（realize）为对称内存张量，PR #173513 引入了参数注册机制：

- [torch/library.py](torch/library.py)：`register_symm_mem_args` + `_resolve_op_name`；
- [torch/_library/simple_registry.py](torch/_library/simple_registry.py)：`SymmMemArgsHolder`、`SimpleLibraryRegistry.get()`；
- [torch/_inductor/ir.py](torch/_inductor/ir.py)：`FallbackKernel._maybe_realize_symm_mem_args`（在 `create` 中调用）。

对应测试：

- [test/inductor/test_symm_mem_registry.py](test/inductor/test_symm_mem_registry.py)（注册系统单测：`register`/`get`/`is_symm_mem_arg`）。
- [test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py)
  `test_custom_op_symm_mem_realization`（端到端验证 custom op 输入被实体化为对称内存）。

```bash
python test/inductor/test_symm_mem_registry.py
python test/inductor/test_low_contention_collectives.py
```

---

## 7. 测试速查表

| 层次 | 测试文件 | 关键类/用例 |
|---|---|---|
| 后端 / 分配 / rendezvous / multimem | [test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py) | `SymmetricMemoryTest` |
| NVSHMEM host 算子 / all-to-all / MemPool | [test/distributed/test_nvshmem.py](test/distributed/test_nvshmem.py) | `NVSHMEMSymmetricMemoryTest`、`NVSHMEMAll2AllTest`、`NVSHMEMTileCommTest`、`DispatchCombineTest` |
| 设备端 SHMEM + Triton | [test/distributed/test_shmem_triton.py](test/distributed/test_shmem_triton.py) | `SHMEMTritonTest` |
| Async-TP 融合算子 | [test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py) | `AsyncTPTest` |
| Inductor micro-pipeline TP | [test/distributed/tensor/parallel/test_micro_pipeline_tp.py](test/distributed/tensor/parallel/test_micro_pipeline_tp.py) | `MicroPipelineTPTest`、`MicroPipelineTP4GPUTest` |
| Inductor low-contention | [test/inductor/test_low_contention_collectives.py](test/inductor/test_low_contention_collectives.py) | — |
| Inductor symm_mem 参数注册 | [test/inductor/test_symm_mem_registry.py](test/inductor/test_symm_mem_registry.py) | — |

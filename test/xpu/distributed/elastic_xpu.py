"""XPU-compatible ElasticBuffer wrapper built on top of deepep_dispatch.

This module mirrors the public CUDA-facing `elastic.py` surface as closely as
practical in the XPU test tree, but the actual dispatch/combine implementation
uses the XPU APIs that already exist in `deepep_dispatch.py`.

Supported today:
- EP dispatch via `deepep_owner_dispatch`
- EP combine via `deepep_owner_combine`
- Basic `EPHandle` caching for dispatch -> combine reuse
- Lightweight stream/event wrappers for compatibility

Not backed by deepep_dispatch yet:
- Engram fetch/write
- PP send/recv
- true AGRS session semantics
- CUDA-specific overlap/runtime bookkeeping

Those methods are kept for interface compatibility and either degrade to local
PyTorch/distributed behavior or raise `NotImplementedError` where the CUDA
semantics do not exist on XPU yet.
"""

from __future__ import annotations

import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

try:
    from .deepep_dispatch import (
        build_combine_rank_output_ptrs,
        deepep_owner_combine,
        deepep_owner_dispatch,
    )
except ImportError:
    from deepep_dispatch import (  # type: ignore
        build_combine_rank_output_ptrs,
        deepep_owner_combine,
        deepep_owner_dispatch,
    )

try:
    from .allgather_local_permute_fusion import (
        compute_scatter_idx,
    )
except ImportError:
    from allgather_local_permute_fusion import (  # type: ignore
        compute_scatter_idx,
    )

_NOTIFY_DISPATCH_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "libnotify_dispatch.so"
)
_HAS_NOTIFY_DISPATCH_KERNEL = False
if os.path.exists(_NOTIFY_DISPATCH_LIB_PATH):
    try:
        torch.ops.load_library(_NOTIFY_DISPATCH_LIB_PATH)
        _HAS_NOTIFY_DISPATCH_KERNEL = hasattr(torch.ops.symm_mem, "notify_dispatch")
    except Exception:
        pass


def value_or(value, default):
    return default if value is None else value


def ceil_div(x: int, y: int) -> int:
    return -(-x // y)


def align(x: int, alignment: int) -> int:
    return ceil_div(x, alignment) * alignment


def _owner_expert_range(rank: int, num_experts: int, world_size: int) -> Tuple[int, int]:
    base = num_experts // world_size
    rem = num_experts % world_size
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return start, start + size


class EventHandle:
    """Small compatibility wrapper around a torch.xpu.Event."""

    def __init__(self, event: Optional[torch.xpu.Event] = None):
        self.event = event or torch.xpu.Event()

    def record(self, stream: Optional[torch.xpu.Stream] = None) -> None:
        if stream is None:
            self.event.record()
        else:
            self.event.record(stream)

    def wait(self, stream: Optional[torch.xpu.Stream] = None) -> None:
        if stream is None:
            self.event.wait()
        else:
            stream.wait_event(self.event)


@dataclass
class EventOverlap:
    """Compatibility wrapper returned by dispatch/combine."""

    event: Optional[torch.xpu.Event] = None

    def wait(self) -> None:
        if self.event is not None:
            self.event.wait()


class EPHandle:
    """Communication handle returned by ElasticBuffer.dispatch.

    Only carries the tensors that combine() needs.
    """

    def __init__(
        self,
        num_experts: int,
        num_sms: int,
        num_tokens_per_rank: int,
        scatter_idx: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        self.num_experts = num_experts
        self.num_sms = num_sms
        self.num_tokens_per_rank = num_tokens_per_rank
        # Global [num_tokens_global, topk] scatter index.
        self.scatter_idx = scatter_idx
        # Received topk routing tensors from dispatch, same first dim as recv_x.
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights


class ElasticBuffer:
    """XPU ElasticBuffer facade backed by deepep_dispatch."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_bytes: Optional[int] = None,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
        prefer_overlap_with_compute: bool = True,
        sl_idx: int = 3,
        num_allocated_qps: int = 0,
        num_cpu_timeout_secs: int = 300,
        num_gpu_timeout_secs: int = 100,
        explicitly_destroy: bool = False,
    ):
        self.group = group
        self.rank_idx = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.use_fp8_dispatch = use_fp8_dispatch
        self.deterministic = deterministic
        self.sl_idx = sl_idx
        self.num_cpu_timeout_secs = num_cpu_timeout_secs
        self.num_gpu_timeout_secs = num_gpu_timeout_secs
        self.explicitly_destroy = explicitly_destroy

        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.dispatch_workspace = None
        self.dispatch_workspace_size_bytes = 0
        self.dispatch_rank_buffers_ptr = None
        self.dispatch_topk_workspace = None
        self.dispatch_topk_workspace_size_bytes = 0
        self.dispatch_topk_weight_workspace = None
        self.dispatch_topk_weight_workspace_size_bytes = 0
        self.topk_weight_rank_buffers_ptr = {}
        self.dispatch_global_topk_offset_elems = 0
        self.topk_rank_ptrs = None
        self.num_bytes = value_or(
            num_bytes,
            self.get_buffer_size_hint(
                group=group,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                hidden=hidden,
                num_topk=num_topk,
                use_fp8_dispatch=use_fp8_dispatch,
                allow_hybrid_mode=allow_hybrid_mode,
                allow_multiple_reduction=allow_multiple_reduction,
            ) if hidden > 0 and num_topk > 0 else 0,
        )

        # The XPU wrapper does not depend on the deep_ep CUDA runtime.
        self.runtime = None
        self.nccl_comm_handle = None
        self._combine_backend_stream = None

        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // max(1, self.num_scaleup_ranks)
        self.scaleup_rank_idx = self.rank_idx % max(1, self.num_scaleup_ranks)
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        if self.num_max_tokens_per_rank > 0 and self.hidden > 0:
            # Size must be at least num_max_tokens_per_rank * hidden * elem_size
            # * num_ranks so that deepep_owner_dispatch (which requests
            # hidden_shard.numel() * element_size * world_size) does not trigger
            # a workspace reallocation that would invalidate cached
            # rank_buffers_ptr pointers.
            self.dispatch_workspace_size_bytes = (
                self.num_max_tokens_per_rank
                * self.hidden
                * (1 if self.use_fp8_dispatch else 2)
                * self.num_ranks
            )
            self.dispatch_workspace = symm_mem.get_symm_mem_workspace(
                self.group.group_name,
                min_size=self.dispatch_workspace_size_bytes,
            )
            self.dispatch_rank_buffers_ptr = self._build_dispatch_rank_buffers_ptr(torch.bfloat16)

        if self.num_max_tokens_per_rank > 0 and self.num_topk > 0:
            if not _HAS_NOTIFY_DISPATCH_KERNEL:
                raise RuntimeError(
                    "notify_dispatch kernel is required but not found. "
                    "Please build and load libnotify_dispatch.so."
                )

            # Create dedicated process groups so that symm_mem can resolve
            # distinct group names for the topk and topk_weight workspaces.
            ranks = list(range(self.num_ranks))
            self._topk_group = dist.new_group(ranks, group_desc="topk")
            self._topk_weight_group = dist.new_group(ranks, group_desc="topk_weight")

            # Per-rank slot layout in the shared topk workspace:
            # 1) local topk:  [num_max_tokens_per_rank, num_topk]
            # 2) global topk: [num_max_tokens_per_rank * num_ranks, num_topk]
            local_topk_elems = self.num_max_tokens_per_rank * self.num_topk
            global_topk_elems = self.num_max_tokens_per_rank * self.num_ranks * self.num_topk
            # Layout per-rank slot in bytes:
            # [local_topk | global_topk]
            self.dispatch_global_topk_offset_elems = local_topk_elems
            self.dispatch_topk_workspace_size_bytes = (
                (local_topk_elems + global_topk_elems) * 4
            )
            self.dispatch_topk_workspace = symm_mem.get_symm_mem_workspace(
                self._topk_group.group_name,
                min_size=self.dispatch_topk_workspace_size_bytes,
            )
            self.topk_rank_ptrs = self._build_topk_rank_buffers_ptr(torch.int32)

            # Per-rank topk weight slot layout:
            # [num_max_tokens_per_rank, num_topk]
            self.dispatch_topk_weight_workspace_size_bytes = local_topk_elems * 4
            self.dispatch_topk_weight_workspace = symm_mem.get_symm_mem_workspace(
                self._topk_weight_group.group_name,
                min_size=self.dispatch_topk_weight_workspace_size_bytes,
            )
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.topk_weight_rank_buffers_ptr[dtype] = self._build_topk_weight_rank_buffers_ptr(dtype)

        torch.xpu.synchronize()
        dist.barrier(group=self.group)
        torch.xpu.synchronize()

    def destroy(self) -> None:
        if self.explicitly_destroy:
            self.runtime = None
            self.nccl_comm_handle = None

    @staticmethod
    def get_buffer_size_hint(
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int = 0,
        use_fp8_dispatch: bool = False,
        allow_hybrid_mode: bool = True,
        allow_multiple_reduction: bool = True,
    ) -> int:
        itemsize = 1 if use_fp8_dispatch else 2
        effective_topk = max(1, num_topk)
        return align(num_max_tokens_per_rank * hidden * effective_topk * itemsize, 32)

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int,
        hidden: int,
        num_max_tokens_per_rank: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0
        num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
        return num_bytes_per_entry * (num_entries + num_max_tokens_per_rank)

    @staticmethod
    def get_pp_buffer_size_hint(
        num_max_tensor_bytes: int,
        num_max_inflight_tensors: int,
    ) -> int:
        num_max_tensor_bytes = align(num_max_tensor_bytes, 32)
        return num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2

    @staticmethod
    def get_agrs_buffer_size_hint(group: dist.ProcessGroup, num_max_session_bytes: int) -> int:
        return num_max_session_bytes

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
        dist.barrier(group=self.group)
        torch.xpu.synchronize()

    @staticmethod
    def capture() -> EventHandle:
        return EventHandle()

    def get_comm_stream(self) -> torch.xpu.Stream:
        return torch.xpu.Stream()

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return 1, self.num_ranks

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return 1, self.num_ranks

    def engram_write(self, storage: torch.Tensor) -> None:
        raise NotImplementedError("Engram is not implemented for elastic_xpu yet")

    def engram_fetch(self, indices: torch.Tensor, num_qps: int = 0) -> Callable:
        raise NotImplementedError("Engram fetch is not implemented for elastic_xpu yet")

    def pp_set_config(self, num_max_tensor_bytes: int, num_max_inflight_tensors: int):
        raise NotImplementedError("PP send/recv is not implemented for elastic_xpu yet")

    def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
        raise NotImplementedError("PP send is not implemented for elastic_xpu yet")

    def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
        raise NotImplementedError("PP recv is not implemented for elastic_xpu yet")

    def create_agrs_session(self) -> None:
        return None

    def destroy_agrs_session(self) -> None:
        return None

    @contextmanager
    def agrs_new_session(self, enabled: bool = True):
        if not enabled:
            yield
            return
        self.create_agrs_session()
        try:
            yield
        finally:
            self.destroy_agrs_session()

    def agrs_set_config(self, num_max_session_bytes: int, num_max_all_gathers_per_session: int) -> None:
        return None

    # noinspection PyTypeChecker
    def agrs_get_inplace_tensor(
        self,
        shapes: Union[Tuple[int, ...], torch.Size, Sequence[Union[Tuple[int, ...], torch.Size]]],
        dtype: torch.dtype,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_batched_mode = isinstance(shapes[0], tuple)
        if not is_batched_mode:
            shapes = (shapes,)  # type: ignore[assignment]
        tensors = tuple(torch.empty(shape, device="xpu", dtype=dtype) for shape in shapes)
        return tensors if is_batched_mode else tensors[0]

    def all_gather(self, t: Union[torch.Tensor, Sequence[torch.Tensor]]):
        if isinstance(t, torch.Tensor):
            gathered = [torch.empty_like(t) for _ in range(self.num_ranks)]
            dist.all_gather(gathered, t, group=self.group)
            return torch.stack(gathered, dim=0), lambda: None

        gathered_list = []
        for tensor in t:
            rank_tensors = [torch.empty_like(tensor) for _ in range(self.num_ranks)]
            dist.all_gather(rank_tensors, tensor, group=self.group)
            gathered_list.append(torch.stack(rank_tensors, dim=0))
        return *gathered_list, (lambda: None)

    def get_theoretical_num_sms(
        self,
        num_experts: int,
        num_topk: int,
        num_scaleout_topk: int = 0,
        rdma_gbs: float = 0,
        nvlink_gbs: float = 0,
        sm_read_gbs: float = 200,
        sm_write_gbs: float = 50,
    ) -> int:
        if self.num_ranks <= 1:
            return 4
        return align(max(4, min(64, self.num_ranks * 4)), 2)

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        return max(1, num_sms)

    def _build_topk_rank_buffers_ptr(self, topk_dtype: torch.dtype) -> torch.Tensor:
        if self.dispatch_topk_workspace is None:
            raise RuntimeError("dispatch topk workspace is not initialized")

        ptr_list = []
        for r in range(self.num_ranks):
            buf = self.dispatch_topk_workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.num_topk),
                topk_dtype,
                storage_offset=0,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}")

    def _build_dispatch_rank_buffers_ptr(self, hidden_dtype: torch.dtype) -> torch.Tensor:
        if self.dispatch_workspace is None:
            raise RuntimeError("dispatch workspace is not initialized")

        ptr_list = []
        for r in range(self.num_ranks):
            buf = self.dispatch_workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.hidden),
                hidden_dtype,
                storage_offset=0,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}")

    def _build_topk_weight_rank_buffers_ptr(self, topk_weight_dtype: torch.dtype) -> torch.Tensor:
        if self.dispatch_topk_weight_workspace is None:
            raise RuntimeError("dispatch topk weight workspace is not initialized")

        ptr_list = []
        for r in range(self.num_ranks):
            buf = self.dispatch_topk_weight_workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.num_topk),
                topk_weight_dtype,
                storage_offset=0,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}")

    def _all_gather_concat_local_shard(self, local_tensor: torch.Tensor) -> torch.Tensor:
        local = local_tensor.contiguous()
        local_flat = local.view(-1)
        numel_per_rank = local_flat.numel()

        workspace_size_bytes = numel_per_rank * self.num_ranks * local_flat.element_size()
        workspace = symm_mem.get_symm_mem_workspace(
            f"{self.group.group_name}_elastic_gather",
            min_size=workspace_size_bytes,
        )

        local_slot = workspace.get_buffer(
            self.rank_idx,
            (numel_per_rank,),
            local_flat.dtype,
            storage_offset=self.rank_idx * numel_per_rank,
        )
        local_slot.copy_(local_flat)
        workspace.barrier()

        gathered_flat = torch.empty(
            numel_per_rank * self.num_ranks,
            device=local.device,
            dtype=local.dtype,
        )
        for r in range(self.num_ranks):
            remote_slot = workspace.get_buffer(
                r,
                (numel_per_rank,),
                local_flat.dtype,
                storage_offset=r * numel_per_rank,
            )
            start = r * numel_per_rank
            gathered_flat[start:start + numel_per_rank].copy_(remote_slot)

        workspace.barrier()

        return gathered_flat.view(self.num_ranks, *local.shape).reshape(
            self.num_ranks * local.shape[0], *local.shape[1:]
        )

    def allgather_local_permute_fusion(
        self,
        hidden_shard: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        scatter_idx: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        remap_hidden_states: Optional[torch.Tensor] = None,
        backend_stream: Optional[torch.xpu.Stream] = None,
        rank_buffers_ptr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens_per_rank = hidden_shard.shape[0]
        if topk_idx.dtype != torch.int32:
            raise ValueError(f"topk_idx must be torch.int32, but got {topk_idx.dtype}")
        if topk_idx.shape[0] != num_tokens_per_rank:
            raise ValueError(
                f"topk_idx shape[0] ({topk_idx.shape[0]}) must equal "
                f"hidden_shard shape[0] ({num_tokens_per_rank})"
            )
        if topk_weights.shape != topk_idx.shape:
            raise ValueError(
                f"topk_weights shape ({topk_weights.shape}) must equal "
                f"topk_idx shape ({topk_idx.shape})"
            )

        global_topk_idx = self._all_gather_concat_local_shard(topk_idx)
        global_topk_weights = self._all_gather_concat_local_shard(topk_weights)
        num_tokens, topk = global_topk_idx.shape
        if num_tokens != num_tokens_per_rank * self.num_ranks:
            raise ValueError(
                f"global topk rows ({num_tokens}) must equal "
                f"num_tokens_per_rank * world_size ({num_tokens_per_rank * self.num_ranks})"
            )
        if global_topk_weights.shape != global_topk_idx.shape:
            raise ValueError(
                f"global topk_weights shape {global_topk_weights.shape} must match "
                f"global topk_idx shape {global_topk_idx.shape}"
            )

        if scatter_idx is None:
            scatter_idx, _ = compute_scatter_idx(global_topk_idx, num_experts=num_experts)
        elif scatter_idx.shape != global_topk_idx.shape:
            raise ValueError(
                f"scatter_idx shape {scatter_idx.shape} must match global topk shape "
                f"{global_topk_idx.shape}"
            )
        if scatter_idx.dtype != torch.int32:
            raise ValueError(
                f"scatter_idx must be torch.int32, but got {scatter_idx.dtype}"
            )

        if remap_hidden_states is None:
            remap_hidden_states = torch.empty(
                (num_tokens * topk, hidden_shard.shape[1]),
                device=hidden_shard.device,
                dtype=hidden_shard.dtype,
            )
        elif remap_hidden_states.shape != (num_tokens * topk, hidden_shard.shape[1]):
            raise ValueError(
                f"remap_hidden_states shape {remap_hidden_states.shape} must be "
                f"({num_tokens * topk}, {hidden_shard.shape[1]})"
            )

        num_tokens_per_rank_local, hidden_size = hidden_shard.shape
        workspace_size_bytes = (
            hidden_shard.numel() * hidden_shard.element_size() * self.num_ranks
        )
        workspace = symm_mem.get_symm_mem_workspace(
            f"{self.group.group_name}_allgather_permute",
            min_size=workspace_size_bytes,
        )

        local_offset = self.rank_idx * num_tokens_per_rank_local * hidden_size
        local_slot = workspace.get_buffer(
            self.rank_idx,
            (num_tokens_per_rank_local, hidden_size),
            hidden_shard.dtype,
            storage_offset=local_offset,
        )
        local_slot.copy_(hidden_shard)
        workspace.barrier()

        has_fused_kernel = hasattr(torch.ops.symm_mem, "allgather_permute")
        if not has_fused_kernel:
            raise RuntimeError(
                "allgather_local_permute_fusion requires torch.ops.symm_mem.allgather_permute"
            )

        if rank_buffers_ptr is None:
            ptr_list = []
            for r in range(self.num_ranks):
                if r == self.rank_idx:
                    ptr_list.append(hidden_shard.data_ptr())
                else:
                    remote_offset = r * num_tokens_per_rank_local * hidden_size
                    remote_buf = workspace.get_buffer(
                        r,
                        (num_tokens_per_rank_local, hidden_size),
                        hidden_shard.dtype,
                        storage_offset=remote_offset,
                    )
                    ptr_list.append(remote_buf.data_ptr())
            signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
            rank_buffers_ptr = torch.tensor(
                signed_ptrs,
                dtype=torch.int64,
                device=hidden_shard.device,
            )

        torch.ops.symm_mem.allgather_permute(
            rank_buffers_ptr,
            scatter_idx,
            remap_hidden_states,
            self.rank_idx,
            self.num_ranks,
        )
        out = remap_hidden_states

        workspace.barrier()

        recv_topk_idx = torch.full(
            (num_tokens * topk, topk),
            -1,
            device=global_topk_idx.device,
            dtype=global_topk_idx.dtype,
        )
        recv_topk_weights = torch.zeros(
            (num_tokens * topk, topk),
            device=global_topk_weights.device,
            dtype=global_topk_weights.dtype,
        )
        flat_scatter = scatter_idx.reshape(-1)
        flat_k = torch.arange(
            topk, device=global_topk_idx.device, dtype=torch.int32
        ).view(1, -1).expand(num_tokens, -1).reshape(-1)
        recv_topk_idx[flat_scatter, flat_k] = global_topk_idx.reshape(-1)
        recv_topk_weights[flat_scatter, flat_k] = global_topk_weights.reshape(-1)

        return out, scatter_idx, recv_topk_idx, recv_topk_weights

    def unpermute_reducescatter_fusion(
        self,
        expert_output: torch.Tensor,
        scatter_idx: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        recv_topk_idx: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        backend_stream: Optional[torch.xpu.Stream] = None,
        rank_buffers_ptr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens, _ = scatter_idx.shape
        if scatter_idx.dtype != torch.int32:
            raise ValueError(
                f"scatter_idx must be torch.int32, but got {scatter_idx.dtype}"
            )
        if num_tokens % self.num_ranks != 0:
            raise ValueError(
                f"scatter_idx rows ({num_tokens}) must be divisible by world_size ({self.num_ranks})"
            )
        num_tokens_per_rank = num_tokens // self.num_ranks

        if recv_topk_weights is not None:
            if recv_topk_weights.shape != (num_tokens * scatter_idx.shape[1], scatter_idx.shape[1]):
                raise ValueError(
                    f"recv_topk_weights shape {recv_topk_weights.shape} must be "
                    f"({num_tokens * scatter_idx.shape[1]}, {scatter_idx.shape[1]})"
                )
            if recv_topk_idx is not None and recv_topk_idx.shape != recv_topk_weights.shape:
                raise ValueError(
                    f"recv_topk_idx shape {recv_topk_idx.shape} must match "
                    f"recv_topk_weights shape {recv_topk_weights.shape}"
                )

            topk = scatter_idx.shape[1]
            gather_rows = scatter_idx
            gather_k = torch.arange(
                topk, device=scatter_idx.device, dtype=torch.int32
            ).view(1, -1).expand(num_tokens, -1)
            global_topk_weights = recv_topk_weights[gather_rows, gather_k].contiguous()

        elif topk_weights is not None and topk_weights.shape[0] == num_tokens_per_rank:
            global_topk_weights = self._all_gather_concat_local_shard(topk_weights)
        elif topk_weights is not None and topk_weights.shape[0] == num_tokens:
            global_topk_weights = topk_weights.contiguous()
        else:
            raise ValueError(
                "Either recv_topk_weights must be provided, or topk_weights must "
                f"have shape[0] equal to num_tokens_per_rank ({num_tokens_per_rank}) "
                f"or num_tokens ({num_tokens})"
            )

        if global_topk_weights.shape != scatter_idx.shape:
            raise ValueError(
                f"global topk_weights shape {global_topk_weights.shape} must match "
                f"scatter_idx shape {scatter_idx.shape}"
            )

        if output is None:
            output = torch.zeros(
                num_tokens_per_rank,
                expert_output.shape[1],
                device=expert_output.device,
                dtype=expert_output.dtype,
            )
        elif output.shape != (num_tokens_per_rank, expert_output.shape[1]):
            raise ValueError(
                f"output shape {output.shape} must be "
                f"({num_tokens_per_rank}, {expert_output.shape[1]})"
            )

        _ = rank_buffers_ptr
        hidden = expert_output.shape[1]
        workspace_size_bytes = (
            self.num_ranks * num_tokens_per_rank * hidden * expert_output.element_size()
        )
        workspace = symm_mem.get_symm_mem_workspace(
            f"{self.group.group_name}_unpermute_rs",
            min_size=workspace_size_bytes,
        )

        has_local_unpermute_kernel = hasattr(torch.ops.symm_mem, "local_unpermute_copy_")

        if has_local_unpermute_kernel:
            my_chunk_start = self.rank_idx * num_tokens_per_rank
            output.zero_()
            torch.ops.symm_mem.local_unpermute_copy_(
                expert_output,
                scatter_idx,
                global_topk_weights,
                my_chunk_start,
                num_tokens_per_rank,
                output,
            )

            if backend_stream is None:
                backend_stream = torch.xpu.Stream()

            local_bufs = [
                torch.empty(
                    num_tokens_per_rank,
                    hidden,
                    device=expert_output.device,
                    dtype=expert_output.dtype,
                ),
                torch.empty(
                    num_tokens_per_rank,
                    hidden,
                    device=expert_output.device,
                    dtype=expert_output.dtype,
                ),
            ]

            for step in range(self.num_ranks - 1):
                target_rank = (self.rank_idx - step - 1) % self.num_ranks
                buf_idx = step % 2
                stream = backend_stream if step % 2 == 0 else torch.xpu.current_stream()
                with torch.xpu.stream(stream):
                    target_chunk_start = target_rank * num_tokens_per_rank
                    torch.ops.symm_mem.local_unpermute_copy_(
                        expert_output,
                        scatter_idx,
                        global_topk_weights,
                        target_chunk_start,
                        num_tokens_per_rank,
                        local_bufs[buf_idx],
                    )

                    target_recv_buf = workspace.get_buffer(
                        target_rank,
                        (self.num_ranks, num_tokens_per_rank, hidden),
                        expert_output.dtype,
                        storage_offset=0,
                    )
                    target_recv_buf[self.rank_idx].copy_(local_bufs[buf_idx])

            torch.xpu.current_stream().wait_stream(backend_stream)
            workspace.barrier()

            my_recv_buf = workspace.get_buffer(
                self.rank_idx,
                (self.num_ranks, num_tokens_per_rank, hidden),
                expert_output.dtype,
                storage_offset=0,
            )
            for i in range(self.num_ranks):
                if i != self.rank_idx:
                    output.add_(my_recv_buf[i])

            workspace.barrier()
            return output, global_topk_weights

        my_chunk_start = self.rank_idx * num_tokens_per_rank
        output.zero_()
        for i in range(num_tokens_per_rank):
            global_i = my_chunk_start + i
            for k in range(scatter_idx.shape[1]):
                src_row = scatter_idx[global_i, k].item()
                output[i] += global_topk_weights[global_i, k] * expert_output[src_row]

        if backend_stream is None:
            backend_stream = torch.xpu.Stream()

        local_bufs = [
            torch.zeros(
                num_tokens_per_rank,
                hidden,
                device=expert_output.device,
                dtype=expert_output.dtype,
            ),
            torch.zeros(
                num_tokens_per_rank,
                hidden,
                device=expert_output.device,
                dtype=expert_output.dtype,
            ),
        ]

        for step in range(self.num_ranks - 1):
            target_rank = (self.rank_idx - step - 1) % self.num_ranks
            buf_idx = step % 2
            stream = backend_stream if step % 2 == 0 else torch.xpu.current_stream()
            with torch.xpu.stream(stream):
                target_chunk_start = target_rank * num_tokens_per_rank
                local_bufs[buf_idx].zero_()
                for i in range(num_tokens_per_rank):
                    global_i = target_chunk_start + i
                    for k in range(scatter_idx.shape[1]):
                        src_row = scatter_idx[global_i, k].item()
                        local_bufs[buf_idx][i] += (
                            global_topk_weights[global_i, k] * expert_output[src_row]
                        )

                target_recv_buf = workspace.get_buffer(
                    target_rank,
                    (self.num_ranks, num_tokens_per_rank, hidden),
                    expert_output.dtype,
                    storage_offset=0,
                )
                target_recv_buf[self.rank_idx].copy_(local_bufs[buf_idx])

        torch.xpu.current_stream().wait_stream(backend_stream)
        workspace.barrier()

        my_recv_buf = workspace.get_buffer(
            self.rank_idx,
            (self.num_ranks, num_tokens_per_rank, hidden),
            expert_output.dtype,
            storage_offset=0,
        )
        for i in range(self.num_ranks):
            if i != self.rank_idx:
                output.add_(my_recv_buf[i])

        workspace.barrier()
        return output, global_topk_weights

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], # [num_tokens_per_rank, hidden]
        topk_idx: Optional[torch.Tensor] = None, # [num_tokens_per_rank, topk]
        topk_weights: Optional[torch.Tensor] = None, # [num_tokens_per_rank, topk]
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: Optional[EventHandle] = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
        handle: Optional[EPHandle] = None,
        do_handle_copy: bool = False,
        do_cpu_sync: Optional[bool] = None,
        do_expand: bool = True,
        use_tma_aligned_col_major_sf: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor], Optional[EPHandle], Optional[EventOverlap]]:
        # starting from 16bit
        if isinstance(x, tuple):
            raise NotImplementedError("FP8 dispatch is not implemented for elastic_xpu yet")

        if not do_expand:
            raise NotImplementedError("not do_expand layout is not supported in MOE kernel")

        # if previous_event is not None:
        #     previous_event.wait()

        if topk_idx is None:
            raise ValueError("topk_idx must not be None")
        if topk_weights is None:
            raise ValueError("topk_weights must not be None")
        if num_experts is None:
            raise ValueError("num_experts must not be None")

        # Handle reuse is disabled for now by design.
        do_cpu_sync = value_or(do_cpu_sync, True)

        x = x.contiguous()
        topk_weights = topk_weights.contiguous()

        num_tokens_per_rank, topk = topk_idx.shape
        hidden = x.shape[1]
        if num_tokens_per_rank != x.shape[0]:
            raise ValueError(
                f"topk_idx shape[0] ({num_tokens_per_rank}) must equal "
                f"x shape[0] ({x.shape[0]})"
            )
        if topk_weights.shape != topk_idx.shape:
            raise ValueError(
                f"topk_weights shape ({topk_weights.shape}) must equal "
                f"topk_idx shape ({topk_idx.shape})"
            )
        num_tokens = num_tokens_per_rank * self.num_ranks

        # Stage 1 (notify_dispatch):
        # 1) local rank writes topk into symmetric topk workspace
        # 2) fused kernel reads remote topk, builds global scatter_idx
        # 3) fused kernel fills local psum_num_recv_tokens_per_expert

        if self.dispatch_workspace is None:
            raise RuntimeError(
                "dispatch workspace is not initialized. Please construct ElasticBuffer with "
                "valid num_max_tokens_per_rank and hidden."
            )
        if num_tokens_per_rank > self.num_max_tokens_per_rank:
            raise ValueError(
                f"x tokens per rank ({num_tokens_per_rank}) exceeds num_max_tokens_per_rank "
                f"({self.num_max_tokens_per_rank})"
            )
        if hidden != self.hidden:
            raise ValueError(
                f"x hidden ({hidden}) does not match ElasticBuffer.hidden ({self.hidden})"
            )
        if self.dispatch_topk_workspace is None:
            raise RuntimeError(
                "topk dispatch workspace is not initialized. Please construct ElasticBuffer with "
                "valid num_max_tokens_per_rank and num_topk."
            )
        if topk_idx.dtype != torch.int32:
            raise ValueError(
                f"topk_idx must be torch.int32 (4-byte), but got {topk_idx.dtype}"
            )
        if topk > self.num_topk:
            raise ValueError(
                f"topk ({topk}) exceeds ElasticBuffer.num_topk ({self.num_topk})"
            )

        local_slot = self.dispatch_workspace.get_buffer(
            self.rank_idx,
            (self.num_max_tokens_per_rank, self.hidden),
            x.dtype,
            storage_offset=0,
        )
        local_slot[:num_tokens_per_rank].copy_(x)

        topk_slot = self.dispatch_topk_workspace.get_buffer(
            self.rank_idx,
            (self.num_max_tokens_per_rank, self.num_topk),
            topk_idx.dtype,
            storage_offset=0,
        )
        topk_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)

        if self.dispatch_topk_weight_workspace is None:
            raise RuntimeError("topk weight dispatch workspace is not initialized")
        topk_weight_slot = self.dispatch_topk_weight_workspace.get_buffer(
            self.rank_idx,
            (self.num_max_tokens_per_rank, self.num_topk),
            topk_weights.dtype,
            storage_offset=0,
        )
        topk_weight_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)

        # A single barrier on any workspace is sufficient – all three
        # workspaces share the same process group, so one barrier
        # guarantees that every rank's copies are visible to all peers.
        self.dispatch_workspace.barrier()

        local_expert_start, local_expert_end = _owner_expert_range(
            self.rank_idx, num_experts, self.num_ranks
        )
        num_local_experts = local_expert_end - local_expert_start

        scatter_idx = torch.empty(
            (num_tokens, topk), device=topk_idx.device, dtype=torch.int32
        )
        global_topk_idx = self.dispatch_topk_workspace.get_buffer(
            self.rank_idx,
            (num_tokens, topk),
            torch.int32,
            storage_offset=self.dispatch_global_topk_offset_elems,
        )
        psum_num_recv_tokens_per_expert = torch.zeros(
            (num_local_experts,), device=topk_idx.device, dtype=torch.int32
        )

        if self.topk_rank_ptrs is None:
            raise RuntimeError("topk_rank_ptrs is not initialized")

        torch.ops.symm_mem.notify_dispatch(
            self.topk_rank_ptrs,
            global_topk_idx,
            scatter_idx,
            psum_num_recv_tokens_per_expert,
            num_tokens_per_rank,
            topk,
            self.num_topk,
            num_experts,
            self.rank_idx,
            self.num_ranks,
        )

        if num_max_tokens_per_rank is not None:
            # Requested layout: num_max_tokens_per_rank * num_tokens * hidden.
            remap_rows = num_max_tokens_per_rank * num_tokens
        else:
            if do_cpu_sync:
                # Per user requirement: read psum on CPU and derive remap_rows.
                psum_cpu = psum_num_recv_tokens_per_expert.to("cpu")
                remap_rows = int(psum_cpu.sum().item())
            else:
                # Without CPU sync we cannot read the exact psum, but the
                # ep_dispatch kernel output is always bounded by
                # dispatch_rows (= num_tokens * topk).  Using 0 here lets
                # work_rows = max(0, dispatch_rows) = dispatch_rows, which
                # is the correct upper-bound allocation.
                remap_rows = 0

        # ep_dispatch currently requires output rows == num_tokens * topk.
        # Keep an internal work buffer that satisfies the kernel contract.
        dispatch_rows = num_tokens * topk
        work_rows = max(remap_rows, dispatch_rows)
        remap_hidden_states = torch.empty(work_rows, hidden, device=x.device, dtype=x.dtype)
        dispatch_view = remap_hidden_states[:dispatch_rows]

        # Build receive-layout topk tensors (shape differs from input topk layout).
        recv_topk_idx = torch.empty(
            (work_rows, topk),
            device=topk_idx.device,
            dtype=topk_idx.dtype,
        )
        recv_topk_weights = torch.empty(
            (work_rows, topk),
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )

        rank_buffers_ptr = self.dispatch_rank_buffers_ptr
        if rank_buffers_ptr is None:
            raise RuntimeError("rank_buffers_ptr is not initialized")
        topk_weight_rank_buffers_ptr = self.topk_weight_rank_buffers_ptr.get(topk_weights.dtype)
        if topk_weight_rank_buffers_ptr is None:
            raise RuntimeError(
                f"topk_weight_rank_buffers_ptr is not initialized for dtype {topk_weights.dtype}"
            )

        # Stage 2: run ep_dispatch kernel directly (bypass deepep_owner_dispatch
        # to skip its 2 redundant barriers — elastic already barrier'd above).
        torch.ops.symm_mem.ep_dispatch(
            rank_buffers_ptr,
            global_topk_idx,
            topk_weight_rank_buffers_ptr,
            scatter_idx,
            dispatch_view,
            recv_topk_idx,
            recv_topk_weights,
            self.num_topk,
            num_experts,
            self.rank_idx,
            self.num_ranks,
        )

        # Keep scatter_idx in handle so combine can consume it directly.
        scatter_idx_for_handle = scatter_idx.clone() if do_handle_copy else scatter_idx

        handle_obj = EPHandle(
            num_experts=num_experts,
            num_sms=num_sms,
            num_tokens_per_rank=num_tokens_per_rank,
            scatter_idx=scatter_idx_for_handle,
            topk_idx=recv_topk_idx,
            topk_weights=recv_topk_weights,
        )

        return remap_hidden_states, recv_topk_idx, recv_topk_weights, handle_obj, None

    @staticmethod
    def _unpack_bias(
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def combine(
        self,
        x: torch.Tensor,  # [num_recv_tokens, hidden]
        handle: EPHandle,
        topk_weights: Optional[torch.Tensor] = None,  # [num_recv_tokens, topk]
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
        num_sms: int = 0,
        num_qps: int = 0,
        previous_event: EventHandle = None,
        previous_event_before_epilogue: Optional[EventHandle] = None,
        async_with_compute_stream: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        if previous_event is not None:
            previous_event.wait()

        num_sms = handle.num_sms if num_sms == 0 else num_sms

        scatter_idx = handle.scatter_idx
        num_tokens_per_rank = handle.num_tokens_per_rank
        topk = handle.topk_idx.shape[1]
        num_tokens = num_tokens_per_rank * self.num_ranks

        if topk_weights is not None and topk_weights.shape[0] != x.shape[0]:
            raise ValueError(
                f"topk_weights shape[0] ({topk_weights.shape[0]}) must match "
                f"x shape[0] ({x.shape[0]})"
            )

        # Reconstruct global topk routing from symmetric memory for the kernel.
        global_topk_idx = self.dispatch_topk_workspace.get_buffer(
            self.rank_idx,
            (num_tokens, topk),
            torch.int32,
            storage_offset=self.dispatch_global_topk_offset_elems,
        )
        global_topk_weights = torch.empty(
            num_tokens, topk, device=x.device, dtype=handle.topk_weights.dtype
        )
        for r in range(self.num_ranks):
            remote_weight_slot = self.dispatch_topk_weight_workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.num_topk),
                handle.topk_weights.dtype,
                storage_offset=0,
            )
            start = r * num_tokens_per_rank
            global_topk_weights[start:start + num_tokens_per_rank, :topk].copy_(
                remote_weight_slot[:num_tokens_per_rank, :topk]
            )

        hidden = x.shape[1]
        output = torch.zeros(num_tokens_per_rank, hidden, device=x.device, dtype=x.dtype)

        deepep_owner_combine(
            expert_output=x.contiguous(),
            topk_idx=global_topk_idx,
            scatter_idx=scatter_idx,
            topk_weights=global_topk_weights.contiguous(),
            output=output,
            num_experts=handle.num_experts,
            group=self.group,
            backend_stream=self._combine_backend_stream,
        )

        bias_0, bias_1 = self._unpack_bias(bias)
        if bias_0 is not None:
            output.add_(bias_0)
        if bias_1 is not None:
            output.add_(bias_1)

        # Return per-rank local topk_weights from symmetric memory.
        local_weight_slot = self.dispatch_topk_weight_workspace.get_buffer(
            self.rank_idx,
            (self.num_max_tokens_per_rank, self.num_topk),
            handle.topk_weights.dtype,
            storage_offset=0,
        )
        combined_topk_weights = local_weight_slot[:num_tokens_per_rank, :topk].clone()

        event = torch.xpu.Event() if async_with_compute_stream else None
        if event is not None:
            event.record()
        return output, combined_topk_weights, EventOverlap(event)

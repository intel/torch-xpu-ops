"""TP+EP owner-based dispatch reference implementation.

This module implements the algorithm from
docs/allgather+local permute fusion.md (TP+EP section):

1. Compute owner rank for each expert.
2. Each rank writes its hidden_shard to symmetric memory.
3. Single kernel launch: every (token, k) pair checks expert ownership,
   reads from the source rank's symmetric memory only if needed,
   and writes into remap_hidden_states.

The C++ kernel (EpDispatch.cpp) does the actual computation.
A Python fallback is provided for environments where the kernel is not built.
"""

from __future__ import annotations

import ctypes
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Try to load the native kernel
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "libep_dispatch.so")
_HAS_NATIVE_KERNEL = False
if os.path.exists(_LIB_PATH):
        try:
                torch.ops.load_library(_LIB_PATH)
                _HAS_NATIVE_KERNEL = hasattr(torch.ops.symm_mem, "ep_dispatch")
        except Exception:
                pass

_COMBINE_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "libep_combine.so")
_HAS_EP_COMBINE_KERNEL = False
if os.path.exists(_COMBINE_LIB_PATH):
        try:
                torch.ops.load_library(_COMBINE_LIB_PATH)
                _HAS_EP_COMBINE_KERNEL = hasattr(torch.ops.symm_mem, "ep_combine")
        except Exception:
                pass

_HAS_EP_COMBINE_LOCAL = False
if _HAS_EP_COMBINE_KERNEL:
        _HAS_EP_COMBINE_LOCAL = hasattr(torch.ops.symm_mem, "ep_combine_local_")


def get_owner_expert_ranges(num_experts: int, tp_world_size: int) -> List[Tuple[int, int]]:
        """Return contiguous expert ranges [start, end) owned by each TP rank."""
        if num_experts <= 0:
                raise ValueError("num_experts must be > 0")
        if tp_world_size <= 0:
                raise ValueError("tp_world_size must be > 0")

        base = num_experts // tp_world_size
        rem = num_experts % tp_world_size

        ranges: List[Tuple[int, int]] = []
        start = 0
        for rank in range(tp_world_size):
                size = base + (1 if rank < rem else 0)
                end = start + size
                ranges.append((start, end))
                start = end
        return ranges


def get_expert_owner(expert_id: int, num_experts: int, tp_world_size: int) -> int:
        """Map an expert id to its owner rank."""
        if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(f"expert_id out of range: {expert_id}")
        ranges = get_owner_expert_ranges(num_experts, tp_world_size)
        for owner, (start, end) in enumerate(ranges):
                if start <= expert_id < end:
                        return owner
        raise RuntimeError("Failed to resolve owner for expert")


def deepep_owner_dispatch(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    num_experts: int,
    scatter_idx: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    rank_buffers_ptr: torch.Tensor = None,
    skip_copy: bool = False,
):
        """
        TP+EP owner-based dispatch using symmetric memory.

        Each rank writes its hidden_shard to symmetric memory, then a single
        ring-ordered kernel reads from all source ranks with coalesced access
        and writes to owned positions in remap_hidden_states.

        Args:
            scatter_idx: [num_tokens, topk] int32 - pre-computed expert-sorted
                write positions (from compute_scatter_idx).
            rank_buffers_ptr: Optional precomputed device tensor of per-rank
                buffer pointers (int64). Pass this to avoid per-call overhead
                when hidden_shard and workspace are stable across calls.
            skip_copy: If True, assume hidden_shard is already written to
                the symmetric memory workspace (e.g., by a preceding matmul).
                NOTE: the barrier mechanism may require the copy; use with caution.
        """
        if group is None:
                group = dist.group.WORLD
        if group_name is None:
                group_name = group.group_name
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        num_tokens_per_rank, hidden_size = hidden_shard.shape
        num_tokens, topk = topk_idx.shape
        assert num_tokens % world_size == 0
        assert num_tokens_per_rank == num_tokens // world_size

        workspace_size_bytes = hidden_shard.numel() * hidden_shard.element_size() * world_size
        workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

        # Write local hidden_shard at rank-specific offset
        local_offset = rank * num_tokens_per_rank * hidden_size
        local_slot = workspace.get_buffer(
                rank, (num_tokens_per_rank, hidden_size),
                hidden_shard.dtype, storage_offset=local_offset,
        )
        if not skip_copy:
                local_slot.copy_(hidden_shard)
        workspace.barrier()

        if _HAS_NATIVE_KERNEL:
                if rank_buffers_ptr is None:
                        # Collect per-rank buffer pointers for the kernel
                        ptr_list = []
                        for r in range(world_size):
                                if r == rank:
                                        ptr_list.append(hidden_shard.data_ptr())
                                else:
                                        offset = r * num_tokens_per_rank * hidden_size
                                        buf = workspace.get_buffer(
                                                r, (num_tokens_per_rank, hidden_size),
                                                hidden_shard.dtype, storage_offset=offset,
                                        )
                                        ptr_list.append(buf.data_ptr())
                        # Device pointers may exceed signed int64 range; wrap to preserve bits
                        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
                        rank_buffers_ptr = torch.tensor(
                                signed_ptrs, dtype=torch.int64,
                        ).to(hidden_shard.device)

                torch.ops.symm_mem.ep_dispatch(
                        rank_buffers_ptr, topk_idx, scatter_idx,
                        remap_hidden_states, num_experts, rank, world_size,
                )
        else:
                # Python fallback
                for step in range(world_size):
                        remote_rank = (rank + step) % world_size
                        if remote_rank == rank:
                                src_buffer = hidden_shard
                        else:
                                remote_offset = remote_rank * num_tokens_per_rank * hidden_size
                                src_buffer = workspace.get_buffer(
                                        remote_rank, (num_tokens_per_rank, hidden_size),
                                        hidden_shard.dtype, storage_offset=remote_offset,
                                )
                        remote_token_offset = remote_rank * num_tokens_per_rank
                        for i in range(num_tokens_per_rank):
                                global_token_idx = remote_token_offset + i
                                for k in range(topk):
                                        expert = int(topk_idx[global_token_idx, k].item())
                                        owner = get_expert_owner(expert, num_experts, world_size)
                                        if owner == rank:
                                                dst = int(scatter_idx[global_token_idx, k].item())
                                                remap_hidden_states[dst].copy_(src_buffer[i])

        workspace.barrier()
        return remap_hidden_states


def deepep_owner_combine(
    expert_output: torch.Tensor,
    topk_idx: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
        rank_output_ptrs: torch.Tensor = None,
):
        """
        TP+EP owner-based combine (DeepEP style) using symmetric memory.

        Pipeline approach (mirrors unpermute_reducescatter_fusion_native):
          1. Compute own chunk with local_unpermute_copy_ → write to output directly
          2. For each remote target rank (two-stream round-robin):
             - Compute that rank's chunk with local_unpermute_copy_ → local temp buf
             - Push temp buf to target rank's symmetric memory receive slot
          3. Barrier (ensure all pushes visible)
          4. Sum received contributions from all other ranks into output
          5. Barrier (ensure reads complete)

        Because expert_output has zeros for non-owned expert rows, local_unpermute_copy_
        automatically produces the correct partial weighted sum (only owned experts
        contribute non-zero values).

        Workspace layout per rank: [world_size, num_tokens_per_rank, hidden]
          slot[i] = partial contribution FROM rank i for this rank's token chunk.

        Args:
            expert_output: [num_tokens * topk, hidden] expert-centric output.
                On each rank, rows for non-owned experts are zeros.
            topk_idx: [num_tokens, topk] int64 - expert assignment per (token, k).
            scatter_idx: [num_tokens, topk] int32 - maps (token, k) to expert_output row.
            topk_weights: [num_tokens, topk] float32 routing weights.
            output: [num_tokens_per_rank, hidden] output shard for this rank.
            num_experts: total number of experts.
            group: TP process group.
            group_name: optional symmetric memory group name.
            backend_stream: optional second stream for pipelining.
            rank_output_ptrs: unused (kept for API compatibility).
        """
        if group is None:
                group = dist.group.WORLD
        if group_name is None:
                group_name = group.group_name
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        num_tokens, topk = topk_idx.shape
        hidden = expert_output.shape[1]
        num_tokens_per_rank = num_tokens // world_size
        assert output.shape[0] == num_tokens_per_rank
        assert output.shape[1] == hidden

        if _HAS_EP_COMBINE_LOCAL:
                # Pipeline path: ownership-filtered native compute + per-chunk push.
                # ep_combine_local_ skips non-owned expert reads, saving ~75% HBM BW.
                recv_workspace_size = (
                        world_size * num_tokens_per_rank * hidden * expert_output.element_size()
                )
                recv_workspace = symm_mem.get_symm_mem_workspace(
                        group_name, min_size=recv_workspace_size
                )

                topk_weights_f32 = topk_weights.float()

                # Phase 1: Compute own chunk → output directly
                my_chunk_start = rank * num_tokens_per_rank
                output.zero_()  # kernel skips tokens with no owned experts
                torch.ops.symm_mem.ep_combine_local_(
                        expert_output, topk_idx, scatter_idx, topk_weights_f32,
                        output, num_experts,
                        my_chunk_start, num_tokens_per_rank, rank, world_size,
                )

                # Phase 2: Compute + push for each remote target rank
                if backend_stream is None:
                        backend_stream = torch.xpu.Stream()

                local_bufs = [
                        torch.empty(num_tokens_per_rank, hidden,
                                    device=expert_output.device, dtype=expert_output.dtype),
                        torch.empty(num_tokens_per_rank, hidden,
                                    device=expert_output.device, dtype=expert_output.dtype),
                ]

                for step in range(world_size - 1):
                        target_rank = (rank - step - 1) % world_size
                        buf_idx = step % 2
                        if step % 2 == 0:
                                stream = backend_stream
                        else:
                                stream = torch.xpu.current_stream()
                        with torch.xpu.stream(stream):
                                target_chunk_start = target_rank * num_tokens_per_rank
                                local_bufs[buf_idx].zero_()
                                torch.ops.symm_mem.ep_combine_local_(
                                        expert_output, topk_idx, scatter_idx, topk_weights_f32,
                                        local_bufs[buf_idx], num_experts,
                                        target_chunk_start, num_tokens_per_rank, rank, world_size,
                                )
                                target_recv_buf = recv_workspace.get_buffer(
                                        target_rank,
                                        (world_size, num_tokens_per_rank, hidden),
                                        expert_output.dtype,
                                        storage_offset=0,
                                )
                                target_recv_buf[rank].copy_(local_bufs[buf_idx])

                torch.xpu.current_stream().wait_stream(backend_stream)
                recv_workspace.barrier()

                # Phase 3: Sum received contributions from all other ranks
                my_recv_buf = recv_workspace.get_buffer(
                        rank,
                        (world_size, num_tokens_per_rank, hidden),
                        expert_output.dtype,
                        storage_offset=0,
                )
                for i in range(world_size):
                        if i != rank:
                                output.add_(my_recv_buf[i])

                recv_workspace.barrier()
        else:
                # Python fallback: two-stage compute then push per target rank.
                recv_workspace_size = (
                        world_size * num_tokens_per_rank * hidden * expert_output.element_size()
                )
                recv_workspace = symm_mem.get_symm_mem_workspace(
                        group_name, min_size=recv_workspace_size
                )

                my_recv_buf = recv_workspace.get_buffer(
                        rank,
                        (world_size, num_tokens_per_rank, hidden),
                        expert_output.dtype,
                        storage_offset=0,
                )
                my_recv_buf.zero_()
                recv_workspace.barrier()

                if backend_stream is None:
                        backend_stream = torch.xpu.Stream()

                local_bufs = [
                        torch.zeros(
                                num_tokens_per_rank, hidden,
                                device=expert_output.device, dtype=expert_output.dtype,
                        ),
                        torch.zeros(
                                num_tokens_per_rank, hidden,
                                device=expert_output.device, dtype=expert_output.dtype,
                        ),
                ]

                for step in range(world_size):
                        target_rank = (rank - step) % world_size
                        buf_idx = step % 2
                        if step % 2 == 0:
                                stream = backend_stream
                        else:
                                stream = torch.xpu.current_stream()

                        with torch.xpu.stream(stream):
                                target_chunk_start = target_rank * num_tokens_per_rank
                                buf = local_bufs[buf_idx]
                                buf.zero_()

                                for i in range(num_tokens_per_rank):
                                        global_i = target_chunk_start + i
                                        for k in range(topk):
                                                expert = int(topk_idx[global_i, k].item())
                                                owner = get_expert_owner(expert, num_experts, world_size)
                                                if owner == rank:
                                                        src_row = int(scatter_idx[global_i, k].item())
                                                        buf[i] += topk_weights[global_i, k] * expert_output[src_row]

                                target_recv_buf = recv_workspace.get_buffer(
                                        target_rank,
                                        (world_size, num_tokens_per_rank, hidden),
                                        expert_output.dtype,
                                        storage_offset=0,
                                )
                                target_recv_buf[rank].copy_(buf)

                torch.xpu.current_stream().wait_stream(backend_stream)
                recv_workspace.barrier()

                output.zero_()
                for src_rank in range(world_size):
                        output.add_(my_recv_buf[src_rank])

                recv_workspace.barrier()
        return output


def build_combine_rank_output_ptrs(
    expert_output: torch.Tensor,
    topk_idx: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
        """Precompute rank_output_ptrs tensor for fused deepep_owner_combine."""
        if group is None:
                group = dist.group.WORLD
        if group_name is None:
                group_name = group.group_name
        world_size = dist.get_world_size(group)

        num_tokens = topk_idx.shape[0]
        num_tokens_per_rank = num_tokens // world_size
        hidden = expert_output.shape[1]

        recv_workspace_size = (
                world_size * num_tokens_per_rank * hidden * expert_output.element_size()
        )
        recv_workspace = symm_mem.get_symm_mem_workspace(
                group_name, min_size=recv_workspace_size
        )

        ptr_list = []
        for r in range(world_size):
                buf = recv_workspace.get_buffer(
                        r,
                        (world_size, num_tokens_per_rank, hidden),
                        expert_output.dtype,
                        storage_offset=0,
                )
                ptr_list.append(buf.data_ptr())

        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(signed_ptrs, dtype=torch.int64).to(expert_output.device)


def build_rank_buffers_ptr(
    hidden_shard: torch.Tensor,
    num_experts: int,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
        """Precompute the rank_buffers_ptr tensor for repeated dispatch calls."""
        if group is None:
                group = dist.group.WORLD
        if group_name is None:
                group_name = group.group_name
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        num_tokens_per_rank, hidden_size = hidden_shard.shape
        workspace_size_bytes = hidden_shard.numel() * hidden_shard.element_size() * world_size
        workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

        ptr_list = []
        for r in range(world_size):
                if r == rank:
                        ptr_list.append(hidden_shard.data_ptr())
                else:
                        offset = r * num_tokens_per_rank * hidden_size
                        buf = workspace.get_buffer(
                                r, (num_tokens_per_rank, hidden_size),
                                hidden_shard.dtype, storage_offset=offset,
                        )
                        ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(signed_ptrs, dtype=torch.int64).to(hidden_shard.device)

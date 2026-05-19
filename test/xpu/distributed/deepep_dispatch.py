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
    group: dist.ProcessGroup = None,
    group_name: str = None,
    rank_buffers_ptr: torch.Tensor = None,
):
        """
        TP+EP owner-based dispatch using symmetric memory.

        Each rank writes its hidden_shard to symmetric memory, then a single
        ring-ordered kernel reads from all source ranks with coalesced access
        and writes to owned positions in remap_hidden_states.

        Args:
            rank_buffers_ptr: Optional precomputed device tensor of per-rank
                buffer pointers (int64). Pass this to avoid per-call overhead
                when hidden_shard and workspace are stable across calls.
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
                        rank_buffers_ptr, topk_idx, remap_hidden_states,
                        num_experts, rank, world_size,
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
                                                dst = global_token_idx * topk + k
                                                remap_hidden_states[dst].copy_(src_buffer[i])

        workspace.barrier()
        return remap_hidden_states


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

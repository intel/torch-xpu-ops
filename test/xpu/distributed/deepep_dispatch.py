"""TP+EP owner-based dispatch reference implementation.

This module implements the algorithm from
docs/allgather+local permute fusion.md (TP+EP section):

1. Compute owner rank for each expert.
2. Build per-owner write_base[src_rank][expert] using prefix-sum.
3. Dispatch hidden states into owner-local remap buffers with fixed offsets.

The implementation is designed for correctness validation and algorithm
prototyping in Python.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


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
):
	"""
	Improved TP+EP owner-based dispatch:
	1. Poll tokens from other devices that map to the current device.
	2. Write these tokens into the local remap_hidden_states based on topk.
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
	symm_buffer = workspace.get_buffer(
		rank,
		(world_size, num_tokens_per_rank, hidden_size),
		hidden_shard.dtype,
		storage_offset=0,
	)

	symm_buffer[rank].copy_(hidden_shard)
	workspace.barrier()

	local_token_offset = rank * num_tokens_per_rank
	for step in range(world_size):
		remote_rank = (rank + step) % world_size
		remote_buffer = symm_buffer[remote_rank] if remote_rank == rank else workspace.get_buffer(
			remote_rank,
			(num_tokens_per_rank, hidden_size),
			hidden_shard.dtype,
			storage_offset=0,
		)
		workspace.barrier()
		for i in range(num_tokens_per_rank):
			global_token_idx = remote_rank * num_tokens_per_rank + i
			for k in range(topk):
				expert = int(topk_idx[global_token_idx, k].item())
				owner = get_expert_owner(expert, num_experts, world_size)
				if owner == rank:
					dst = global_token_idx * topk + k
					remap_hidden_states[dst].copy_(remote_buffer[i])
	workspace.barrier()
	return remap_hidden_states

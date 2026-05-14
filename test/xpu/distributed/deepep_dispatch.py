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
	group=None,
	group_name=None,
	num_experts: int = None,
	topk_w: torch.Tensor = None,
):
	"""
	TP+EP owner-based dispatch: dispatch hidden_shard into remap_hidden_states according to topk_idx and expert ownership.
	Args:
		hidden_shard: [num_tokens_per_rank, hidden_size] (local input)
		topk_idx: [num_tokens, topk] (global, all ranks have the same)
		remap_hidden_states: [num_tokens * topk, hidden_size] (output, symmetric memory)
		group: TP process group
		group_name: Optional, for symmetric memory workspace
		num_experts: total expert count
		topk_w: [num_tokens, topk] (optional, route weights)
	Returns:
		remap_hidden_states: [num_tokens * topk, hidden_size] (filled)
	"""
	import torch.distributed as dist
	import torch.distributed._symmetric_memory as symm_mem
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
	if num_experts is None:
		num_experts = topk_idx.max().item() + 1

	# Get symmetric memory workspace
	workspace_size_bytes = hidden_shard.numel() * hidden_shard.element_size() * world_size
	workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)
	symm_buffer = workspace.get_buffer(
		rank,
		(world_size, num_tokens_per_rank, hidden_size),
		hidden_shard.dtype,
		storage_offset=0,
	)

	# Step 1: copy local hidden_shard into symmetric memory slot
	symm_buffer[rank].copy_(hidden_shard)
	workspace.barrier()

	# Step 2: owner-based remap
	local_token_offset = rank * num_tokens_per_rank
	for i in range(num_tokens_per_rank):
		global_token_idx = local_token_offset + i
		for k in range(topk):
			expert = int(topk_idx[global_token_idx, k].item())
			# Only write if this rank owns the expert
			owner = get_expert_owner(expert, num_experts, world_size)
			if owner == rank:
				dst = global_token_idx * topk + k
				remap_hidden_states[dst].copy_(hidden_shard[i])
	workspace.barrier()
	return remap_hidden_states

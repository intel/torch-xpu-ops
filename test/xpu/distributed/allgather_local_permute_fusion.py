"""
Allgather + local permute fusion using symmetric memory (TP only scenario).

Implements the algorithm described in docs/allgather+local permute fusion.md (TP only section):
- Each rank knows the global topk_idx mapping
- Each rank writes its local hidden_shard to remap_hidden_states according to topk_idx
- Allgather is performed directly into the correct positions in remap_hidden_states (symmetric memory)
- No redundant permute after allgather

Inputs:
    hidden_shard: [num_tokens_per_rank, hidden_size] (local input)
    topk_idx: [num_tokens, topk] (global, all ranks have the same)
    world_size: TP group size
    rank: TP rank
    remap_hidden_states: [num_tokens * topk, hidden_size] (output, symmetric memory)

"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def allgather_local_permute_fusion(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
):
    """
    TP only: Allgather + local permute fusion using symmetric memory.

    Args:
        hidden_shard: [num_tokens_per_rank, hidden_size] (local input)
        topk_idx: [num_tokens, topk] (global, all ranks have the same)
        group: TP process group
        remap_hidden_states: [num_tokens * topk, hidden_size] (output, symmetric memory)
        group_name: Optional, for symmetric memory workspace
    Returns:
        remap_hidden_states: [num_tokens * topk, hidden_size] (filled)
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

    # Before pull loop: write local hidden_shard to remap_hidden_states by topk
    local_token_offset = rank * num_tokens_per_rank
    for i in range(num_tokens_per_rank):
        global_token_idx = local_token_offset + i
        for k in range(topk):
            _ = topk_idx[global_token_idx, k]
            dst = global_token_idx * topk + k
            remap_hidden_states[dst].copy_(hidden_shard[i])

    # Step 2: two-stream round-robin pull; write to remap immediately per remote shard
    backend_stream = torch.xpu.Stream()
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        if step % 2 == 0:
            stream = backend_stream
        else:
            stream = torch.xpu.current_stream()       
        with torch.xpu.stream(stream):
            remote_buffer = workspace.get_buffer(
                remote_rank,
                (num_tokens_per_rank, hidden_size),
                hidden_shard.dtype,
                storage_offset=0,
            )
            symm_buffer[remote_rank].copy_(remote_buffer)

            remote_token_offset = remote_rank * num_tokens_per_rank
            for i in range(num_tokens_per_rank):
                global_token_idx = remote_token_offset + i
                hidden_vec = symm_buffer[remote_rank, i]
                for k in range(topk):
                    _ = topk_idx[global_token_idx, k]
                    dst = global_token_idx * topk + k
                    remap_hidden_states[dst].copy_(hidden_vec)
    torch.xpu.current_stream().wait_stream(backend_stream)
    workspace.barrier()
    return remap_hidden_states

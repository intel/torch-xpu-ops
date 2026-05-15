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

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Try to load the native local_permute_copy kernel
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so")
_HAS_LOCAL_PERMUTE_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_permute_copy_")
    except Exception:
        pass


def allgather_local_permute_fusion_native(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
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

    # Before pull loop: one fused kernel for local shard remap
    local_token_offset = rank * num_tokens_per_rank
    torch.ops.symm_mem.local_permute_copy_(
        hidden_shard,
        topk_idx,
        local_token_offset,
        remap_hidden_states,
    )

    # Step 2: two-stream round-robin pull; write to remap immediately per remote shard
    if backend_stream is None:
        backend_stream = torch.xpu.Stream()
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        if step % 2 == 0:
            stream = backend_stream
        else:
            stream = torch.xpu.current_stream()
        with torch.xpu.stream(stream):
            remote_offset = remote_rank * num_tokens_per_rank * hidden_size
            remote_buffer = workspace.get_buffer(
                remote_rank,
                (num_tokens_per_rank, hidden_size),
                hidden_shard.dtype,
                storage_offset=remote_offset,
            )
            symm_buffer[remote_rank].copy_(remote_buffer)

            remote_token_offset = remote_rank * num_tokens_per_rank
            torch.ops.symm_mem.local_permute_copy_(
                symm_buffer[remote_rank],
                topk_idx,
                remote_token_offset,
                remap_hidden_states,
            )
    torch.xpu.current_stream().wait_stream(backend_stream)
    workspace.barrier()
    return remap_hidden_states


def allgather_local_permute_fusion_python(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
):
    """
    TP only: Allgather + local permute fusion using Python copy loops.

    This keeps the original Python implementation for correctness checks
    and A/B performance comparison with the native fused kernel API.
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
    for i in range(num_tokens_per_rank):
        global_token_idx = local_token_offset + i
        for k in range(topk):
            _ = topk_idx[global_token_idx, k]
            dst = global_token_idx * topk + k
            remap_hidden_states[dst].copy_(hidden_shard[i])

    if backend_stream is None:
        backend_stream = torch.xpu.Stream()
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        if step % 2 == 0:
            stream = backend_stream
        else:
            stream = torch.xpu.current_stream()
        with torch.xpu.stream(stream):
            remote_offset = remote_rank * num_tokens_per_rank * hidden_size
            remote_buffer = workspace.get_buffer(
                remote_rank,
                (num_tokens_per_rank, hidden_size),
                hidden_shard.dtype,
                storage_offset=remote_offset,
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


def allgather_local_permute_fusion(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
):
    """Default API: uses native kernel if available, otherwise Python fallback."""
    if _HAS_LOCAL_PERMUTE_KERNEL:
        return allgather_local_permute_fusion_native(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            remap_hidden_states=remap_hidden_states,
            group=group,
            group_name=group_name,
            backend_stream=backend_stream,
        )
    return allgather_local_permute_fusion_python(
        hidden_shard=hidden_shard,
        topk_idx=topk_idx,
        remap_hidden_states=remap_hidden_states,
        group=group,
        group_name=group_name,
        backend_stream=backend_stream,
    )


def allgather_with_symm_mem(
    input_shard: torch.Tensor,
    output_tensor: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
):
    """
    Pure allgather using symmetric memory (no reduction).
    Each rank contributes input_shard, all ranks gather into output_tensor.
    Args:
        input_shard: [numel_per_rank, ...] (local input)
        output_tensor: [numel_per_rank * world_size, ...] (output, symmetric memory)
        group: process group
        group_name: Optional, for symmetric memory workspace
    Returns:
        output_tensor: filled with allgathered data
    """
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    input_flat = input_shard.contiguous().view(-1)
    output_flat = output_tensor.contiguous().view(-1)
    numel_per_rank = input_flat.numel()
    assert output_flat.numel() == numel_per_rank * world_size

    import torch.distributed._symmetric_memory as symm_mem
    workspace_size_bytes = numel_per_rank * world_size * input_flat.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Barrier to ensure previous allgather is complete before writing
    workspace.barrier()

    # Step 1: Each rank writes its shard to its slot in symm buffer
    my_slot = workspace.get_buffer(rank, (numel_per_rank,), input_flat.dtype, storage_offset=rank * numel_per_rank)
    my_slot.copy_(input_flat)

    workspace.barrier()

    # Step 2: Ring allgather (pull-based, like allreduce_with_pull)
    # Each rank pulls chunk[remote_rank] from remote_rank's symm buffer
    output_flat[rank * numel_per_rank:(rank + 1) * numel_per_rank].copy_(my_slot)
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        remote_buffer = workspace.get_buffer(
            remote_rank,
            (numel_per_rank,),
            input_flat.dtype,
            storage_offset=remote_rank * numel_per_rank
        )
        output_flat[remote_rank * numel_per_rank:(remote_rank + 1) * numel_per_rank].copy_(remote_buffer)
    workspace.barrier()
    return output_tensor

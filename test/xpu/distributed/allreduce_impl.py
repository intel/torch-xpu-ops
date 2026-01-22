"""
Allreduce implementation using symmetric memory.

This module implements allreduce using symmetric memory primitives:
1. Reduce-scatter: Each rank reduces a chunk and scatters to all ranks
2. Kernel reduce: Perform local reduction using kernel on scattered data
3. Allgather: Gather reduced results from all ranks
4. Barrier: Synchronize using symmetric memory barrier

Reference: https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/__init__.py

Usage example:
    import torch
    import torch.distributed as dist
    from allreduce_impl import allreduce_with_symm_mem

    # Initialize process group
    dist.init_process_group(backend="xccl")

    # Create input tensor
    tensor = torch.randn(1024, device="xpu")

    # Perform allreduce
    result = allreduce_with_symm_mem(tensor, op="sum", group_name="0")
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def allreduce_with_symm_mem(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Perform allreduce using symmetric memory with ring reduce-scatter + allgather.

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)

    Returns:
        Reduced tensor with the same shape as input (in-place modification)
    """
    if op != "sum":
        raise ValueError(f"Only 'sum' operation is supported, got '{op}'")

    # Get group info
    if group is None:
        group = dist.group.WORLD

    group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # Work with flattened view
    tensor_flat = tensor.view(-1)
    numel = tensor_flat.numel()

    # Check divisibility
    if numel % world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by world_size ({world_size})"
        )

    chunk_size = numel // world_size

    # Get symmetric memory workspace
    workspace_size_bytes = chunk_size * world_size * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Barrier to ensure previous allreduce is complete before writing
    workspace.barrier()

    # Step 1: Ring Scatter - Each rank pushes to remote ranks in ring fashion
    # Reference: PyTorch _low_contention_reduce_scatter_with_workspace implementation
    # Each rank pushes chunk[remote_rank] to remote_rank's symm buffer
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        remote_buffer = workspace.get_buffer(
            remote_rank,
            (chunk_size,),
            tensor.dtype,
            storage_offset=rank * chunk_size
        )
        remote_buffer.copy_(tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size])

    # Barrier after each ring step to ensure data is received before next push
    workspace.barrier()

    # Step 2: Reduce - Single kernel reduction
    chunk_start = rank * chunk_size
    chunk_end = (rank + 1) * chunk_size

    # First, copy local chunk to symm[rank] position (symm[rank] was 0)
    my_slot = workspace.get_buffer(
        rank, (chunk_size,), tensor.dtype, storage_offset=rank * chunk_size
    )
    my_slot.copy_(tensor_flat[chunk_start:chunk_end])

    # View symm as [world_size, chunk_size] and reduce with single kernel
    my_symm_2d = workspace.get_buffer(
        rank, (world_size, chunk_size), tensor.dtype, storage_offset=0
    )

    # Single reduction kernel: sum along dim 0
    torch.sum(my_symm_2d, dim=0, out=tensor_flat[chunk_start:chunk_end])

    # Write reduced result back to symm[rank] for allgather
    my_slot.copy_(tensor_flat[chunk_start:chunk_end])

    workspace.barrier()

    # Step 3: Ring Allgather - Each rank pulls from remote ranks in ring fashion
    # Reference: PyTorch _low_contention_all_gather implementation
    # Each rank pulls chunk[remote_rank] from remote_rank's symm buffer
    # for step in range(world_size - 1):
    #     remote_rank = (rank - step - 1) % world_size
    #     remote_buffer = workspace.get_buffer(
    #         remote_rank,
    #         (chunk_size,),
    #         tensor.dtype,
    #         storage_offset=remote_rank * chunk_size
    #     )
    #     tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size].copy_(remote_buffer)
    # Using push instead of pull
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        remote_buffer = workspace.get_buffer(
            remote_rank,
            (chunk_size,),
            tensor.dtype,
            storage_offset=rank * chunk_size
        )
        remote_buffer.copy_(tensor_flat[chunk_start:chunk_end])

    workspace.barrier()
    tensor.copy_(my_symm_2d.view(-1))

    return tensor


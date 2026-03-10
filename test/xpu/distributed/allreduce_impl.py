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


# =============================================================================
# Hardcoded Topology: 8 GPUs, 2 Sockets, 4 PCIe Switches
#
#   Socket 0                    Socket 1
#   ┌─────────────────┐         ┌─────────────────┐
#   │ Switch 0: GPU 0,1 │         │ Switch 2: GPU 4,5 │
#   │ Switch 1: GPU 2,3 │         │ Switch 3: GPU 6,7 │
#   └─────────────────┘         └─────────────────┘
#
# Communication speed hierarchy:
#   1. Same PCIe Switch (fastest): 0↔1, 2↔3, 4↔5, 6↔7
#   2. Same Socket, different Switch (medium): 0↔2, 0↔3, 1↔2, 1↔3, etc.
#   3. Cross Socket (slowest): 0↔4, 0↔5, 0↔6, 0↔7, etc.
# =============================================================================

NUM_SOCKETS = 2
GPUS_PER_SOCKET = 4
SWITCHES_PER_SOCKET = 2
GPUS_PER_SWITCH = 2
WORLD_SIZE = NUM_SOCKETS * GPUS_PER_SOCKET  # 8

# Socket membership
SOCKET_RANKS = {
    0: (0, 1, 2, 3),  # Socket 0
    1: (4, 5, 6, 7),  # Socket 1
}

# PCIe Switch membership (GPUs under same switch have fastest P2P)
SWITCH_RANKS = {
    0: (0, 1),  # Switch 0 @ Socket 0
    1: (2, 3),  # Switch 1 @ Socket 0
    2: (4, 5),  # Switch 2 @ Socket 1
    3: (6, 7),  # Switch 3 @ Socket 1
}

# Cross-socket peer mapping: GPU[i] <-> GPU[i+4]
CROSS_SOCKET_PEER = {
    0: 4, 1: 5, 2: 6, 3: 7,
    4: 0, 5: 1, 6: 2, 7: 3,
}

# Same-switch peer mapping (for intra-switch communication)
SAME_SWITCH_PEER = {
    0: 1, 1: 0,
    2: 3, 3: 2,
    4: 5, 5: 4,
    6: 7, 7: 6,
}


def get_socket_id(rank: int) -> int:
    """Get socket ID for a given rank. Socket 0: 0-3, Socket 1: 4-7."""
    return rank // GPUS_PER_SOCKET


def get_switch_id(rank: int) -> int:
    """Get PCIe switch ID for a given rank."""
    return rank // GPUS_PER_SWITCH


def get_socket_ranks(rank: int) -> tuple[int, ...]:
    """Get all ranks in the same socket as the given rank."""
    socket_id = get_socket_id(rank)
    return SOCKET_RANKS[socket_id]


def get_switch_ranks(rank: int) -> tuple[int, ...]:
    """Get all ranks under the same PCIe switch."""
    switch_id = get_switch_id(rank)
    return SWITCH_RANKS[switch_id]


def get_cross_socket_peer(rank: int) -> int:
    """Get corresponding peer rank in the other socket."""
    return CROSS_SOCKET_PEER[rank]


def get_same_switch_peer(rank: int) -> int:
    """Get peer rank under the same PCIe switch."""
    return SAME_SWITCH_PEER[rank]

def allreduce_cross_switch(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Perform allreduce using symmetric memory with ring reduce-scatter + allgather.

    Optimized version: reduces copy operations by working directly on symm buffer.

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

    # For 8 ranks (2 sockets): local_world_size = 4 (per socket)
    # For 4 ranks or less: local_world_size = world_size (single socket)
    local_world_size = world_size // 2 if world_size > 4 else world_size
    chunk_size = numel // local_world_size

    # Get symmetric memory workspace
    workspace_size_bytes = chunk_size * local_world_size * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Topology info
    switch_id = rank // 4  # 0 for ranks 0-3, 1 for ranks 4-7
    local_rank = rank % 4  # 0-3 within each switch
    cross_switch_peer = rank ^ 4  # 0↔4, 1↔5, 2↔6, 3↔7
    half_chunk_size = chunk_size // 2

    # My chunk position
    my_chunk_start = local_rank * chunk_size

    # Barrier to ensure previous allreduce is complete before writing
    workspace.barrier()

    # Step 1: Ring Scatter - Each rank pushes to remote ranks in ring fashion
    for step in range(local_world_size - 1):
        remote_rank = (local_rank - step - 1) % local_world_size
        remote_buffer = workspace.get_buffer(
            remote_rank + switch_id * 4,
            (chunk_size,),
            tensor.dtype,
            storage_offset=local_rank * chunk_size
        )
        remote_buffer.copy_(tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size])

    # Barrier after scatter to ensure data is received
    workspace.barrier()

    # Step 2: Reduce
    # Get my slot in symm buffer
    my_slot = workspace.get_buffer(
        rank, (chunk_size,), tensor.dtype, storage_offset=my_chunk_start
    )

    # First, copy local chunk to my_slot (my_symm_2d[local_rank] was not written by scatter)
    my_slot.copy_(tensor_flat[my_chunk_start:my_chunk_start + chunk_size])

    # View symm as [local_world_size, chunk_size] and reduce with single kernel
    my_symm_2d = workspace.get_buffer(
        rank, (local_world_size, chunk_size), tensor.dtype, storage_offset=0
    )

    # Reduce: sum all chunks from symm buffer
    # Use tensor_flat[my_chunk] as temp output to avoid read-write conflict
    # (my_slot is my_symm_2d[local_rank], writing to it while reading would be unsafe)
    my_chunk_in_tensor = tensor_flat[my_chunk_start:my_chunk_start + chunk_size]
    torch.sum(my_symm_2d, dim=0, out=my_chunk_in_tensor)

    # Copy reduced result back to my_slot for cross-switch and allgather
    my_slot.copy_(my_chunk_in_tensor)

    workspace.barrier()

    # Cross-switch exchange for 8 ranks
    if world_size > 4:
        # Step 3a: Exchange halves between sockets
        # Socket 0 sends back half, Socket 1 sends front half
        # Use my_slot directly (already in symm buffer)
        my_front_half = my_slot[:half_chunk_size]
        my_back_half = my_slot[half_chunk_size:]

        if switch_id == 0:
            # Socket 0: send back half to peer's temp slot
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=local_rank * chunk_size
            )
            remote_buffer.copy_(my_back_half)
        else:
            # Socket 1: send front half to peer's temp slot
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=local_rank * chunk_size
            )
            remote_buffer.copy_(my_front_half)

        # Barrier 1: ensure all sends complete
        workspace.barrier()

        # Step 3b: Reduce with received data and write result to peer
        received = workspace.get_buffer(
            rank, (half_chunk_size,), tensor.dtype,
            storage_offset=local_rank * chunk_size
        )

        if switch_id == 0:
            # Socket 0: reduce front half, write to self and peer
            my_front_half.add_(received)
            # Write to peer's front half position
            peer_front_half = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start
            )
            peer_front_half.copy_(my_front_half)
        else:
            # Socket 1: reduce back half, write to self and peer
            my_back_half.add_(received)
            # Write to peer's back half position
            peer_back_half = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start + half_chunk_size
            )
            peer_back_half.copy_(my_back_half)

        # Barrier 2: ensure all writes complete before reading
        workspace.barrier()

        # Step 3c: Read the half written by peer (directly into my_slot)
        if switch_id == 0:
            # Socket 0: peer wrote back half to my symm buffer
            # my_back_half already points to the right location, just need to sync
            pass  # Data already in my_slot[half_chunk_size:]
        else:
            # Socket 1: peer wrote front half to my symm buffer
            pass  # Data already in my_slot[:half_chunk_size]

    # Step 4: Ring Allgather - pull from remote ranks
    # Copy my reduced chunk from symm buffer to tensor_flat first
    tensor_flat[my_chunk_start:my_chunk_start + chunk_size].copy_(my_slot)

    for step in range(local_world_size - 1):
        remote_rank = (local_rank - step - 1) % local_world_size
        remote_buffer = workspace.get_buffer(
            remote_rank + switch_id * 4,
            (chunk_size,),
            tensor.dtype,
            storage_offset=remote_rank * chunk_size
        )
        tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size].copy_(remote_buffer)

    workspace.barrier()
    return tensor

def allreduce_cross_switch_pipeline(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
    num_pipelines: int = 4,
) -> torch.Tensor:
    """
    Perform allreduce optimized for dual-socket topology with PCIe switches.

    Topology:
        Socket 0: GPU 0,1,2,3 (PCIe switch, ring within)
        Socket 1: GPU 4,5,6,7 (PCIe switch, ring within)
        Cross-socket: UPI pairs (0↔4, 1↔5, 2↔6, 3↔7)

    Algorithm (Pull-based, minimal barriers):
        1. Copy to symm buffer
        2. Intra-socket ring reduce-scatter (PCIe pull)
        3. Cross-socket reduce (UPI)
        4. Intra-socket ring allgather (PCIe pull)

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)
        num_pipelines: Not used, kept for API compatibility

    Returns:
        Reduced tensor with the same shape as input (in-place modification)
    """
    if op != "sum":
        raise ValueError(f"Only 'sum' operation is supported, got '{op}'")

    if group is None:
        group = dist.group.WORLD

    group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    tensor_flat = tensor.view(-1)
    numel = tensor_flat.numel()

    # For 8 ranks: 4 GPUs per socket
    local_world_size = world_size // 2 if world_size > 4 else world_size

    if numel % local_world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by local_world_size ({local_world_size})"
        )

    chunk_size = numel // local_world_size

    # Get symmetric memory workspace (2x for cross-socket buffer)
    workspace_size_bytes = numel * tensor.element_size() * 2
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Topology info
    local_rank = rank % local_world_size  # 0-3 within socket
    socket_base = (rank // local_world_size) * local_world_size  # 0 or 4
    prev_local = (local_rank - 1 + local_world_size) % local_world_size
    prev_rank = socket_base + prev_local
    peer_rank = rank ^ local_world_size  # 0↔4, 1↔5, 2↔6, 3↔7

    workspace.barrier()

    # =========================================================================
    # Phase 1: Copy local data to symmetric memory buffer
    # =========================================================================
    my_symm_buffer = workspace.get_buffer(rank, (numel,), tensor.dtype, storage_offset=0)
    my_symm_buffer.copy_(tensor_flat)

    workspace.barrier()

    # =========================================================================
    # Phase 2: Intra-socket Ring Reduce-Scatter (Pull-based)
    # Each rank pulls from prev_rank within same socket
    # =========================================================================
    for step in range(local_world_size - 1):
        idx = (local_rank - step + local_world_size) % local_world_size
        start = idx * chunk_size
        end = start + chunk_size

        prev_buffer = workspace.get_buffer(
            prev_rank, (chunk_size,), tensor.dtype, storage_offset=start
        )
        my_symm_buffer[start:end].add_(prev_buffer)

        workspace.barrier()

    # After RS, my reduced chunk is at index (local_rank + 1) % local_world_size
    my_chunk_idx = (local_rank + 1) % local_world_size
    my_chunk_start = my_chunk_idx * chunk_size
    my_chunk_end = my_chunk_start + chunk_size

    # =========================================================================
    # Phase 3: Cross-socket Reduce (Pull-based)
    # Each rank exchanges with peer in other socket
    # =========================================================================
    if world_size > 4:
        # Push my chunk to peer's buffer (use second half of workspace)
        remote_buffer = workspace.get_buffer(
            peer_rank, (chunk_size,), tensor.dtype,
            storage_offset=numel + my_chunk_start
        )
        remote_buffer.copy_(my_symm_buffer[my_chunk_start:my_chunk_end])

        workspace.barrier()

        # Pull from peer and reduce
        peer_buffer = workspace.get_buffer(
            rank, (chunk_size,), tensor.dtype,
            storage_offset=numel + my_chunk_start
        )
        my_symm_buffer[my_chunk_start:my_chunk_end].add_(peer_buffer)

        workspace.barrier()

    # =========================================================================
    # Phase 4: Intra-socket Ring Allgather (Pull-based)
    # Each rank pulls from prev_rank within same socket
    # =========================================================================
    for step in range(local_world_size - 1):
        idx = (my_chunk_idx - step + local_world_size) % local_world_size
        start = idx * chunk_size
        end = start + chunk_size

        prev_buffer = workspace.get_buffer(
            prev_rank, (chunk_size,), tensor.dtype, storage_offset=start
        )
        my_symm_buffer[start:end].copy_(prev_buffer)

        workspace.barrier()

    # Final copy: symm buffer -> tensor_flat
    tensor_flat.copy_(my_symm_buffer)

    return tensor


def allreduce_with_pull(
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
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        remote_buffer = workspace.get_buffer(
            remote_rank,
            (chunk_size,),
            tensor.dtype,
            storage_offset=remote_rank * chunk_size
        )
        tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size].copy_(remote_buffer)
    
    workspace.barrier()
    return tensor

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


def allreduce_with_ring_pull(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Perform allreduce using symmetric memory with ring reduce-scatter + ring allgather.

    This implementation uses a pull-based ring pattern:
    - Reduce-Scatter (Ring Pull): Each rank pulls from prev_rank and accumulates
    - Allgather (Ring Pull): Each rank pulls from prev_rank

    Ring pattern for world_size=N:
    - Each rank only reads from prev_rank = (rank - 1 + N) % N
    - Data index starts from rank and rotates backwards
    - Each phase has (N-1) steps

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

    # Ring neighbor for allgather phase
    prev_rank = (rank - 1 + world_size) % world_size

    # Barrier to ensure previous allreduce is complete before writing
    workspace.barrier()

    # ==========================================================================
    # Phase 1: Copy local data to symmetric memory buffer
    # Each rank writes its entire tensor to its own symm buffer
    # ==========================================================================
    my_symm_buffer = workspace.get_buffer(
        rank, (numel,), tensor.dtype, storage_offset=0
    )
    my_symm_buffer.copy_(tensor_flat)

    workspace.barrier()

    # ==========================================================================
    # Phase 2: Ring Reduce-Scatter (Pull-based)
    # Each rank only reads from prev_rank to avoid write contention.
    # Data index starts from rank and rotates backwards.
    #
    # Step s: rank r pulls chunk[(r - s + world_size) % N] from prev_rank,
    #         adds to own symm buffer, writes result for next rank to pull.
    #
    # After (world_size - 1) steps, chunk[(rank + 1) % N] has full reduction.
    # ==========================================================================
    for step in range(world_size - 1):
        # Which chunk to reduce this step
        chunk_idx = (rank - step + world_size) % world_size
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size

        # Pull partial sum from prev_rank's symm buffer
        prev_buffer = workspace.get_buffer(
            prev_rank, (chunk_size,), tensor.dtype, storage_offset=chunk_start
        )

        # Add prev_rank's partial sum to my symm buffer (in-place)
        my_symm_buffer[chunk_start:chunk_end].add_(prev_buffer)

        workspace.barrier()

    # ==========================================================================
    # Phase 3: Ring Allgather (Pull-based)
    # Each rank only reads from its prev_rank.
    # After (world_size - 1) steps, each rank has all reduced chunks.
    #
    # After reduce-scatter, each rank r has reduced chunk[(r + 1) % N].
    # Data index starts from (rank + 1) and rotates backwards.
    # Step s: rank r pulls chunk[(r + 1 - s + world_size) % N] from prev_rank
    # ==========================================================================
    for step in range(world_size - 1):
        # Which chunk to gather this step
        # Start from (rank + 1) which is the reduced chunk we own
        chunk_idx = (rank + 1 - step + world_size) % world_size
        chunk_start = chunk_idx * chunk_size
        chunk_end = chunk_start + chunk_size

        # Pull the chunk from prev_rank's symm buffer
        prev_buffer = workspace.get_buffer(
            prev_rank, (chunk_size,), tensor.dtype, storage_offset=chunk_start
        )

        # Copy to local symm buffer (for next rank to pull in next step)
        my_symm_buffer[chunk_start:chunk_end].copy_(prev_buffer)

        workspace.barrier()

    # Final copy: symm buffer -> tensor_flat (local memory copy)
    tensor_flat.copy_(my_symm_buffer)

    return tensor


# =============================================================================
# Two-Level Hierarchical Allreduce Implementation
# Level 1: Intra-Socket (all GPUs within socket treated as equal bandwidth)
# Level 2: Cross-Socket (slowest, minimized to N/4 of original data)
# =============================================================================


def _intra_socket_reduce_scatter(
    tensor: torch.Tensor,
    workspace,
    rank: int,
) -> tuple[torch.Tensor, int]:
    """
    Level 1: Reduce-scatter within socket.
    Socket 0: GPUs 0,1,2,3
    Socket 1: GPUs 4,5,6,7

    Each GPU ends up with 1/4 of the reduced data from its socket.
    Uses ring reduce-scatter pattern.
    """
    socket_ranks = get_socket_ranks(rank)
    local_size = len(socket_ranks)  # 4
    local_rank = socket_ranks.index(rank)
    numel = tensor.numel()
    chunk_size = numel // local_size  # N/4

    # Working buffer
    data = tensor.clone()

    # Ring reduce-scatter: (local_size - 1) steps
    for step in range(local_size - 1):
        # Ring neighbors within socket
        send_to_local = (local_rank + 1) % local_size
        recv_from_local = (local_rank - 1 + local_size) % local_size
        send_to_rank = socket_ranks[send_to_local]

        # Which chunk to send this step
        send_chunk_idx = (local_rank - step + local_size) % local_size
        send_start = send_chunk_idx * chunk_size

        # Push chunk to next rank in ring
        remote_buffer = workspace.get_buffer(
            send_to_rank,
            (chunk_size,),
            tensor.dtype,
            storage_offset=local_rank * chunk_size
        )
        remote_buffer.copy_(data[send_start:send_start + chunk_size])

        workspace.barrier()

        # Receive chunk from prev rank and accumulate
        recv_chunk_idx = (local_rank - step - 1 + local_size) % local_size
        recv_start = recv_chunk_idx * chunk_size

        peer_buffer = workspace.get_buffer(
            rank,
            (chunk_size,),
            tensor.dtype,
            storage_offset=recv_from_local * chunk_size
        )

        # Accumulate
        data[recv_start:recv_start + chunk_size] += peer_buffer

        workspace.barrier()

    # Each rank now owns reduced chunk at index (local_rank + 1) % local_size
    my_chunk_idx = (local_rank + 1) % local_size
    my_start = my_chunk_idx * chunk_size
    reduced_chunk = data[my_start:my_start + chunk_size].clone()

    return reduced_chunk, my_chunk_idx


def _cross_socket_reduce(
    local_chunk: torch.Tensor,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 2: Reduce across sockets.
    Pairs: 0↔4, 1↔5, 2↔6, 3↔7

    Each GPU exchanges its chunk with corresponding peer in other socket and reduces.
    After this, each chunk contains the sum from all 8 GPUs.
    """
    peer_rank = get_cross_socket_peer(rank)
    chunk_size = local_chunk.numel()

    # Use socket-local rank as slot (0-3)
    socket_local_rank = rank % GPUS_PER_SOCKET
    peer_socket_local_rank = peer_rank % GPUS_PER_SOCKET

    # Push to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (chunk_size,),
        local_chunk.dtype,
        storage_offset=socket_local_rank * chunk_size
    )
    remote_buffer.copy_(local_chunk)

    workspace.barrier()

    # Read and reduce
    peer_buffer = workspace.get_buffer(
        rank,
        (chunk_size,),
        local_chunk.dtype,
        storage_offset=peer_socket_local_rank * chunk_size
    )

    result = local_chunk + peer_buffer
    return result


def _intra_socket_allgather(
    my_chunk: torch.Tensor,
    my_chunk_idx: int,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 1: Allgather within socket.
    Each GPU has 1 fully-reduced chunk, gather all 4 chunks from socket peers.
    Returns tensor with all 4 chunks (N elements).
    """
    socket_ranks = get_socket_ranks(rank)
    local_size = len(socket_ranks)  # 4
    local_rank = socket_ranks.index(rank)
    chunk_size = my_chunk.numel()
    total_size = chunk_size * local_size  # N

    # Initialize result
    result = torch.zeros(total_size, dtype=my_chunk.dtype, device=my_chunk.device)

    # Place my chunk
    my_start = my_chunk_idx * chunk_size
    result[my_start:my_start + chunk_size] = my_chunk

    # Broadcast my chunk to all socket peers
    for target_local in range(local_size):
        if target_local == local_rank:
            continue
        target_rank = socket_ranks[target_local]

        remote_buffer = workspace.get_buffer(
            target_rank,
            (chunk_size,),
            my_chunk.dtype,
            storage_offset=local_rank * chunk_size
        )
        remote_buffer.copy_(my_chunk)

    workspace.barrier()

    # Receive chunks from all socket peers
    for sender_local in range(local_size):
        if sender_local == local_rank:
            continue

        # Get sender's chunk index
        sender_chunk_idx = (sender_local + 1) % local_size
        sender_start = sender_chunk_idx * chunk_size

        peer_buffer = workspace.get_buffer(
            rank,
            (chunk_size,),
            my_chunk.dtype,
            storage_offset=sender_local * chunk_size
        )
        result[sender_start:sender_start + chunk_size] = peer_buffer

    return result


def hierarchical_allreduce_with_symm_mem(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
    num_pipelines: int = 4,
) -> torch.Tensor:
    """
    Perform 2-level hierarchical allreduce with PCIe/UPI bandwidth overlap.

    Topology (hardcoded):
        Socket 0: GPU 0, 1, 2, 3 (PCIe switch)
        Socket 1: GPU 4, 5, 6, 7 (PCIe switch)
        Cross-socket: UPI (slow)

    Algorithm with Pipeline Overlap:
        Split data into num_pipelines chunks, then for each chunk:
        1. Intra-socket reduce-scatter (PCIe): N/P → N/(P*4) per GPU
        2. Cross-socket reduce (UPI): Exchange with peer (0↔4, 1↔5, 2↔6, 3↔7)
        3. Intra-socket allgather (PCIe): N/(P*4) → N/P per GPU

        Pipeline overlap ensures PCIe and UPI are used simultaneously:
        - While chunk[i] does cross-socket (UPI), chunk[i+1] does intra-socket (PCIe)

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)
        num_pipelines: Number of pipeline stages for overlap (default: 4)

    Returns:
        Reduced tensor with the same shape as input (in-place modification)
    """
    if op != "sum":
        raise ValueError(f"Only 'sum' operation is supported, got '{op}'")

    if group is None:
        group = dist.group.WORLD

    group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if world_size != WORLD_SIZE:
        raise ValueError(
            f"World size ({world_size}) must be {WORLD_SIZE} for this topology"
        )

    tensor_flat = tensor.view(-1)
    numel = tensor_flat.numel()

    # Must be divisible by GPUS_PER_SOCKET * num_pipelines
    if numel % (GPUS_PER_SOCKET * num_pipelines) != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by {GPUS_PER_SOCKET * num_pipelines}"
        )

    # Pipeline chunk sizes
    pipe_size = numel // num_pipelines  # Size per pipeline stage
    chunk_size = pipe_size // GPUS_PER_SOCKET  # Size per GPU after reduce-scatter

    # Get symmetric memory workspace (need 2x for double buffering)
    workspace_size_bytes = numel * tensor.element_size() * 2
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Topology info
    local_rank = rank % GPUS_PER_SOCKET  # 0-3 within socket
    socket_ranks = get_socket_ranks(rank)
    prev_local = (local_rank - 1 + GPUS_PER_SOCKET) % GPUS_PER_SOCKET
    prev_rank = socket_ranks[prev_local]
    peer_rank = get_cross_socket_peer(rank)

    workspace.barrier()

    # =========================================================================
    # Phase 1: Copy local data to symmetric memory buffer
    # =========================================================================
    my_symm_buffer = workspace.get_buffer(rank, (numel,), tensor.dtype, storage_offset=0)
    my_symm_buffer.copy_(tensor_flat)

    workspace.barrier()

    # =========================================================================
    # Pipelined execution: overlap PCIe (RS/AG) with UPI (XS)
    #
    # Each pipeline goes through: RS (3 steps) -> XS (2 steps) -> AG (3 steps)
    # Stagger pipelines so PCIe and UPI operations overlap.
    #
    # Pipeline state machine:
    #   State 0-2: RS step 0-2 (PCIe pull from prev_rank)
    #   State 3:   XS send (UPI push to peer)
    #   State 4:   XS recv+reduce (UPI pull from peer)
    #   State 5-7: AG step 0-2 (PCIe pull from prev_rank)
    #   State 8:   Done
    #
    # Each global step: advance all pipelines by 1 state, barrier
    # Pipelines are staggered by their index for overlap.
    # =========================================================================

    RS_STEPS = GPUS_PER_SOCKET - 1  # 3
    AG_STEPS = GPUS_PER_SOCKET - 1  # 3
    TOTAL_STATES = RS_STEPS + 2 + AG_STEPS  # 3 + 2 + 3 = 8

    def do_pipeline_step(pipe_idx: int, state: int):
        """Execute one step for a pipeline based on its current state."""
        if state < 0 or state >= TOTAL_STATES:
            return  # Not started or already done

        pipe_offset = pipe_idx * pipe_size
        my_chunk_idx = (local_rank + 1) % GPUS_PER_SOCKET

        if state < RS_STEPS:
            # RS step
            rs_step = state
            idx = (local_rank - rs_step + GPUS_PER_SOCKET) % GPUS_PER_SOCKET
            start = pipe_offset + idx * chunk_size
            end = start + chunk_size

            prev_buffer = workspace.get_buffer(
                prev_rank, (chunk_size,), tensor.dtype, storage_offset=start
            )
            my_symm_buffer[start:end].add_(prev_buffer)

        elif state == RS_STEPS:
            # XS send
            start = pipe_offset + my_chunk_idx * chunk_size
            end = start + chunk_size

            remote_buffer = workspace.get_buffer(
                peer_rank, (chunk_size,), tensor.dtype,
                storage_offset=numel + start
            )
            remote_buffer.copy_(my_symm_buffer[start:end])

        elif state == RS_STEPS + 1:
            # XS recv + reduce
            start = pipe_offset + my_chunk_idx * chunk_size
            end = start + chunk_size

            peer_buffer = workspace.get_buffer(
                rank, (chunk_size,), tensor.dtype,
                storage_offset=numel + start
            )
            my_symm_buffer[start:end].add_(peer_buffer)

        else:
            # AG step
            ag_step = state - RS_STEPS - 2
            idx = (my_chunk_idx - ag_step + GPUS_PER_SOCKET) % GPUS_PER_SOCKET
            start = pipe_offset + idx * chunk_size
            end = start + chunk_size

            prev_buffer = workspace.get_buffer(
                prev_rank, (chunk_size,), tensor.dtype, storage_offset=start
            )
            my_symm_buffer[start:end].copy_(prev_buffer)

    # Stagger pipeline starts to maximize overlap
    # Pipeline i starts at global_step = i
    # Total global steps = num_pipelines - 1 + TOTAL_STATES
    total_global_steps = num_pipelines - 1 + TOTAL_STATES

    for global_step in range(total_global_steps):
        # Each pipeline's current state
        for pipe_idx in range(num_pipelines):
            pipe_state = global_step - pipe_idx
            do_pipeline_step(pipe_idx, pipe_state)

        workspace.barrier()

    # Final copy: symm buffer -> tensor_flat
    tensor_flat.copy_(my_symm_buffer)

    return tensor

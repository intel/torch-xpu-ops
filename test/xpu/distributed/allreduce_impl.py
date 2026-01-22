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


# =============================================================================
# Three-Level Hierarchical Allreduce Implementation
# Level 1: Intra-Switch (fastest P2P)
# Level 2: Intra-Socket cross-Switch (medium speed)
# Level 3: Cross-Socket (slowest, minimized)
# =============================================================================

def _intra_switch_reduce(
    tensor: torch.Tensor,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 1: Reduce within PCIe switch (fastest).
    GPU pairs: 0↔1, 2↔3, 4↔5, 6↔7

    Each GPU exchanges half of its data with its switch peer and reduces.
    After this step, each GPU has reduced data from both GPUs in the switch.
    """
    peer_rank = get_same_switch_peer(rank)
    numel = tensor.numel()
    half_size = numel // 2

    # Determine which half this rank is responsible for
    # Lower rank in switch handles first half, higher rank handles second half
    switch_ranks = get_switch_ranks(rank)
    is_lower_rank = (rank == switch_ranks[0])
    local_pos = 0 if is_lower_rank else 1  # Position within switch pair (0 or 1)
    peer_local_pos = 1 if is_lower_rank else 0

    my_half_start = 0 if is_lower_rank else half_size

    # Push my responsible half to peer's workspace
    # Use local_pos as slot identifier (0 or 1), multiplied by half_size
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (half_size,),
        tensor.dtype,
        storage_offset=local_pos * half_size
    )
    remote_buffer.copy_(tensor[my_half_start:my_half_start + half_size])

    workspace.barrier()

    # Read peer's contribution and reduce
    peer_buffer = workspace.get_buffer(
        rank,
        (half_size,),
        tensor.dtype,
        storage_offset=peer_local_pos * half_size
    )

    # Reduce: my_half = my_data + peer_data
    result = tensor[my_half_start:my_half_start + half_size] + peer_buffer

    return result, my_half_start


def _intra_socket_cross_switch_reduce(
    local_chunk: torch.Tensor,
    chunk_offset: int,
    workspace,
    rank: int,
) -> tuple[torch.Tensor, int]:
    """
    Level 2: Reduce across switches within socket (medium speed).
    Switch pairs: (0,1)↔(2,3) in Socket 0, (4,5)↔(6,7) in Socket 1

    After switch-level reduce, we now reduce across switches in the same socket.
    """
    socket_id = get_socket_id(rank)
    switch_id = get_switch_id(rank)
    chunk_size = local_chunk.numel()
    half_size = chunk_size // 2

    # Determine peer switch within same socket
    # Socket 0: Switch 0 ↔ Switch 1, Socket 1: Switch 2 ↔ Switch 3
    if socket_id == 0:
        peer_switch = 1 if switch_id == 0 else 0
    else:
        peer_switch = 3 if switch_id == 2 else 2

    # Get corresponding rank in peer switch (same position within switch)
    switch_ranks = get_switch_ranks(rank)
    local_pos_in_switch = switch_ranks.index(rank)
    peer_switch_ranks = SWITCH_RANKS[peer_switch]
    peer_rank = peer_switch_ranks[local_pos_in_switch]

    # Determine which half this switch handles
    is_lower_switch = (switch_id < peer_switch)
    my_half_start = 0 if is_lower_switch else half_size

    # Use socket-local slot: 0,1 for lower switch ranks, 2,3 for higher switch ranks
    # Each rank in socket gets a unique slot: 0,1,2,3
    socket_local_rank = rank % GPUS_PER_SOCKET  # 0-3 for each socket
    peer_socket_local_rank = peer_rank % GPUS_PER_SOCKET

    # Push my half to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (half_size,),
        local_chunk.dtype,
        storage_offset=socket_local_rank * half_size
    )
    remote_buffer.copy_(local_chunk[my_half_start:my_half_start + half_size])

    workspace.barrier()

    # Read and reduce
    peer_buffer = workspace.get_buffer(
        rank,
        (half_size,),
        local_chunk.dtype,
        storage_offset=peer_socket_local_rank * half_size
    )

    result = local_chunk[my_half_start:my_half_start + half_size] + peer_buffer
    new_offset = chunk_offset + my_half_start

    return result, new_offset


def _cross_socket_reduce(
    local_chunk: torch.Tensor,
    chunk_offset: int,
    workspace,
    rank: int,
) -> tuple[torch.Tensor, int]:
    """
    Level 3: Reduce across sockets (slowest - minimized data!).
    Pairs: 0↔4, 1↔5, 2↔6, 3↔7

    At this point, each GPU only has 1/4 of the original data to exchange.
    """
    peer_rank = get_cross_socket_peer(rank)
    chunk_size = local_chunk.numel()
    half_size = chunk_size // 2

    socket_id = get_socket_id(rank)
    is_lower_socket = (socket_id == 0)
    my_half_start = 0 if is_lower_socket else half_size

    # Use socket-local rank as slot identifier (0-3)
    socket_local_rank = rank % GPUS_PER_SOCKET
    peer_socket_local_rank = peer_rank % GPUS_PER_SOCKET

    # Push my half to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (half_size,),
        local_chunk.dtype,
        storage_offset=socket_local_rank * half_size
    )
    remote_buffer.copy_(local_chunk[my_half_start:my_half_start + half_size])

    workspace.barrier()

    # Read and reduce
    peer_buffer = workspace.get_buffer(
        rank,
        (half_size,),
        local_chunk.dtype,
        storage_offset=peer_socket_local_rank * half_size
    )

    result = local_chunk[my_half_start:my_half_start + half_size] + peer_buffer
    new_offset = chunk_offset + my_half_start

    return result, new_offset


def _cross_socket_allgather(
    reduced_chunk: torch.Tensor,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 3 Allgather: Broadcast back across sockets.
    """
    peer_rank = get_cross_socket_peer(rank)
    chunk_size = reduced_chunk.numel()

    # Use socket-local rank as slot identifier (0-3)
    socket_local_rank = rank % GPUS_PER_SOCKET
    peer_socket_local_rank = peer_rank % GPUS_PER_SOCKET

    # Push to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (chunk_size,),
        reduced_chunk.dtype,
        storage_offset=socket_local_rank * chunk_size
    )
    remote_buffer.copy_(reduced_chunk)

    workspace.barrier()

    # Read peer's data
    peer_buffer = workspace.get_buffer(
        rank,
        (chunk_size,),
        reduced_chunk.dtype,
        storage_offset=peer_socket_local_rank * chunk_size
    )

    # Combine: create tensor with both halves
    socket_id = get_socket_id(rank)
    is_lower_socket = (socket_id == 0)

    combined = torch.empty(chunk_size * 2, dtype=reduced_chunk.dtype, device=reduced_chunk.device)
    if is_lower_socket:
        combined[:chunk_size] = reduced_chunk
        combined[chunk_size:] = peer_buffer
    else:
        combined[:chunk_size] = peer_buffer
        combined[chunk_size:] = reduced_chunk

    return combined


def _intra_socket_cross_switch_allgather(
    chunk: torch.Tensor,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 2 Allgather: Broadcast back across switches within socket.
    """
    socket_id = get_socket_id(rank)
    switch_id = get_switch_id(rank)
    chunk_size = chunk.numel()

    # Determine peer switch
    if socket_id == 0:
        peer_switch = 1 if switch_id == 0 else 0
    else:
        peer_switch = 3 if switch_id == 2 else 2

    switch_ranks = get_switch_ranks(rank)
    local_pos = switch_ranks.index(rank)
    peer_switch_ranks = SWITCH_RANKS[peer_switch]
    peer_rank = peer_switch_ranks[local_pos]

    # Use socket-local rank as slot identifier (0-3)
    socket_local_rank = rank % GPUS_PER_SOCKET
    peer_socket_local_rank = peer_rank % GPUS_PER_SOCKET

    # Push to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (chunk_size,),
        chunk.dtype,
        storage_offset=socket_local_rank * chunk_size
    )
    remote_buffer.copy_(chunk)

    workspace.barrier()

    # Read peer's data
    peer_buffer = workspace.get_buffer(
        rank,
        (chunk_size,),
        chunk.dtype,
        storage_offset=peer_socket_local_rank * chunk_size
    )

    # Combine
    is_lower_switch = (switch_id < peer_switch)
    combined = torch.empty(chunk_size * 2, dtype=chunk.dtype, device=chunk.device)
    if is_lower_switch:
        combined[:chunk_size] = chunk
        combined[chunk_size:] = peer_buffer
    else:
        combined[:chunk_size] = peer_buffer
        combined[chunk_size:] = chunk

    return combined


def _intra_switch_allgather(
    chunk: torch.Tensor,
    workspace,
    rank: int,
) -> torch.Tensor:
    """
    Level 1 Allgather: Broadcast back within PCIe switch.
    """
    peer_rank = get_same_switch_peer(rank)
    chunk_size = chunk.numel()

    # Use switch-local position as slot (0 or 1)
    switch_ranks = get_switch_ranks(rank)
    is_lower_rank = (rank == switch_ranks[0])
    local_pos = 0 if is_lower_rank else 1
    peer_local_pos = 1 if is_lower_rank else 0

    # Push to peer
    remote_buffer = workspace.get_buffer(
        peer_rank,
        (chunk_size,),
        chunk.dtype,
        storage_offset=local_pos * chunk_size
    )
    remote_buffer.copy_(chunk)

    workspace.barrier()

    # Read peer's data
    peer_buffer = workspace.get_buffer(
        rank,
        (chunk_size,),
        chunk.dtype,
        storage_offset=peer_local_pos * chunk_size
    )

    # Combine
    combined = torch.empty(chunk_size * 2, dtype=chunk.dtype, device=chunk.device)
    if is_lower_rank:
        combined[:chunk_size] = chunk
        combined[chunk_size:] = peer_buffer
    else:
        combined[:chunk_size] = peer_buffer
        combined[chunk_size:] = chunk

    return combined


def hierarchical_allreduce_with_symm_mem(
    tensor: torch.Tensor,
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Perform 3-level hierarchical allreduce optimized for 8-GPU dual-socket topology.

    Topology (hardcoded):
        Socket 0: Switch 0 (GPU 0,1), Switch 1 (GPU 2,3)
        Socket 1: Switch 2 (GPU 4,5), Switch 3 (GPU 6,7)

    Algorithm (Reduce phase):
        1. Intra-Switch reduce: 0↔1, 2↔3, 4↔5, 6↔7 (fastest P2P)
        2. Intra-Socket cross-Switch reduce: SW0↔SW1, SW2↔SW3 (medium)
        3. Cross-Socket reduce: Socket0↔Socket1 (slowest, only 1/4 data!)

    Algorithm (Allgather phase):
        4. Cross-Socket allgather: Socket0↔Socket1
        5. Intra-Socket cross-Switch allgather: SW0↔SW1, SW2↔SW3
        6. Intra-Switch allgather: 0↔1, 2↔3, 4↔5, 6↔7

    Cross-socket data transfer is reduced to 1/4 of original!

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

    # Validate world size matches hardcoded topology
    if world_size != WORLD_SIZE:
        raise ValueError(
            f"World size ({world_size}) must be {WORLD_SIZE} for this topology"
        )

    # Work with flattened view
    tensor_flat = tensor.view(-1)
    numel = tensor_flat.numel()

    # Check divisibility by world_size (for reduce-scatter/allgather)
    if numel % world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by world_size ({world_size})"
        )

    # Get symmetric memory workspace
    workspace_size_bytes = numel * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Initial barrier
    workspace.barrier()

    # =========================================================================
    # REDUCE PHASE: Bottom-up (Switch → Socket → Cross-Socket)
    # =========================================================================

    # Step 1: Intra-Switch reduce (fastest - same PCIe switch)
    # Data size: N → N/2 per GPU
    switch_reduced, offset1 = _intra_switch_reduce(tensor_flat, workspace, rank)

    workspace.barrier()

    # Step 2: Intra-Socket cross-Switch reduce (medium speed)
    # Data size: N/2 → N/4 per GPU
    socket_reduced, offset2 = _intra_socket_cross_switch_reduce(
        switch_reduced, offset1, workspace, rank
    )

    workspace.barrier()

    # Step 3: Cross-Socket reduce (slowest - only N/4 data transferred!)
    # Data size: N/4 → N/8 per GPU (final reduced chunk)
    final_reduced, _ = _cross_socket_reduce(
        socket_reduced, offset2, workspace, rank
    )

    workspace.barrier()

    # =========================================================================
    # ALLGATHER PHASE: Top-down (Cross-Socket → Socket → Switch)
    # =========================================================================

    # Step 4: Cross-Socket allgather
    # Data size: N/8 → N/4 per GPU
    cross_gathered = _cross_socket_allgather(final_reduced, workspace, rank)

    workspace.barrier()

    # Step 5: Intra-Socket cross-Switch allgather
    # Data size: N/4 → N/2 per GPU
    socket_gathered = _intra_socket_cross_switch_allgather(cross_gathered, workspace, rank)

    workspace.barrier()

    # Step 6: Intra-Switch allgather (fastest)
    # Data size: N/2 → N per GPU
    full_result = _intra_switch_allgather(socket_gathered, workspace, rank)

    # Copy result back to original tensor
    tensor.copy_(full_result.view(tensor.shape))

    return tensor

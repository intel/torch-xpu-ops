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
    debug: bool = False,
) -> torch.Tensor:
    """
    Perform allreduce using symmetric memory optimized for 8-GPU dual-switch topology.

    Topology (8 GPUs, 2 Switches):
        Switch 0: GPU 0, 1, 2, 3
        Switch 1: GPU 4, 5, 6, 7

    Bandwidth characteristics:
        - Intra-switch read:  28 GB/s (prefer read/pull for intra-switch)
        - Intra-switch write: 22 GB/s
        - Cross-switch write: 22 GB/s (prefer write/push for cross-switch)
        - Cross-switch read:  17 GB/s

    Algorithm using RING pattern to avoid bandwidth congestion:
        Phase 1: Ring reduce-scatter within each switch (3 steps)
                 - Uses push (22 GB/s write) in ring pattern: 0→1→2→3→0
                 - Each step: push 1 chunk to next rank, reduce 1 chunk from prev rank
                 - Result: each rank has partial sum of its chunk from 4 ranks in switch

        Phase 2: Cross-switch exchange (1 step)
                 - Uses push (22 GB/s write) to exchange partial sums
                 - Rank i ↔ Rank i+4 exchange their chunks
                 - Result: each rank has fully reduced chunk (sum from all 8 ranks)

        Phase 3: Ring allgather within switch + cross-switch exchange
                 - Push reduced chunk to cross_peer (22 GB/s write)
                 - Ring allgather using pull (28 GB/s read): 3 steps
                 - Result: all ranks have all 8 reduced chunks

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)
        debug: If True, print timing information for each phase

    Returns:
        Reduced tensor with the same shape as input (in-place modification)
    """
    import time

    def log_time(msg, start_time, rank, do_sync=True):
        if do_sync:
            torch.xpu.synchronize()
        elapsed = (time.perf_counter() - start_time) * 1000
        if rank == 0:
            print(f"[Rank {rank}] {msg}: {elapsed:.3f} ms")
        return time.perf_counter()

    if op != "sum":
        raise ValueError(f"Only 'sum' operation is supported, got '{op}'")

    # Get group info
    if group is None:
        group = dist.group.WORLD

    group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if world_size != 8:
        raise ValueError(
            f"World size ({world_size}) must be 8 for dual-switch topology"
        )

    # Work with flattened view
    tensor_flat = tensor.view(-1)
    numel = tensor_flat.numel()

    # Check divisibility by world_size
    if numel % world_size != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by world_size ({world_size})"
        )

    chunk_size = numel // world_size

    # Get symmetric memory workspace
    workspace_size_bytes = numel * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Barrier to ensure previous allreduce is complete
    workspace.barrier()

    if debug:
        torch.xpu.synchronize()
        t_start = time.perf_counter()

    # Topology info
    switch_id = rank // 4  # 0 for ranks 0-3, 1 for ranks 4-7
    cross_switch_peer = rank ^ 4  # 0↔4, 1↔5, 2↔6, 3↔7

    # Switch base rank
    switch_base = switch_id * 4

    # Chunk positions
    my_chunk_start = rank * chunk_size
    my_chunk_end = my_chunk_start + chunk_size

    # ==========================================================================
    # Phase 1: Ring-based reduce-scatter within each switch
    # Use ring pattern to avoid bandwidth congestion
    # Each step: push chunk to next rank in ring, receive from previous rank
    # local_rank: 0,1,2,3 within each switch
    # Ring: 0→1→2→3→0
    # ==========================================================================
    local_rank = rank % 4

    # Ring reduce-scatter: 3 steps for 4 ranks in each switch
    for step in range(3):
        # Ring pattern: send to (local_rank + 1) % 4, receive from (local_rank - 1) % 4
        send_to_local = (local_rank + 1) % 4
        send_to = switch_base + send_to_local

        # Which chunk to send: in step i, send chunk that will end up at rank (local_rank - step) % 4
        chunk_idx_local = (local_rank - step) % 4
        chunk_idx = switch_base + chunk_idx_local
        chunk_start = chunk_idx * chunk_size

        # Push to next rank in ring
        remote_buffer = workspace.get_buffer(
            send_to, (chunk_size,), tensor.dtype,
            storage_offset=chunk_idx * chunk_size  # store at the chunk's natural position
        )
        remote_buffer.copy_(tensor_flat[chunk_start:chunk_start + chunk_size])

        if debug and step == 0:
            t_start = log_time("Phase1: ring push step 0", t_start, rank)

        workspace.barrier()

        if debug and step == 0:
            t_start = log_time("Phase1: barrier step 0", t_start, rank)

        # Reduce received chunk into my tensor
        recv_chunk_idx_local = (local_rank - step - 1) % 4
        recv_chunk_idx = switch_base + recv_chunk_idx_local
        recv_chunk_start = recv_chunk_idx * chunk_size

        received = workspace.get_buffer(
            rank, (chunk_size,), tensor.dtype,
            storage_offset=recv_chunk_idx * chunk_size
        )
        tensor_flat[recv_chunk_start:recv_chunk_start + chunk_size] += received

    if debug:
        t_start = log_time("Phase1: ring reduce-scatter (3 steps)", t_start, rank)

    # Now each rank has partial sum for its chunk (sum of 4 ranks in same switch)
    # my_chunk = tensor_flat[my_chunk_start:my_chunk_end] has sum from my switch

    # ==========================================================================
    # Phase 2: Cross-switch exchange
    # Each rank exchanges its partial sum with cross-switch peer
    # Use push (22 GB/s) instead of pull (17 GB/s)
    # ==========================================================================

    # Push my partial sum (my_chunk) to cross_peer
    remote_buffer = workspace.get_buffer(
        cross_switch_peer, (chunk_size,), tensor.dtype,
        storage_offset=rank * chunk_size  # store at my rank's slot
    )
    remote_buffer.copy_(tensor_flat[my_chunk_start:my_chunk_end])

    if debug:
        t_start = log_time("Phase2: cross-switch push", t_start, rank)

    workspace.barrier()

    if debug:
        t_start = log_time("Phase2: barrier", t_start, rank)

    # Reduce received partial sum from cross_peer
    received = workspace.get_buffer(
        rank, (chunk_size,), tensor.dtype,
        storage_offset=cross_switch_peer * chunk_size
    )
    tensor_flat[my_chunk_start:my_chunk_end] += received

    if debug:
        t_start = log_time("Phase2: cross-switch reduce", t_start, rank)

    # Now tensor_flat[my_chunk] has the fully reduced result (sum from all 8 ranks)

    # ==========================================================================
    # Phase 3: Ring-based allgather within each switch + cross-switch exchange
    # First: write my reduced chunk to my buffer and push to cross_peer
    # Then: ring allgather within switch
    # ==========================================================================

    # Write my reduced chunk to my symm buffer
    my_slot = workspace.get_buffer(
        rank, (chunk_size,), tensor.dtype, storage_offset=my_chunk_start
    )
    my_slot.copy_(tensor_flat[my_chunk_start:my_chunk_end])

    # Push my reduced chunk to cross_switch_peer (22 GB/s write)
    remote_buffer = workspace.get_buffer(
        cross_switch_peer, (chunk_size,), tensor.dtype,
        storage_offset=my_chunk_start
    )
    remote_buffer.copy_(tensor_flat[my_chunk_start:my_chunk_end])

    if debug:
        t_start = log_time("Phase3: write local + cross-switch push", t_start, rank)

    workspace.barrier()

    if debug:
        t_start = log_time("Phase3: barrier", t_start, rank)

    # Read cross_peer's chunk from my buffer (it was pushed to me)
    cross_peer_chunk_start = cross_switch_peer * chunk_size
    tensor_flat[cross_peer_chunk_start:cross_peer_chunk_start + chunk_size].copy_(
        workspace.get_buffer(rank, (chunk_size,), tensor.dtype, storage_offset=cross_peer_chunk_start)
    )

    if debug:
        t_start = log_time("Phase3: read cross-peer from local", t_start, rank)

    # Ring allgather within switch: 3 steps for 4 ranks
    # Use pull (28 GB/s read) which is faster than push (22 GB/s write)
    for step in range(3):
        # Ring pattern: pull from (local_rank - 1 - step) % 4
        pull_from_local = (local_rank - 1 - step) % 4
        pull_from = switch_base + pull_from_local
        pull_chunk_start = pull_from * chunk_size

        peer_buffer = workspace.get_buffer(
            pull_from, (chunk_size,), tensor.dtype, storage_offset=pull_chunk_start
        )
        tensor_flat[pull_chunk_start:pull_chunk_start + chunk_size].copy_(peer_buffer)

        if debug and step == 0:
            t_start = log_time("Phase3: ring pull step 0", t_start, rank)

        # Need barrier between steps to ensure data is ready
        if step < 2:
            workspace.barrier()

    if debug:
        t_start = log_time("Phase3: ring allgather (3 steps)", t_start, rank)

    workspace.barrier()

    if debug:
        log_time("Phase3: final barrier", t_start, rank)

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
) -> torch.Tensor:
    """
    Perform 2-level hierarchical allreduce optimized for 8-GPU dual-socket topology.

    Topology (hardcoded):
        Socket 0: GPU 0, 1, 2, 3 (equal bandwidth within socket)
        Socket 1: GPU 4, 5, 6, 7 (equal bandwidth within socket)

    Algorithm:
        1. Intra-socket reduce-scatter: Each GPU gets 1/4 of socket-reduced data
        2. Cross-socket reduce: Exchange & reduce with peer (0↔4, 1↔5, 2↔6, 3↔7)
           - After this, each chunk is FULLY reduced (sum of all 8 GPUs)
           - Both sockets now have the same 4 chunks (no exchange needed!)
        3. Intra-socket allgather: Gather all 4 chunks within socket

    Cross-socket data transfer is reduced to N/4 (only 1 chunk per GPU)!

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)

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

    # Must be divisible by GPUS_PER_SOCKET (4) for reduce-scatter
    if numel % GPUS_PER_SOCKET != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by {GPUS_PER_SOCKET}"
        )

    # Get symmetric memory workspace
    workspace_size_bytes = numel * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    workspace.barrier()

    # =========================================================================
    # Step 1: Intra-socket reduce-scatter
    # Data: N → N/4 per GPU (reduced within socket)
    # =========================================================================
    my_chunk, my_chunk_idx = _intra_socket_reduce_scatter(tensor_flat, workspace, rank)

    workspace.barrier()

    # =========================================================================
    # Step 2: Cross-socket reduce (MINIMIZED - only N/4 data!)
    # Data: N/4 per GPU, now fully reduced across all 8 GPUs
    # =========================================================================
    fully_reduced = _cross_socket_reduce(my_chunk, workspace, rank)

    workspace.barrier()

    # =========================================================================
    # Step 3: Intra-socket allgather
    # Each socket has 4 GPUs with 4 different fully-reduced chunks
    # Gather all 4 chunks within socket → N elements
    # No cross-socket exchange needed! Both sockets have same 4 chunks.
    # =========================================================================
    result = _intra_socket_allgather(fully_reduced, my_chunk_idx, workspace, rank)

    workspace.barrier()

    # Copy result back to original tensor
    tensor.copy_(result.view(tensor.shape))

    return tensor

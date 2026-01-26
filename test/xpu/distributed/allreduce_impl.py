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

    # Barrier to ensure previous allreduce is complete before writing
    workspace.barrier()

    # Step 1: Ring Scatter - Each rank pushes to remote ranks in ring fashion
    # Reference: PyTorch _low_contention_reduce_scatter_with_workspace implementation
    # Each rank pushes chunk[remote_rank] to remote_rank's symm buffer
    for step in range(local_world_size - 1):
        remote_rank = (local_rank - step - 1) % local_world_size
        remote_buffer = workspace.get_buffer(
            remote_rank + switch_id * 4,
            (chunk_size,),
            tensor.dtype,
            storage_offset=local_rank * chunk_size
        )
        remote_buffer.copy_(tensor_flat[remote_rank * chunk_size:(remote_rank + 1) * chunk_size])

    # Barrier after each ring step to ensure data is received before next push
    workspace.barrier()

    # Step 2: Reduce - Single kernel reduction
    chunk_start = local_rank * chunk_size
    chunk_end = (local_rank + 1) * chunk_size

    # First, copy local chunk to symm[rank] position (symm[rank] was 0)
    my_slot = workspace.get_buffer(
        rank, (chunk_size,), tensor.dtype, storage_offset=local_rank * chunk_size
    )
    my_slot.copy_(tensor_flat[chunk_start:chunk_end])

    # View symm as [world_size, chunk_size] and reduce with single kernel
    my_symm_2d = workspace.get_buffer(
        rank, (local_world_size, chunk_size), tensor.dtype, storage_offset=0
    )

    # Single reduction kernel: sum along dim 0
    torch.sum(my_symm_2d, dim=0, out=tensor_flat[chunk_start:chunk_end])

    # Write reduced result back to symm[rank] for allgather
    my_slot.copy_(tensor_flat[chunk_start:chunk_end])

    workspace.barrier()

    my_chunk_idx = local_rank
    my_chunk_start = my_chunk_idx * chunk_size
    my_chunk_end = my_chunk_start + chunk_size

    # Calculate offsets for front and back halves of my chunk
    front_half_start = my_chunk_start
    front_half_end = my_chunk_start + half_chunk_size
    back_half_start = my_chunk_start + half_chunk_size
    back_half_end = my_chunk_end

    # Step 2a: Exchange - send the half I don't own to peer
    # Timing for cross-socket exchange (lines 217-271)
    import time
    torch.xpu.synchronize()
    cross_switch_start = time.perf_counter()

    # Cross-switch exchange for 8 ranks
    if world_size > 4:
        # Step 2a: Exchange halves between sockets
        # Socket 0 sends back half, Socket 1 sends front half
        if switch_id == 0:
            # Socket 0 rank (ranks 0-3): send back half to cross_switch_peer
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=local_rank * chunk_size  # use local_rank slot
            )
            remote_buffer.copy_(tensor_flat[back_half_start:back_half_end])
        else:
            # Socket 1 rank (ranks 4-7): send front half to cross_switch_peer
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=local_rank * chunk_size  # use local_rank slot
            )
            remote_buffer.copy_(tensor_flat[front_half_start:front_half_end])

        # Barrier 1: ensure all initial sends are complete
        workspace.barrier()

        # Step 2b: Reduce the half I own with received data
        received = workspace.get_buffer(
            rank, (half_chunk_size,), tensor.dtype,
            storage_offset=local_rank * chunk_size
        )

        if switch_id == 0:
            # Socket 0 rank: reduce front half with received front half from peer
            tensor_flat[front_half_start:front_half_end] += received

            # Socket 0 rank: update my own symm buffer for allgather
            my_slot = workspace.get_buffer(
                rank, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start  # front half position
            )
            my_slot.copy_(tensor_flat[front_half_start:front_half_end])

            # Socket 0 rank: write front half result to peer's front half position
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start  # front half position in peer's buffer
            )
            remote_buffer.copy_(tensor_flat[front_half_start:front_half_end])
        else:
            # Socket 1 rank: reduce back half with received back half from peer
            tensor_flat[back_half_start:back_half_end] += received

            # Socket 1 rank: update my own symm buffer for allgather
            my_slot = workspace.get_buffer(
                rank, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start + half_chunk_size  # back half position
            )
            my_slot.copy_(tensor_flat[back_half_start:back_half_end])

            # Socket 1 rank: write back half result to peer's back half position
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start + half_chunk_size  # back half position
            )
            remote_buffer.copy_(tensor_flat[back_half_start:back_half_end])

        # Barrier 2: ensure all cross-switch writes are complete
        workspace.barrier()

        # Step 2c: Read the half written by peer
        if switch_id == 0:
            # Socket 0 rank: read back half from symm buffer (written by peer)
            received_back_half = workspace.get_buffer(
                rank, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start + half_chunk_size  # back half position
            )
            tensor_flat[back_half_start:back_half_end].copy_(received_back_half)
        else:
            # Socket 1 rank: read front half from symm buffer (written by peer)
            received_front_half = workspace.get_buffer(
                rank, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start  # front half position
            )
            tensor_flat[front_half_start:front_half_end].copy_(received_front_half)
    cross_switch_end = time.perf_counter()
    cross_switch_cost_ms = (cross_switch_end - cross_switch_start) * 1000
    if rank == 0:
        print(f"[Rank {rank}] Cross-switch exchange cost: {cross_switch_cost_ms:.3f} ms")

    # Step 3: Ring Allgather - Each rank pulls from remote ranks in ring fashion
    # Reference: PyTorch _low_contention_all_gather implementation
    # Each rank pulls chunk[remote_rank] from remote_rank's symm buffer
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
    num_pipelines: int = 2,
) -> torch.Tensor:
    """
    Perform allreduce with pipelined cross-socket and intra-socket communication.

    This implementation overlaps cross-socket communication with intra-socket
    communication by splitting the data into multiple pipeline stages.

    Pipeline design (for 2 stages, 8 ranks):
    =========================================================================
    Time ->
    -------------------------------------------------------------------------
    Pipe 0: [Scatter] -> [Reduce] -> [Cross-Socket Send] -> [Cross-Socket Recv+Reduce] -> [Allgather]
    Pipe 1:              [Scatter] -> [Reduce] -> [Cross-Socket Send] -> [Cross-Socket Recv+Reduce] -> [Allgather]
    -------------------------------------------------------------------------

    The key insight: While Pipe 0 is doing cross-socket communication (slow),
    Pipe 1 can do intra-socket scatter/reduce (fast). This hides cross-socket latency.

    Args:
        tensor: Input tensor to reduce (must be on XPU device)
        op: Reduction operation (only "sum" is supported)
        group: Process group (default: None, uses WORLD group)
        num_pipelines: Number of pipeline stages (default: 2)

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
    if numel % (world_size * num_pipelines) != 0:
        raise ValueError(
            f"Tensor size ({numel}) must be divisible by world_size * num_pipelines ({world_size * num_pipelines})"
        )

    # For 8 ranks (2 sockets): local_world_size = 4 (per socket)
    # For 4 ranks or less: local_world_size = world_size (single socket)
    local_world_size = world_size // 2 if world_size > 4 else world_size

    # Each pipeline stage handles a portion of the data
    pipeline_chunk_size = numel // num_pipelines
    chunk_size = pipeline_chunk_size // local_world_size
    half_chunk_size = chunk_size // 2

    # Get symmetric memory workspace (need space for all pipeline stages)
    workspace_size_bytes = numel * tensor.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Topology info
    switch_id = rank // 4  # 0 for ranks 0-3, 1 for ranks 4-7
    local_rank = rank % 4  # 0-3 within each switch
    cross_switch_peer = rank ^ 4  # 0↔4, 1↔5, 2↔6, 3↔7

    # Barrier to ensure previous allreduce is complete before writing
    workspace.barrier()

    # Helper functions for each pipeline stage operation
    def do_scatter(pipe_idx: int):
        """Ring scatter for a pipeline stage."""
        pipe_offset = pipe_idx * pipeline_chunk_size
        for step in range(local_world_size - 1):
            remote_rank = (local_rank - step - 1) % local_world_size
            remote_buffer = workspace.get_buffer(
                remote_rank + switch_id * 4,
                (chunk_size,),
                tensor.dtype,
                storage_offset=pipe_offset + local_rank * chunk_size
            )
            src_start = pipe_offset + remote_rank * chunk_size
            remote_buffer.copy_(tensor_flat[src_start:src_start + chunk_size])

    def do_local_reduce(pipe_idx: int):
        """Local reduce for a pipeline stage."""
        pipe_offset = pipe_idx * pipeline_chunk_size
        chunk_start = pipe_offset + local_rank * chunk_size
        chunk_end = chunk_start + chunk_size

        # Copy local chunk to symm buffer
        my_slot = workspace.get_buffer(
            rank, (chunk_size,), tensor.dtype,
            storage_offset=pipe_offset + local_rank * chunk_size
        )
        my_slot.copy_(tensor_flat[chunk_start:chunk_end])

        # Reduce from all local ranks
        my_symm_2d = workspace.get_buffer(
            rank, (local_world_size, chunk_size), tensor.dtype,
            storage_offset=pipe_offset
        )
        torch.sum(my_symm_2d, dim=0, out=tensor_flat[chunk_start:chunk_end])

        # Write back for allgather
        my_slot.copy_(tensor_flat[chunk_start:chunk_end])

    def do_cross_socket_send(pipe_idx: int):
        """Cross-socket send for a pipeline stage."""
        if world_size <= 4:
            return
        pipe_offset = pipe_idx * pipeline_chunk_size
        my_chunk_start = pipe_offset + local_rank * chunk_size
        front_half_start = my_chunk_start
        back_half_start = my_chunk_start + half_chunk_size

        if switch_id:
            # Socket 1: send front half
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=pipe_offset + local_rank * chunk_size
            )
            remote_buffer.copy_(tensor_flat[front_half_start:front_half_start + half_chunk_size])
        else:
            # Socket 0: send back half
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=pipe_offset + local_rank * chunk_size
            )
            remote_buffer.copy_(tensor_flat[back_half_start:back_half_start + half_chunk_size])

    def do_cross_socket_recv_reduce(pipe_idx: int):
        """Cross-socket receive and reduce for a pipeline stage."""
        if world_size <= 4:
            return
        pipe_offset = pipe_idx * pipeline_chunk_size
        my_chunk_start = pipe_offset + local_rank * chunk_size
        front_half_start = my_chunk_start
        back_half_start = my_chunk_start + half_chunk_size

        received = workspace.get_buffer(
            rank, (half_chunk_size,), tensor.dtype,
            storage_offset=pipe_offset + local_rank * chunk_size
        )

        if switch_id:
            # Socket 1: reduce back half, send to peer
            tensor_flat[back_half_start:back_half_start + half_chunk_size] += received
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start + half_chunk_size
            )
            remote_buffer.copy_(tensor_flat[back_half_start:back_half_start + half_chunk_size])
        else:
            # Socket 0: reduce front half, send to peer
            tensor_flat[front_half_start:front_half_start + half_chunk_size] += received
            remote_buffer = workspace.get_buffer(
                cross_switch_peer, (half_chunk_size,), tensor.dtype,
                storage_offset=my_chunk_start
            )
            remote_buffer.copy_(tensor_flat[front_half_start:front_half_start + half_chunk_size])

    def do_allgather(pipe_idx: int):
        """Ring allgather for a pipeline stage."""
        pipe_offset = pipe_idx * pipeline_chunk_size
        for step in range(local_world_size - 1):
            remote_rank = (local_rank - step - 1) % local_world_size
            remote_buffer = workspace.get_buffer(
                remote_rank + switch_id * 4,
                (chunk_size,),
                tensor.dtype,
                storage_offset=pipe_offset + remote_rank * chunk_size
            )
            dst_start = pipe_offset + remote_rank * chunk_size
            tensor_flat[dst_start:dst_start + chunk_size].copy_(remote_buffer)

    # ==========================================================================
    # Pipelined Execution with Stream Overlap
    # ==========================================================================
    # Create streams for each pipeline stage to enable true overlap
    streams = [torch.xpu.Stream() for _ in range(num_pipelines)]
    current_stream = torch.xpu.current_stream()

    # Phase 1: Scatter - each pipeline on its own stream
    for pipe_idx in range(num_pipelines):
        streams[pipe_idx].wait_stream(current_stream)
        with torch.xpu.stream(streams[pipe_idx]):
            do_scatter(pipe_idx)

    # Current stream waits for all pipeline streams before barrier
    for s in streams:
        current_stream.wait_stream(s)
    workspace.barrier()

    # Phase 2: Local reduce - each pipeline on its own stream
    for pipe_idx in range(num_pipelines):
        streams[pipe_idx].wait_stream(current_stream)
        with torch.xpu.stream(streams[pipe_idx]):
            do_local_reduce(pipe_idx)

    # Current stream waits for all pipeline streams before barrier
    for s in streams:
        current_stream.wait_stream(s)
    workspace.barrier()

    # Phase 3: Pipelined cross-socket exchange
    if world_size > 4:
        # Send phase: all pipelines send on their streams (overlapped)
        for pipe_idx in range(num_pipelines):
            streams[pipe_idx].wait_stream(current_stream)
            with torch.xpu.stream(streams[pipe_idx]):
                do_cross_socket_send(pipe_idx)

        # Current stream waits for all pipeline streams before barrier
        for s in streams:
            current_stream.wait_stream(s)
        workspace.barrier()

        # Recv+Reduce phase: all pipelines receive and reduce (overlapped)
        for pipe_idx in range(num_pipelines):
            streams[pipe_idx].wait_stream(current_stream)
            with torch.xpu.stream(streams[pipe_idx]):
                do_cross_socket_recv_reduce(pipe_idx)

        # Current stream waits for all pipeline streams before barrier
        for s in streams:
            current_stream.wait_stream(s)
        workspace.barrier()

    # Phase 4: Allgather - each pipeline on its own stream
    for pipe_idx in range(num_pipelines):
        streams[pipe_idx].wait_stream(current_stream)
        with torch.xpu.stream(streams[pipe_idx]):
            do_allgather(pipe_idx)

    # Current stream waits for all pipeline streams before final barrier
    for s in streams:
        current_stream.wait_stream(s)
    workspace.barrier()

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

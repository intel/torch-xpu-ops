"""
Unpermute + reduce-scatter fusion using symmetric memory (TP only scenario).

This is the reverse of allgather + local permute fusion:
- Input: expert_output [num_tokens * topk, hidden] in expert-centric layout
         (all tokens for expert 0 first, then expert 1, etc.)
- Unpermute: gather from expert-sorted positions back to token order, weighted by topk_weights
         result[i] = sum_k( topk_weights[i,k] * expert_output[scatter_idx[i,k]] )
- Result: [num_tokens, hidden] partial sums (need reduce across TP ranks)

In TP, expert weights are column/row-parallelized across ranks, so each rank produces
partial sums that must be reduced. We fuse unpermute + reduce-scatter:
- Each rank computes its local unpermute chunk by chunk (num_tokens / world_size per chunk)
- After computing each chunk, immediately push to the target rank's symmetric memory
- Two streams alternate: while one stream pushes chunk[i], the other computes chunk[i+1]
- After all pushes + barrier, each rank sums received contributions from all ranks

Pipeline pattern (mirrors allgather_local_permute_fusion_python):
  allgather_permute:   pull remote shard → scatter-write (per remote rank, two-stream)
  unpermute_reduce:    compute chunk → push to remote (per remote rank, two-stream)

Inputs:
    expert_output: [num_tokens * topk, hidden] - output from expert computation
    scatter_idx: [num_tokens, topk] int32 - maps (token_i, k) to row in expert_output
    topk_weights: [num_tokens, topk] float - routing weights for weighted sum
    group: TP process group
    output: [num_tokens_per_rank, hidden] - reduced result for this rank's token shard
"""

import os

import ctypes

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Try to load the native unpermute kernel
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "libunpermute_reduce_scatter.so")
_HAS_LOCAL_UNPERMUTE_KERNEL = False
_HAS_UNPERMUTE_RS_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_UNPERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_unpermute_copy_")
        _HAS_UNPERMUTE_RS_KERNEL = hasattr(torch.ops.symm_mem, "unpermute_reduce_scatter")
    except Exception:
        pass


def unpermute_allreduce_simple(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    group: dist.ProcessGroup = None,
):
    """
    Simple unpermute + allreduce (no fusion, no pipeline).

    Steps:
      1. Weighted gather: for each token i, sum over k:
         result[i] = sum_k( topk_weights[i,k] * expert_output[scatter_idx[i,k]] )
      2. Allreduce result across TP group
      3. Return this rank's token shard

    Args:
        expert_output: [num_tokens * topk, hidden] expert-centric layout
        scatter_idx: [num_tokens, topk] int32 - source row for each (token, k)
        topk_weights: [num_tokens, topk] float - routing weights
        output: [num_tokens_per_rank, hidden] pre-allocated output buffer
        group: TP process group

    Returns:
        output: [num_tokens_per_rank, hidden] reduced result for this rank's shard
    """
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    num_tokens, topk = scatter_idx.shape
    hidden = expert_output.shape[1]
    num_tokens_per_rank = num_tokens // world_size

    # Step 1: Unpermute - weighted gather from expert-centric to token-centric
    full_result = torch.zeros(
        num_tokens, hidden, device=expert_output.device, dtype=expert_output.dtype
    )
    for i in range(num_tokens):
        for k in range(topk):
            src_row = scatter_idx[i, k].item()
            full_result[i] += topk_weights[i, k] * expert_output[src_row]

    # Step 2: Allreduce across TP group
    dist.all_reduce(full_result, group=group)

    # Step 3: Each rank takes its own token shard
    my_start = rank * num_tokens_per_rank
    output.copy_(full_result[my_start : my_start + num_tokens_per_rank])
    return output


def unpermute_reducescatter_fusion(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
):
    """
    Pipelined unpermute + reduce-scatter using symmetric memory and two streams.

    Algorithm:
      1. Split num_tokens into world_size chunks (each chunk = one rank's token shard)
      2. Compute unpermute for own chunk first → write directly to output
      3. For each remote rank (two-stream round-robin):
         - Compute unpermute for that rank's chunk
         - Push result to that rank's symmetric memory receive slot
      4. Barrier (ensure all pushes visible)
      5. Sum all received contributions into output

    The two-stream round-robin gives compute-comm overlap:
      - Step i on stream A: compute + push for target_rank_a
      - Step i+1 on stream B: compute + push for target_rank_b
      (adjacent steps execute concurrently on different streams)

    Workspace layout per rank: [world_size, num_tokens_per_rank, hidden]
      slot[i] = contribution received FROM rank i for this rank's token chunk

    Args:
        expert_output: [num_tokens * topk, hidden] expert-centric layout
        scatter_idx: [num_tokens, topk] int32 - source row for each (token, k)
        topk_weights: [num_tokens, topk] float - routing weights
        output: [num_tokens_per_rank, hidden] pre-allocated output buffer
        group: TP process group
        group_name: optional group name for symmetric memory workspace
        backend_stream: optional second stream for pipelining

    Returns:
        output: [num_tokens_per_rank, hidden] reduced result for this rank's shard
    """
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    num_tokens, topk = scatter_idx.shape
    hidden = expert_output.shape[1]
    num_tokens_per_rank = num_tokens // world_size

    # Symmetric memory: each rank has [world_size, num_tokens_per_rank, hidden]
    # rank r's slot[i] will receive rank i's partial unpermute for chunk r
    workspace_size_bytes = (
        world_size * num_tokens_per_rank * hidden * expert_output.element_size()
    )
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Phase 1: Compute own chunk's unpermute → write directly to output
    my_chunk_start = rank * num_tokens_per_rank
    output.zero_()
    for i in range(num_tokens_per_rank):
        global_i = my_chunk_start + i
        for k in range(topk):
            src_row = scatter_idx[global_i, k].item()
            output[i] += topk_weights[global_i, k] * expert_output[src_row]

    # Phase 2: Compute + push for each remote rank (two-stream round-robin)
    if backend_stream is None:
        backend_stream = torch.xpu.Stream()

    # Double buffer to avoid WAR hazard between adjacent steps on different streams
    local_bufs = [
        torch.zeros(num_tokens_per_rank, hidden, device=expert_output.device, dtype=expert_output.dtype),
        torch.zeros(num_tokens_per_rank, hidden, device=expert_output.device, dtype=expert_output.dtype),
    ]

    for step in range(world_size - 1):
        target_rank = (rank - step - 1) % world_size
        buf_idx = step % 2
        if step % 2 == 0:
            stream = backend_stream
        else:
            stream = torch.xpu.current_stream()
        with torch.xpu.stream(stream):
            # Compute unpermute for target_rank's token chunk
            target_chunk_start = target_rank * num_tokens_per_rank
            local_bufs[buf_idx].zero_()
            for i in range(num_tokens_per_rank):
                global_i = target_chunk_start + i
                for k in range(topk):
                    src_row = scatter_idx[global_i, k].item()
                    local_bufs[buf_idx][i] += topk_weights[global_i, k] * expert_output[src_row]

            # Push to target_rank's receive buffer at my slot
            target_recv_buf = workspace.get_buffer(
                target_rank,
                (world_size, num_tokens_per_rank, hidden),
                expert_output.dtype,
                storage_offset=0,
            )
            target_recv_buf[rank].copy_(local_bufs[buf_idx])

    torch.xpu.current_stream().wait_stream(backend_stream)
    workspace.barrier()

    # Phase 3: Sum all received contributions for my chunk
    my_recv_buf = workspace.get_buffer(
        rank,
        (world_size, num_tokens_per_rank, hidden),
        expert_output.dtype,
        storage_offset=0,
    )
    for i in range(world_size):
        if i != rank:
            output.add_(my_recv_buf[i])

    workspace.barrier()
    return output


def unpermute_reducescatter_fusion_native(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
    rank_buffers_ptr: torch.Tensor = None,
):
    """
        Three-phase unpermute + reduce-scatter using symmetric memory.

        This implementation follows the requested ordering strictly:
            1) Local reduction first (weighted unpermute on local expert_output)
            2) Write local reduced result to symmetric memory + barrier
            3) Read remote symmetric-memory reduced results for this rank's token shard,
                 then add with local shard result

    Args:
        expert_output: [num_tokens * topk, hidden] expert-centric layout
        scatter_idx: [num_tokens, topk] int32 - source row for each (token, k)
        topk_weights: [num_tokens, topk] float32 - routing weights
        output: [num_tokens_per_rank, hidden] pre-allocated output buffer
        group: TP process group
        group_name: optional group name for symmetric memory workspace
        backend_stream: unused in this implementation (kept for API compatibility)
        rank_buffers_ptr: unused in this implementation (kept for API compatibility)

    Returns:
        output: [num_tokens_per_rank, hidden] reduced result for this rank's shard
    """
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    num_tokens, topk = scatter_idx.shape
    hidden = expert_output.shape[1]
    num_tokens_per_rank = num_tokens // world_size

    # Phase 1: local reduction for all tokens (weighted unpermute on local expert_output)
    local_reduced = torch.empty(
        num_tokens,
        hidden,
        device=expert_output.device,
        dtype=expert_output.dtype,
    )
    for chunk_rank in range(world_size):
        chunk_start = chunk_rank * num_tokens_per_rank
        torch.ops.symm_mem.local_unpermute_copy_(
            expert_output,
            scatter_idx,
            topk_weights,
            chunk_start,
            num_tokens_per_rank,
            local_reduced[chunk_start : chunk_start + num_tokens_per_rank],
        )

    # Phase 2: write local reduced result to symmetric memory and synchronize
    reduced_workspace_size_bytes = local_reduced.numel() * local_reduced.element_size()
    reduced_workspace = symm_mem.get_symm_mem_workspace(
        group_name + "_local_reduced",
        min_size=reduced_workspace_size_bytes,
    )
    local_slot = reduced_workspace.get_buffer(
        rank,
        local_reduced.shape,
        local_reduced.dtype,
        storage_offset=0,
    )
    local_slot.copy_(local_reduced)
    reduced_workspace.barrier()

    # Phase 3: read remote reduced results for this rank's token shard and accumulate
    my_start = rank * num_tokens_per_rank
    output.copy_(local_reduced[my_start : my_start + num_tokens_per_rank])
    for step in range(world_size - 1):
        remote_rank = (rank - step - 1) % world_size
        remote_reduced = reduced_workspace.get_buffer(
            remote_rank,
            local_reduced.shape,
            local_reduced.dtype,
            storage_offset=0,
        )
        output.add_(remote_reduced[my_start : my_start + num_tokens_per_rank])

    reduced_workspace.barrier()
    return output


def build_unpermute_rank_buffers_ptr(
    expert_output: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
    """Precompute rank_buffers_ptr tensor for repeated unpermute_reduce_scatter calls."""
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    workspace_size_bytes = expert_output.numel() * expert_output.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    ptr_list = []
    for r in range(world_size):
        if r == rank:
            ptr_list.append(expert_output.data_ptr())
        else:
            buf = workspace.get_buffer(
                r, expert_output.shape,
                expert_output.dtype, storage_offset=0,
            )
            ptr_list.append(buf.data_ptr())
    signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
    return torch.tensor(signed_ptrs, dtype=torch.int64).to(expert_output.device)


def unpermute_reducescatter(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
    rank_buffers_ptr: torch.Tensor = None,
):
    """
    Default API: uses native kernel if available, otherwise Python fallback.

    Args:
        expert_output: [num_tokens * topk, hidden] expert-centric layout
        scatter_idx: [num_tokens, topk] int32 - source row for each (token, k)
        topk_weights: [num_tokens, topk] float32 - routing weights
        output: [num_tokens_per_rank, hidden] pre-allocated output buffer
        group: TP process group
        group_name: optional group name for symmetric memory workspace
        backend_stream: optional second stream for pipelining
        rank_buffers_ptr: optional precomputed buffer pointers (fused kernel only)

    Returns:
        output: [num_tokens_per_rank, hidden] reduced result for this rank's shard
    """
    # Prefer single-kernel fused path when available.
    # local_unpermute_copy_ alone implies a multi-kernel fallback only.
    if _HAS_UNPERMUTE_RS_KERNEL:
        return unpermute_reducescatter_fusion_native(
            expert_output=expert_output,
            scatter_idx=scatter_idx,
            topk_weights=topk_weights,
            output=output,
            group=group,
            group_name=group_name,
            backend_stream=backend_stream,
            rank_buffers_ptr=rank_buffers_ptr,
        )

    # Fallback path (non-fused): compute+push pipeline in Python.
    return unpermute_reducescatter_fusion(
        expert_output=expert_output,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=output,
        group=group,
        group_name=group_name,
        backend_stream=backend_stream,
    )

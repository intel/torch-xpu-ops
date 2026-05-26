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
_HAS_REDUCE_SCATTER_SUM = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_UNPERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_unpermute_copy_")
        _HAS_UNPERMUTE_RS_KERNEL = hasattr(torch.ops.symm_mem, "unpermute_reduce_scatter")
        _HAS_REDUCE_SCATTER_SUM = hasattr(torch.ops.symm_mem, "reduce_scatter_sum")
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
    Pipelined unpermute + reduce-scatter using symmetric memory and native kernels.

    Optimized pipeline:
      1. Own chunk compute on backend stream (overlaps with first push step)
         → writes to local recv slot my_recv_buf[rank]
      2. Remote chunks compute+push on alternating streams (main first)
      3. Barrier (ensure all pushes visible)
      4. Single sum of all recv slots → output

    Workspace layout per rank: [world_size, num_tokens_per_rank, hidden]
      slot[i] = contribution received FROM rank i for this rank's token chunk
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

    # Workspace: each rank has [world_size, num_tokens_per_rank, hidden] receive slots
    workspace_size_bytes = (
        world_size * num_tokens_per_rank * hidden * expert_output.element_size()
    )
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    if backend_stream is None:
        backend_stream = torch.xpu.Stream()

    my_chunk_start = rank * num_tokens_per_rank

    # Precompute workspace buffer views (avoid Python overhead in loop)
    my_recv_buf = workspace.get_buffer(
        rank, (world_size, num_tokens_per_rank, hidden),
        expert_output.dtype, storage_offset=0,
    )
    target_recv_bufs = []
    target_ranks = []
    for step in range(world_size - 1):
        tr = (rank - step - 1) % world_size
        target_ranks.append(tr)
        target_recv_bufs.append(workspace.get_buffer(
            tr, (world_size, num_tokens_per_rank, hidden),
            expert_output.dtype, storage_offset=0,
        ))

    local_bufs = [
        torch.empty(num_tokens_per_rank, hidden, device=expert_output.device, dtype=expert_output.dtype),
        torch.empty(num_tokens_per_rank, hidden, device=expert_output.device, dtype=expert_output.dtype),
    ]

    # Pipeline: remote chunks compute+push, then own-chunk compute overlaps with last push.
    # By placing own-chunk compute at the END (instead of before the pipeline), it
    # overlaps with the last step's DMA push, saving one full compute latency (~0.288ms).
    #
    # Timeline (C=compute, P=push, both ~0.29ms):
    #   Step0(C) → [Step0_push || Step1(C)] → [Step1_push || Step2(C)]
    #            → [Step2_push || own_chunk(C)] → wait → adds
    #   Total = C + 3P + overhead  (was 2C + 3P + overhead)
    for step in range(world_size - 1):
        buf_idx = step % 2
        if step % 2 == 0:
            stream = backend_stream
        else:
            stream = torch.xpu.current_stream()
        with torch.xpu.stream(stream):
            target_chunk_start = target_ranks[step] * num_tokens_per_rank
            torch.ops.symm_mem.local_unpermute_copy_(
                expert_output, scatter_idx, topk_weights,
                target_chunk_start, num_tokens_per_rank, local_bufs[buf_idx],
            )
            target_recv_bufs[step][rank].copy_(local_bufs[buf_idx])

    # Own-chunk compute on main stream — overlaps with last step's push on backend.
    # Writes to output directly (local memory, no push needed for own rank).
    torch.ops.symm_mem.local_unpermute_copy_(
        expert_output, scatter_idx, topk_weights,
        my_chunk_start, num_tokens_per_rank, output,
    )

    torch.xpu.current_stream().wait_stream(backend_stream)
    workspace.barrier()

    # Sum received contributions from all other ranks
    for i in range(world_size):
        if i != rank:
            output.add_(my_recv_buf[i])

    workspace.barrier()
    return output


def build_unpermute_rank_buffers_ptr(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
    """Precompute rank_buffers_ptr for reduce_scatter_sum kernel.

    Points to each rank's local_reduced [num_tokens, hidden] slot in
    symmetric memory.
    """
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    num_tokens = scatter_idx.size(0)
    hidden = expert_output.size(1)
    dtype = expert_output.dtype

    workspace_size_bytes = num_tokens * hidden * expert_output.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    ptr_list = []
    for r in range(world_size):
        buf = workspace.get_buffer(
            r, (num_tokens, hidden), dtype, storage_offset=0,
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
    # Prefer native pipeline path: local_unpermute + push + sum.
    if _HAS_LOCAL_UNPERMUTE_KERNEL:
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

"""
Allgather + local permute fusion using symmetric memory (TP only scenario).

Implements the algorithm described in docs/allgather+local permute fusion.md (TP only section):
- Each rank knows the global topk_idx mapping
- Each rank writes its local hidden_shard to remap_hidden_states according to topk_idx
- Allgather is performed directly into the correct positions in remap_hidden_states (symmetric memory)
- No redundant permute after allgather

The output remap_hidden_states is organized in expert-centric layout: all tokens assigned
to expert 0 come first, then expert 1, etc. This allows expert computation to directly
consume contiguous token blocks.

Inputs:
    hidden_shard: [num_tokens_per_rank, hidden_size] (local input)
    topk_idx: [num_tokens, topk] (global, all ranks have the same)
    scatter_idx: [num_tokens, topk] int32 - pre-computed output positions (expert-sorted)
    world_size: TP group size
    rank: TP rank
    remap_hidden_states: [num_tokens * topk, hidden_size] (output)

"""

import os

import ctypes

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Try to load the native local_permute_copy kernel
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so")
_HAS_LOCAL_PERMUTE_KERNEL = False
_HAS_ALLGATHER_PERMUTE_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_permute_copy_")
        _HAS_ALLGATHER_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute")
    except Exception:
        pass

_ALLGATHER_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "liballgather_with_symm_mem.so"
)
_HAS_ALLGATHER_WITH_SYMM_MEM_KERNEL = False
if os.path.exists(_ALLGATHER_LIB_PATH):
    try:
        torch.ops.load_library(_ALLGATHER_LIB_PATH)
        _HAS_ALLGATHER_WITH_SYMM_MEM_KERNEL = hasattr(
            torch.ops.symm_mem, "allgather_with_symm_mem"
        )
    except Exception:
        pass

_ALLGATHER_ISHMEM_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "liballgather_permute_ishmem.so"
)
_HAS_ALLGATHER_PERMUTE_ISHMEM_KERNEL = False
if os.path.exists(_ALLGATHER_ISHMEM_LIB_PATH):
    try:
        torch.ops.load_library(_ALLGATHER_ISHMEM_LIB_PATH)
        _HAS_ALLGATHER_PERMUTE_ISHMEM_KERNEL = hasattr(
            torch.ops.symm_mem, "allgather_permute_ishmem"
        )
    except Exception:
        pass


def compute_scatter_idx(topk_idx: torch.Tensor, num_experts: int = None):
    """
    Compute expert-centric scatter indices from global topk routing.

    The output layout groups tokens by expert: all tokens routed to expert 0
    are placed first, then expert 1, etc. Within each expert's block, tokens
    appear in the order determined by stable sort of (expert_id, original_position).

    Args:
        topk_idx: [num_tokens, topk] int32 - expert assignment for each (token, k)
        num_experts: total number of experts (optional, inferred from topk_idx if None)

    Returns:
        scatter_idx: [num_tokens, topk] int32 - output position for each (token, k)
        expert_offsets: [num_experts + 1] int64 - start offset of each expert in output
    """
    num_tokens, topk = topk_idx.shape
    topk_flat = topk_idx.reshape(-1)  # [num_tokens * topk]

    if num_experts is None:
        num_experts = int(topk_flat.max().item()) + 1

    # Sort by expert (stable: preserves token order within each expert)
    _, sort_indices = topk_flat.sort(stable=True)

    # Compute inverse permutation: scatter_idx[original_pos] = sorted_pos
    scatter_idx = torch.empty_like(sort_indices, dtype=torch.int32)
    scatter_idx[sort_indices] = torch.arange(
        len(topk_flat), device=topk_idx.device, dtype=torch.int32
    )

    # Expert offsets via bincount + cumsum
    expert_counts = torch.bincount(topk_flat.to(torch.int64), minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=topk_idx.device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    scatter_idx = scatter_idx.reshape(num_tokens, topk)
    return scatter_idx, expert_offsets


def allgather_local_permute_fusion_native(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    scatter_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
    rank_buffers_ptr: torch.Tensor = None,
):
    """
    TP only: Allgather + local permute fusion using symmetric memory.

    When the fused allgather_permute kernel is available, this uses a single
    kernel launch that reads directly from all ranks' symmetric memory buffers
    (ring-ordered, like EP dispatch) and writes all (token, k) positions.

    Falls back to multi-kernel path (per-rank copy + local_permute_copy_)
    when the fused kernel is not built.

    Args:
        hidden_shard: [num_tokens_per_rank, hidden_size] (local input)
        topk_idx: [num_tokens, topk] (global, all ranks have the same)
        scatter_idx: [num_tokens, topk] int32 - pre-computed expert-sorted positions
        remap_hidden_states: [num_tokens * topk, hidden_size] (output)
        group: TP process group
        group_name: Optional, for symmetric memory workspace
        backend_stream: Optional, only used in multi-kernel fallback path
        rank_buffers_ptr: Optional precomputed device tensor of per-rank
            buffer pointers (int64). Pass to avoid per-call overhead.
    Returns:
        remap_hidden_states: [num_tokens * topk, hidden_size] (filled, expert-centric layout)
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

    # Write local hidden_shard at rank-specific offset
    local_offset = rank * num_tokens_per_rank * hidden_size
    local_slot = workspace.get_buffer(
        rank, (num_tokens_per_rank, hidden_size),
        hidden_shard.dtype, storage_offset=local_offset,
    )
    local_slot.copy_(hidden_shard)
    workspace.barrier()

    if _HAS_ALLGATHER_PERMUTE_KERNEL:
        assert rank_buffers_ptr is not None, (
            "rank_buffers_ptr is required; use build_allgather_rank_buffers_ptr() to create it"
        )
        torch.ops.symm_mem.allgather_permute(
            rank_buffers_ptr, scatter_idx, remap_hidden_states,
            rank, world_size,
        )
    else:
        # Multi-kernel fallback: per-rank copy + local_permute_copy_
        symm_buffer = workspace.get_buffer(
            rank,
            (world_size, num_tokens_per_rank, hidden_size),
            hidden_shard.dtype,
            storage_offset=0,
        )

        local_token_offset = rank * num_tokens_per_rank
        torch.ops.symm_mem.local_permute_copy_(
            hidden_shard,
            scatter_idx,
            local_token_offset,
            remap_hidden_states,
        )

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
                    scatter_idx,
                    remote_token_offset,
                    remap_hidden_states,
                )
        torch.xpu.current_stream().wait_stream(backend_stream)

    workspace.barrier()
    return remap_hidden_states


def allgather_local_permute_fusion_python(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    scatter_idx: torch.Tensor,
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
            dst = scatter_idx[global_token_idx, k].item()
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
                    dst = scatter_idx[global_token_idx, k].item()
                    remap_hidden_states[dst].copy_(hidden_vec)

    torch.xpu.current_stream().wait_stream(backend_stream)
    workspace.barrier()
    return remap_hidden_states


def build_allgather_rank_buffers_ptr(
    hidden_shard: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
    """Precompute rank_buffers_ptr tensor for repeated allgather_permute calls."""
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    num_tokens_per_rank, hidden_size = hidden_shard.shape
    workspace_size_bytes = hidden_shard.numel() * hidden_shard.element_size() * world_size
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    ptr_list = []
    for r in range(world_size):
        if r == rank:
            ptr_list.append(hidden_shard.data_ptr())
        else:
            offset = r * num_tokens_per_rank * hidden_size
            buf = workspace.get_buffer(
                r, (num_tokens_per_rank, hidden_size),
                hidden_shard.dtype, storage_offset=offset,
            )
            ptr_list.append(buf.data_ptr())
    signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
    return torch.tensor(signed_ptrs, dtype=torch.int64).to(hidden_shard.device)


def build_allgather_with_symm_mem_rank_buffers_ptr(
    input_shard: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
) -> torch.Tensor:
    """Precompute rank_buffers_ptr tensor for allgather_with_symm_mem kernel.

    rank_buffers_ptr[rank] is unused by the kernel (it reads input_shard directly).
    rank_buffers_ptr[r != rank] points to rank r's symm_mem slot.
    """
    if group is None:
        group = dist.group.WORLD
    if group_name is None:
        group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    input_flat = input_shard.contiguous().view(-1)
    numel_per_rank = input_flat.numel()

    workspace_size_bytes = numel_per_rank * world_size * input_flat.element_size()
    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    ptr_list = []
    for r in range(world_size):
        if r == rank:
            ptr_list.append(input_shard.data_ptr())
        else:
            buf = workspace.get_buffer(
                r, (numel_per_rank,),
                input_flat.dtype, storage_offset=r * numel_per_rank,
            )
            ptr_list.append(buf.data_ptr())
    signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
    return torch.tensor(signed_ptrs, dtype=torch.int64).to(input_shard.device)


def allgather_local_permute_fusion(
    hidden_shard: torch.Tensor,
    topk_idx: torch.Tensor,
    scatter_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
    group_name: str = None,
    backend_stream: torch.xpu.Stream = None,
    rank_buffers_ptr: torch.Tensor = None,
):
    """
    Default API: uses native kernel if available, otherwise Python fallback.

    Args:
        hidden_shard: [num_tokens_per_rank, hidden_size] local input
        topk_idx: [num_tokens, topk] global expert assignments
        scatter_idx: [num_tokens, topk] int32 - pre-computed output positions
                     (from compute_scatter_idx)
        remap_hidden_states: [num_tokens * topk, hidden_size] output buffer
        group: process group
        group_name: optional group name for symmetric memory
        backend_stream: optional stream for overlapping (multi-kernel fallback only)
        rank_buffers_ptr: optional precomputed buffer pointers (fused kernel only)

    Returns:
        remap_hidden_states filled with expert-centric layout
    """
    if _HAS_LOCAL_PERMUTE_KERNEL:
        return allgather_local_permute_fusion_native(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            scatter_idx=scatter_idx,
            remap_hidden_states=remap_hidden_states,
            group=group,
            group_name=group_name,
            backend_stream=backend_stream,
            rank_buffers_ptr=rank_buffers_ptr,
        )
    return allgather_local_permute_fusion_python(
        hidden_shard=hidden_shard,
        topk_idx=topk_idx,
        scatter_idx=scatter_idx,
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
    rank_buffers_ptr: torch.Tensor = None,
):
    """
    Pure allgather using symmetric memory (no reduction).
    Each rank contributes input_shard, all ranks gather into output_tensor.

    Uses native SYCL kernel when available, otherwise falls back to Python.

    Args:
        input_shard: [numel_per_rank, ...] (local input)
        output_tensor: [numel_per_rank * world_size, ...] (output)
        group: process group
        group_name: Optional, for symmetric memory workspace
        rank_buffers_ptr: Optional precomputed device tensor of per-rank
            buffer pointers (int64). Pass to avoid per-call overhead.
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

    import torch.distributed._symmetric_memory as symm_mem_mod
    workspace_size_bytes = numel_per_rank * world_size * input_flat.element_size()
    workspace = symm_mem_mod.get_symm_mem_workspace(group_name, min_size=workspace_size_bytes)

    # Barrier to ensure previous allgather is complete before writing
    workspace.barrier()

    # Step 1: Each rank writes its shard to its slot in symm buffer
    my_slot = workspace.get_buffer(
        rank, (numel_per_rank,), input_flat.dtype,
        storage_offset=rank * numel_per_rank,
    )
    my_slot.copy_(input_flat)

    workspace.barrier()

    # Step 2: Ring allgather
    if _HAS_ALLGATHER_WITH_SYMM_MEM_KERNEL:
        if rank_buffers_ptr is None:
            rank_buffers_ptr = build_allgather_with_symm_mem_rank_buffers_ptr(
                input_shard, group, group_name,
            )
        torch.ops.symm_mem.allgather_with_symm_mem(
            input_flat, rank_buffers_ptr, output_flat, rank, world_size,
        )
    else:
        # Python fallback: pull-based ring allgather
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


def allgather_permute_ishmem(
    input_shard: torch.Tensor,
    scatter_idx: torch.Tensor,
    remap_hidden_states: torch.Tensor,
    group: dist.ProcessGroup = None,
):
    """Allgather + permute via the ISHMEM kernel."""
    if not _HAS_ALLGATHER_PERMUTE_ISHMEM_KERNEL:
        raise RuntimeError(
            "allgather_permute_ishmem kernel is unavailable; build "
            "test/xpu/csrc/liballgather_permute_ishmem.so first"
        )
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    return torch.ops.symm_mem.allgather_permute_ishmem(
        input_shard.contiguous(),
        scatter_idx.contiguous(),
        remap_hidden_states,
        rank,
        world_size,
    )

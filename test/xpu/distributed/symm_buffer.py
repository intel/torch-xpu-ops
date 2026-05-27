"""SymmBuffer: symmetric-memory based fusion APIs for MoE dispatch/combine.

Provides two fused operations built on top of torch symmetric memory:
1. allgather_local_permute_fusion  — allgather hidden states + permute to expert-centric layout
2. unpermute_reducescatter_fusion  — unpermute expert output + reduce-scatter back to token-centric layout

Core workspace views, pointers, and streams are pre-built in __init__.
Per-call output tensors are created in the fused APIs.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Load native kernels
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so")
_HAS_LOCAL_PERMUTE_KERNEL = False
_HAS_ALLGATHER_PERMUTE_KERNEL = False
_HAS_ALLGATHER_PERMUTE_TOPK_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_permute_copy_")
        _HAS_ALLGATHER_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute")
        _HAS_ALLGATHER_PERMUTE_TOPK_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute_topk")
    except Exception:
        pass

_UNPERMUTE_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "libunpermute_reduce_scatter.so"
)
_HAS_LOCAL_UNPERMUTE_KERNEL = False
if os.path.exists(_UNPERMUTE_LIB_PATH):
    try:
        torch.ops.load_library(_UNPERMUTE_LIB_PATH)
        _HAS_LOCAL_UNPERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_unpermute_copy_")
    except Exception:
        pass

_NOTIFY_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "libnotify_dispatch.so"
)
_HAS_NOTIFY_DISPATCH_KERNEL = False
_HAS_NOTIFY_DISPATCH_V2_KERNEL = False
if os.path.exists(_NOTIFY_LIB_PATH):
    try:
        torch.ops.load_library(_NOTIFY_LIB_PATH)
        _HAS_NOTIFY_DISPATCH_KERNEL = hasattr(torch.ops.symm_mem, "notify_dispatch")
        _HAS_NOTIFY_DISPATCH_V2_KERNEL = hasattr(torch.ops.symm_mem, "notify_dispatch_v2")
    except Exception:
        pass

_HAS_ALLGATHER_PERMUTE_V2_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        _HAS_ALLGATHER_PERMUTE_V2_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute_v2")
    except Exception:
        pass

_HAS_LOCAL_UNPERMUTE_V2_KERNEL = False
if os.path.exists(_UNPERMUTE_LIB_PATH):
    try:
        _HAS_LOCAL_UNPERMUTE_V2_KERNEL = hasattr(torch.ops.symm_mem, "local_unpermute_copy_v2")
    except Exception:
        pass


@dataclass
class SymmHandle:
    """Handle that carries permute outputs for use by unpermute.

    Returned by allgather_local_permute_fusion and accepted by
    unpermute_reducescatter_fusion so the caller doesn't need to
    pass scatter_idx / recv_topk_idx / recv_topk_weights manually.

    In v2 mode, scatter_idx holds expert-relative positions and
    global_topk_idx / rows_per_expert are populated for v2 kernels.
    """

    scatter_idx: torch.Tensor         # [num_tokens, topk] int32
    recv_topk_idx: torch.Tensor       # [num_tokens * topk, topk] int32 (recv layout)
    recv_topk_weights: torch.Tensor   # [num_tokens, topk] float32 (gathered global weights)
    num_tokens_per_rank: int
    # V2 fields (None when using v1 path)
    global_topk_idx: Optional[torch.Tensor] = None    # [num_tokens, topk] int32
    rows_per_expert: Optional[torch.Tensor] = None    # [num_experts] int32
    is_v2: bool = False


class SymmBuffer:
    """Symmetric-memory buffer with pre-allocated state for MoE fusion kernels.

    Args:
        group: Process group for collective operations.
        num_max_tokens_per_rank: Maximum number of tokens each rank can hold.
        hidden: Hidden dimension size.
        num_topk: Number of top-k experts per token.
        hidden_dtype: Data type for hidden states (default: torch.bfloat16).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        hidden_dtype: torch.dtype = torch.bfloat16,
    ):
        self.group = group
        self.rank_idx = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_topk = num_topk
        self.hidden_dtype = hidden_dtype

        device = f"xpu:{self.rank_idx}"
        num_tokens_max = num_max_tokens_per_rank * self.num_ranks
        topk = num_topk
        hidden_elem_size = torch.tensor([], dtype=hidden_dtype).element_size()

        # Allocate one big symmetric memory workspace for all data:
        #   Section 1 — hidden states:  num_ranks slots of [M, H] in hidden_dtype
        #   Section 2 — topk indices:   num_ranks slots of [M, topk] in int32
        #   Section 3 — topk weights:   num_ranks slots of [M, topk] in float32
        hidden_section_bytes = (
            num_max_tokens_per_rank * hidden * hidden_elem_size * self.num_ranks
        )
        topk_section_bytes = (
            num_max_tokens_per_rank * num_topk * 4 * self.num_ranks
        )
        weights_section_bytes = (
            num_max_tokens_per_rank * num_topk * 4 * self.num_ranks
        )
        self.workspace_size_bytes = (
            hidden_section_bytes + topk_section_bytes + weights_section_bytes
        )
        self.workspace = symm_mem.get_symm_mem_workspace(
            self.group.group_name,
            min_size=self.workspace_size_bytes,
        )

        # Topk section base offset (in int32 elements)
        self._topk_base_offset = hidden_section_bytes // 4

        # --- allgather_permute pre-allocations ---
        self._allgather_permute_rank_buffers_ptr = self._build_rank_buffers_ptr()
        local_offset = self.rank_idx * num_max_tokens_per_rank * hidden
        self._allgather_local_slot = self.workspace.get_buffer(
            self.rank_idx,
            (num_max_tokens_per_rank, hidden),
            hidden_dtype,
            storage_offset=local_offset,
        )
        self._scatter_idx = torch.empty(
            num_tokens_max, topk,
            device=device, dtype=torch.int32,
        )
        self._global_topk_idx_buf = torch.empty(
            num_tokens_max, topk, device=device, dtype=torch.int32,
        )
        self._global_topk_weights_buf = torch.empty(
            num_tokens_max, topk, device=device, dtype=torch.float32,
        )

        # --- notify_dispatch pre-allocations (topk in shared workspace) ---
        topk_local_offset = (
            self._topk_base_offset
            + self.rank_idx * num_max_tokens_per_rank * num_topk
        )
        self._topk_local_slot = self.workspace.get_buffer(
            self.rank_idx,
            (num_max_tokens_per_rank, num_topk),
            torch.int32,
            storage_offset=topk_local_offset,
        )
        self._topk_rank_ptrs = self._build_topk_rank_ptrs()
        self._psum_buf = torch.zeros(512, device=device, dtype=torch.int32)
        self._rows_per_expert_buf = torch.zeros(512, device=device, dtype=torch.int32)

        # --- notify_dispatch pre-allocations (weights in shared workspace) ---
        self._weights_base_offset = (
            hidden_section_bytes + topk_section_bytes
        ) // 4  # in float32 elements
        weights_local_offset = (
            self._weights_base_offset
            + self.rank_idx * num_max_tokens_per_rank * num_topk
        )
        self._weights_local_slot = self.workspace.get_buffer(
            self.rank_idx,
            (num_max_tokens_per_rank, num_topk),
            torch.float32,
            storage_offset=weights_local_offset,
        )
        self._weights_rank_ptrs = self._build_weights_rank_ptrs()

        # --- unpermute_reducescatter pre-allocations ---
        self._backend_stream = torch.xpu.Stream()
        self._unpermute_local_bufs = [
            torch.empty(
                num_max_tokens_per_rank, hidden,
                device=device, dtype=hidden_dtype,
            ),
            torch.empty(
                num_max_tokens_per_rank, hidden,
                device=device, dtype=hidden_dtype,
            ),
        ]
        self._unpermute_my_recv_buf = self.workspace.get_buffer(
            self.rank_idx,
            (self.num_ranks, num_max_tokens_per_rank, hidden),
            hidden_dtype,
            storage_offset=0,
        )
        self._unpermute_target_ranks = []
        self._unpermute_target_recv_bufs = []
        for step in range(self.num_ranks - 1):
            tr = (self.rank_idx - step - 1) % self.num_ranks
            self._unpermute_target_ranks.append(tr)
            self._unpermute_target_recv_bufs.append(
                self.workspace.get_buffer(
                    tr,
                    (self.num_ranks, num_max_tokens_per_rank, hidden),
                    hidden_dtype,
                    storage_offset=0,
                )
            )
        self._gather_k = (
            torch.arange(topk, device=device, dtype=torch.int32)
            .view(1, -1).expand(num_tokens_max, -1)
        )

    def _build_rank_buffers_ptr(self) -> torch.Tensor:
        """Build stable rank_buffers_ptr for allgather_permute kernel.

        All pointers (including local rank) point to workspace slots.
        Layout: rank r's data at storage_offset = r * num_max_tokens_per_rank * hidden.
        """
        ptr_list = []
        for r in range(self.num_ranks):
            offset = r * self.num_max_tokens_per_rank * self.hidden
            buf = self.workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.hidden),
                self.hidden_dtype,
                storage_offset=offset,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(
            signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}"
        )

    def _build_topk_rank_ptrs(self) -> torch.Tensor:
        """Build rank pointers into topk section of shared workspace."""
        ptr_list = []
        for r in range(self.num_ranks):
            offset = (
                self._topk_base_offset
                + r * self.num_max_tokens_per_rank * self.num_topk
            )
            buf = self.workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.num_topk),
                torch.int32,
                storage_offset=offset,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(
            signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}"
        )

    def _build_weights_rank_ptrs(self) -> torch.Tensor:
        """Build rank pointers into weights section of shared workspace."""
        ptr_list = []
        for r in range(self.num_ranks):
            offset = (
                self._weights_base_offset
                + r * self.num_max_tokens_per_rank * self.num_topk
            )
            buf = self.workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank, self.num_topk),
                torch.float32,
                storage_offset=offset,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(
            signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}"
        )

    def allgather_local_permute_fusion(
        self,
        hidden_shard: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        remap_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SymmHandle]:
        """Fused allgather + local permute using symmetric memory.

        Writes per-rank topk_idx to symmetric memory, then uses the
        notify_dispatch kernel to compute scatter_idx directly (like
        elastic_xpu dispatch). No external scatter_idx needed.

        Args:
            hidden_shard: [num_tokens_per_rank, hidden] local hidden states.
            topk_idx: [num_tokens_per_rank, topk] int32, per-rank expert
                assignments.
            topk_weights: [num_tokens_per_rank, topk] float32, per-rank
                routing weights.
            num_experts: Total number of experts.
            remap_hidden_states: Pre-allocated output buffer
                [num_tokens * topk, hidden].

        Returns:
            (remap_hidden_states, recv_topk_idx, recv_topk_weights, handle)
        """
        num_tokens_per_rank = hidden_shard.shape[0]
        topk = topk_idx.shape[1]
        num_tokens = num_tokens_per_rank * self.num_ranks

        expected_rows = num_tokens * topk
        if remap_hidden_states.shape[0] < expected_rows:
            raise ValueError(
                f"remap_hidden_states rows ({remap_hidden_states.shape[0]}) must be >= "
                f"num_tokens * topk ({expected_rows})"
            )
        if remap_hidden_states.shape[1] != self.hidden:
            raise ValueError(
                f"remap_hidden_states hidden ({remap_hidden_states.shape[1]}) must equal "
                f"self.hidden ({self.hidden})"
            )

        # Write local topk_idx, topk_weights, and hidden_shard to symmetric memory
        self._topk_local_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)
        self._weights_local_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)
        self._allgather_local_slot.copy_(hidden_shard)
        self.workspace.barrier()

        # Determine whether to use v2 path
        use_v2 = (
            _HAS_NOTIFY_DISPATCH_V2_KERNEL
            and _HAS_ALLGATHER_PERMUTE_V2_KERNEL
            and _HAS_LOCAL_UNPERMUTE_V2_KERNEL
        )

        # Compute scatter_idx and gather topk_weights via notify_dispatch kernel
        # (reads topk_idx and weights from all ranks' symm mem)
        scatter_idx = self._scatter_idx[:num_tokens, :topk]
        global_topk_idx_buf = self._global_topk_idx_buf[:num_tokens, :topk]
        global_topk_weights_buf = self._global_topk_weights_buf[:num_tokens, :topk]

        if use_v2:
            # V2 path: single-kernel notify_dispatch_v2 + allgather_permute_v2
            rows_per_expert_buf = self._rows_per_expert_buf[:num_experts]

            torch.ops.symm_mem.notify_dispatch_v2(
                self._topk_rank_ptrs,
                global_topk_idx_buf,
                scatter_idx,
                rows_per_expert_buf,
                num_tokens_per_rank,
                topk,
                self.num_topk,
                num_experts,
                self.rank_idx,
                self.num_ranks,
                self._weights_rank_ptrs,
                global_topk_weights_buf,
            )

            # Build recv_topk_idx/weights using absolute positions derived from
            # expert-relative scatter_idx + rows_per_expert cumsum
            recv_topk_idx = torch.empty(
                (num_tokens * topk, topk),
                device=topk_idx.device,
                dtype=topk_idx.dtype,
            )
            recv_topk_weights = torch.empty(
                (num_tokens * topk, topk),
                device=topk_weights.device,
                dtype=topk_weights.dtype,
            )

            # Compute exclusive prefix-sum for absolute position conversion
            expert_cumsum = torch.zeros(num_experts, device=hidden_shard.device, dtype=torch.int32)
            expert_cumsum[1:] = rows_per_expert_buf[:-1]
            expert_cumsum = expert_cumsum.cumsum(0)

            flat_expert = global_topk_idx_buf.reshape(-1)
            flat_rel = scatter_idx.reshape(-1)
            flat_scatter_abs = expert_cumsum[flat_expert] + flat_rel
            flat_k = (
                torch.arange(topk, device=hidden_shard.device, dtype=torch.int32)
                .view(1, -1).expand(num_tokens, -1).reshape(-1)
            )
            recv_topk_idx[flat_scatter_abs, flat_k] = flat_expert
            recv_topk_weights[flat_scatter_abs, flat_k] = global_topk_weights_buf.reshape(-1)

            # Allgather + permute using v2 kernel (relative scatter_idx)
            torch.ops.symm_mem.allgather_permute_v2(
                self._allgather_permute_rank_buffers_ptr,
                scatter_idx,
                global_topk_idx_buf,
                rows_per_expert_buf,
                remap_hidden_states,
                self.rank_idx,
                self.num_ranks,
            )

            self.workspace.barrier()

            handle = SymmHandle(
                scatter_idx=scatter_idx,
                recv_topk_idx=recv_topk_idx,
                recv_topk_weights=global_topk_weights_buf,
                num_tokens_per_rank=num_tokens_per_rank,
                global_topk_idx=global_topk_idx_buf,
                rows_per_expert=rows_per_expert_buf,
                is_v2=True,
            )

            return remap_hidden_states, recv_topk_idx, recv_topk_weights, handle

        # V1 path (original): two-kernel notify_dispatch + allgather_permute
        base = num_experts // self.num_ranks
        rem = num_experts % self.num_ranks
        num_local_experts = base + (1 if self.rank_idx < rem else 0)
        psum_buf = self._psum_buf[:num_local_experts]

        torch.ops.symm_mem.notify_dispatch(
            self._topk_rank_ptrs,
            global_topk_idx_buf,
            scatter_idx,
            psum_buf,
            num_tokens_per_rank,
            topk,
            self.num_topk,
            num_experts,
            self.rank_idx,
            self.num_ranks,
            self._weights_rank_ptrs,
            global_topk_weights_buf,
        )

        # Build recv-topk outputs in recv layout to match ep_dispatch style.
        recv_topk_idx = torch.empty(
            (num_tokens * topk, topk),
            device=topk_idx.device,
            dtype=topk_idx.dtype,
        )
        recv_topk_weights = torch.empty(
            (num_tokens * topk, topk),
            device=topk_weights.device,
            dtype=topk_weights.dtype,
        )

        if _HAS_ALLGATHER_PERMUTE_TOPK_KERNEL:
            torch.ops.symm_mem.allgather_permute_topk(
                self._allgather_permute_rank_buffers_ptr,
                global_topk_idx_buf,
                self._weights_rank_ptrs,
                scatter_idx,
                remap_hidden_states,
                recv_topk_idx,
                recv_topk_weights,
                self.num_topk,
                num_experts,
                self.rank_idx,
                self.num_ranks,
            )
        else:
            flat_k = (
                torch.arange(topk, device=hidden_shard.device, dtype=torch.int32)
                .view(1, -1).expand(num_tokens, -1).reshape(-1)
            )
            flat_scatter = scatter_idx.reshape(-1)
            recv_topk_idx[flat_scatter, flat_k] = (
                global_topk_idx_buf.reshape(-1)
            )
            recv_topk_weights[flat_scatter, flat_k] = (
                global_topk_weights_buf.reshape(-1)
            )

            # Hidden data already in symm mem (written + barrier'd above)
            torch.ops.symm_mem.allgather_permute(
                self._allgather_permute_rank_buffers_ptr,
                scatter_idx,
                remap_hidden_states,
                self.rank_idx,
                self.num_ranks,
            )

        self.workspace.barrier()

        handle = SymmHandle(
            scatter_idx=scatter_idx,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=global_topk_weights_buf,
            num_tokens_per_rank=num_tokens_per_rank,
        )

        return remap_hidden_states, recv_topk_idx, recv_topk_weights, handle

    def unpermute_reducescatter_fusion(
        self,
        expert_output: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        handle: SymmHandle,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Fused unpermute + reduce-scatter using symmetric memory.

        Pipelined implementation: computes local unpermute for remote chunks
        and pushes them via symmetric memory, then computes own chunk at the
        end to overlap with the last push.

        Args:
            expert_output: [num_tokens * topk, hidden] expert-centric layout.
            recv_topk_idx: [num_tokens * topk, topk] int32 (recv layout).
            recv_topk_weights: [num_tokens, topk] routing weights.
            handle: SymmHandle from allgather_local_permute_fusion.
            output: Pre-allocated output [num_tokens_per_rank, hidden].

        Returns:
            output
        """
        if not _HAS_LOCAL_UNPERMUTE_KERNEL:
            raise RuntimeError(
                "local_unpermute_copy_ kernel is required for "
                "unpermute_reducescatter_fusion"
            )

        scatter_idx = handle.scatter_idx
        global_topk_weights = recv_topk_weights

        num_tokens = scatter_idx.shape[0]
        num_tokens_per_rank = num_tokens // self.num_ranks
        if output.shape[0] != num_tokens_per_rank:
            raise ValueError(
                f"output rows ({output.shape[0]}) must equal num_tokens_per_rank ({num_tokens_per_rank})"
            )

        my_chunk_start = self.rank_idx * num_tokens_per_rank

        # Pipeline: remote chunks compute+push first, own-chunk at END
        # to overlap with last step's DMA push.
        stream = self._backend_stream
        for step in range(self.num_ranks - 1):
            buf_idx = step % 2
            cur_stream = stream if step % 2 == 0 else torch.xpu.current_stream()
            with torch.xpu.stream(cur_stream):
                target_chunk_start = self._unpermute_target_ranks[step] * num_tokens_per_rank
                if handle.is_v2:
                    torch.ops.symm_mem.local_unpermute_copy_v2(
                        expert_output, scatter_idx,
                        handle.global_topk_idx, handle.rows_per_expert,
                        global_topk_weights,
                        target_chunk_start, num_tokens_per_rank,
                        self._unpermute_local_bufs[buf_idx],
                    )
                else:
                    torch.ops.symm_mem.local_unpermute_copy_(
                        expert_output, scatter_idx, global_topk_weights,
                        target_chunk_start, num_tokens_per_rank,
                        self._unpermute_local_bufs[buf_idx],
                    )
                self._unpermute_target_recv_bufs[step][self.rank_idx].copy_(
                    self._unpermute_local_bufs[buf_idx]
                )

        # Own-chunk compute on main stream — overlaps with last push
        if handle.is_v2:
            torch.ops.symm_mem.local_unpermute_copy_v2(
                expert_output, scatter_idx,
                handle.global_topk_idx, handle.rows_per_expert,
                global_topk_weights,
                my_chunk_start, num_tokens_per_rank, output,
            )
        else:
            torch.ops.symm_mem.local_unpermute_copy_(
                expert_output, scatter_idx, global_topk_weights,
                my_chunk_start, num_tokens_per_rank, output,
            )

        torch.xpu.current_stream().wait_stream(stream)
        self.workspace.barrier()

        for i in range(self.num_ranks):
            if i != self.rank_idx:
                output.add_(self._unpermute_my_recv_buf[i])

        self.workspace.barrier()
        return output

        

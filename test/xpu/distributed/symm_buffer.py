"""SymmBuffer: symmetric-memory based fusion APIs for MoE dispatch/combine.

Provides two fused operations built on top of torch symmetric memory:
1. allgather_local_permute_fusion  — allgather hidden states + permute to expert-centric layout
2. unpermute_reducescatter_fusion  — unpermute expert output + reduce-scatter back to token-centric layout

All per-call allocations (workspace views, rank_buffers_ptr, scratch buffers,
streams) are pre-built in __init__ so the hot path has zero Python overhead.
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
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_LOCAL_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_permute_copy_")
        _HAS_ALLGATHER_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute")
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
if os.path.exists(_NOTIFY_LIB_PATH):
    try:
        torch.ops.load_library(_NOTIFY_LIB_PATH)
        _HAS_NOTIFY_DISPATCH_KERNEL = hasattr(torch.ops.symm_mem, "notify_dispatch")
    except Exception:
        pass


@dataclass
class SymmHandle:
    """Handle that carries permute outputs for use by unpermute.

    Returned by allgather_local_permute_fusion and accepted by
    unpermute_reducescatter_fusion so the caller doesn't need to
    pass scatter_idx / recv_topk_idx / recv_topk_weights manually.
    """

    scatter_idx: torch.Tensor         # [num_tokens, topk] int32
    recv_topk_idx: torch.Tensor       # [num_tokens * topk, topk] int32
    recv_topk_weights: torch.Tensor   # [num_tokens * topk, topk] float32
    num_tokens_per_rank: int


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
        self._remap_hidden_states = torch.empty(
            num_tokens_max * topk, hidden,
            device=device, dtype=hidden_dtype,
        )
        self._recv_topk_idx = torch.full(
            (num_tokens_max * topk, topk), -1,
            device=device, dtype=torch.int32,
        )
        self._recv_topk_weights = torch.zeros(
            num_tokens_max * topk, topk,
            device=device, dtype=torch.float32,
        )
        self._flat_k = (
            torch.arange(topk, device=device, dtype=torch.int32)
            .view(1, -1).expand(num_tokens_max, -1).reshape(-1)
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
        remap_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SymmHandle]:
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
            remap_hidden_states: Optional pre-allocated output buffer
                [num_tokens * topk, hidden]. Uses internal buffer if None.

        Returns:
            (remap_hidden_states, handle)
        """
        num_tokens_per_rank = hidden_shard.shape[0]
        topk = topk_idx.shape[1]
        num_tokens = num_tokens_per_rank * self.num_ranks

        # Write local topk_idx, topk_weights, and hidden_shard to symmetric memory
        self._topk_local_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)
        self._weights_local_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)
        self._allgather_local_slot.copy_(hidden_shard)
        self.workspace.barrier()

        # Compute scatter_idx and gather topk_weights via notify_dispatch kernel
        # (reads topk_idx and weights from all ranks' symm mem)
        base = num_experts // self.num_ranks
        rem = num_experts % self.num_ranks
        num_local_experts = base + (1 if self.rank_idx < rem else 0)
        psum_buf = self._psum_buf[:num_local_experts]

        scatter_idx = torch.empty(
            num_tokens, topk, device=hidden_shard.device, dtype=torch.int32,
        )
        torch.ops.symm_mem.notify_dispatch(
            self._topk_rank_ptrs,
            self._global_topk_idx_buf,
            scatter_idx,
            psum_buf,
            num_tokens_per_rank,
            topk,
            self.num_topk,
            num_experts,
            self.rank_idx,
            self.num_ranks,
            self._weights_rank_ptrs,
            self._global_topk_weights_buf,
        )

        # Build recv_topk_idx / recv_topk_weights
        self._recv_topk_idx.fill_(-1)
        self._recv_topk_weights.zero_()
        flat_scatter = scatter_idx.reshape(-1)
        self._recv_topk_idx[flat_scatter, self._flat_k] = (
            self._global_topk_idx_buf.reshape(-1)
        )
        self._recv_topk_weights[flat_scatter, self._flat_k] = (
            self._global_topk_weights_buf.reshape(-1)
        )
        recv_topk_idx = self._recv_topk_idx
        recv_topk_weights = self._recv_topk_weights

        if remap_hidden_states is None:
            remap_hidden_states = self._remap_hidden_states

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
            recv_topk_weights=recv_topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
        )

        return remap_hidden_states, handle

    def unpermute_reducescatter_fusion(
        self,
        expert_output: torch.Tensor,
        scatter_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        recv_topk_idx: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        handle: Optional[SymmHandle] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused unpermute + reduce-scatter using symmetric memory.

        Pipelined implementation: computes local unpermute for remote chunks
        and pushes them via symmetric memory, then computes own chunk at the
        end to overlap with the last push.

        Args:
            expert_output: [num_tokens * topk, hidden] expert-centric layout.
            scatter_idx: [num_tokens, topk] int32, source positions.
                Can be omitted if handle is provided.
            topk_weights: Optional [num_tokens_per_rank, topk] or [num_tokens, topk]
                routing weights. Used when neither handle nor recv_topk_weights
                is provided.
            recv_topk_idx: Optional, unused (kept for API compatibility).
            recv_topk_weights: Optional [num_tokens * topk, topk] remapped weights.
                If provided, global_topk_weights is reconstructed from this.
            output: Optional pre-allocated output [num_tokens_per_rank, hidden].
            handle: Optional SymmHandle from allgather_local_permute_fusion.
                When provided, scatter_idx and recv_topk_weights are taken
                from the handle (explicit arguments still override).

        Returns:
            (output, global_topk_weights)
        """
        # Unpack handle fields (explicit args override)
        if handle is not None:
            if scatter_idx is None:
                scatter_idx = handle.scatter_idx
            if recv_topk_weights is None:
                recv_topk_weights = handle.recv_topk_weights
            if recv_topk_idx is None:
                recv_topk_idx = handle.recv_topk_idx

        if scatter_idx is None:
            raise ValueError(
                "scatter_idx is required — provide it directly or via handle"
            )
        num_tokens = scatter_idx.shape[0]
        num_tokens_per_rank = num_tokens // self.num_ranks
        topk = scatter_idx.shape[1]

        # Reconstruct global_topk_weights
        if recv_topk_weights is not None:
            global_topk_weights = recv_topk_weights[
                scatter_idx, self._gather_k[:num_tokens]
            ].contiguous()
        elif topk_weights is not None and topk_weights.shape[0] == num_tokens_per_rank:
            dist.all_gather_into_tensor(
                self._global_topk_weights_buf, topk_weights, group=self.group,
            )
            global_topk_weights = self._global_topk_weights_buf
        elif topk_weights is not None and topk_weights.shape[0] == num_tokens:
            global_topk_weights = topk_weights
        else:
            raise ValueError(
                "Either recv_topk_weights or topk_weights must be provided"
            )

        if output is None:
            output = torch.zeros(
                num_tokens_per_rank,
                expert_output.shape[1],
                device=expert_output.device,
                dtype=expert_output.dtype,
            )

        hidden = expert_output.shape[1]

        if _HAS_LOCAL_UNPERMUTE_KERNEL:
            my_chunk_start = self.rank_idx * num_tokens_per_rank

            # Pipeline: remote chunks compute+push first, own-chunk at END
            # to overlap with last step's DMA push.
            stream = self._backend_stream
            for step in range(self.num_ranks - 1):
                buf_idx = step % 2
                cur_stream = stream if step % 2 == 0 else torch.xpu.current_stream()
                with torch.xpu.stream(cur_stream):
                    target_chunk_start = self._unpermute_target_ranks[step] * num_tokens_per_rank
                    torch.ops.symm_mem.local_unpermute_copy_(
                        expert_output, scatter_idx, global_topk_weights,
                        target_chunk_start, num_tokens_per_rank,
                        self._unpermute_local_bufs[buf_idx],
                    )
                    self._unpermute_target_recv_bufs[step][self.rank_idx].copy_(
                        self._unpermute_local_bufs[buf_idx]
                    )

            # Own-chunk compute on main stream — overlaps with last push
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
            return output, global_topk_weights

        # Python fallback (no native kernel)
        my_chunk_start = self.rank_idx * num_tokens_per_rank
        output.zero_()
        for i in range(num_tokens_per_rank):
            global_i = my_chunk_start + i
            for k in range(topk):
                src_row = scatter_idx[global_i, k].item()
                output[i] += global_topk_weights[global_i, k] * expert_output[src_row]

        stream = self._backend_stream
        local_bufs = self._unpermute_local_bufs

        for step in range(self.num_ranks - 1):
            target_rank = self._unpermute_target_ranks[step]
            buf_idx = step % 2
            cur_stream = stream if step % 2 == 0 else torch.xpu.current_stream()
            with torch.xpu.stream(cur_stream):
                target_chunk_start = target_rank * num_tokens_per_rank
                local_bufs[buf_idx].zero_()
                for i in range(num_tokens_per_rank):
                    global_i = target_chunk_start + i
                    for k in range(topk):
                        src_row = scatter_idx[global_i, k].item()
                        local_bufs[buf_idx][i] += (
                            global_topk_weights[global_i, k] * expert_output[src_row]
                        )
                self._unpermute_target_recv_bufs[step][self.rank_idx].copy_(
                    local_bufs[buf_idx]
                )

        torch.xpu.current_stream().wait_stream(stream)
        self.workspace.barrier()

        for i in range(self.num_ranks):
            if i != self.rank_idx:
                output.add_(self._unpermute_my_recv_buf[i])

        self.workspace.barrier()
        return output, global_topk_weights

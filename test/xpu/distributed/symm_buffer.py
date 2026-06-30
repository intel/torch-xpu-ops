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
from typing import Tuple

import logging
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Debug logging controlled by SYMM_BUFFER_DEBUG environment variable.
# Set SYMM_BUFFER_DEBUG=1 to enable detailed input/output logging.
_SYMM_BUFFER_DEBUG = os.environ.get("SYMM_BUFFER_DEBUG", "0") == "1"
_logger = logging.getLogger("SymmBuffer")
if _SYMM_BUFFER_DEBUG and not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[SymmBuffer %(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)

# Load native kernels
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so")
_HAS_ALLGATHER_PERMUTE_KERNEL = False
_HAS_LOCAL_PERMUTE_KERNEL = False
if os.path.exists(_LIB_PATH):
    try:
        torch.ops.load_library(_LIB_PATH)
        _HAS_ALLGATHER_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "allgather_permute")
        _HAS_LOCAL_PERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_permute_copy_")
    except Exception:
        pass

_UNPERMUTE_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "libunpermute_reduce_scatter.so"
)
_HAS_LOCAL_UNPERMUTE_KERNEL = False
_HAS_SUM_REDUCTION_KERNEL = False
if os.path.exists(_UNPERMUTE_LIB_PATH):
    try:
        torch.ops.load_library(_UNPERMUTE_LIB_PATH)
        _HAS_LOCAL_UNPERMUTE_KERNEL = hasattr(torch.ops.symm_mem, "local_unpermute_copy_")
        _HAS_SUM_REDUCTION_KERNEL = hasattr(torch.ops.symm_mem, "sum_reduction")
    except Exception:
        pass

_NOTIFY_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "csrc", "libnotify_dispatch.so"
)
_HAS_NOTIFY_DISPATCH_V2_KERNEL = False
if os.path.exists(_NOTIFY_LIB_PATH):
    try:
        torch.ops.load_library(_NOTIFY_LIB_PATH)
        _HAS_NOTIFY_DISPATCH_V2_KERNEL = hasattr(torch.ops.symm_mem, "notify_dispatch_v2")
    except Exception:
        pass

# Ring (single-kernel, pipelined) fused collectives — native op availability.
_RING_LIB_NAMES = (
    "libring_allgather_permute.so",
    "libring_reduce_scatter_unpermute.so",
)
for _lib in _RING_LIB_NAMES:
    _path = os.path.join(os.path.dirname(__file__), "..", "csrc", _lib)
    if os.path.exists(_path):
        try:
            torch.ops.load_library(_path)
        except Exception:
            pass
_HAS_RING_ALLGATHER_PERMUTE = hasattr(torch.ops.symm_mem, "ring_allgather_permute")
_HAS_RING_REDUCE_SCATTER_UNPERMUTE = hasattr(
    torch.ops.symm_mem, "ring_reduce_scatter_unpermute"
)

# Upper bound on the number of work-groups any single-kernel ring collective
# launches; sizes the signal-pad region.
_RING_MAX_WG = 1024

# Monotonically increasing signal tag per SymmBuffer instance.
_ring_iter_counters: dict = {}

# Select the fused-kernel implementation.  When FUSION_RING=1 (the default), the
# allgather_local_permute_fusion / unpermute_reducescatter_fusion APIs use the
# pipelined single-kernel ring collectives; set FUSION_RING=0 to use the
# notify_dispatch + allgather_permute / staged-unpermute implementation.
_FUSION_RING = os.environ.get("FUSION_RING", "1") == "1"
# When FUSION_RING_PUSH=1, the ring dispatch uses the PUSH kernel (posted writes
# to the right peer interleaved with permute) instead of the default PULL kernel.
_FUSION_RING_PUSH = os.environ.get("FUSION_RING_PUSH", "0") == "1"


@dataclass
class SymmHandle:
    """Handle that carries permute outputs for use by unpermute.

    Returned by allgather_local_permute_fusion and accepted by
    unpermute_reducescatter_fusion so the caller doesn't need to
    pass individual tensors manually.

    scatter_idx holds expert-relative positions; abs_scatter_idx
    holds the absolute positions used by the staged unpermute kernel.
    """

    scatter_idx: torch.Tensor         # [num_tokens, topk] int32 (expert-relative)
    global_topk_weights: torch.Tensor # [num_tokens, topk] float32 (gathered global weights)
    num_tokens_per_rank: int
    global_topk_idx: torch.Tensor     # [num_tokens, topk] int32
    rows_per_expert: torch.Tensor     # [num_experts] int32
    abs_scatter_idx: torch.Tensor     # [num_tokens, topk] int32 (absolute, for staged unpermute)
    num_experts: int = 0              # total experts
    global_scale: torch.Tensor = None       # [num_tokens] float32 gathered per-token scale (FP8)
    permuted_scale: torch.Tensor = None      # [remap_rows] float32 per-row scale aligned to remap


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
        self.count = 0

        device = f"xpu:{self.rank_idx}"
        num_tokens_max = num_max_tokens_per_rank * self.num_ranks
        topk = num_topk
        hidden_elem_size = torch.tensor([], dtype=hidden_dtype).element_size()

        # Allocate one big symmetric memory workspace for all data:
        #   Section 1 — hidden states:  num_ranks slots of [M, H] in hidden_dtype
        #   Section 2 — topk indices:   num_ranks slots of [M, topk] in int32
        #   Section 3 — topk weights:   num_ranks slots of [M, topk] in float32
        #   Section 4 — per-token scale: num_ranks slots of [M] in float32 (FP8 path)
        hidden_section_bytes = (
            num_max_tokens_per_rank * hidden * hidden_elem_size * self.num_ranks
        )
        topk_section_bytes = (
            num_max_tokens_per_rank * num_topk * 4 * self.num_ranks
        )
        weights_section_bytes = (
            num_max_tokens_per_rank * num_topk * 4 * self.num_ranks
        )
        # Per-token scale: one float32 per token (not per top-k slot).
        scale_section_bytes = (
            num_max_tokens_per_rank * 4 * self.num_ranks
        )
        # Signal pad for ring collectives: placed after the four data sections.
        ring_pad_slots = self.num_ranks * _RING_MAX_WG
        ring_pad_bytes = ring_pad_slots * 4
        self._ring_pad_offset_bytes = (
            (
                hidden_section_bytes
                + topk_section_bytes
                + weights_section_bytes
                + scale_section_bytes
                + 127
            )
            // 128 * 128
        )
        self.workspace_size_bytes = self._ring_pad_offset_bytes + ring_pad_bytes
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

        # --- notify_dispatch pre-allocations (per-token scale, FP8 path) ---
        # One float32 per token, gathered the same way as weights but indexed
        # per token (not per top-k slot).
        self._scale_base_offset = (
            hidden_section_bytes + topk_section_bytes + weights_section_bytes
        ) // 4  # in float32 elements
        scale_local_offset = (
            self._scale_base_offset + self.rank_idx * num_max_tokens_per_rank
        )
        self._scale_local_slot = self.workspace.get_buffer(
            self.rank_idx,
            (num_max_tokens_per_rank,),
            torch.float32,
            storage_offset=scale_local_offset,
        )
        self._scale_rank_ptrs = self._build_scale_rank_ptrs()
        self._global_scale_buf = torch.empty(
            num_tokens_max, device=device, dtype=torch.float32,
        )

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

        # --- ring fused-collective pre-allocations ---
        # Built last so SymmBuffer's (larger) workspace already exists; the ring
        # data region (num_tokens_per_rank * hidden * world_size elements) is no
        # larger than the hidden section, so reusing the cached workspace does
        # not grow/invalidate it.
        self._fusion_ring = (
            _FUSION_RING
            and _HAS_RING_ALLGATHER_PERMUTE
            and _HAS_RING_REDUCE_SCATTER_UNPERMUTE
            and _HAS_NOTIFY_DISPATCH_V2_KERNEL
        )
        self._ring_rank_buffers_ptr = None
        self._ring_signal_pads_ptr = None
        self._ring_local_data = None
        self._ring_local_pad = None
        if self._fusion_ring:
            # Ring rank_buffers_ptr: all ranks at offset 0 (flat gather buffer).
            data_numel = num_max_tokens_per_rank * hidden * self.num_ranks
            ring_data_ptrs = []
            ring_pad_ptrs = []
            pad_offset_i32 = self._ring_pad_offset_bytes // 4
            pad_slots = self.num_ranks * _RING_MAX_WG
            for r in range(self.num_ranks):
                dbuf = self.workspace.get_buffer(
                    r, (data_numel,), hidden_dtype, storage_offset=0
                )
                pbuf = self.workspace.get_buffer(
                    r, (pad_slots,), torch.int32, storage_offset=pad_offset_i32
                )
                ring_data_ptrs.append(dbuf.data_ptr())
                ring_pad_ptrs.append(pbuf.data_ptr())
                if r == self.rank_idx:
                    self._ring_local_data = dbuf
                    self._ring_local_pad = pbuf
            signed_data = [ctypes.c_int64(p).value for p in ring_data_ptrs]
            signed_pad = [ctypes.c_int64(p).value for p in ring_pad_ptrs]
            self._ring_rank_buffers_ptr = torch.tensor(
                signed_data, dtype=torch.int64, device=device
            )
            self._ring_signal_pads_ptr = torch.tensor(
                signed_pad, dtype=torch.int64, device=device
            )

    def _next_ring_iter(self) -> int:
        """Return a monotonically increasing ring iteration tag."""
        key = self.group.group_name
        v = _ring_iter_counters.get(key, 0) + 1
        _ring_iter_counters[key] = v
        return v

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

    def _build_scale_rank_ptrs(self) -> torch.Tensor:
        """Build rank pointers into per-token scale section of shared workspace."""
        ptr_list = []
        for r in range(self.num_ranks):
            offset = (
                self._scale_base_offset
                + r * self.num_max_tokens_per_rank
            )
            buf = self.workspace.get_buffer(
                r,
                (self.num_max_tokens_per_rank,),
                torch.float32,
                storage_offset=offset,
            )
            ptr_list.append(buf.data_ptr())
        signed_ptrs = [ctypes.c_int64(p).value for p in ptr_list]
        return torch.tensor(
            signed_ptrs, dtype=torch.int64, device=f"xpu:{self.rank_idx}"
        )

    @staticmethod
    def _log_tensor(name: str, t: torch.Tensor) -> None:
        """Log tensor metadata and a data snippet."""
        if t is None:
            _logger.debug("%s: None", name)
            return
        _logger.debug(
            "%s: shape=%s dtype=%s device=%s\n  data=%s",
            name, list(t.shape), t.dtype, t.device, t,
        )


    # ================================================================== #
    #  Public API: allgather + local permute (dispatch)                   #
    # ================================================================== #

    def allgather_local_permute_fusion(
        self,
        hidden_shard: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        remap_hidden_states: torch.Tensor,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, SymmHandle]:
        """Fused allgather + local permute (MoE dispatch).

        Dispatches at the entry point to one of two fully independent
        implementations: the ring single-kernel path (FUSION_RING=1, default) or
        the notify_dispatch + allgather_permute path (FUSION_RING=0).

        Supports FP8 hidden states (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``)
        in addition to 16-bit/float; the permute is a pure byte copy so any
        element width works.

        Args:
            hidden_shard: [num_tokens_per_rank, hidden] local hidden states.
            topk_idx: [num_tokens_per_rank, topk] int32 per-rank expert ids.
            topk_weights: [num_tokens_per_rank, topk] float32 routing weights.
            num_experts: Total number of experts.
            remap_hidden_states: Pre-allocated output [num_tokens * topk, hidden].
            scale: Optional [num_tokens_per_rank] float32 per-token quantization
                scale (FP8). When provided it is allgathered (via
                notify_dispatch_v2) into ``handle.global_scale`` [num_tokens] and
                permuted into ``handle.permuted_scale`` [remap_rows] aligned with
                ``remap_hidden_states``.

        Returns:
            (remap_hidden_states, handle).
        """
        self.count += 1
        if remap_hidden_states.shape[1] != self.hidden:
            raise ValueError(
                f"remap_hidden_states hidden ({remap_hidden_states.shape[1]}) must equal "
                f"self.hidden ({self.hidden})"
            )
        if scale is not None:
            if scale.dim() != 1 or scale.shape[0] != hidden_shard.shape[0]:
                raise ValueError(
                    f"scale must be 1D [num_tokens_per_rank={hidden_shard.shape[0]}], "
                    f"got shape {list(scale.shape)}"
                )
        if self._fusion_ring:
            return self._allgather_local_permute_fusion_ring(
                hidden_shard, topk_idx, topk_weights, num_experts,
                remap_hidden_states, scale,
            )
        return self._allgather_local_permute_fusion_notify(
            hidden_shard, topk_idx, topk_weights, num_experts,
            remap_hidden_states, scale,
        )

    def _permute_scale(
        self,
        global_scale: torch.Tensor,
        abs_scatter_idx: torch.Tensor,
        remap_rows: int,
        num_tokens: int,
        topk: int,
    ) -> torch.Tensor:
        """Scatter per-token global scale into expert-row layout.

        permuted_scale[abs_scatter_idx[t, k]] = global_scale[t] for every valid
        (t, k) destination row, mirroring the hidden-state permute.
        """
        permuted_scale = torch.zeros(
            remap_rows, device=global_scale.device, dtype=torch.float32
        )
        flat_dst = abs_scatter_idx.reshape(-1)
        token_scale = (
            global_scale.reshape(num_tokens, 1)
            .expand(num_tokens, topk)
            .reshape(-1)
        )
        valid = flat_dst >= 0
        permuted_scale[flat_dst[valid].long()] = token_scale[valid]
        return permuted_scale

    # ------------------------------------------------------------------ #
    #  Ring implementation                                                 #
    # ------------------------------------------------------------------ #

    def _allgather_local_permute_fusion_ring(
        self,
        hidden_shard: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        remap_hidden_states: torch.Tensor,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, SymmHandle]:
        """Single-kernel ring allgather + expert-centric permute.

        Uses notify_dispatch_v2 to build global routing tables (expert-relative
        scatter_idx + rows_per_expert), converts to abs_scatter_idx via
        expert_cumsum, then the ring kernel uses it directly.
        """
        num_tokens_per_rank = hidden_shard.shape[0]
        topk = topk_idx.shape[1]
        num_tokens = num_tokens_per_rank * self.num_ranks

        # Stage this rank's topk_idx / weights into symmetric memory so the
        # kernel can read every rank's slice via the per-rank pointer tables.
        self._topk_local_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)
        self._weights_local_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)
        if scale is not None:
            self._scale_local_slot[:num_tokens_per_rank].copy_(scale)
        self.workspace.barrier()

        global_topk_idx = self._global_topk_idx_buf[:num_tokens, :topk]
        global_topk_weights = self._global_topk_weights_buf[:num_tokens, :topk]
        scatter_idx = self._scatter_idx[:num_tokens, :topk]
        rows_per_expert = self._rows_per_expert_buf[:num_experts]
        scale_rank_ptrs = self._scale_rank_ptrs if scale is not None else None
        global_scale = (
            self._global_scale_buf[:num_tokens] if scale is not None else None
        )

        torch.ops.symm_mem.notify_dispatch_v2(
            self._topk_rank_ptrs,
            global_topk_idx,
            scatter_idx,
            rows_per_expert,
            num_tokens_per_rank,
            topk,
            self.num_topk,
            num_experts,
            self.rank_idx,
            self.num_ranks,
            self._weights_rank_ptrs,
            global_topk_weights,
            scale_rank_ptrs,
            global_scale,
        )

        # Compute absolute scatter_idx from expert-relative positions.
        expert_cumsum = torch.zeros(
            num_experts, device=hidden_shard.device, dtype=torch.int32
        )
        expert_cumsum[1:] = rows_per_expert[:-1]
        expert_cumsum = expert_cumsum.cumsum(0).to(torch.int32)
        flat_expert = global_topk_idx.reshape(-1)
        flat_rel = scatter_idx.reshape(-1)
        abs_scatter_idx = (
            expert_cumsum[flat_expert] + flat_rel
        ).reshape(num_tokens, topk)

        iteration = self._next_ring_iter()
        actual_data_numel = num_tokens_per_rank * self.hidden * self.num_ranks
        ring_gather_output = self._ring_local_data[:actual_data_numel]
        ring_op = (
            torch.ops.symm_mem.ring_allgather_permute_push
            if _FUSION_RING_PUSH
            else torch.ops.symm_mem.ring_allgather_permute
        )
        ring_op(
            hidden_shard,
            self._ring_rank_buffers_ptr,
            self._ring_signal_pads_ptr,
            ring_gather_output,
            remap_hidden_states,
            abs_scatter_idx.contiguous(),
            self.rank_idx,
            self.num_ranks,
            iteration,
        )

        permuted_scale = None
        if scale is not None:
            permuted_scale = self._permute_scale(
                global_scale, abs_scatter_idx,
                remap_hidden_states.shape[0], num_tokens, topk,
            )

        handle = SymmHandle(
            scatter_idx=scatter_idx,
            global_topk_weights=global_topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            global_topk_idx=global_topk_idx,
            rows_per_expert=rows_per_expert,
            abs_scatter_idx=abs_scatter_idx,
            num_experts=num_experts,
            global_scale=global_scale,
            permuted_scale=permuted_scale,
        )

        if _SYMM_BUFFER_DEBUG:
            _logger.debug(
                "=== allgather_local_permute_fusion OUTPUT (rank=%d, ring) ===",
                self.rank_idx,
            )
            self._log_tensor("remap_hidden_states", remap_hidden_states)
            self._log_tensor("handle.abs_scatter_idx", abs_scatter_idx)
            self._log_tensor("handle.global_topk_idx", global_topk_idx)
            self._log_tensor("handle.global_topk_weights", global_topk_weights)
            self._log_tensor("handle.rows_per_expert", rows_per_expert)

        return remap_hidden_states, handle

    # ------------------------------------------------------------------ #
    #  notify_dispatch implementation                                      #
    # ------------------------------------------------------------------ #

    def _allgather_local_permute_fusion_notify(
        self,
        hidden_shard: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        remap_hidden_states: torch.Tensor,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, SymmHandle]:
        """notify_dispatch_v2 dispatch + allgather_permute via symmetric memory."""
        if not _HAS_NOTIFY_DISPATCH_V2_KERNEL:
            raise RuntimeError("notify_dispatch_v2 kernel is required")

        num_tokens_per_rank = hidden_shard.shape[0]
        topk = topk_idx.shape[1]
        num_tokens = num_tokens_per_rank * self.num_ranks

        expected_rows = num_tokens * topk
        if remap_hidden_states.shape[0] < expected_rows:
            raise ValueError(
                f"remap_hidden_states rows ({remap_hidden_states.shape[0]}) must be >= "
                f"num_tokens * topk ({expected_rows})"
            )

        # Write local topk_idx / weights / hidden to symmetric memory.
        self._topk_local_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)
        self._weights_local_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)
        self._allgather_local_slot[:num_tokens_per_rank].copy_(hidden_shard)
        if scale is not None:
            self._scale_local_slot[:num_tokens_per_rank].copy_(scale)
        self.workspace.barrier()

        scatter_idx = self._scatter_idx[:num_tokens, :topk]
        global_topk_idx_buf = self._global_topk_idx_buf[:num_tokens, :topk]
        global_topk_weights_buf = self._global_topk_weights_buf[:num_tokens, :topk]
        rows_per_expert_buf = self._rows_per_expert_buf[:num_experts]
        scale_rank_ptrs = self._scale_rank_ptrs if scale is not None else None
        global_scale = (
            self._global_scale_buf[:num_tokens] if scale is not None else None
        )

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
            scale_rank_ptrs,
            global_scale,
        )

        # Absolute scatter_idx from expert-relative positions.
        expert_cumsum = torch.zeros(
            num_experts, device=hidden_shard.device, dtype=torch.int32
        )
        expert_cumsum[1:] = rows_per_expert_buf[:-1]
        expert_cumsum = expert_cumsum.cumsum(0).to(torch.int32)
        flat_expert = global_topk_idx_buf.reshape(-1)
        flat_rel = scatter_idx.reshape(-1)
        abs_scatter_idx = (
            expert_cumsum[flat_expert] + flat_rel
        ).reshape(num_tokens, topk)

        torch.ops.symm_mem.allgather_permute(
            self._allgather_permute_rank_buffers_ptr,
            abs_scatter_idx,
            remap_hidden_states,
            self.rank_idx,
            self.num_ranks,
        )
        self.workspace.barrier()

        permuted_scale = None
        if scale is not None:
            permuted_scale = self._permute_scale(
                global_scale, abs_scatter_idx,
                remap_hidden_states.shape[0], num_tokens, topk,
            )

        handle = SymmHandle(
            scatter_idx=scatter_idx,
            global_topk_weights=global_topk_weights_buf,
            num_tokens_per_rank=num_tokens_per_rank,
            global_topk_idx=global_topk_idx_buf,
            rows_per_expert=rows_per_expert_buf,
            abs_scatter_idx=abs_scatter_idx,
            num_experts=num_experts,
            global_scale=global_scale,
            permuted_scale=permuted_scale,
        )

        if _SYMM_BUFFER_DEBUG:
            _logger.debug(
                "=== allgather_local_permute_fusion OUTPUT (rank=%d, notify) ===",
                self.rank_idx,
            )
            self._log_tensor("remap_hidden_states", remap_hidden_states)
            self._log_tensor("handle.scatter_idx", scatter_idx)
            self._log_tensor("handle.abs_scatter_idx", abs_scatter_idx)

        return remap_hidden_states, handle

    # ================================================================== #
    #  Public API: unpermute + reduce-scatter (dispatch)                  #
    # ================================================================== #

    def unpermute_reducescatter_fusion(
        self,
        expert_output: torch.Tensor,
        handle: SymmHandle,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Fused unpermute + reduce-scatter (MoE combine).

        Dispatches at the entry point to the ring single-kernel path
        (FUSION_RING=1, default) or the staged symmetric-memory path
        (FUSION_RING=0).

        Args:
            expert_output: this rank's expert outputs.
            handle: SymmHandle from allgather_local_permute_fusion.
            output: Pre-allocated output [num_tokens_per_rank, hidden].

        Returns:
            output
        """
        self.count += 1
        # The reduce-scatter folds partials by accumulation, so only 16-bit
        # floating types (FP16 / BF16) are supported here — FP8 has no usable
        # additive accumulation path.
        if self.hidden_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "unpermute_reducescatter_fusion only supports 16-bit hidden "
                f"dtypes (float16/bfloat16); got {self.hidden_dtype}"
            )
        num_tokens = handle.global_topk_idx.shape[0]
        num_tokens_per_rank = num_tokens // self.num_ranks
        if output.shape[0] != num_tokens_per_rank:
            raise ValueError(
                f"output rows ({output.shape[0]}) must equal num_tokens_per_rank ({num_tokens_per_rank})"
            )
        if self._fusion_ring:
            return self._unpermute_reducescatter_fusion_ring(
                expert_output, handle, output, num_tokens_per_rank
            )
        return self._unpermute_reducescatter_fusion_staged(
            expert_output, handle, output, num_tokens_per_rank
        )

    # ------------------------------------------------------------------ #
    #  Ring implementation                                                 #
    # ------------------------------------------------------------------ #

    def _unpermute_reducescatter_fusion_ring(
        self,
        expert_output: torch.Tensor,
        handle: SymmHandle,
        output: torch.Tensor,
        num_tokens_per_rank: int,
    ) -> torch.Tensor:
        """Single-kernel fused unpermute + ring reduce-scatter."""
        self._ring_local_pad.zero_()
        self.workspace.barrier()

        iteration = self._next_ring_iter()
        actual_data_numel = num_tokens_per_rank * self.hidden * self.num_ranks
        ring_acc_output = self._ring_local_data[:actual_data_numel]
        torch.ops.symm_mem.ring_reduce_scatter_unpermute(
            expert_output,
            self._ring_rank_buffers_ptr,
            self._ring_signal_pads_ptr,
            ring_acc_output,
            output,
            handle.abs_scatter_idx.contiguous(),
            handle.global_topk_weights.contiguous(),
            self.rank_idx,
            self.num_ranks,
            iteration,
        )
        if _SYMM_BUFFER_DEBUG:
            _logger.debug(
                "=== unpermute_reducescatter_fusion OUTPUT (rank=%d, ring) ===",
                self.rank_idx,
            )
            self._log_tensor("output", output)
        return output

    # ------------------------------------------------------------------ #
    #  Staged symmetric-memory implementation                              #
    # ------------------------------------------------------------------ #

    def _unpermute_reducescatter_fusion_staged(
        self,
        expert_output: torch.Tensor,
        handle: SymmHandle,
        output: torch.Tensor,
        num_tokens_per_rank: int,
    ) -> torch.Tensor:
        """Pipelined unpermute (per remote chunk) + symmetric-memory reduce."""
        if not _HAS_LOCAL_UNPERMUTE_KERNEL:
            raise RuntimeError(
                "local_unpermute_copy_ kernel is required for "
                "unpermute_reducescatter_fusion"
            )

        scatter_idx = handle.abs_scatter_idx
        global_topk_weights = handle.global_topk_weights

        self.workspace.barrier()
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
                    self._unpermute_local_bufs[buf_idx][:num_tokens_per_rank],
                )
                self._unpermute_target_recv_bufs[step][self.rank_idx, :num_tokens_per_rank].copy_(
                    self._unpermute_local_bufs[buf_idx][:num_tokens_per_rank]
                )

        # Own-chunk compute on main stream — overlaps with last push
        torch.ops.symm_mem.local_unpermute_copy_(
            expert_output, scatter_idx, global_topk_weights,
            my_chunk_start, num_tokens_per_rank, output,
        )

        torch.xpu.current_stream().wait_stream(stream)
        self.workspace.barrier()

        if _HAS_SUM_REDUCTION_KERNEL and self.num_ranks > 2:
            # Slicing the middle dim produces a non-contiguous view;
            # avoid .contiguous() (which copies) — try the strided view first.
            recv_view = self._unpermute_my_recv_buf[:, :num_tokens_per_rank, :]
            try:
                torch.ops.symm_mem.sum_reduction(
                    recv_view, output, self.rank_idx, self.num_ranks
                )
            except RuntimeError:
                # Kernel requires contiguous input — fall back to per-rank add
                for i in range(self.num_ranks):
                    if i != self.rank_idx:
                        output.add_(self._unpermute_my_recv_buf[i, :num_tokens_per_rank])
        else:
            for i in range(self.num_ranks):
                if i != self.rank_idx:
                    output.add_(self._unpermute_my_recv_buf[i, :num_tokens_per_rank])

        self.workspace.barrier()

        if _SYMM_BUFFER_DEBUG:
            _logger.debug(
                "=== unpermute_reducescatter_fusion OUTPUT (rank=%d, staged) ===",
                self.rank_idx,
            )
            self._log_tensor("output", output)

        return output

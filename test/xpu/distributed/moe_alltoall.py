"""MoE all-to-all (Expert-Parallel) dispatch + combine on symmetric memory.

This module is the EP counterpart of ``symm_buffer.py`` (which implements the
TP-style MoE fusion: allgather+permute dispatch and unpermute+reduce-scatter
combine).  Under expert parallelism each device owns only
``num_experts / world_size`` experts, so:

* **dispatch** (``ep_dispatch``): every ``(token, k)`` assignment is sent only
  to the *owner* device of its expert, instead of all-gathering every token to
  every rank.
* **combine** (this module's ``combine`` / ``ep_combine_fusion``): each expert
  output is combined back to the token's *original* device only, instead of a
  dense reduce-scatter across all ranks.  This is the reverse of
  ``unpermute + ring reduce-scatter fusion`` (see
  ``unpermute_reducescatter_fusion.py``), but ownership-filtered: a rank only
  contributes the partial weighted sum of the experts it actually owns.

Both paths keep their symmetric-memory workspace, pointer tensors, receive-slot
views and the second (pipeline) stream pre-built in ``__init__`` — mirroring the
``SymmBuffer`` design — so the per-call cost measures the collective itself.

Native kernels used (from ``../csrc``):
    libep_dispatch.so   -> symm_mem::ep_dispatch
    libep_combine.so    -> symm_mem::ep_combine_local_
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# ---------------------------------------------------------------------------
# Native kernel loading
# ---------------------------------------------------------------------------
_CSRC = os.path.join(os.path.dirname(__file__), "..", "csrc")


def _try_load(lib_name: str) -> None:
    path = os.path.join(_CSRC, lib_name)
    if os.path.exists(path):
        try:
            torch.ops.load_library(path)
        except Exception:
            pass


_try_load("libep_dispatch.so")
_try_load("libep_combine.so")
_try_load("libring_reduce_scatter_unpermute.so")

_HAS_EP_DISPATCH = hasattr(torch.ops.symm_mem, "ep_dispatch")
_HAS_RING_RS_UNPERMUTE_EP = hasattr(
    torch.ops.symm_mem, "ring_reduce_scatter_unpermute_ep"
)
# Owner-based (sparse) pull combine: the exact reverse of ep_dispatch.  Phase 1
# pre-aggregates this rank's owned experts locally into a symmetric partial
# buffer; phase 2 gathers only the contributing owner ranks' partial rows
# (deduped) into the output -> cross-device traffic scales with owned data
# (~dispatch volume) instead of the dense (world_size-1)*T*H ring push.
_HAS_EP_COMBINE_PULL = hasattr(
    torch.ops.symm_mem, "ep_combine_pull_partial"
) and hasattr(torch.ops.symm_mem, "ep_combine_pull_gather")

# Fused single-kernel pull combine: overlaps phase-1 (owner pre-aggregation) and
# phase-2 (remote gather) as producer/consumer work-groups of ONE grid, using
# per-token cross-rank ready flags instead of a mid barrier (same-grid WGs DO
# overlap on this HW, unlike cross-stream kernels).  Enabled by default when the
# op is available.
_HAS_EP_COMBINE_PULL_FUSED = hasattr(torch.ops.symm_mem, "ep_combine_pull_fused")

# Combine backend selection: "pull" (sparse, default) or "ring" (dense).
_MOE_COMBINE_BACKEND = os.environ.get("MOE_COMBINE", "pull").lower()

# Use the fused overlapped single-kernel pull combine (default on when built).
_MOE_COMBINE_FUSED = os.environ.get("MOE_COMBINE_FUSED", "1") == "1"

# Pull-combine pipelining: split each rank's tokens into this many slices and
# overlap phase-1 (owner pre-aggregation, compute stream) of later slices with
# phase-2 (remote gather, comm stream) of earlier slices.  1 = no pipeline.
#
# NOTE: measured on this XPU, kernels from different streams do NOT execute
# concurrently (each phase already saturates the tile), so chunking only adds
# per-slice barrier overhead and regresses.  Left configurable for other HW /
# experimentation, but defaults to 1 (serial two-barrier path).
_MOE_COMBINE_CHUNKS = int(os.environ.get("MOE_COMBINE_CHUNKS", "1"))

# Ring signal-pad capacity, must match RING_MAX_WG in RingReduceScatterUnpermute.cpp.
_RING_MAX_WG = 1024

# Monotonic ring-iteration tag per process group (signal-pad generation).
_ring_iter_counters: dict = {}


def get_owner_expert_ranges(num_experts: int, world_size: int):
    """Contiguous expert ranges ``[start, end)`` owned by each rank."""
    base = num_experts // world_size
    rem = num_experts % world_size
    ranges = []
    start = 0
    for r in range(world_size):
        size = base + (1 if r < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def get_expert_owner(expert_id: int, num_experts: int, world_size: int) -> int:
    """Map an expert id to its owner rank."""
    for owner, (start, end) in enumerate(get_owner_expert_ranges(num_experts, world_size)):
        if start <= expert_id < end:
            return owner
    raise ValueError(f"expert_id out of range: {expert_id}")


@dataclass
class MoEDispatchHandle:
    """Outputs of :meth:`MoEAllToAll.dispatch` needed by the combine step."""

    remap_hidden_states: torch.Tensor  # [num_tokens * topk, hidden] owner-filled rows
    recv_topk_idx: torch.Tensor        # [num_tokens * topk, topk] int32
    recv_topk_weights: torch.Tensor    # [num_tokens * topk, topk] float32
    global_topk_idx: torch.Tensor      # [num_tokens, topk] int32
    scatter_idx: torch.Tensor          # [num_tokens, topk] int32
    global_topk_weights: torch.Tensor  # [num_tokens, topk] float32


class MoEAllToAll:
    """Expert-parallel MoE all-to-all (dispatch + combine) on symmetric memory.

    Args:
        group: process group (expert-parallel group).
        num_max_tokens_per_rank: max tokens each rank holds (``T``).
        hidden: hidden dim (``H``).
        num_topk: experts per token (``K``).
        num_experts: total number of experts (``E``).
        hidden_dtype: hidden dtype (default bfloat16).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        num_experts: int,
        hidden_dtype: torch.dtype = torch.bfloat16,
    ):
        if not _HAS_EP_DISPATCH:
            raise RuntimeError("ep_dispatch kernel unavailable; build libep_dispatch.so")
        if not _HAS_RING_RS_UNPERMUTE_EP:
            raise RuntimeError(
                "ring_reduce_scatter_unpermute_ep kernel unavailable; "
                "rebuild libring_reduce_scatter_unpermute.so"
            )

        self.group = group
        self.group_name = group.group_name
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.T = num_max_tokens_per_rank
        self.hidden = hidden
        self.topk = num_topk
        self.num_experts = num_experts
        self.dtype = hidden_dtype
        self.device = f"xpu:{self.rank}"

        W, T, H = self.world_size, self.T, self.hidden
        elem = torch.empty(0, dtype=hidden_dtype).element_size()

        # One symmetric workspace holding two regions:
        #   [0 .. data_bytes)          flat data buffer [W*T*H] (dispatch hidden
        #                              slot [T,H] aliases its head; combine ring
        #                              acc uses the whole thing)
        #   [pad_offset .. +pad_bytes) ring signal pad  [W*_RING_MAX_WG] int32
        data_bytes = W * T * H * elem
        self._ring_pad_offset_bytes = ((data_bytes + 127) // 128) * 128
        pad_slots = W * _RING_MAX_WG
        pad_bytes = pad_slots * 4
        self.workspace = symm_mem.get_symm_mem_workspace(
            self.group_name, min_size=self._ring_pad_offset_bytes + pad_bytes
        )

        # --- dispatch: per-rank hidden buffer pointers (offset 0, shape [T, H]) ---
        self._hidden_ptr = self._build_ptrs((T, H), hidden_dtype)

        # --- combine: ring reduce-scatter resources (mirrors symm_buffer) ---
        data_numel = W * T * H
        pad_offset_i32 = self._ring_pad_offset_bytes // 4
        ring_data_ptrs, ring_pad_ptrs = [], []
        for r in range(W):
            dbuf = self.workspace.get_buffer(
                r, (data_numel,), hidden_dtype, storage_offset=0
            )
            pbuf = self.workspace.get_buffer(
                r, (pad_slots,), torch.int32, storage_offset=pad_offset_i32
            )
            ring_data_ptrs.append(ctypes.c_int64(dbuf.data_ptr()).value)
            ring_pad_ptrs.append(ctypes.c_int64(pbuf.data_ptr()).value)
            if r == self.rank:
                self._ring_local_data = dbuf   # doubles as this rank's acc buffer
                self._ring_local_pad = pbuf
        self._ring_rank_buffers_ptr = torch.tensor(
            ring_data_ptrs, dtype=torch.int64, device=self.device
        )
        self._ring_signal_pads_ptr = torch.tensor(
            ring_pad_ptrs, dtype=torch.int64, device=self.device
        )

        # --- dispatch: dedicated topk-weight workspace (separate group so it
        #     does not alias the hidden/ring workspace) ---
        self._tw_group = dist.new_group(
            list(range(W)), group_desc=f"moe_alltoall_tw_{self.group_name}"
        )
        self._tw_ws = symm_mem.get_symm_mem_workspace(
            self._tw_group.group_name, min_size=T * num_topk * 4
        )
        self._tw_slot = self._tw_ws.get_buffer(
            self.rank, (T, num_topk), torch.float32, storage_offset=0
        )
        self._tw_ptr = self._build_tw_ptrs()

        # --- combine (pull): symmetric partial buffer [W*T, H] + peer ptrs ---
        #     Phase 1 writes this rank's owner-aggregated partials here; phase 2
        #     reads peers' partials.  Dedicated group so it never aliases the
        #     hidden/ring or topk-weight workspaces.
        self._pull_group = None
        self._pull_ws = None
        self._partial_slot = None
        self._partial_rank_ptrs = None
        if _HAS_EP_COMBINE_PULL:
            self._pull_group = dist.new_group(
                list(range(W)), group_desc=f"moe_alltoall_pull_{self.group_name}"
            )
            self._pull_ws = symm_mem.get_symm_mem_workspace(
                self._pull_group.group_name, min_size=W * T * H * elem
            )
            self._partial_slot = self._pull_ws.get_buffer(
                self.rank, (W * T, H), hidden_dtype, storage_offset=0
            )
            partial_ptrs = [
                ctypes.c_int64(
                    self._pull_ws.get_buffer(
                        r, (W * T, H), hidden_dtype, storage_offset=0
                    ).data_ptr()
                ).value
                for r in range(W)
            ]
            self._partial_rank_ptrs = torch.tensor(
                partial_ptrs, dtype=torch.int64, device=self.device
            )
            # Dedicated compute stream so pipelined phase-1 (owner aggregation)
            # can run ahead of phase-2 (remote gather) on the default stream.
            self._pull_compute_stream = torch.xpu.Stream(device=self.device)

            # --- fused combine: per-global-token ready flags [W*T] int32 ---
            #     Producers set ready[g]=tag after publishing partial[g];
            #     consumers spin until the owner's ready[g]==tag before reading.
            self._fused_tag = 0
            self._ready_slot = None
            self._ready_rank_ptrs = None
            if _HAS_EP_COMBINE_PULL_FUSED:
                self._ready_group = dist.new_group(
                    list(range(W)),
                    group_desc=f"moe_alltoall_ready_{self.group_name}",
                )
                self._ready_ws = symm_mem.get_symm_mem_workspace(
                    self._ready_group.group_name, min_size=W * T * 4
                )
                self._ready_slot = self._ready_ws.get_buffer(
                    self.rank, (W * T,), torch.int32, storage_offset=0
                )
                self._ready_slot.zero_()
                ready_ptrs = [
                    ctypes.c_int64(
                        self._ready_ws.get_buffer(
                            r, (W * T,), torch.int32, storage_offset=0
                        ).data_ptr()
                    ).value
                    for r in range(W)
                ]
                self._ready_rank_ptrs = torch.tensor(
                    ready_ptrs, dtype=torch.int64, device=self.device
                )

    def _next_ring_iter(self) -> int:
        key = self.group_name
        v = _ring_iter_counters.get(key, 0) + 1
        _ring_iter_counters[key] = v
        return v

    # ------------------------------------------------------------------ #
    #  Pointer builders                                                    #
    # ------------------------------------------------------------------ #
    def _build_ptrs(self, shape, dtype) -> torch.Tensor:
        ptrs = []
        for r in range(self.world_size):
            buf = self.workspace.get_buffer(r, shape, dtype, storage_offset=0)
            ptrs.append(ctypes.c_int64(buf.data_ptr()).value)
        return torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    def _build_tw_ptrs(self) -> torch.Tensor:
        ptrs = []
        for r in range(self.world_size):
            buf = self._tw_ws.get_buffer(
                r, (self.T, self.topk), torch.float32, storage_offset=0
            )
            ptrs.append(ctypes.c_int64(buf.data_ptr()).value)
        return torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------ #
    #  Dispatch (ep_dispatch)                                              #
    # ------------------------------------------------------------------ #
    def dispatch(
        self,
        hidden_shard: torch.Tensor,
        global_topk_idx: torch.Tensor,
        global_topk_weights: torch.Tensor,
        scatter_idx: torch.Tensor,
        remap_hidden_states: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        recv_topk_weights: torch.Tensor,
    ) -> MoEDispatchHandle:
        """Owner-based all-to-all dispatch.

        Publishes this rank's hidden shard and per-token weights to symmetric
        memory, then a single kernel launch pulls every owned ``(token, k)``
        row from the source rank into ``remap_hidden_states``.

        Args:
            hidden_shard: [T, H] this rank's local hidden states.
            global_topk_idx: [num_tokens, topk] int32 routing (all ranks).
            global_topk_weights: [num_tokens, topk] float32 routing weights.
            scatter_idx: [num_tokens, topk] int32 expert-sorted write positions.
            remap_hidden_states: [num_tokens * topk, H] output (owner rows filled).
            recv_topk_idx / recv_topk_weights: [num_tokens * topk, topk] outputs.
        """
        W, T, H = self.world_size, self.T, self.hidden

        # Publish hidden shard.
        local_slot = self.workspace.get_buffer(
            self.rank, (T, H), self.dtype, storage_offset=0
        )
        local_slot.copy_(hidden_shard)
        self.workspace.barrier()

        # Publish this rank's local weight shard for remote reads.
        start = self.rank * T
        self._tw_slot.copy_(global_topk_weights[start : start + T].contiguous())
        self._tw_ws.barrier()

        topk_idx_i32 = (
            global_topk_idx
            if global_topk_idx.dtype == torch.int32
            else global_topk_idx.to(torch.int32)
        )
        recv_topk_idx.fill_(-1)
        recv_topk_weights.zero_()
        torch.ops.symm_mem.ep_dispatch(
            self._hidden_ptr,
            topk_idx_i32,
            self._tw_ptr,
            scatter_idx,
            remap_hidden_states,
            recv_topk_idx,
            recv_topk_weights,
            self.topk,          # topk_weights_stride
            self.num_experts,
            self.rank,
            self.world_size,
        )
        self.workspace.barrier()

        return MoEDispatchHandle(
            remap_hidden_states=remap_hidden_states,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            global_topk_idx=topk_idx_i32,
            scatter_idx=scatter_idx,
            global_topk_weights=global_topk_weights,
        )

    # ------------------------------------------------------------------ #
    #  Combine (owner-based, sparse pull -- reverse of ep_dispatch)         #
    # ------------------------------------------------------------------ #
    def combine(
        self,
        expert_output: torch.Tensor,
        handle: MoEDispatchHandle,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Owner-based MoE combine (the exact reverse of ``dispatch``).

        Default backend (``MOE_COMBINE=pull``): a two-phase *sparse* pull that
        mirrors ep_dispatch's owner-based all-to-all, so combine cross-device
        traffic scales with owned data (~dispatch volume) instead of the dense
        ``(world_size-1)*T*H`` ring push:

          * phase 1 (``ep_combine_pull_partial``, local): pre-aggregate the
            weighted sum of the experts THIS rank owns into a symmetric partial
            buffer laid out ``[world_size*T, H]`` (one row per global token);
          * phase 2 (``ep_combine_pull_gather``, remote): for each of this rank's
            own tokens, read + sum only the partial rows of the ranks that own
            one of its experts (deduped owner mask) -> at most ``#owner`` sparse
            coalesced remote reads per token, no incast, local-only writes.

        Setting ``MOE_COMBINE=ring`` selects the legacy dense
        ``ring_reduce_scatter_unpermute_ep`` backend.

        Args:
            expert_output: [num_tokens * topk, H] expert outputs; only rows for
                experts owned by this rank need to be valid.
            handle: the dispatch handle (carries routing tensors).
            output: [T, H] pre-allocated output for this rank's token shard.
        """
        topk_idx_i32 = (
            handle.global_topk_idx
            if handle.global_topk_idx.dtype == torch.int32
            else handle.global_topk_idx.to(torch.int32)
        ).contiguous()
        tw_f32 = (
            handle.global_topk_weights
            if handle.global_topk_weights.dtype == torch.float32
            else handle.global_topk_weights.float()
        ).contiguous()

        use_pull = _HAS_EP_COMBINE_PULL and _MOE_COMBINE_BACKEND != "ring"
        if use_pull:
            return self._combine_pull(expert_output, handle, output, topk_idx_i32, tw_f32)
        return self._combine_ring(expert_output, handle, output, topk_idx_i32, tw_f32)

    def _combine_pull(
        self, expert_output, handle, output, topk_idx_i32, tw_f32
    ) -> torch.Tensor:
        scatter = handle.scatter_idx.contiguous()
        if _MOE_COMBINE_FUSED and self._ready_slot is not None:
            return self._combine_pull_fused(
                expert_output, output, topk_idx_i32, tw_f32, scatter
            )
        T = output.size(0)
        chunks = _MOE_COMBINE_CHUNKS
        if chunks > 1 and T % chunks == 0:
            return self._combine_pull_pipelined(
                expert_output, output, topk_idx_i32, tw_f32, scatter, chunks
            )
        return self._combine_pull_serial(
            expert_output, output, topk_idx_i32, tw_f32, scatter
        )

    def _combine_pull_fused(
        self, expert_output, output, topk_idx_i32, tw_f32, scatter
    ) -> torch.Tensor:
        # Single-kernel overlapped combine.  ONE start barrier protects the
        # previous iteration's readers and aligns all ranks; the per-token ready
        # flags (monotonic tag) replace the phase1/phase2 mid barrier.
        self._fused_tag += 1
        self._pull_ws.barrier()
        torch.ops.symm_mem.ep_combine_pull_fused(
            expert_output,
            self._partial_slot,
            self._partial_rank_ptrs,
            self._ready_slot,
            self._ready_rank_ptrs,
            topk_idx_i32,
            scatter,
            tw_f32,
            output,
            self.num_experts,
            self.rank,
            self.world_size,
            self._fused_tag,
        )
        return output

    def _combine_pull_serial(
        self, expert_output, output, topk_idx_i32, tw_f32, scatter
    ) -> torch.Tensor:
        # Phase 1: owner-side local pre-aggregation into this rank's symmetric
        # partial buffer.  Barrier before, so no peer is still reading last
        # iteration's partials while we overwrite them.
        self._pull_ws.barrier()
        torch.ops.symm_mem.ep_combine_pull_partial(
            expert_output,
            self._partial_slot,
            topk_idx_i32,
            scatter,
            tw_f32,
            self.num_experts,
            self.rank,
            self.world_size,
        )
        # Publish partials, then phase 2: sparse gather from owner ranks.
        self._pull_ws.barrier()
        torch.ops.symm_mem.ep_combine_pull_gather(
            self._partial_rank_ptrs,
            topk_idx_i32,
            output,
            expert_output,
            scatter,
            tw_f32,
            self.num_experts,
            self.rank,
            self.world_size,
        )
        return output

    def _combine_pull_pipelined(
        self, expert_output, output, topk_idx_i32, tw_f32, scatter, chunks
    ) -> torch.Tensor:
        # Overlap phase-1 (owner aggregation, local) of later token slices with
        # phase-2 (remote gather) of earlier slices.  phase-1 runs on a compute
        # stream that races ahead; the comm (default) stream gates each slice's
        # gather on (a) local phase-1 completion via an event and (b) all-rank
        # phase-1 completion via a per-slice symmetric barrier.  No in-kernel
        # peer spin -> no cross-rank dependency chain / tail latency.
        T = output.size(0)
        step = T // chunks
        ws = self.world_size
        comm_stream = torch.xpu.current_stream()
        compute_stream = self._pull_compute_stream

        # Protect previous iteration's readers before overwriting partials, and
        # make the compute stream observe the inputs produced on the comm stream.
        self._pull_ws.barrier()
        compute_stream.wait_stream(comm_stream)

        events = [torch.xpu.Event() for _ in range(chunks)]
        with torch.xpu.stream(compute_stream):
            for c in range(chunks):
                torch.ops.symm_mem.ep_combine_pull_partial(
                    expert_output,
                    self._partial_slot,
                    topk_idx_i32,
                    scatter,
                    tw_f32,
                    self.num_experts,
                    self.rank,
                    ws,
                    c * step,
                    step,
                )
                events[c].record(compute_stream)

        for c in range(chunks):
            comm_stream.wait_event(events[c])
            self._pull_ws.barrier(channel=c + 1)
            torch.ops.symm_mem.ep_combine_pull_gather(
                self._partial_rank_ptrs,
                topk_idx_i32,
                output,
                expert_output,
                scatter,
                tw_f32,
                self.num_experts,
                self.rank,
                ws,
                c * step,
                step,
            )
        return output

    def _combine_ring(
        self, expert_output, handle, output, topk_idx_i32, tw_f32
    ) -> torch.Tensor:
        """Legacy dense ring reduce-scatter combine (kept for comparison)."""
        # Fresh signal-pad generation for this ring iteration.
        self._ring_local_pad.zero_()
        self.workspace.barrier()

        iteration = self._next_ring_iter()
        acc = self._ring_local_data  # [W*T*H], this rank's symmetric acc buffer
        torch.ops.symm_mem.ring_reduce_scatter_unpermute_ep(
            expert_output,
            self._ring_rank_buffers_ptr,
            self._ring_signal_pads_ptr,
            acc,
            output,
            handle.scatter_idx.contiguous(),
            topk_idx_i32,
            tw_f32,
            self.num_experts,
            self.rank,
            self.world_size,
            iteration,
        )
        return output

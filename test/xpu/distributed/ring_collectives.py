"""
Ring allgather and ring reduce-scatter built on symmetric memory.

These wrap the native SYCL kernels in test/xpu/csrc/RingAllgather.cpp and
test/xpu/csrc/RingReduceScatter.cpp, which implement the pipelined ring
algorithms (one chunk per link per step) modeled after oneCCL's
allgatherv_large_sycl_ring / reduce_scatter_large_sycl_ring.

Cross-rank ordering is handled on-device with per-step signal pads, so the
only host-side synchronization needed is a single workspace barrier to publish
the zeroed signal pads before each collective.

Usage (see test_ring_collectives_dist.py):
    out = ring_allgather(input_shard)            # [world_size * chunk]
    out = ring_reduce_scatter(full_input)        # [chunk] (reduced block `rank`)
"""

import ctypes
import json
import os
import threading

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import env

# ---------------------------------------------------------------------------
# Load native libraries
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(__file__)

_HAS_RING_ALLGATHER = False
_HAS_RING_REDUCE_SCATTER = False

for _lib, _attr in (
    ("libring_allgather.so", "ring_allgather"),
    ("libring_reduce_scatter.so", "ring_reduce_scatter"),
    ("libring_allgather_permute.so", "ring_allgather_permute"),
    ("libring_reduce_scatter_unpermute.so", "ring_reduce_scatter_unpermute"),
    (
        "libring_reduce_scatter_unpermute_two_stage.so",
        "ring_reduce_scatter_unpermute_two_stage",
    ),
):
    _path = os.path.join(_BASE, "..", "csrc", _lib)
    if os.path.exists(_path):
        try:
            torch.ops.load_library(_path)
        except Exception:
            pass

_HAS_RING_ALLGATHER = hasattr(torch.ops.symm_mem, "ring_allgather")
_HAS_RING_REDUCE_SCATTER = hasattr(torch.ops.symm_mem, "ring_reduce_scatter")
_HAS_RING_ALLGATHER_PERMUTE = hasattr(torch.ops.symm_mem, "ring_allgather_permute")
_HAS_RING_REDUCE_SCATTER_UNPERMUTE = hasattr(
    torch.ops.symm_mem, "ring_reduce_scatter_unpermute"
)
_HAS_RING_REDUCE_SCATTER_UNPERMUTE_TWO_STAGE = hasattr(
    torch.ops.symm_mem, "ring_reduce_scatter_unpermute_two_stage"
)

# Monotonically increasing signal tag per group (kept for robustness even
# though pads are zeroed before each call).
_iter_counters = {}

# Upper bound on the number of work-groups any single-kernel ring collective
# launches; it sizes the signal-pad region.  Must be >= the largest RING_MAX_WG
# used by any ring kernel (RingAllgather.cpp / RingReduceScatter.cpp use 64;
# RingAllgatherPermute.cpp uses 256, since more work-groups expose the
# parallelism needed to saturate cross-GPU read bandwidth on the pull ring).
# The single-kernel ring uses per-work-group signal slots laid out as
# slot(phase, wg) = phase * num_wg + wg, with phase up to world_size-1 and
# wg up to num_wg-1 (num_wg <= RING_MAX_WG). Allocating world_size * _RING_MAX_WG
# uint32 slots per rank is always sufficient.
_RING_MAX_WG = 256


def _next_iter(group_name):
    v = _iter_counters.get(group_name, 0) + 1
    _iter_counters[group_name] = v
    return v


# ---------------------------------------------------------------------------
# Work-group autotuning
# ---------------------------------------------------------------------------
# The ring reduce-scatter kernels are memory-latency-bound on the scattered
# top-k gather; the optimal work-group count depends on the GPU (EU count,
# resident-subgroup capacity) and the problem shape.  Rather than bake a magic
# constant, we autotune it once per (platform, shape) and persist the result so
# a *new platform* pays the search cost exactly once.
#
# CRITICAL: this is a symmetric collective -- work-group `i` on rank R signals
# work-group `i` on rank R+1, so num_wg MUST be identical on every rank.  The
# tuner therefore all-reduces candidate timings and every rank picks the same
# argmin (never a per-rank local decision).
_RING_MAX_WG = 256  # upper bound; must match the native RING_MAX_WG / pad size.

# Toggle: "0"/"false" -> use the native analytic formula (num_wg_hint=-1).
_WG_AUTOTUNE_ENABLED = env.ring_wg_autotune()
_WG_TUNE_WARMUP = env.ring_wg_tune_warmup()
_WG_TUNE_ITERS = env.ring_wg_tune_iters()

_WG_CACHE_PATH = env.ring_wg_cache_path()

_wg_cache = None  # dict: str(key) -> int(num_wg)
_wg_cache_lock = threading.Lock()
_wg_inflight = set()  # keys currently being tuned (guards re-entrancy)


def _wg_cache_load():
    global _wg_cache
    if _wg_cache is not None:
        return _wg_cache
    _wg_cache = {}
    try:
        with open(_WG_CACHE_PATH, "r") as f:
            _wg_cache = json.load(f)
    except Exception:
        _wg_cache = {}
    return _wg_cache


def _wg_cache_store(key, value):
    cache = _wg_cache_load()
    cache[key] = int(value)
    try:
        os.makedirs(os.path.dirname(_WG_CACHE_PATH), exist_ok=True)
        tmp = _WG_CACHE_PATH + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=0, sort_keys=True)
        os.replace(tmp, _WG_CACHE_PATH)
    except Exception:
        pass  # cache persistence is best-effort


def _platform_tag(device):
    try:
        p = torch.xpu.get_device_properties(device)
        return f"{p.name}|eu{getattr(p, 'gpu_eu_count', 0)}"
    except Exception:
        return "unknown-xpu"


def _wg_cache_key(device, dtype, world_size, tokens, hidden, topk):
    return "|".join(
        str(x)
        for x in (
            _platform_tag(device),
            str(dtype),
            world_size,
            tokens,
            hidden,
            topk,
        )
    )


def _wg_candidates(tokens):
    """Candidate num_wg values from power-of-two token splits (even division
    avoids the load-imbalance spikes seen with odd tokens/wg)."""
    cands = set()
    tpw = 1
    while tpw <= max(1, tokens):
        nwg = (tokens + tpw - 1) // tpw
        if 1 <= nwg <= _RING_MAX_WG:
            cands.add(int(nwg))
        tpw *= 2
    if not cands:
        cands.add(max(1, min(tokens, _RING_MAX_WG)))
    return sorted(cands)


def _time_candidate(invoke, num_wg, device, group):
    """Median latency (ms) of `invoke(num_wg)`, agreed across ranks by summing
    per-rank times so every rank selects the identical winner."""
    for _ in range(_WG_TUNE_WARMUP):
        invoke(num_wg)
    torch.xpu.synchronize()
    begins = [torch.xpu.Event(enable_timing=True) for _ in range(_WG_TUNE_ITERS)]
    ends = [torch.xpu.Event(enable_timing=True) for _ in range(_WG_TUNE_ITERS)]
    for i in range(_WG_TUNE_ITERS):
        begins[i].record()
        invoke(num_wg)
        ends[i].record()
    torch.xpu.synchronize()
    lat = sorted(begins[i].elapsed_time(ends[i]) for i in range(_WG_TUNE_ITERS))
    local_median = lat[len(lat) // 2]
    t = torch.tensor([local_median], dtype=torch.float64, device=device)
    if world_size_of(group) > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t.item()


def world_size_of(group):
    try:
        return dist.get_world_size(group)
    except Exception:
        return 1


def autotune_num_wg(key, invoke, tokens, device, group):
    """Return the best num_wg for `key`, tuning (once) if not cached.

    `invoke(num_wg)` must run one full collective kernel launch with the given
    work-group hint.  All ranks must call this in lock-step (it issues collective
    ops); the returned value is identical on every rank.
    """
    cache = _wg_cache_load()
    if key in cache:
        return int(cache[key])
    with _wg_cache_lock:
        if key in cache:
            return int(cache[key])
        # Re-entrancy guard: if we're already tuning this key, fall back to auto.
        if key in _wg_inflight:
            return -1
        _wg_inflight.add(key)
    try:
        best_wg, best_t = None, None
        for c in _wg_candidates(tokens):
            t = _time_candidate(invoke, c, device, group)
            if best_t is None or t < best_t:
                best_t, best_wg = t, c
        if best_wg is None:
            best_wg = -1
        _wg_cache_store(key, best_wg)
        return int(best_wg)
    finally:
        _wg_inflight.discard(key)


def _align_up(x, a):
    return (x + a - 1) // a * a


def _signed_ptr_tensor(ptrs, device):
    signed = [ctypes.c_int64(p).value for p in ptrs]
    return torch.tensor(signed, dtype=torch.int64, device=device)


def _build_ring_resources(group, group_name, dtype, data_numel, world_size, rank):
    """Allocate (or reuse) a symmetric workspace holding a per-rank data region
    of `data_numel` elements followed by a signal-pad region.

    Returns:
        workspace: the symm_mem workspace handle (for barrier()).
        local_data: this rank's data buffer view [data_numel] (dtype).
        rank_buffers_ptr: int64[world_size] pointers to each rank's data region.
        signal_pads_ptr: int64[world_size] pointers to each rank's pad region.
        local_pad: this rank's pad view [pad_slots] (int32).
    """
    elem_size = torch.empty(0, dtype=dtype).element_size()
    data_bytes = data_numel * elem_size
    pad_slots = world_size * _RING_MAX_WG
    pad_offset_bytes = _align_up(data_bytes, 128)
    total_bytes = pad_offset_bytes + pad_slots * 4

    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=total_bytes)

    pad_offset_i32 = pad_offset_bytes // 4

    data_ptrs = []
    pad_ptrs = []
    local_data = None
    local_pad = None
    for r in range(world_size):
        data_buf = workspace.get_buffer(r, (data_numel,), dtype, storage_offset=0)
        pad_buf = workspace.get_buffer(
            r, (pad_slots,), torch.int32, storage_offset=pad_offset_i32
        )
        data_ptrs.append(data_buf.data_ptr())
        pad_ptrs.append(pad_buf.data_ptr())
        if r == rank:
            local_data = data_buf
            local_pad = pad_buf

    device = local_data.device
    rank_buffers_ptr = _signed_ptr_tensor(data_ptrs, device)
    signal_pads_ptr = _signed_ptr_tensor(pad_ptrs, device)
    return workspace, local_data, rank_buffers_ptr, signal_pads_ptr, local_pad


def _group_info(group):
    if group is None:
        group = dist.group.WORLD
    group_name = group.group_name
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    return group, group_name, rank, world_size


def build_ring_allgather_resources(
    input_shard: torch.Tensor, group: dist.ProcessGroup = None
):
    """Pre-allocate the symmetric workspace and pointer tensors for
    ring_allgather, so the (host-side) buffer/pointer setup can be hoisted out
    of a timed loop.

    Args:
        input_shard: a representative local shard; its dtype and numel define
            the workspace size.
        group: process group (default: WORLD).

    Returns:
        An opaque resources object to pass as `resources=` to ring_allgather.
    """
    assert _HAS_RING_ALLGATHER, "ring_allgather native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_flat = input_shard.contiguous().view(-1)
    chunk = input_flat.numel()
    data_numel = chunk * world_size

    (
        workspace,
        output,
        rank_buffers_ptr,
        signal_pads_ptr,
        local_pad,
    ) = _build_ring_resources(
        group, group_name, input_flat.dtype, data_numel, world_size, rank
    )
    return {
        "group_name": group_name,
        "rank": rank,
        "world_size": world_size,
        "chunk": chunk,
        "dtype": input_flat.dtype,
        "workspace": workspace,
        "output": output,
        "rank_buffers_ptr": rank_buffers_ptr,
        "signal_pads_ptr": signal_pads_ptr,
        "local_pad": local_pad,
    }


def ring_allgather(
    input_shard: torch.Tensor,
    group: dist.ProcessGroup = None,
    resources: dict = None,
) -> torch.Tensor:
    """Pipelined ring allgather.

    Args:
        input_shard: 1D (or flattenable) local shard of `chunk` elements.
        group: process group (default: WORLD).
        resources: optional precomputed resources from
            build_ring_allgather_resources() to skip per-call workspace/pointer
            setup (useful for benchmarking the kernel in isolation).

    Returns:
        A 1D tensor of `world_size * chunk` elements (this rank's symmetric
        output buffer) holding all gathered shards in rank order.
    """
    assert _HAS_RING_ALLGATHER, "ring_allgather native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_flat = input_shard.contiguous().view(-1)
    chunk = input_flat.numel()
    data_numel = chunk * world_size

    if resources is not None:
        assert resources["group_name"] == group_name, "resources group mismatch"
        assert resources["chunk"] == chunk, "resources chunk size mismatch"
        assert resources["dtype"] == input_flat.dtype, "resources dtype mismatch"
        workspace = resources["workspace"]
        output = resources["output"]
        rank_buffers_ptr = resources["rank_buffers_ptr"]
        signal_pads_ptr = resources["signal_pads_ptr"]
        local_pad = resources["local_pad"]
    else:
        (
            workspace,
            output,
            rank_buffers_ptr,
            signal_pads_ptr,
            local_pad,
        ) = _build_ring_resources(
            group, group_name, input_flat.dtype, data_numel, world_size, rank
        )

    # Publish zeroed signal pads to all ranks before the collective starts.
    local_pad.zero_()
    workspace.barrier()

    iteration = _next_iter(group_name)
    torch.ops.symm_mem.ring_allgather(
        input_flat,
        rank_buffers_ptr,
        signal_pads_ptr,
        output,
        rank,
        world_size,
        iteration,
    )
    return output


def build_ring_reduce_scatter_resources(
    input: torch.Tensor, group: dist.ProcessGroup = None
):
    """Pre-allocate the symmetric workspace, pointer tensors and output buffer
    for ring_reduce_scatter, so the (host-side) buffer/pointer setup can be
    hoisted out of a timed loop.

    Args:
        input: a representative full input tensor of `world_size * chunk`
            elements; its dtype and numel define the workspace size.
        group: process group (default: WORLD).

    Returns:
        An opaque resources object to pass as `resources=` to
        ring_reduce_scatter.
    """
    assert _HAS_RING_REDUCE_SCATTER, "ring_reduce_scatter native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_flat = input.contiguous().view(-1)
    total_numel = input_flat.numel()
    assert total_numel % world_size == 0, "input numel must be divisible by world_size"
    chunk = total_numel // world_size

    (
        workspace,
        acc,
        rank_buffers_ptr,
        signal_pads_ptr,
        local_pad,
    ) = _build_ring_resources(
        group, group_name, input_flat.dtype, total_numel, world_size, rank
    )
    output = torch.empty(chunk, dtype=input_flat.dtype, device=input_flat.device)
    return {
        "group_name": group_name,
        "rank": rank,
        "world_size": world_size,
        "chunk": chunk,
        "dtype": input_flat.dtype,
        "workspace": workspace,
        "acc": acc,
        "output": output,
        "rank_buffers_ptr": rank_buffers_ptr,
        "signal_pads_ptr": signal_pads_ptr,
        "local_pad": local_pad,
    }


def ring_reduce_scatter(
    input: torch.Tensor,
    group: dist.ProcessGroup = None,
    resources: dict = None,
) -> torch.Tensor:
    """Pipelined ring reduce-scatter (sum).

    Args:
        input: 1D (or flattenable) tensor of `world_size * chunk` elements
            (the full data, identical layout on every rank).
        group: process group (default: WORLD).
        resources: optional precomputed resources from
            build_ring_reduce_scatter_resources() to skip per-call
            workspace/pointer setup (useful for benchmarking the kernel in
            isolation).

    Returns:
        A 1D tensor of `chunk` elements containing the reduced (summed) block
        with index == rank.
    """
    assert _HAS_RING_REDUCE_SCATTER, "ring_reduce_scatter native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_flat = input.contiguous().view(-1)
    total_numel = input_flat.numel()
    assert total_numel % world_size == 0, "input numel must be divisible by world_size"
    chunk = total_numel // world_size

    if resources is not None:
        assert resources["group_name"] == group_name, "resources group mismatch"
        assert resources["chunk"] == chunk, "resources chunk size mismatch"
        assert resources["dtype"] == input_flat.dtype, "resources dtype mismatch"
        workspace = resources["workspace"]
        acc = resources["acc"]
        output = resources["output"]
        rank_buffers_ptr = resources["rank_buffers_ptr"]
        signal_pads_ptr = resources["signal_pads_ptr"]
        local_pad = resources["local_pad"]
    else:
        (
            workspace,
            acc,
            rank_buffers_ptr,
            signal_pads_ptr,
            local_pad,
        ) = _build_ring_resources(
            group, group_name, input_flat.dtype, total_numel, world_size, rank
        )
        output = torch.empty(chunk, dtype=input_flat.dtype, device=input_flat.device)

    local_pad.zero_()
    workspace.barrier()

    iteration = _next_iter(group_name)
    torch.ops.symm_mem.ring_reduce_scatter(
        input_flat,
        rank_buffers_ptr,
        signal_pads_ptr,
        acc,
        output,
        rank,
        world_size,
        iteration,
    )
    return output


def build_ring_allgather_permute_resources(
    input_shard: torch.Tensor, group: dist.ProcessGroup = None
):
    """Pre-allocate the symmetric workspace and pointer tensors for
    ring_allgather_permute, so the (host-side) buffer/pointer setup can be
    hoisted out of a timed loop.

    Args:
        input_shard: a representative [num_tokens_per_rank, hidden] local shard;
            its dtype and shape define the workspace size.
        group: process group (default: WORLD).

    Returns:
        An opaque resources object to pass as `resources=` to
        ring_allgather_permute.
    """
    assert _HAS_RING_ALLGATHER_PERMUTE, "ring_allgather_permute native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_shard = input_shard.contiguous()
    assert input_shard.dim() == 2, "input_shard must be [tokens, hidden]"
    num_tokens_per_rank, hidden = input_shard.shape
    data_numel = num_tokens_per_rank * hidden * world_size

    (
        workspace,
        gather_output,
        rank_buffers_ptr,
        signal_pads_ptr,
        local_pad,
    ) = _build_ring_resources(
        group, group_name, input_shard.dtype, data_numel, world_size, rank
    )
    return {
        "group_name": group_name,
        "rank": rank,
        "world_size": world_size,
        "num_tokens_per_rank": num_tokens_per_rank,
        "hidden": hidden,
        "dtype": input_shard.dtype,
        "workspace": workspace,
        "gather_output": gather_output,
        "rank_buffers_ptr": rank_buffers_ptr,
        "signal_pads_ptr": signal_pads_ptr,
        "local_pad": local_pad,
    }


def ring_allgather_permute(
    input_shard: torch.Tensor,
    scatter_idx: torch.Tensor,
    remap_rows: int,
    group: dist.ProcessGroup = None,
    resources: dict = None,
    remap_output: torch.Tensor = None,
) -> torch.Tensor:
    """Fused ring allgather + MoE permute (dispatch).

    Args:
        input_shard: [num_tokens_per_rank, hidden] local token shard.
        scatter_idx: [world_size * num_tokens_per_rank, topk] int32 absolute
            destination rows (same semantics as allgather_permute in
            LocalPermuteCopy.cpp).
        remap_rows: number of rows in the expert-sorted output.
        group: process group (default: WORLD).
        resources: optional precomputed resources from
            build_ring_allgather_permute_resources() to skip per-call
            workspace/pointer setup (useful for benchmarking the kernel in
            isolation).
        remap_output: optional preallocated [remap_rows, hidden] output buffer
            to avoid per-call allocation in a timed loop.

    Returns:
        remap_output: [remap_rows, hidden] expert-sorted output.
    """
    assert _HAS_RING_ALLGATHER_PERMUTE, "ring_allgather_permute native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    input_shard = input_shard.contiguous()
    assert input_shard.dim() == 2, "input_shard must be [tokens, hidden]"
    num_tokens_per_rank, hidden = input_shard.shape
    data_numel = num_tokens_per_rank * hidden * world_size

    if resources is not None:
        assert resources["group_name"] == group_name, "resources group mismatch"
        assert resources["num_tokens_per_rank"] == num_tokens_per_rank, (
            "resources num_tokens_per_rank mismatch"
        )
        assert resources["hidden"] == hidden, "resources hidden size mismatch"
        assert resources["dtype"] == input_shard.dtype, "resources dtype mismatch"
        workspace = resources["workspace"]
        gather_output = resources["gather_output"]
        rank_buffers_ptr = resources["rank_buffers_ptr"]
        signal_pads_ptr = resources["signal_pads_ptr"]
        local_pad = resources["local_pad"]
    else:
        (
            workspace,
            gather_output,
            rank_buffers_ptr,
            signal_pads_ptr,
            local_pad,
        ) = _build_ring_resources(
            group, group_name, input_shard.dtype, data_numel, world_size, rank
        )

    if remap_output is None:
        remap_output = torch.empty(
            (remap_rows, hidden), dtype=input_shard.dtype, device=input_shard.device
        )

    # local_pad.zero_()
    # workspace.barrier()

    iteration = _next_iter(group_name)
    # Topology choice is fabric- and scale-dependent. Measured on 8x B60 (PCIe),
    # with the world_size-gated LSC store in the push kernel (LSC auto-off at
    # ws<=4, see RingAllgatherPermute.cpp):
    #   world_size=8: PUSH ~2.44 ms vs PULL ~3.31 ms -> push wins (~26%)
    #   world_size=4: PUSH ~0.775 ms vs PULL ~0.88 ms -> push wins (~12%)
    # PUSH's posted writes win at every measured scale once the low-rank LSC
    # regression is gated out, so default to PUSH whenever available.
    # Override explicitly with RING_AGP_PUSH=1 / =0.
    _have_push = hasattr(torch.ops.symm_mem, "ring_allgather_permute_push")
    _use_push = env.ring_agp_push() and _have_push
    _agp_op = (
        torch.ops.symm_mem.ring_allgather_permute_push
        if _use_push
        else torch.ops.symm_mem.ring_allgather_permute
    )
    _agp_op(
        input_shard,
        rank_buffers_ptr,
        signal_pads_ptr,
        gather_output,
        remap_output,
        scatter_idx.contiguous(),
        rank,
        world_size,
        iteration,
    )
    return remap_output


def build_ring_reduce_scatter_unpermute_resources(
    expert_output: torch.Tensor,
    num_tokens_per_rank: int,
    group: dist.ProcessGroup = None,
):
    """Pre-allocate the symmetric workspace and pointer tensors for
    ring_reduce_scatter_unpermute, so the (host-side) buffer/pointer setup can
    be hoisted out of a timed loop.

    Args:
        expert_output: a representative [remap_rows, hidden] tensor; its dtype
            and hidden size define the workspace size.
        num_tokens_per_rank: number of tokens this rank receives back.
        group: process group (default: WORLD).

    Returns:
        An opaque resources object to pass as `resources=` to
        ring_reduce_scatter_unpermute.
    """
    assert (
        _HAS_RING_REDUCE_SCATTER_UNPERMUTE
    ), "ring_reduce_scatter_unpermute native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    expert_output = expert_output.contiguous()
    hidden = expert_output.size(1)
    data_numel = num_tokens_per_rank * hidden * world_size

    (
        workspace,
        acc,
        rank_buffers_ptr,
        signal_pads_ptr,
        local_pad,
    ) = _build_ring_resources(
        group, group_name, expert_output.dtype, data_numel, world_size, rank
    )
    return {
        "group_name": group_name,
        "rank": rank,
        "world_size": world_size,
        "num_tokens_per_rank": num_tokens_per_rank,
        "hidden": hidden,
        "dtype": expert_output.dtype,
        "workspace": workspace,
        "acc": acc,
        "rank_buffers_ptr": rank_buffers_ptr,
        "signal_pads_ptr": signal_pads_ptr,
        "local_pad": local_pad,
    }


def ring_reduce_scatter_unpermute(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: int,
    group: dist.ProcessGroup = None,
    resources: dict = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Fused MoE unpermute + ring reduce-scatter (combine).

    Args:
        expert_output: [num_tokens*topk, hidden] this rank's expert outputs.
        scatter_idx: [world_size * num_tokens_per_rank, topk] int32 absolute
            destination rows (same semantics as allgather_permute).
        topk_weights: [world_size * num_tokens_per_rank, topk] float32 weights.
        num_tokens_per_rank: number of tokens this rank receives back.
        group: process group (default: WORLD).
        resources: optional precomputed resources from
            build_ring_reduce_scatter_unpermute_resources() to skip per-call
            workspace/pointer setup (useful for benchmarking the kernel in
            isolation).
        output: optional preallocated [num_tokens_per_rank, hidden] output
            buffer to avoid per-call allocation in a timed loop.

    Returns:
        output: [num_tokens_per_rank, hidden] fully combined tokens of this rank.
    """
    assert (
        _HAS_RING_REDUCE_SCATTER_UNPERMUTE
    ), "ring_reduce_scatter_unpermute native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    expert_output = expert_output.contiguous()
    hidden = expert_output.size(1)
    data_numel = num_tokens_per_rank * hidden * world_size

    if resources is not None:
        assert resources["group_name"] == group_name, "resources group mismatch"
        assert resources["num_tokens_per_rank"] == num_tokens_per_rank, (
            "resources num_tokens_per_rank mismatch"
        )
        assert resources["hidden"] == hidden, "resources hidden size mismatch"
        assert resources["dtype"] == expert_output.dtype, "resources dtype mismatch"
        workspace = resources["workspace"]
        acc = resources["acc"]
        rank_buffers_ptr = resources["rank_buffers_ptr"]
        signal_pads_ptr = resources["signal_pads_ptr"]
        local_pad = resources["local_pad"]
    else:
        (
            workspace,
            acc,
            rank_buffers_ptr,
            signal_pads_ptr,
            local_pad,
        ) = _build_ring_resources(
            group, group_name, expert_output.dtype, data_numel, world_size, rank
        )

    if output is None:
        output = torch.empty(
            (num_tokens_per_rank, hidden),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )

    local_pad.zero_()
    workspace.barrier()

    scatter_idx_c = scatter_idx.contiguous()
    topk_weights_c = topk_weights.contiguous()

    def _invoke(num_wg_hint):
        local_pad.zero_()
        workspace.barrier()
        iteration = _next_iter(group_name)
        torch.ops.symm_mem.ring_reduce_scatter_unpermute(
            expert_output,
            rank_buffers_ptr,
            signal_pads_ptr,
            acc,
            output,
            scatter_idx_c,
            topk_weights_c,
            rank,
            world_size,
            iteration,
            num_wg_hint,
        )
        return output

    num_wg_hint = -1
    if _WG_AUTOTUNE_ENABLED:
        topk = scatter_idx_c.size(1)
        key = _wg_cache_key(
            expert_output.device,
            expert_output.dtype,
            world_size,
            num_tokens_per_rank,
            hidden,
            topk,
        )
        num_wg_hint = autotune_num_wg(
            key, _invoke, num_tokens_per_rank, expert_output.device, group
        )

    _invoke(num_wg_hint)
    return output


def build_ring_reduce_scatter_unpermute_two_stage_resources(
    expert_output: torch.Tensor,
    num_tokens_per_rank: int,
    group: dist.ProcessGroup = None,
):
    """Resources for ring_reduce_scatter_unpermute_two_stage.

    Identical to build_ring_reduce_scatter_unpermute_resources plus a local
    [2, num_tokens_per_rank, hidden] `scratch` ping-pong buffer used by the
    two-stage kernel to prefetch each block's gather one ring step ahead.
    """
    assert (
        _HAS_RING_REDUCE_SCATTER_UNPERMUTE_TWO_STAGE
    ), "ring_reduce_scatter_unpermute_two_stage native kernel not available"
    resources = build_ring_reduce_scatter_unpermute_resources(
        expert_output, num_tokens_per_rank, group=group
    )
    hidden = resources["hidden"]
    resources["scratch"] = torch.empty(
        (2, num_tokens_per_rank, hidden),
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    return resources


def ring_reduce_scatter_unpermute_two_stage(
    expert_output: torch.Tensor,
    scatter_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens_per_rank: int,
    group: dist.ProcessGroup = None,
    resources: dict = None,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Two-stage (software-pipelined) fused MoE unpermute + ring reduce-scatter.

    Same interface and semantics as ring_reduce_scatter_unpermute, but the
    native kernel issues the local weighted gather (Stage A) before waiting on
    the incoming ring partial so the gather overlaps the cross-rank transfer.
    Requires a `scratch` buffer, provided via
    build_ring_reduce_scatter_unpermute_two_stage_resources().
    """
    assert (
        _HAS_RING_REDUCE_SCATTER_UNPERMUTE_TWO_STAGE
    ), "ring_reduce_scatter_unpermute_two_stage native kernel not available"
    group, group_name, rank, world_size = _group_info(group)

    expert_output = expert_output.contiguous()
    hidden = expert_output.size(1)
    data_numel = num_tokens_per_rank * hidden * world_size

    if resources is None:
        resources = build_ring_reduce_scatter_unpermute_two_stage_resources(
            expert_output, num_tokens_per_rank, group=group
        )
    assert resources["group_name"] == group_name, "resources group mismatch"
    assert resources["num_tokens_per_rank"] == num_tokens_per_rank, (
        "resources num_tokens_per_rank mismatch"
    )
    assert resources["hidden"] == hidden, "resources hidden size mismatch"
    assert resources["dtype"] == expert_output.dtype, "resources dtype mismatch"
    assert "scratch" in resources, "two-stage resources missing scratch buffer"

    workspace = resources["workspace"]
    acc = resources["acc"]
    rank_buffers_ptr = resources["rank_buffers_ptr"]
    signal_pads_ptr = resources["signal_pads_ptr"]
    local_pad = resources["local_pad"]
    scratch = resources["scratch"]

    if output is None:
        output = torch.empty(
            (num_tokens_per_rank, hidden),
            dtype=expert_output.dtype,
            device=expert_output.device,
        )

    local_pad.zero_()
    workspace.barrier()

    iteration = _next_iter(group_name)
    torch.ops.symm_mem.ring_reduce_scatter_unpermute_two_stage(
        expert_output,
        rank_buffers_ptr,
        signal_pads_ptr,
        acc,
        scratch,
        output,
        scatter_idx.contiguous(),
        topk_weights.contiguous(),
        rank,
        world_size,
        iteration,
    )
    return output

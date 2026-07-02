"""
Performance benchmark for ring_allgather_permute (fused dispatch: symmetric
-memory single-kernel ring allgather + expert-centric MoE permute).

Compares the on-device fused ring_allgather_permute against an unfused
reference that performs the framework collective dist.all_gather_into_tensor
followed by an indexed scatter (the local permute), and reports a roofline
projection based on the cross-GPU link and HBM bandwidths.

Usage:
    mpirun -n 2 python benchmark_ring_allgather_permute_dist.py
    mpirun -n 4 python benchmark_ring_allgather_permute_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

import env

from ring_collectives import (
    _HAS_RING_ALLGATHER_PERMUTE,
    build_ring_allgather_permute_resources,
    ring_allgather_permute,
)

# Importing this module loads the native local_permute_copy_ op used by the
# reference path (same op as test_allgather_local_permute_fusion_dist.py).
from allgather_local_permute_fusion import _HAS_LOCAL_PERMUTE_KERNEL, compute_scatter_idx

TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
DTYPE = torch.bfloat16

LOOP = 40
WARMUP = 20
ENABLE_PROFILE = False
ENABLE_PROJECTION = True

PCIE_DISCOUNT = 0.85
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count, bw_gbps):
    # GB/s is interpreted as 1e9 bytes/s for bandwidth projection.
    return bytes_count / (bw_gbps * 1e9) * 1e3


def init_distributed():
    env.setup_distributed_env(
        master_addr="127.0.0.1", master_port="29803", overwrite=False
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def build_routing(num_tokens, topk, num_experts, device):
    """Build global routing tables (identical on every rank): topk_idx and
    scatter_idx (absolute expert-sorted destination rows).

    Returns:
        topk_idx: [num_tokens, topk] int32 expert assignments.
        scatter_idx: [num_tokens, topk] int32 absolute destination rows.
        remap_rows: total rows in the expert-sorted output (= num_tokens * topk).
    """
    g = torch.Generator().manual_seed(123)
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, topk), generator=g, dtype=torch.int32
    )

    scatter_idx, _ = compute_scatter_idx(topk_idx.to(device), num_experts)
    remap_rows = num_tokens * topk

    return topk_idx.to(device), scatter_idx.to(device), remap_rows


def benchmark_ring_allgather_permute():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not _HAS_RING_ALLGATHER_PERMUTE:
        if rank == 0:
            print("Native ring_allgather_permute kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden = HIDDEN_SIZE
    topk = TOPK
    num_experts = NUM_EXPERTS
    num_tokens = num_tokens_per_rank * world_size

    topk_idx, scatter_idx, remap_rows = build_routing(
        num_tokens, topk, num_experts, device
    )

    # Global token table is identical on every rank (rank-independent seed), so
    # the all-gathered tokens equal `full_tokens` and the reference can be
    # computed locally; each rank simply contributes its own contiguous shard.
    torch.manual_seed(1234)
    full_tokens = torch.randn(num_tokens, hidden, device=device, dtype=DTYPE)
    shard = full_tokens[rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank].contiguous()

    elem_size = shard.element_size()

    # Reference path: all_gather + native local_permute_copy_ with absolute
    # scatter_idx, writing every (token, k) slot into [num_tokens*topk, hidden].
    all_hidden_ref = torch.empty(num_tokens, hidden, device=device, dtype=DTYPE)
    ref_output = torch.empty(remap_rows, hidden, device=device, dtype=DTYPE)

    def run_reference():
        dist.all_gather_into_tensor(all_hidden_ref, shard, group=group)
        torch.ops.symm_mem.local_permute_copy_(
            all_hidden_ref, scatter_idx, 0, ref_output
        )
        return ref_output

    # Hoist the fused kernel's symmetric workspace / pointer setup and output
    # buffer out of the timed loop so only the device-side work is measured
    # (matches how the reference test hoists rank_buffers_ptr).
    resources = build_ring_allgather_permute_resources(shard, group=group)
    remap_output = torch.empty(
        (remap_rows, hidden), dtype=DTYPE, device=device
    )

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    if ENABLE_PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # Warm up fused ring_allgather_permute.
        for _ in range(WARMUP):
            out = ring_allgather_permute(
                shard, scatter_idx, remap_rows,
                group=group, resources=resources, remap_output=remap_output,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # Timed fused path.
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = ring_allgather_permute(
                shard, scatter_idx, remap_rows,
                group=group, resources=resources, remap_output=remap_output,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up unfused reference (all_gather + native local permute).
        for _ in range(WARMUP):
            run_reference()
        torch.xpu.synchronize()
        dist.barrier()

        # Timed reference path.
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            run_reference()
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [
        ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)
    ]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_ring_allgather_permute_rank{rank}.json")

    # Accuracy check (single fresh pass): every (token, k) writes to its
    # absolute scatter_idx position; must match the reference local_permute_copy_.
    out = ring_allgather_permute(
        shard, scatter_idx, remap_rows, group=group, resources=resources,
    ).clone()
    ref = run_reference().clone()
    torch.xpu.synchronize()
    ok = torch.equal(out, ref)

    print(f"[Ring allgather_permute time in rank {rank}] {latencies} ms")
    print(f"[Reference allgather+local_permute time in rank {rank}] {ref_latencies} ms")

    if rank == 0:
        avg_ring = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"[Accuracy] ring_allgather_permute match={ok} remap_rows={remap_rows}")
        print(
            f"[Summary] avg_fused={avg_ring:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_ring:.3f}x"
        )

        if ENABLE_PROJECTION:
            # Per-rank cross-GPU payload received from peers for the allgather.
            allgather_bytes = (world_size - 1) * num_tokens_per_rank * hidden * elem_size
            # Permute write traffic: one hidden vector per (token, k) slot.
            write_bytes = remap_rows * hidden * elem_size

            proj_allgather_ms = project_time_ms(allgather_bytes, CROSS_GPU_BW_GBPS)
            proj_write_ms = project_time_ms(write_bytes, HBM_BW_GBPS)
            proj_lower_bound = proj_allgather_ms + proj_write_ms

            print(
                f"[Projection] allgather_bytes={bytes_to_mb(allgather_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_allgather_ms:.3f} ms"
            )
            print(
                f"[Projection] permute_write_bytes={bytes_to_mb(write_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_write_ms:.3f} ms"
            )
            print(f"[Projection] fused_lower_bound={proj_lower_bound:.3f} ms")
            print(
                f"[Gap Analysis] actual={avg_ring:.3f} ms vs projected={proj_lower_bound:.3f} ms, "
                f"ratio={avg_ring / proj_lower_bound:.2f}x, "
                f"efficiency={proj_lower_bound / avg_ring * 100:.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ring_allgather_permute()

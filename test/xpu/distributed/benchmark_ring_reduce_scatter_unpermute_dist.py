"""
Performance benchmark for ring_reduce_scatter_unpermute (fused combine:
symmetric-memory single-kernel MoE unpermute + ring reduce-scatter).

Compares the on-device fused ring_reduce_scatter_unpermute against an unfused
reference: native local_unpermute_copy_ (weighted gather, expert-centric ->
token-centric) followed by the framework collective dist.reduce_scatter_tensor.
Reports a roofline projection based on the cross-GPU link bandwidth.

Usage:
    mpirun -n 2 python benchmark_ring_reduce_scatter_unpermute_dist.py
    mpirun -n 4 python benchmark_ring_reduce_scatter_unpermute_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

import env

from ring_collectives import (
    _HAS_RING_REDUCE_SCATTER_UNPERMUTE,
    build_ring_reduce_scatter_unpermute_resources,
    ring_reduce_scatter_unpermute,
)

from allgather_local_permute_fusion import compute_scatter_idx

# Importing this module loads the native local_unpermute_copy_ op used by the
# reference path.
from unpermute_reducescatter_fusion import _HAS_LOCAL_UNPERMUTE_KERNEL

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
        master_addr="127.0.0.1", master_port="29805", overwrite=False
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def build_routing(num_tokens, topk, num_experts, device):
    """Build global routing tables (identical on every rank): topk_idx,
    scatter_idx (absolute expert-sorted destination rows), topk_weights.

    Returns:
        topk_idx: [num_tokens, topk] int32 expert assignments.
        scatter_idx: [num_tokens, topk] int32 absolute destination rows.
        topk_weights: [num_tokens, topk] float32 routing weights.
        remap_rows: total rows in the expert-sorted layout (= num_tokens * topk).
    """
    g = torch.Generator().manual_seed(123)
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, topk), generator=g, dtype=torch.int32
    )
    topk_weights = torch.rand(num_tokens, topk, generator=g, dtype=torch.float32)

    scatter_idx, _ = compute_scatter_idx(topk_idx.to(device), num_experts)
    remap_rows = num_tokens * topk

    return (
        topk_idx.to(device),
        scatter_idx.to(device),
        topk_weights.to(device),
        remap_rows,
    )


def benchmark_ring_reduce_scatter_unpermute():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not _HAS_RING_REDUCE_SCATTER_UNPERMUTE:
        if rank == 0:
            print("Native ring_reduce_scatter_unpermute kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return
    if not _HAS_LOCAL_UNPERMUTE_KERNEL:
        if rank == 0:
            print("Native local_unpermute_copy_ kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden = HIDDEN_SIZE
    topk = TOPK
    num_experts = NUM_EXPERTS
    num_tokens = num_tokens_per_rank * world_size

    topk_idx, scatter_idx, topk_weights, remap_rows = build_routing(
        num_tokens, topk, num_experts, device
    )

    # Expert output: full [num_tokens*topk, hidden] buffer (MoE TP: every rank
    # owns all experts).
    torch.manual_seed(1000)
    expert_output = torch.randn(remap_rows, hidden, device=device, dtype=DTYPE)

    elem_size = expert_output.element_size()

    # Reference path: local_unpermute_copy_ (absolute scatter_idx) +
    # reduce_scatter_tensor.
    def run_reference():
        full_result = torch.empty(num_tokens, hidden, device=device, dtype=DTYPE)
        torch.ops.symm_mem.local_unpermute_copy_(
            expert_output, scatter_idx, topk_weights,
            0, num_tokens, full_result,
        )
        out = torch.zeros(num_tokens_per_rank, hidden, device=device, dtype=DTYPE)
        dist.reduce_scatter_tensor(out, full_result, group=group)
        return out

    # Hoist the fused kernel's symmetric workspace / pointer setup and output
    # buffer out of the timed loop so only the device-side work is measured.
    resources = build_ring_reduce_scatter_unpermute_resources(
        expert_output, num_tokens_per_rank, group=group
    )
    fused_output = torch.empty(num_tokens_per_rank, hidden, device=device, dtype=DTYPE)

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
        # Warm up fused ring_reduce_scatter_unpermute.
        for _ in range(WARMUP):
            out = ring_reduce_scatter_unpermute(
                expert_output, scatter_idx, topk_weights,
                num_tokens_per_rank, group=group, resources=resources,
                output=fused_output,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # Timed fused path.
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = ring_reduce_scatter_unpermute(
                expert_output, scatter_idx, topk_weights,
                num_tokens_per_rank, group=group, resources=resources,
                output=fused_output,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up unfused reference (local_unpermute_copy_ + reduce_scatter).
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
        prof.export_chrome_trace(f"./profile_ring_reduce_scatter_unpermute_rank{rank}.json")

    # Accuracy check: fused vs reference (local_unpermute_copy_ + reduce_scatter).
    out = ring_reduce_scatter_unpermute(
        expert_output, scatter_idx, topk_weights,
        num_tokens_per_rank, group=group, resources=resources,
    ).clone()
    ref = run_reference().clone()
    torch.xpu.synchronize()
    max_err = (out.float() - ref.float()).abs().max().item()
    # The fused ring folds the running partial through `world_size` hops, and for
    # low-precision dtypes (e.g. bf16) the intermediate sum is rounded to dtype on
    # every hop. These sequential roundings accumulate, so the worst-case element
    # error grows with the number of hops. Scale the relative tolerance by
    # sqrt(world_size) to track that growth (fp32 stays well within tolerance).
    rel_tol = 1e-2 * (world_size ** 0.5)
    tol = rel_tol * ref.float().abs().max().clamp_min(1.0).item()
    ok = max_err <= tol

    print(f"[Ring reduce_scatter_unpermute time in rank {rank}] {latencies} ms")
    print(f"[Reference unpermute+reduce_scatter time in rank {rank}] {ref_latencies} ms")

    if rank == 0:
        avg_ring = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"[Accuracy] ring_reduce_scatter_unpermute match={ok} max_err={max_err} tol={tol}")
        print(
            f"[Summary] avg_fused={avg_ring:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_ring:.3f}x"
        )

        if ENABLE_PROJECTION:
            rs_bytes = (world_size - 1) * num_tokens_per_rank * hidden * elem_size
            read_bytes = num_tokens * topk * hidden * elem_size
            write_bytes = num_tokens * hidden * elem_size

            proj_rs_ms = project_time_ms(rs_bytes, CROSS_GPU_BW_GBPS)
            proj_compute_ms = project_time_ms(read_bytes + write_bytes, HBM_BW_GBPS)
            proj_lower_bound = proj_rs_ms + proj_compute_ms

            print(
                f"[Projection] reduce_scatter_bytes={bytes_to_mb(rs_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_rs_ms:.3f} ms"
            )
            print(
                f"[Projection] unpermute_compute (read+write)={bytes_to_mb(read_bytes + write_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_compute_ms:.3f} ms"
            )
            print(f"[Projection] fused_lower_bound={proj_lower_bound:.3f} ms")
            print(
                f"[Gap Analysis] actual={avg_ring:.3f} ms vs projected={proj_lower_bound:.3f} ms, "
                f"ratio={avg_ring / proj_lower_bound:.2f}x, "
                f"efficiency={proj_lower_bound / avg_ring * 100:.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ring_reduce_scatter_unpermute()
